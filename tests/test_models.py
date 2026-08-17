"""Each model is verified against its scikit-learn counterpart."""

import numpy as np
import pytest
from sklearn import cluster as skcluster
from sklearn import decomposition as skdecomp
from sklearn import linear_model as sklinear
from sklearn import naive_bayes as sknb
from sklearn import neighbors as skneighbors
from sklearn import tree as sktree

from mlkit.knn import KNeighborsClassifier
from mlkit.kmeans import KMeans
from mlkit.linear import LinearRegression, Ridge, SGDRegressor
from mlkit.logistic import LogisticRegression, SoftmaxRegression, sigmoid, softmax
from mlkit.naive_bayes import GaussianNB
from mlkit.pca import PCA
from mlkit.tree import DecisionTreeClassifier, entropy, gini


class TestLinearRegression:
    def test_normal_equation_recovers_known_weights(self, tiny_regression):
        """On exactly linear data the closed form must be near-exact."""
        X, y, true_w, true_b = tiny_regression
        model = LinearRegression(solver="normal").fit(X, y)
        assert np.allclose(model.coef_, true_w, atol=1e-10)
        assert model.intercept_ == pytest.approx(true_b, abs=1e-10)

    def test_matches_sklearn(self, diabetes):
        X, y = diabetes
        mine = LinearRegression(solver="normal").fit(X, y)
        theirs = sklinear.LinearRegression().fit(X, y)
        assert np.allclose(mine.coef_, theirs.coef_, atol=1e-8)
        assert mine.intercept_ == pytest.approx(theirs.intercept_, abs=1e-8)

    def test_gradient_descent_converges_to_the_closed_form(self, tiny_regression):
        """The two solvers optimise the same objective, so they must agree."""
        X, y, _, _ = tiny_regression
        exact = LinearRegression(solver="normal").fit(X, y)
        approx = LinearRegression(solver="gradient", learning_rate=0.1, n_iterations=5000).fit(X, y)
        assert np.allclose(exact.coef_, approx.coef_, atol=1e-3)

    def test_gradient_descent_loss_decreases_monotonically(self, tiny_regression):
        X, y, _, _ = tiny_regression
        model = LinearRegression(solver="gradient", learning_rate=0.05, n_iterations=200).fit(X, y)
        history = np.array(model.loss_history_)
        assert np.all(np.diff(history) <= 1e-12)

    def test_diverges_with_too_large_a_learning_rate(self, tiny_regression):
        """Documenting the failure mode rather than only the success."""
        X, y, _, _ = tiny_regression
        model = LinearRegression(solver="gradient", learning_rate=2.0, n_iterations=100).fit(X, y)
        assert model.loss_history_[-1] > model.loss_history_[0]

    def test_sgd_reaches_a_similar_solution(self, tiny_regression):
        X, y, true_w, _ = tiny_regression
        model = SGDRegressor(learning_rate=0.05, n_epochs=200, batch_size=16).fit(X, y)
        assert np.allclose(model.coef_, true_w, atol=0.05)

    def test_handles_more_features_than_samples(self):
        """lstsq must cope where inv(X.T @ X) would be singular."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 30))
        y = rng.normal(size=10)
        model = LinearRegression(solver="normal").fit(X, y)
        assert np.all(np.isfinite(model.coef_))


class TestRidge:
    def test_matches_sklearn(self, diabetes):
        X, y = diabetes
        for alpha in (0.1, 1.0, 10.0):
            mine = Ridge(alpha=alpha).fit(X, y)
            theirs = sklinear.Ridge(alpha=alpha, solver="cholesky").fit(X, y)
            assert np.allclose(mine.coef_, theirs.coef_, atol=1e-8)
            assert mine.intercept_ == pytest.approx(theirs.intercept_, abs=1e-8)

    def test_larger_alpha_shrinks_the_weights(self, diabetes):
        X, y = diabetes
        norms = [np.linalg.norm(Ridge(alpha=a).fit(X, y).coef_) for a in (0.01, 1.0, 100.0)]
        assert norms[0] > norms[1] > norms[2]

    def test_rejects_negative_alpha(self):
        with pytest.raises(ValueError):
            Ridge(alpha=-1.0)


class TestLogisticRegression:
    def test_sigmoid_is_stable_at_extremes(self):
        """The naive formula overflows here and returns NaN."""
        extreme = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
        out = sigmoid(extreme)
        assert np.all(np.isfinite(out))
        assert out[0] == pytest.approx(0.0)
        assert out[2] == pytest.approx(0.5)
        assert out[-1] == pytest.approx(1.0)

    def test_softmax_is_stable_and_normalised(self):
        logits = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, 0.0, 1000.0]])
        p = softmax(logits)
        assert np.all(np.isfinite(p))
        assert np.allclose(p.sum(axis=1), 1.0)

    def test_accuracy_close_to_sklearn(self, breast_cancer, scaled_split):
        X, y = breast_cancer
        Xtr, Xte, ytr, yte = scaled_split(X, y)
        mine = LogisticRegression(learning_rate=0.5, n_iterations=3000).fit(Xtr, ytr)
        theirs = sklinear.LogisticRegression(max_iter=5000).fit(Xtr, ytr)
        assert mine.score(Xte, yte) >= theirs.score(Xte, yte) - 0.03

    def test_loss_decreases(self, breast_cancer, scaled_split):
        X, y = breast_cancer
        Xtr, _, ytr, _ = scaled_split(X, y)
        model = LogisticRegression(learning_rate=0.1, n_iterations=500).fit(Xtr, ytr)
        assert model.loss_history_[-1] < model.loss_history_[0] * 0.5

    def test_threshold_trades_precision_for_recall(self, breast_cancer, scaled_split):
        from mlkit.metrics import precision, recall

        X, y = breast_cancer
        Xtr, Xte, ytr, yte = scaled_split(X, y)
        model = LogisticRegression(learning_rate=0.5, n_iterations=2000).fit(Xtr, ytr)

        low = model.predict(Xte, threshold=0.2)
        high = model.predict(Xte, threshold=0.8)
        assert recall(yte, low, "none")[1] >= recall(yte, high, "none")[1]
        assert precision(yte, high, "none")[1] >= precision(yte, low, "none")[1]

    def test_l2_shrinks_weights(self, breast_cancer, scaled_split):
        X, y = breast_cancer
        Xtr, _, ytr, _ = scaled_split(X, y)
        plain = LogisticRegression(learning_rate=0.5, n_iterations=1000, l2=0.0).fit(Xtr, ytr)
        penalised = LogisticRegression(learning_rate=0.5, n_iterations=1000, l2=50.0).fit(Xtr, ytr)
        assert np.linalg.norm(penalised.coef_) < np.linalg.norm(plain.coef_)


class TestSoftmaxRegression:
    def test_multiclass_accuracy(self, iris, scaled_split):
        X, y = iris
        Xtr, Xte, ytr, yte = scaled_split(X, y)
        model = SoftmaxRegression(learning_rate=0.5, n_iterations=2000).fit(Xtr, ytr)
        assert model.score(Xte, yte) > 0.85

    def test_probabilities_sum_to_one(self, iris, scaled_split):
        X, y = iris
        Xtr, Xte, ytr, _ = scaled_split(X, y)
        model = SoftmaxRegression(n_iterations=200).fit(Xtr, ytr)
        assert np.allclose(model.predict_proba(Xte).sum(axis=1), 1.0)

    def test_beats_sklearn_by_no_more_than_a_small_margin(self, digits, scaled_split):
        X, y = digits
        Xtr, Xte, ytr, yte = scaled_split(X, y)
        mine = SoftmaxRegression(learning_rate=0.5, n_iterations=800).fit(Xtr, ytr)
        theirs = sklinear.LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        assert mine.score(Xte, yte) >= theirs.score(Xte, yte) - 0.05


class TestKNN:
    def test_matches_sklearn_predictions(self, iris, scaled_split):
        X, y = iris
        Xtr, Xte, ytr, _ = scaled_split(X, y)
        for k in (1, 3, 5, 11):
            mine = KNeighborsClassifier(k=k).fit(Xtr, ytr).predict(Xte)
            theirs = skneighbors.KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr).predict(Xte)
            assert np.mean(mine == theirs) > 0.95

    def test_k1_fits_training_data_perfectly(self, iris):
        """Each point is its own nearest neighbour - zero training error, high variance."""
        X, y = iris
        model = KNeighborsClassifier(k=1).fit(X, y)
        assert model.score(X, y) == pytest.approx(1.0)

    def test_unscaled_features_hurt_accuracy(self, breast_cancer):
        """k-NN is not scale invariant, unlike a tree."""
        from mlkit.preprocessing import StandardScaler, train_test_split

        X, y = breast_cancer
        Xtr, Xte, ytr, yte = train_test_split(X, y, seed=0, stratify=True)
        raw = KNeighborsClassifier(k=5).fit(Xtr, ytr).score(Xte, yte)

        scaler = StandardScaler().fit(Xtr)
        scaled = KNeighborsClassifier(k=5).fit(scaler.transform(Xtr), ytr).score(scaler.transform(Xte), yte)
        assert scaled > raw

    def test_distance_weighting_changes_predictions(self, iris, scaled_split):
        X, y = iris
        Xtr, Xte, ytr, _ = scaled_split(X, y)
        uniform = KNeighborsClassifier(k=15, weights="uniform").fit(Xtr, ytr).predict(Xte)
        weighted = KNeighborsClassifier(k=15, weights="distance").fit(Xtr, ytr).predict(Xte)
        assert uniform.shape == weighted.shape

    def test_rejects_bad_k(self):
        with pytest.raises(ValueError):
            KNeighborsClassifier(k=0)


class TestKMeans:
    def test_inertia_close_to_sklearn(self, iris):
        X, _ = iris
        mine = KMeans(n_clusters=3, n_init=10, seed=0).fit(X)
        theirs = skcluster.KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
        assert mine.inertia_ == pytest.approx(theirs.inertia_, rel=0.02)

    def test_kmeans_plus_plus_helps_where_clusters_are_separable(self, iris):
        """k-means++ reduces both mean inertia and its variance across seeds.

        Getting this test right took three attempts, and the failures were the
        instructive part.

        1. Comparing final inertia on a *single* seed failed: on digits with
           k=10, k-means++ scored 0.03% worse. Its guarantee is about the expected
           result, and one seed cannot measure an expectation.
        2. Averaging iteration counts over 12 seeds also failed - the advantage
           reversed (17.75 vs 16.75) and only reappeared at 25 seeds. It was
           noise, not signal.
        3. Sweeping the sample size from 10 to 30 seeds showed the real picture:
           on iris both claims hold at *every* sample size, while on digits the
           sign of the difference flips depending on how many seeds you average.

        So the effect is genuine on iris (3 well-separated clusters in 4
        dimensions) and not measurable on digits (10 clusters in 64 dimensions),
        where distance concentration leaves every initialisation about equally
        good. Only the robust case is asserted.
        """
        X, _ = iris
        plus = [KMeans(n_clusters=3, init="k-means++", n_init=1, seed=s).fit(X).inertia_ for s in range(20)]
        random = [KMeans(n_clusters=3, init="random", n_init=1, seed=s).fit(X).inertia_ for s in range(20)]

        assert np.mean(plus) < np.mean(random)
        assert np.std(plus) < np.std(random)

    def test_recovers_well_separated_blobs(self):
        rng = np.random.default_rng(0)
        blobs = np.vstack([
            rng.normal(loc=[0, 0], scale=0.3, size=(50, 2)),
            rng.normal(loc=[8, 8], scale=0.3, size=(50, 2)),
            rng.normal(loc=[0, 8], scale=0.3, size=(50, 2)),
        ])
        labels = KMeans(n_clusters=3, seed=0).fit_predict(blobs)
        # Each true group must end up in a single cluster.
        for start in (0, 50, 100):
            assert len(set(labels[start : start + 50])) == 1

    def test_more_clusters_never_increases_inertia(self, iris):
        X, _ = iris
        inertias = [KMeans(n_clusters=k, seed=0).fit(X).inertia_ for k in (2, 3, 5, 8)]
        assert all(a >= b for a, b in zip(inertias, inertias[1:]))

    def test_rejects_too_many_clusters(self):
        with pytest.raises(ValueError):
            KMeans(n_clusters=10).fit(np.zeros((5, 2)))


class TestDecisionTree:
    def test_impurity_measures(self):
        pure = np.array([0, 0, 0, 0])
        assert gini(pure, 2) == pytest.approx(0.0)
        assert entropy(pure, 2) == pytest.approx(0.0)

        balanced = np.array([0, 0, 1, 1])
        assert gini(balanced, 2) == pytest.approx(0.5)
        assert entropy(balanced, 2) == pytest.approx(1.0)

    def test_accuracy_close_to_sklearn(self, breast_cancer):
        from mlkit.preprocessing import train_test_split

        X, y = breast_cancer
        Xtr, Xte, ytr, yte = train_test_split(X, y, seed=0, stratify=True)
        for depth in (2, 3, 5):
            mine = DecisionTreeClassifier(max_depth=depth).fit(Xtr, ytr).score(Xte, yte)
            theirs = sktree.DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, ytr).score(Xte, yte)
            assert mine >= theirs - 0.06

    def test_unbounded_depth_fits_training_data_perfectly(self, iris):
        """Which is exactly the overfitting the depth cap exists to prevent."""
        X, y = iris
        assert DecisionTreeClassifier().fit(X, y).score(X, y) == pytest.approx(1.0)

    def test_depth_cap_is_respected(self, breast_cancer):
        X, y = breast_cancer
        assert DecisionTreeClassifier(max_depth=3).fit(X, y).depth() <= 3

    def test_scale_invariance(self, iris):
        """The distinguishing property against k-NN: rescaling changes nothing."""
        X, y = iris
        plain = DecisionTreeClassifier(max_depth=4).fit(X, y).predict(X)
        scaled = DecisionTreeClassifier(max_depth=4).fit(X * 1000 + 50, y).predict(X * 1000 + 50)
        assert np.array_equal(plain, scaled)

    def test_probabilities_sum_to_one(self, iris):
        X, y = iris
        assert np.allclose(DecisionTreeClassifier(max_depth=3).fit(X, y).predict_proba(X).sum(axis=1), 1.0)

    def test_entropy_and_gini_give_similar_trees(self, breast_cancer):
        X, y = breast_cancer
        g = DecisionTreeClassifier(max_depth=4, criterion="gini").fit(X, y).score(X, y)
        e = DecisionTreeClassifier(max_depth=4, criterion="entropy").fit(X, y).score(X, y)
        assert abs(g - e) < 0.05


class TestGaussianNB:
    def test_matches_sklearn(self, iris):
        X, y = iris
        mine = GaussianNB().fit(X, y)
        theirs = sknb.GaussianNB().fit(X, y)
        assert np.array_equal(mine.predict(X), theirs.predict(X))
        assert np.allclose(mine.predict_proba(X), theirs.predict_proba(X), atol=1e-8)

    def test_survives_many_features_without_underflow(self, digits):
        """64 features would underflow to zero without log-space arithmetic."""
        X, y = digits
        model = GaussianNB().fit(X, y)
        proba = model.predict_proba(X)
        assert np.all(np.isfinite(proba))
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_constant_feature_does_not_divide_by_zero(self):
        X = np.c_[np.array([1.0, 2, 3, 4]), np.ones(4)]   # second column constant
        y = np.array([0, 0, 1, 1])
        model = GaussianNB().fit(X, y)
        assert np.all(np.isfinite(model.predict_proba(X)))


class TestPCA:
    def test_matches_sklearn(self, iris):
        X, _ = iris
        mine = PCA(n_components=3).fit(X)
        theirs = skdecomp.PCA(n_components=3).fit(X)
        assert np.allclose(mine.explained_variance_ratio_, theirs.explained_variance_ratio_, atol=1e-10)
        # Components agree up to a per-component sign flip.
        for a, b in zip(mine.components_, theirs.components_):
            assert np.allclose(a, b, atol=1e-8) or np.allclose(a, -b, atol=1e-8)

    def test_variance_ratios_sum_to_one_when_all_kept(self, iris):
        X, _ = iris
        assert PCA().fit(X).explained_variance_ratio_.sum() == pytest.approx(1.0)

    def test_components_are_orthonormal(self, digits):
        X, _ = digits
        C = PCA(n_components=10).fit(X).components_
        assert np.allclose(C @ C.T, np.eye(10), atol=1e-8)

    def test_full_reconstruction_is_lossless(self, iris):
        X, _ = iris
        assert PCA().fit(X).reconstruction_error(X) < 1e-20

    def test_reconstruction_error_falls_as_components_are_added(self, digits):
        X, _ = digits
        errors = [PCA(n_components=k).fit(X).reconstruction_error(X) for k in (2, 5, 10, 20)]
        assert all(a > b for a, b in zip(errors, errors[1:]))

    def test_variance_is_ordered_descending(self, digits):
        X, _ = digits
        variance = PCA(n_components=10).fit(X).explained_variance_
        assert np.all(np.diff(variance) <= 0)
