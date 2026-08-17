"""Preprocessing, and the data leakage that scaling wrongly causes."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler as SkScaler

from mlkit.preprocessing import KFold, MinMaxScaler, StandardScaler, add_bias, one_hot, train_test_split


class TestScalers:
    def test_matches_sklearn(self, breast_cancer):
        X, _ = breast_cancer
        assert np.allclose(StandardScaler().fit_transform(X), SkScaler().fit_transform(X))

    def test_output_is_standardised(self, breast_cancer):
        X, _ = breast_cancer
        Z = StandardScaler().fit_transform(X)
        assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(Z.std(axis=0), 1.0, atol=1e-10)

    def test_constant_feature_does_not_produce_nan(self):
        """Zero variance would divide by zero and poison everything downstream."""
        X = np.c_[np.array([1.0, 2, 3]), np.ones(3)]
        Z = StandardScaler().fit_transform(X)
        assert np.all(np.isfinite(Z))
        assert np.allclose(Z[:, 1], 0.0)

    def test_inverse_transform_round_trips(self, iris):
        X, _ = iris
        scaler = StandardScaler().fit(X)
        assert np.allclose(scaler.inverse_transform(scaler.transform(X)), X)

    def test_minmax_maps_to_unit_range(self, iris):
        X, _ = iris
        Z = MinMaxScaler().fit_transform(X)
        assert np.allclose(Z.min(axis=0), 0.0) and np.allclose(Z.max(axis=0), 1.0)

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            StandardScaler().transform(np.zeros((2, 2)))

    def test_test_set_is_not_forced_to_zero_mean(self, breast_cancer):
        """The signature of a correctly applied scaler.

        Fitted on train only, the test set's mean is *near* zero but not exactly
        zero. If it came out at exactly zero, the scaler had seen the test data -
        which is the leak this whole file exists to prevent.
        """
        X, y = breast_cancer
        Xtr, Xte, _, _ = train_test_split(X, y, seed=0)
        scaler = StandardScaler().fit(Xtr)
        test_mean = np.abs(scaler.transform(Xte).mean(axis=0)).max()
        assert 0 < test_mean < 0.5


class TestSplit:
    def test_sizes_and_no_overlap(self, breast_cancer):
        X, y = breast_cancer
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, seed=0)
        assert len(Xtr) + len(Xte) == len(X)
        assert len(ytr) + len(yte) == len(y)
        # Every row must appear exactly once across the two halves.
        combined = np.vstack([Xtr, Xte])
        assert len(np.unique(combined, axis=0)) == len(np.unique(X, axis=0))

    def test_is_deterministic_for_a_given_seed(self, iris):
        X, y = iris
        a = train_test_split(X, y, seed=7)
        b = train_test_split(X, y, seed=7)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[3], b[3])

    def test_different_seeds_give_different_splits(self, iris):
        X, y = iris
        assert not np.array_equal(train_test_split(X, y, seed=1)[0], train_test_split(X, y, seed=2)[0])

    def test_stratify_preserves_class_proportions(self):
        """On imbalanced data an unstratified split can starve a class."""
        y = np.r_[np.zeros(180), np.ones(20)].astype(int)
        X = np.arange(200).reshape(-1, 1).astype(float)

        _, _, ytr, yte = train_test_split(X, y, test_size=0.25, seed=0, stratify=True)
        assert ytr.mean() == pytest.approx(0.1, abs=0.02)
        assert yte.mean() == pytest.approx(0.1, abs=0.02)


class TestKFold:
    def test_every_sample_validates_exactly_once(self):
        X = np.arange(97).reshape(-1, 1)
        seen = np.concatenate([validation for _, validation in KFold(5, seed=0).split(X)])
        assert np.array_equal(np.sort(seen), np.arange(97))

    def test_train_and_validation_never_overlap(self):
        X = np.arange(50).reshape(-1, 1)
        for train, validation in KFold(5, seed=0).split(X):
            assert not set(train) & set(validation)

    def test_fold_sizes_differ_by_at_most_one(self):
        """97 into 5 folds must be 20,20,19,19,19 - not 19,19,19,19,21."""
        X = np.arange(97).reshape(-1, 1)
        sizes = [len(v) for _, v in KFold(5, seed=0).split(X)]
        assert max(sizes) - min(sizes) <= 1
        assert sum(sizes) == 97

    def test_rejects_too_few_splits(self):
        with pytest.raises(ValueError):
            KFold(n_splits=1)


class TestHelpers:
    def test_one_hot(self):
        encoded = one_hot(np.array([0, 2, 1]), n_classes=3)
        assert np.array_equal(encoded, np.eye(3)[[0, 2, 1]])
        assert np.allclose(encoded.sum(axis=1), 1.0)

    def test_add_bias_prepends_ones(self):
        Xb = add_bias(np.zeros((4, 2)))
        assert Xb.shape == (4, 3)
        assert np.all(Xb[:, 0] == 1.0)
