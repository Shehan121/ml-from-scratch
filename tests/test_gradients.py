"""Verify every analytical gradient against finite differences.

The most important test file here. A wrong gradient does not raise - the network
still trains, just worse - so hand-derived backpropagation is only trustworthy
once it has been checked numerically.
"""

import numpy as np
import pytest

from mlkit.gradcheck import check_gradients, numerical_gradient, relative_error
from mlkit.neural_net import Dense, MLPClassifier, ReLU, Sigmoid, Tanh, softmax_cross_entropy


@pytest.fixture
def small_problem():
    """Deliberately tiny: gradient checking is O(number of parameters) forward
    passes, so a real network would take minutes."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 4))
    y = rng.integers(0, 3, size=8)
    return X, y


class TestGradCheckItself:
    """The checker has to be verified before it can verify anything."""

    def test_recovers_a_known_derivative(self):
        """d/dx of sum(x^2) is 2x."""
        x = np.array([1.0, -2.0, 3.5])
        grad = numerical_gradient(lambda: float(np.sum(x**2)), x)
        assert np.allclose(grad, 2 * x, atol=1e-6)

    def test_recovers_a_known_matrix_derivative(self):
        """d/dW of sum(W) is all ones."""
        W = np.arange(6, dtype=float).reshape(2, 3)
        grad = numerical_gradient(lambda: float(np.sum(W)), W)
        assert np.allclose(grad, np.ones((2, 3)), atol=1e-6)

    def test_restores_the_parameter(self):
        """The perturbation must leave no trace, or later checks are corrupted."""
        x = np.array([1.0, 2.0, 3.0])
        original = x.copy()
        numerical_gradient(lambda: float(np.sum(x**3)), x)
        assert np.array_equal(x, original)

    def test_relative_error_is_scale_free(self):
        assert relative_error(np.array([1e6]), np.array([1e6])) == 0.0
        assert relative_error(np.array([1e-8]), np.array([1e-8])) == 0.0
        assert relative_error(np.array([1.0]), np.array([2.0])) == pytest.approx(0.5)

    def test_detects_a_deliberately_wrong_gradient(self):
        """If the checker cannot fail, it is not testing anything."""
        assert relative_error(np.array([1.0, 2.0]), np.array([1.0, -2.0])) > 1e-3


class TestLayerGradients:
    def test_dense_gradients(self):
        rng = np.random.default_rng(1)
        layer = Dense(4, 3, seed=0)
        x = rng.normal(size=(6, 4))
        upstream = rng.normal(size=(6, 3))

        out = layer.forward(x)
        layer.backward(upstream)

        # A scalar loss is needed for finite differences; sum(out * upstream)
        # has exactly `upstream` as its derivative with respect to `out`.
        def loss():
            return float(np.sum(layer.forward(x) * upstream))

        assert relative_error(layer.dW, numerical_gradient(loss, layer.W)) < 1e-7
        assert relative_error(layer.db, numerical_gradient(loss, layer.b)) < 1e-7

    def test_dense_input_gradient(self):
        """dL/dx is what gets passed to the previous layer - and to an attacker."""
        rng = np.random.default_rng(2)
        layer = Dense(4, 3, seed=0)
        x = rng.normal(size=(5, 4))
        upstream = rng.normal(size=(5, 3))

        layer.forward(x)
        analytic = layer.backward(upstream)

        def loss():
            return float(np.sum(layer.forward(x) * upstream))

        assert relative_error(analytic, numerical_gradient(loss, x)) < 1e-7

    @pytest.mark.parametrize("activation_cls", [ReLU, Sigmoid, Tanh])
    def test_activation_gradients(self, activation_cls):
        rng = np.random.default_rng(3)
        layer = activation_cls()
        # Avoid values near zero for ReLU, whose derivative is undefined there and
        # where a finite difference straddles the kink and gives a wrong answer.
        x = rng.normal(size=(6, 4)) + np.sign(rng.normal(size=(6, 4))) * 0.5
        upstream = rng.normal(size=(6, 4))

        layer.forward(x)
        analytic = layer.backward(upstream)

        def loss():
            return float(np.sum(layer.forward(x) * upstream))

        assert relative_error(analytic, numerical_gradient(loss, x)) < 1e-6

    def test_softmax_cross_entropy_gradient(self):
        """The fused loss - where p - y comes from."""
        rng = np.random.default_rng(4)
        logits = rng.normal(size=(7, 4))
        y = rng.integers(0, 4, size=7)

        _, analytic = softmax_cross_entropy(logits, y)

        def loss():
            return softmax_cross_entropy(logits, y)[0]

        assert relative_error(analytic, numerical_gradient(loss, logits)) < 1e-7

    def test_relu_gradient_is_zero_where_input_was_negative(self):
        layer = ReLU()
        x = np.array([[-2.0, 3.0, -0.5, 1.0]])
        layer.forward(x)
        grad = layer.backward(np.ones_like(x))
        assert np.array_equal(grad, np.array([[0.0, 1.0, 0.0, 1.0]]))


class TestNetworkGradients:
    """End-to-end: the whole stack's gradients, not just individual layers."""

    @pytest.mark.parametrize("hidden", [(5,), (6, 4), (8, 6, 4)])
    def test_mlp_gradients_match_finite_differences(self, small_problem, hidden):
        X, y = small_problem
        model = MLPClassifier(hidden=hidden, seed=0)
        model._build(X.shape[1], 3)

        errors = check_gradients(model, X, y)
        assert errors, "no parameters were checked"
        worst = max(errors.values())
        assert worst < 1e-6, f"worst relative error {worst:.2e} in {errors}"

    @pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
    def test_gradients_correct_for_every_activation(self, small_problem, activation):
        X, y = small_problem
        model = MLPClassifier(hidden=(6, 4), activation=activation, seed=1)
        model._build(X.shape[1], 3)
        assert max(check_gradients(model, X, y).values()) < 1e-6

    def test_l2_penalty_gradient_is_also_correct(self, small_problem):
        """Regularisation adds a term to both the loss and the gradient; if only
        one is updated the check catches it immediately."""
        X, y = small_problem
        model = MLPClassifier(hidden=(6,), l2=0.1, seed=2)
        model._build(X.shape[1], 3)
        assert max(check_gradients(model, X, y).values()) < 1e-6

    def test_input_gradient_matches_finite_differences(self, small_problem):
        """The gradient adversarial attacks rely on."""
        X, y = small_problem
        model = MLPClassifier(hidden=(6,), seed=3)
        model._build(X.shape[1], 3)

        analytic = model.input_gradient(X, y)
        Xc = X.copy()

        def loss():
            return softmax_cross_entropy(model.forward(Xc, training=True), y)[0]

        assert relative_error(analytic, numerical_gradient(loss, Xc)) < 1e-6


class TestNumericalStability:
    """Regression tests for two real failures found while building this."""

    def test_relu_propagates_nan_rather_than_laundering_it(self):
        """`np.where(x > 0, x, 0)` would silently turn NaN into 0.0.

        That is not a cosmetic difference. When a too-large learning rate drove
        the first layer's weights to NaN, a `np.where`-based ReLU replaced them
        with clean zeros, so downstream layers saw valid input and the loss stayed
        finite at ln(10) while the model predicted at chance. The numerical failure
        was invisible to any guard watching the loss.
        """
        layer = ReLU()
        out = layer.forward(np.array([[np.nan, 1.0, -1.0]]))
        assert np.isnan(out[0, 0]), "NaN must propagate, not become 0.0"
        assert out[0, 1] == 1.0
        assert out[0, 2] == 0.0

    def test_nan_fraction_distinguishes_divergence_from_dead_units(self):
        """`dead_fraction` alone cannot tell the two apart, because NaN > 0 is False."""
        layer = ReLU()
        layer.forward(np.array([[np.nan, np.nan, np.nan, np.nan]]))
        assert layer.dead_fraction() == 1.0      # looks exactly like dying ReLU
        assert layer.nan_fraction() == 1.0       # but this reveals the real cause

        layer.forward(np.array([[-1.0, -2.0, -3.0, -4.0]]))
        assert layer.dead_fraction() == 1.0      # genuinely dead
        assert layer.nan_fraction() == 0.0

    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_divergence_raises_instead_of_returning_a_chance_level_model(self):
        """A silent failure is worse than an exception.

        lr=0.5 on a depth-3 ReLU network overflows the weights around epoch 8.
        Before the guard existed, `fit` completed and returned a model scoring 10%
        on 10 classes, which is easily mistaken for an architecture problem.
        """
        from sklearn.datasets import load_digits

        from mlkit.preprocessing import StandardScaler, train_test_split

        digits = load_digits()
        Xtr, _, ytr, _ = train_test_split(digits.data, digits.target, seed=0, stratify=True)
        Xtr = StandardScaler().fit_transform(Xtr)

        with pytest.raises(FloatingPointError, match="diverged"):
            MLPClassifier(
                hidden=(32, 32, 32), activation="relu", learning_rate=0.5,
                n_epochs=60, batch_size=32, seed=0,
            ).fit(Xtr, ytr)

    def test_healthy_training_does_not_trip_the_guard(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 6))
        y = (X[:, 0] > 0).astype(int)
        model = MLPClassifier(hidden=(16,), learning_rate=0.1, n_epochs=30, seed=0).fit(X, y)
        assert np.all(np.isfinite(model.loss_history_))
        assert model.score(X, y) > 0.9
