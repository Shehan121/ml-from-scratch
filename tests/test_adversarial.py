"""Adversarial robustness of the from-scratch network."""

import numpy as np
import pytest
from sklearn.datasets import load_digits

from mlkit.adversarial import accuracy_under_attack, fgsm, pgd
from mlkit.metrics import accuracy
from mlkit.neural_net import MLPClassifier
from mlkit.preprocessing import MinMaxScaler, train_test_split


@pytest.fixture(scope="module")
def trained():
    """A genuinely accurate model, so any collapse is caused by the attack.

    Scaled to [0, 1] so pixel values stay in a valid range after perturbation and
    epsilon is interpretable as a fraction of the full intensity scale.
    """
    d = load_digits()
    X = MinMaxScaler().fit_transform(d.data)
    Xtr, Xte, ytr, yte = train_test_split(X, d.target, test_size=0.3, seed=0, stratify=True)
    model = MLPClassifier(hidden=(64,), learning_rate=0.5, n_epochs=60, batch_size=32, seed=0)
    model.fit(Xtr, ytr)
    return model, Xte, yte


class TestFGSM:
    def test_model_is_accurate_before_attack(self, trained):
        model, Xte, yte = trained
        assert model.score(Xte, yte) > 0.92

    def test_attack_reduces_accuracy(self, trained):
        model, Xte, yte = trained
        clean = model.score(Xte, yte)
        attacked = accuracy(yte, model.predict(fgsm(model, Xte, yte, epsilon=0.1, clip=(0, 1))))
        assert attacked < clean

    def test_perturbation_respects_the_epsilon_budget(self, trained):
        """The L-infinity constraint: no feature may move more than epsilon."""
        model, Xte, yte = trained
        epsilon = 0.08
        X_adv = fgsm(model, Xte, yte, epsilon=epsilon)
        assert np.max(np.abs(X_adv - Xte)) <= epsilon + 1e-12

    def test_clipping_keeps_inputs_valid(self, trained):
        model, Xte, yte = trained
        X_adv = fgsm(model, Xte, yte, epsilon=0.5, clip=(0.0, 1.0))
        assert X_adv.min() >= 0.0 and X_adv.max() <= 1.0

    def test_larger_epsilon_is_at_least_as_damaging(self, trained):
        model, Xte, yte = trained
        curve = accuracy_under_attack(model, Xte, yte, [0.0, 0.05, 0.1, 0.2], clip=(0, 1))
        accuracies = [a for _, a in curve]
        # Allow a tiny non-monotonicity; FGSM is a single linear step, not optimal.
        assert accuracies[0] > accuracies[-1]
        assert accuracies[-1] < 0.5

    def test_epsilon_zero_is_the_clean_input(self, trained):
        """A check that the attack machinery is not corrupting inputs by itself."""
        model, Xte, yte = trained
        curve = accuracy_under_attack(model, Xte, yte, [0.0], clip=(0, 1))
        assert curve[0][1] == pytest.approx(model.score(Xte, yte))

    def test_sign_is_what_matters_not_magnitude(self, trained):
        """FGSM uses sign(grad), so every perturbation entry is exactly +/-eps."""
        model, Xte, yte = trained
        delta = fgsm(model, Xte, yte, epsilon=0.1) - Xte
        assert np.allclose(np.abs(delta), 0.1)


class TestPGD:
    def test_stronger_than_fgsm_at_equal_budget(self, trained):
        """Many small steps follow the curvature better than one large one."""
        model, Xte, yte = trained
        epsilon = 0.1
        fgsm_acc = accuracy(yte, model.predict(fgsm(model, Xte, yte, epsilon, clip=(0, 1))))
        pgd_acc = accuracy(
            yte, model.predict(pgd(model, Xte, yte, epsilon, step_size=0.025, n_steps=10, clip=(0, 1)))
        )
        assert pgd_acc <= fgsm_acc

    def test_projection_keeps_the_perturbation_in_budget(self, trained):
        """Ten steps of 0.025 could travel 0.25; the projection caps it at 0.1."""
        model, Xte, yte = trained
        epsilon = 0.1
        X_adv = pgd(model, Xte, yte, epsilon, step_size=0.025, n_steps=10)
        assert np.max(np.abs(X_adv - Xte)) <= epsilon + 1e-9
