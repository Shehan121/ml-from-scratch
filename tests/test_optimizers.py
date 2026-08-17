"""Optimiser behaviour, including the bias correction that is easy to omit."""

import numpy as np
import pytest

from mlkit.optimizers import SGD, Adam, Momentum, make_optimizer


def quadratic_grad(p):
    """Gradient of sum((p - 3)^2), whose minimum is at p = 3."""
    return 2 * (p - 3.0)


def descend(optimizer, steps=400, start=0.0):
    p = np.array([start])
    for _ in range(steps):
        optimizer.step([p], [quadratic_grad(p)])
    return p[0]


class TestConvergence:
    @pytest.mark.parametrize(
        "optimizer",
        [SGD(0.05), Momentum(0.01, 0.9), Adam(0.05)],
        ids=["sgd", "momentum", "adam"],
    )
    def test_all_reach_the_minimum(self, optimizer):
        assert descend(optimizer) == pytest.approx(3.0, abs=1e-2)

    def test_updates_happen_in_place(self):
        """The layers hold references to their parameter arrays, so an optimiser
        that rebinds instead of mutating would silently train nothing."""
        p = np.array([1.0, 2.0])
        before = p.copy()
        SGD(0.1).step([p], [np.ones(2)])
        assert not np.array_equal(p, before)

    def test_momentum_outpaces_sgd_at_the_same_rate(self):
        """The accumulated velocity is worth roughly 1/(1-beta) in step size."""
        rate, steps = 0.01, 40
        assert abs(descend(Momentum(rate, 0.9), steps) - 3.0) < abs(descend(SGD(rate), steps) - 3.0)

    def test_sgd_diverges_when_the_rate_is_too_large(self):
        """For this quadratic, any rate above 1.0 overshoots and grows."""
        assert abs(descend(SGD(1.1), steps=50)) > 1e3


class TestAdam:
    def test_bias_correction_makes_the_first_step_full_sized(self):
        """Without dividing by (1 - beta^t), the first step is ~10% of intended.

        At t=1 the corrected update is lr * g/|g| = lr * sign(g), regardless of
        gradient magnitude - that scale invariance is the whole point of Adam.
        """
        p = np.array([1.0])
        gradient = np.array([5.0])
        Adam(learning_rate=0.1).step([p], [gradient])
        assert 1.0 - p[0] == pytest.approx(0.1, abs=1e-6)

    def test_step_size_is_scale_invariant(self):
        """A gradient 1000x larger produces essentially the same first step.

        On the first step m_hat = g and v_hat = g^2, so the update reduces to
        lr * g / (|g| + eps) - independent of |g|, which is what lets Adam use one
        learning rate across layers whose gradients differ by orders of magnitude.
        """
        moves = []
        for magnitude in (1.0, 100.0, 1000.0):
            p = np.array([0.0])
            Adam(learning_rate=0.01).step([p], [np.array([magnitude])])
            moves.append(-p[0])
        assert all(m == pytest.approx(moves[0], rel=1e-6) for m in moves)

    def test_epsilon_breaks_scale_invariance_for_tiny_gradients(self):
        """The limit of the previous property, and why eps is not free.

        eps = 1e-8 sits in the denominator to prevent division by zero, so it
        perturbs the step by roughly eps/|g|. For a gradient of 1.0 that is 1e-8
        and invisible; for 0.001 it is 1e-5 and measurable. Scale invariance is
        therefore approximate, and degrades exactly where gradients are smallest -
        the vanishing-gradient regime where you most want it to hold.
        """
        steps = {}
        for magnitude in (0.001, 1.0):
            p = np.array([0.0])
            Adam(learning_rate=0.01).step([p], [np.array([magnitude])])
            steps[magnitude] = -p[0]

        assert steps[1.0] == pytest.approx(0.01, rel=1e-6)
        # Measurably below the ideal step, but only in the fifth decimal place.
        assert steps[0.001] < steps[1.0]
        assert steps[0.001] == pytest.approx(0.01, rel=1e-3)

    def test_state_is_lazily_shaped_to_the_parameters(self):
        adam = Adam()
        assert adam.m is None
        adam.step([np.zeros((2, 3))], [np.ones((2, 3))])
        assert adam.m[0].shape == (2, 3)


class TestFactory:
    @pytest.mark.parametrize("name,cls", [("sgd", SGD), ("momentum", Momentum), ("adam", Adam)])
    def test_builds_by_name(self, name, cls):
        assert isinstance(make_optimizer(name, 0.1), cls)

    def test_adam_gets_a_smaller_default_rate(self):
        """A rate that suits SGD makes Adam overshoot, because Adam normalises."""
        assert make_optimizer("adam", 0.1).learning_rate < 0.1

    def test_rejects_unknown_name(self):
        with pytest.raises(ValueError):
            make_optimizer("rmsprop", 0.1)
