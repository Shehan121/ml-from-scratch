"""Gradient descent variants, in the order they were invented.

Each one fixes a specific failure of the previous. Reading them as a sequence is
more useful than reading them as a menu:

SGD       follows the gradient. Slow through ravines, where the surface is much
          steeper across than along, so the iterate zig-zags.
Momentum  accumulates a velocity, damping the zig-zag and accelerating along
          consistent directions.
Adam      keeps a *per-parameter* step size from the gradient's own history, so
          rarely-updated parameters still move.

`scripts/run_experiments.py` measures all three on the same network and seed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SGD", "Momentum", "Adam", "make_optimizer"]


class SGD:
    """Plain gradient descent: `p -= lr * grad`.

    One hyperparameter and no state. The learning rate is doing everything, which
    is why it is so sensitive: too small and training crawls, too large and the
    loss diverges. `reports/` shows both failure modes on the same network.
    """

    def __init__(self, learning_rate: float = 0.1) -> None:
        self.learning_rate = learning_rate

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        for p, g in zip(params, grads):
            # In-place so the layers keep their references. p = p - lr*g would
            # rebind the local name and silently update nothing.
            p -= self.learning_rate * g


class Momentum:
    """SGD with a velocity term.

        v = beta * v + grad
        p -= lr * v

    The velocity is an exponentially weighted sum of past gradients. Components
    that keep pointing the same way accumulate; components that oscillate cancel.
    With beta = 0.9 the effective step along a consistent direction is roughly
    1/(1-beta) = 10x larger than plain SGD's, which is where the speed-up comes
    from.
    """

    def __init__(self, learning_rate: float = 0.1, beta: float = 0.9) -> None:
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocities: list[np.ndarray] | None = None

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            self.velocities[i] = self.beta * self.velocities[i] + g
            p -= self.learning_rate * self.velocities[i]


class Adam:
    """Adaptive moment estimation - momentum plus per-parameter scaling.

    Tracks two exponential moving averages: `m` of the gradient (a mean, like
    momentum) and `v` of its square (an uncentred variance). The update divides by
    sqrt(v), so a parameter with consistently large gradients takes proportionally
    smaller steps and a rarely-updated one still moves.

    **The bias correction is not optional.** Both averages start at zero, so early
    estimates are biased toward zero - at t=1 with beta1=0.9, `m` is only 10% of
    the true gradient. Dividing by `1 - beta^t` corrects it. Omitting the
    correction gives a near-frozen first few hundred steps, a bug that looks like
    a bad learning rate. `tests/test_optimizers.py` asserts the corrected first
    step is close to the plain SGD step.

    Defaults beta1=0.9, beta2=0.999, eps=1e-8 are the paper's, and work well
    enough that they are rarely tuned.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: list[np.ndarray] | None = None
        self.v: list[np.ndarray] | None = None
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g**2)

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # eps sits outside the sqrt, matching the reference implementation.
            p -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


def make_optimizer(name: str, learning_rate: float):
    """Build an optimiser by name, with a sensible default rate per method.

    Adam's default is 100x smaller than SGD's, which is not arbitrary: because it
    normalises by the gradient magnitude, its steps are roughly unit-scaled, so a
    rate that suits SGD makes Adam wildly overshoot.
    """
    name = name.lower()
    if name == "sgd":
        return SGD(learning_rate)
    if name == "momentum":
        return Momentum(learning_rate)
    if name == "adam":
        return Adam(learning_rate if learning_rate < 0.01 else 0.001)
    raise ValueError(f"unknown optimizer: {name}")
