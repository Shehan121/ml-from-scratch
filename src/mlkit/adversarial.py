"""Adversarial examples against the from-scratch network.

Where "learning ML" meets "securing ML". A model can reach high test accuracy and
still be broken by input perturbations too small to see - not because it is
undertrained, but because of how it generalises.

The Fast Gradient Sign Method (Goodfellow et al., 2014) is the simplest
demonstration. Training moves the *weights* down the loss gradient; FGSM moves the
*input* up it:

    x_adv = x + epsilon * sign(dL/dx)

Two details carry the insight:

* **`sign`, not the raw gradient.** The perturbation is constrained by its
  L-infinity norm - no single feature may move more than epsilon - so the optimal
  move under that budget is to push every feature to its limit in the direction
  that increases loss. Magnitude is irrelevant; only direction matters.
* **It needs `dL/dx`.** Backpropagation already computes this on its way to the
  first layer and normally discards it. The attack is not a new algorithm - it is
  a by-product of training, which is exactly why it is so cheap.

`scripts/run_experiments.py` measures accuracy against epsilon on real digits.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fgsm", "pgd", "accuracy_under_attack"]


def fgsm(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
    clip: tuple[float, float] | None = None,
) -> np.ndarray:
    """One-step L-infinity attack. Returns the perturbed inputs.

    `clip` keeps the result inside the valid input range - without it an
    "adversarial image" can contain impossible pixel values, which makes the
    attack look stronger than it is because the inputs are no longer images.
    """
    X = np.asarray(X, float)
    gradient = model.input_gradient(X, y)
    adversarial = X + epsilon * np.sign(gradient)

    if clip is not None:
        adversarial = np.clip(adversarial, clip[0], clip[1])
    return adversarial


def pgd(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
    step_size: float = 0.02,
    n_steps: int = 10,
    clip: tuple[float, float] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Projected gradient descent - FGSM applied iteratively.

    Strictly stronger than FGSM for the same epsilon, because a single large step
    overshoots the local geometry while many small steps follow the curvature. The
    projection back into the epsilon-ball after each step is what keeps the total
    perturbation within budget; without it this is just untargeted optimisation and
    the "adversarial" input drifts arbitrarily far from the original.

    Starting from a random point inside the ball, rather than from `x` itself,
    avoids a degenerate starting gradient and is standard practice.
    """
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    adversarial = X + rng.uniform(-epsilon, epsilon, size=X.shape)

    for _ in range(n_steps):
        gradient = model.input_gradient(adversarial, y)
        adversarial = adversarial + step_size * np.sign(gradient)

        # Project back into the L-infinity ball around the original input.
        adversarial = np.clip(adversarial, X - epsilon, X + epsilon)
        if clip is not None:
            adversarial = np.clip(adversarial, clip[0], clip[1])

    return adversarial


def accuracy_under_attack(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilons: list[float],
    attack: str = "fgsm",
    clip: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    """Accuracy at each perturbation budget - the robustness curve.

    epsilon = 0 recovers clean accuracy, so the first point doubles as a check
    that the attack machinery is not corrupting inputs on its own.
    """
    from mlkit.metrics import accuracy

    out: list[tuple[float, float]] = []
    for epsilon in epsilons:
        if epsilon == 0:
            X_adv = X
        elif attack == "fgsm":
            X_adv = fgsm(model, X, y, epsilon, clip)
        elif attack == "pgd":
            X_adv = pgd(model, X, y, epsilon, step_size=epsilon / 4, n_steps=10, clip=clip)
        else:
            raise ValueError(f"unknown attack: {attack}")
        out.append((epsilon, accuracy(y, model.predict(X_adv))))
    return out
