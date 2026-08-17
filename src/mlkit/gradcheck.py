"""Verify analytical gradients against finite differences.

The tool that makes hand-derived backpropagation trustworthy. A wrong gradient
rarely errors - the network still trains, just worse - so without this check a
subtle sign or transpose mistake is nearly invisible.

The method: for each parameter, perturb it by +h and -h, measure the loss both
times, and compare the resulting slope with what backpropagation claimed.

    numerical = (L(p + h) - L(p - h)) / (2h)

The **central** difference is used rather than the forward difference
`(L(p+h) - L(p))/h`. Central differences have error O(h^2) against the forward
version's O(h); at h = 1e-5 that is the difference between ~1e-10 and ~1e-5 error,
which decides whether a real bug is distinguishable from floating-point noise.

Choosing h is a genuine trade-off. Too large and the difference quotient stops
approximating the derivative; too small and catastrophic cancellation in
`L(p+h) - L(p-h)` destroys the precision. Around 1e-5 balances them for float64.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["numerical_gradient", "check_gradients", "relative_error"]


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Scale-free difference: |a-b| / max(|a|, |b|).

    An absolute difference is useless here because gradient magnitudes vary by
    orders of magnitude across layers. The conventional reading:

        < 1e-7   correct
        < 1e-5   probably fine for float64
        > 1e-3   almost certainly a bug
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    denominator = np.maximum(np.abs(a), np.abs(b))
    denominator = np.where(denominator == 0, 1.0, denominator)
    return float(np.max(np.abs(a - b) / denominator))


def numerical_gradient(loss_fn: Callable[[], float], param: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of `loss_fn` with respect to `param`.

    `param` is modified in place and restored, so `loss_fn` must read the same
    array object rather than a copy - which is why the layers here expose their
    parameter arrays directly.
    """
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        index = it.multi_index
        original = param[index]

        param[index] = original + h
        plus = loss_fn()

        param[index] = original - h
        minus = loss_fn()

        param[index] = original  # restore before moving on
        grad[index] = (plus - minus) / (2 * h)
        it.iternext()

    return grad


def check_gradients(
    model,
    X: np.ndarray,
    y: np.ndarray,
    h: float = 1e-5,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> dict[str, float]:
    """Check every parameter array of `model` against finite differences.

    Returns a mapping of parameter name to relative error. `model` must expose
    `loss_and_grads(X, y)` which performs a forward and backward pass and leaves
    gradients on the layers.
    """
    from mlkit.neural_net import Dense

    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]

    # Snapshot *every* analytical gradient before perturbing anything.
    #
    # Capturing them lazily per layer is a subtle trap, and one this checker
    # originally fell into. `numerical_gradient` calls the loss repeatedly, and
    # each call runs a full forward *and backward* pass, overwriting every layer's
    # gradient buffer. By the time the sweep reached the second layer, its `dW`
    # held values from the last perturbed pass of the first layer rather than from
    # the clean state.
    #
    # The symptom was diagnostic: layer 0 checked out at 1e-8 while layer 1 came
    # in at 2.5e-4 — small enough to look like a plausible numerical tolerance,
    # large enough to hide a real bug. A gradient checker that is itself subtly
    # wrong is worse than none, because it grants false confidence.
    model.loss_and_grads(X, y)
    analytic = [(layer.dW.copy(), layer.db.copy()) for layer in dense_layers]

    results: dict[str, float] = {}

    for index, layer in enumerate(dense_layers):
        for name, param, expected in (("W", layer.W, analytic[index][0]), ("b", layer.b, analytic[index][1])):
            numeric = numerical_gradient(lambda: model.loss_and_grads(X, y), param, h)
            error = relative_error(expected, numeric)
            key = f"layer{index}.{name}"
            results[key] = error
            if verbose:
                verdict = "OK" if error < tolerance else "FAIL"
                print(f"  {key:<14} rel err {error:.3e}  {verdict}")

    return results
