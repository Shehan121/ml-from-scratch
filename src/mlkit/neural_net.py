"""A multilayer perceptron with backpropagation, written out layer by layer.

This is the centre of the project. Backpropagation is not a separate algorithm —
it is the chain rule applied to a composition of functions, organised so that
each layer receives the gradient of the loss with respect to its own output and
returns the gradient with respect to its input.

Each layer therefore implements exactly two methods:

``forward(x)``      compute the output, caching whatever ``backward`` will need
``backward(grad)``  given dL/d(output), return dL/d(input) and store dL/d(params)

Every gradient in this file is verified numerically in
``tests/test_gradients.py`` — a finite-difference check against the analytical
derivative. Hand-derived backprop that merely *looks* right is the single
easiest way to spend a day debugging a network that trains slowly for no visible
reason, so the derivation is checked rather than trusted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from mlkit.logistic import softmax
from mlkit.preprocessing import one_hot

__all__ = [
    "Dense",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "MLPClassifier",
    "softmax_cross_entropy",
]


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------


class Layer(ABC):
    """A layer with a forward pass and a gradient-returning backward pass."""

    @abstractmethod
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray: ...

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray: ...

    @property
    def params(self) -> list[np.ndarray]:
        return []

    @property
    def grads(self) -> list[np.ndarray]:
        return []


class Dense(Layer):
    r"""Fully connected layer: :math:`y = xW + b`.

    The three gradients, all following from the chain rule on that one expression:

    .. code-block:: text

        dL/dW = x^T  @ dL/dy        (n_in  x n_out)
        dL/db = sum(dL/dy, axis=0)  (n_out,)     -- summed over the batch
        dL/dx = dL/dy @ W^T         (batch x n_in) -- passed to the previous layer

    The bias gradient is *summed* over the batch rather than averaged because the
    same bias contributed to every sample in it. Averaging here while summing in
    ``dL/dW`` is a classic mismatch that makes the bias learn at a different
    effective rate from the weights.

    **He initialisation** (``sqrt(2 / n_in)``) is used because these networks use
    ReLU. Initialising all weights to zero would make every neuron in a layer
    compute the same thing and receive the same gradient forever — the symmetry is
    never broken, and the layer behaves as a single neuron. Too large a scale
    saturates activations instead. The ``2`` in the numerator compensates for ReLU
    discarding half its input; a tanh network wants Xavier's ``1 / n_in``.
    """

    def __init__(self, n_in: int, n_out: int, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._x = x  # needed for dL/dW
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.dW = self._x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T

    @property
    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    @property
    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]


class ReLU(Layer):
    """``max(0, x)``.

    Its derivative is 1 where the input was positive and 0 elsewhere — cheap, and
    crucially it does not shrink the gradient on the active path. Sigmoid's
    derivative peaks at 0.25, so a ten-layer sigmoid network multiplies gradients
    by at most 0.25^10 ≈ 1e-6 on the way back: the vanishing-gradient problem, and
    the reason ReLU replaced sigmoid in hidden layers.

    The cost is dead neurons — a unit whose input is always negative receives zero
    gradient forever and never recovers. ``dead_fraction`` reports how many are in
    that state, and ``reports/`` measures it against the learning rate.

    The gradient at exactly 0 is undefined; 0 is the conventional choice.
    """

    def __init__(self) -> None:
        self._positive: np.ndarray | None = None
        self._last_input: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._last_input = x
        self._positive = x > 0
        # np.maximum, not np.where(x > 0, x, 0.0). The two agree on every finite
        # input and differ on NaN: `np.where` treats `NaN > 0` as False and
        # substitutes a clean 0.0, laundering the NaN away, while np.maximum
        # propagates it.
        #
        # That laundering caused a real failure here. When a too-large learning
        # rate drove the first layer's weights to NaN, this layer quietly replaced
        # them with zeros, so the rest of the network saw valid input, the loss
        # stayed finite at ln(10), and the model simply predicted at chance. A
        # divergence guard watching the loss could not see it. Propagating the NaN
        # makes the failure loud, which is the whole point.
        return np.maximum(x, 0.0)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._positive

    def dead_fraction(self) -> float:
        """Fraction of units that produced no output on the last forward pass.

        .. warning::

           Read this together with :meth:`nan_fraction`. ``NaN > 0`` evaluates to
           ``False``, so a network whose weights have diverged to NaN reports
           **100% dead** here — which looks exactly like dying ReLU and is a
           completely different failure.

           That misdiagnosis actually happened while building this project: a
           depth-3 ReLU network collapsed to chance accuracy, this method returned
           100% for every layer, and the obvious conclusion was dying ReLU. Tracing
           the loss per epoch showed the real cause — a loss spike at epoch 7 blew
           the weights up by 5x, and they became NaN at epoch 8.
        """
        if self._positive is None:
            return 0.0
        return float(1.0 - self._positive.mean())

    def nan_fraction(self) -> float:
        """Fraction of the last pre-activation that was not finite.

        Non-zero here means the network has diverged, and any ``dead_fraction``
        reading should be disregarded.
        """
        if self._last_input is None:
            return 0.0
        return float(np.mean(~np.isfinite(self._last_input)))


class Sigmoid(Layer):
    """``1 / (1 + exp(-x))``, with derivative ``s(1 - s)``.

    Caching the output rather than the input is the useful trick: the derivative is
    expressible entirely in terms of the value already computed, so the backward
    pass needs no exponentials at all.
    """

    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        from mlkit.logistic import sigmoid

        self._out = sigmoid(x)
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._out * (1.0 - self._out)


class Tanh(Layer):
    """``tanh(x)``, with derivative ``1 - tanh^2(x)``.

    Zero-centred, unlike sigmoid, which keeps the mean activation near zero and
    makes the following layer's optimisation better conditioned. Its derivative
    still peaks at 1.0 and decays, so deep tanh stacks vanish too — just less
    sharply than sigmoid.
    """

    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1.0 - self._out**2)


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    r"""Combined softmax + cross-entropy, returning ``(loss, dL/dlogits)``.

    Fused deliberately, because the composition simplifies to something both
    simpler and more stable than either half:

    .. math::  \frac{\partial L}{\partial z} = \frac{p - y}{n}

    Implementing softmax and cross-entropy as separate layers means computing the
    softmax Jacobian (a full K x K matrix per sample) and then multiplying by the
    loss derivative, where terms cancel to give this same expression. Doing it
    separately is more work, less numerically stable, and produces the identical
    answer.

    That ``p - y`` is also the third appearance of the same gradient form in this
    project, after linear and logistic regression — the pattern is not a
    coincidence but a consequence of pairing each output activation with its
    matching log-likelihood loss.
    """
    n = len(logits)
    p = softmax(logits)
    y = np.asarray(y, dtype=int).ravel()

    loss = float(-np.mean(np.log(np.clip(p[np.arange(n), y], 1e-15, 1.0))))

    grad = p.copy()
    grad[np.arange(n), y] -= 1.0
    return loss, grad / n


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------


class MLPClassifier:
    """Multilayer perceptron trained by mini-batch gradient descent.

    ``hidden=(32, 16)`` builds ``Dense(d, 32) → ReLU → Dense(32, 16) → ReLU →
    Dense(16, K)``, with softmax + cross-entropy applied to the final logits.

    The output layer has no activation of its own: the softmax lives inside the
    loss, for the stability reason given above. Applying softmax in the network
    *and* using a cross-entropy that expects probabilities is a common
    double-application bug that produces a model which trains but plateaus early.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (32,),
        activation: str = "relu",
        learning_rate: float = 0.1,
        n_epochs: int = 100,
        batch_size: int = 32,
        optimizer: str = "sgd",
        l2: float = 0.0,
        seed: int = 0,
    ) -> None:
        if activation not in {"relu", "sigmoid", "tanh"}:
            raise ValueError("activation must be relu, sigmoid or tanh")
        self.hidden = hidden
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer_name = optimizer
        self.l2 = l2
        self.seed = seed

        self.layers: list[Layer] = []
        self.loss_history_: list[float] = []
        self.n_classes_: int = 0
        self._optimizer = None

    def _activation_layer(self) -> Layer:
        return {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}[self.activation]()

    def _build(self, n_features: int, n_classes: int) -> None:
        from mlkit.optimizers import make_optimizer

        rng = np.random.default_rng(self.seed)
        sizes = [n_features, *self.hidden]
        self.layers = []

        for i in range(len(sizes) - 1):
            # A distinct seed per layer, so two layers of equal shape do not get
            # identical initial weights.
            self.layers.append(Dense(sizes[i], sizes[i + 1], seed=int(rng.integers(1 << 31))))
            self.layers.append(self._activation_layer())
        self.layers.append(Dense(sizes[-1], n_classes, seed=int(rng.integers(1 << 31))))

        self._optimizer = make_optimizer(self.optimizer_name, self.learning_rate)

    # ---- forward / backward ---------------------------------------------

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        out = np.asarray(X, float)
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _all_params_and_grads(self):
        params, grads = [], []
        for layer in self.layers:
            params.extend(layer.params)
            grads.extend(layer.grads)
        return params, grads

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray) -> float:
        """One forward and backward pass; leaves gradients on the layers.

        Separated out so the gradient checker can drive exactly the same code path
        the optimiser does.
        """
        logits = self.forward(X, training=True)
        loss, grad = softmax_cross_entropy(logits, y)

        if self.l2:
            # Penalise weights but never biases: a bias shifts the decision
            # boundary rather than controlling model complexity.
            weight_norm = sum(float(np.sum(layer.W**2)) for layer in self.layers if isinstance(layer, Dense))
            loss += 0.5 * self.l2 * weight_norm

        self.backward(grad)

        if self.l2:
            for layer in self.layers:
                if isinstance(layer, Dense):
                    layer.dW += self.l2 * layer.W
        return loss

    # ---- training --------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        X = np.asarray(X, float)
        y = np.asarray(y, dtype=int).ravel()
        self.n_classes_ = int(y.max()) + 1
        self._build(X.shape[1], self.n_classes_)

        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.loss_history_ = []

        for _ in range(self.n_epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                loss = self.loss_and_grads(X[batch], y[batch])

                params, grads = self._all_params_and_grads()
                self._optimizer.step(params, grads)

                epoch_loss += loss
                n_batches += 1

            mean_loss = epoch_loss / n_batches
            self.loss_history_.append(mean_loss)

            # Fail loudly on divergence. Without this the run completes and
            # returns a model that predicts uniformly at chance level - a silent
            # failure that is easy to mistake for an architecture problem. The
            # loss going non-finite is unambiguous, so there is no reason to
            # discover it later from the accuracy.
            # Check the parameters, not only the loss. A finite loss is not
            # evidence of a healthy network: an activation can mask non-finite
            # weights upstream and leave the loss looking respectable.
            params, _ = self._all_params_and_grads()
            diverged = not np.isfinite(mean_loss) or any(not np.all(np.isfinite(p)) for p in params)

            if diverged:
                raise FloatingPointError(
                    f"training diverged at epoch {len(self.loss_history_)}: loss "
                    f"{mean_loss:.4g}, non-finite weights present. The learning rate "
                    f"({self.learning_rate}) is too large for this architecture - the "
                    "weights overflowed. Reduce it."
                )

        return self

    # ---- prediction ------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.forward(X, training=False))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X, training=False), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from mlkit.metrics import accuracy

        return accuracy(y, self.predict(X))

    def input_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """dL/dX — the gradient with respect to the *input*, not the weights.

        Normally a by-product thrown away at the first layer. It is what
        adversarial attacks need: it says which way to nudge a sample to increase
        the loss. See :mod:`mlkit.adversarial`.
        """
        logits = self.forward(X, training=True)
        _, grad = softmax_cross_entropy(logits, y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def n_parameters(self) -> int:
        return sum(p.size for p in self._all_params_and_grads()[0])
