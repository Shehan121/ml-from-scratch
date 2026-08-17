"""Linear regression, solved three ways, plus ridge.

The three solvers exist to be *compared*. They optimise the same objective and
should land on the same weights, so agreement between them is a correctness check
you can run without a reference implementation — and where they disagree tells you
something real about conditioning or learning rates.

===================  ==================  =========================
solver               cost                notes
===================  ==================  =========================
normal equation      O(d^3 + n d^2)      exact, no hyperparameters
batch gradient       O(iters * n * d)    scales to large n
stochastic gradient  O(iters * d)        noisy, cheapest per step
===================  ==================  =========================
"""

from __future__ import annotations

import numpy as np

from mlkit.preprocessing import add_bias

__all__ = ["LinearRegression", "Ridge", "SGDRegressor"]


class LinearRegression:
    """Ordinary least squares.

    Two solvers behind one interface:

    ``normal``    solve the normal equations directly — exact, and the right
                  choice whenever the feature count is small enough to invert.
    ``gradient``  batch gradient descent — needed once d is large enough that a
                  d x d solve is impractical, and the mechanism every neural
                  network uses.
    """

    def __init__(
        self,
        solver: str = "normal",
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tol: float = 1e-9,
    ) -> None:
        if solver not in {"normal", "gradient"}:
            raise ValueError("solver must be 'normal' or 'gradient'")
        self.solver = solver
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol

        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.loss_history_: list[float] = []
        self.n_iter_: int = 0

    # ---- fitting ---------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        if self.solver == "normal":
            self._fit_normal(X, y)
        else:
            self._fit_gradient(X, y)
        return self

    def _fit_normal(self, X: np.ndarray, y: np.ndarray) -> None:
        r"""Solve :math:`(X^T X) w = X^T y`.

        ``np.linalg.lstsq`` rather than ``inv(X.T @ X) @ X.T @ y``. The textbook
        formula is a genuine numerical trap: forming ``X^T X`` squares the
        condition number, so a merely awkward problem becomes an unsolvable one,
        and the matrix is singular outright whenever features are collinear or
        d > n. ``lstsq`` goes through an SVD, which handles both — returning the
        minimum-norm solution instead of raising.
        """
        Xb = add_bias(X)
        weights, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]
        self.n_iter_ = 1

    def _fit_gradient(self, X: np.ndarray, y: np.ndarray) -> None:
        """Batch gradient descent on mean squared error.

        The gradient of MSE is ``(2/n) * X^T (Xw - y)``. The ``2/n`` matters: drop
        the ``1/n`` and the step size becomes dependent on the dataset size, so a
        learning rate tuned on 100 rows explodes on 100,000.
        """
        n, d = X.shape
        Xb = add_bias(X)
        weights = np.zeros(d + 1)
        self.loss_history_ = []

        for iteration in range(self.n_iterations):
            residual = Xb @ weights - y
            gradient = (2.0 / n) * (Xb.T @ residual)
            weights -= self.learning_rate * gradient

            loss = float(np.mean(residual**2))
            self.loss_history_.append(loss)
            self.n_iter_ = iteration + 1

            # Stop when progress stalls, rather than always burning every
            # iteration. Checking the loss delta is cheaper than a gradient norm.
            if iteration > 0 and abs(self.loss_history_[-2] - loss) < self.tol:
                break

        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]

    # ---- prediction ------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("call fit before predict")
        return np.asarray(X, float) @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from mlkit.metrics import r2_score

        return r2_score(y, self.predict(X))


class Ridge:
    r"""Least squares with an L2 penalty: minimise
    :math:`\|Xw - y\|^2 + \alpha \|w\|^2`.

    Two things the penalty buys:

    * **Solvability.** Adding ``alpha * I`` makes ``X^T X`` invertible even when
      features are collinear or d > n. Ridge was invented for this, before
      overfitting was the headline motivation.
    * **Variance reduction.** Shrinking weights trades a little bias for less
      variance, which usually improves test error.

    The intercept is deliberately **not** penalised. Shrinking it would pull
    predictions toward zero rather than toward the data's mean, which is not what
    regularisation is for — hence the zeroed first entry of the penalty matrix.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        Xb = add_bias(X)

        penalty = self.alpha * np.eye(Xb.shape[1])
        penalty[0, 0] = 0.0  # never regularise the intercept

        weights = np.linalg.solve(Xb.T @ Xb + penalty, Xb.T @ y)
        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("call fit before predict")
        return np.asarray(X, float) @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from mlkit.metrics import r2_score

        return r2_score(y, self.predict(X))


class SGDRegressor:
    """Stochastic gradient descent — one sample (or mini-batch) per update.

    Batch gradient descent computes an exact gradient over all n samples for each
    step. SGD estimates it from a handful, so each step is far cheaper and far
    noisier. The noise is not purely a cost: it lets the iterate escape shallow
    regions that a deterministic path would settle into, which is why SGD rather
    than full-batch descent trains real networks.

    The learning rate is decayed as ``lr / (1 + decay * epoch)``. With a constant
    rate the noise never lets the iterate settle, and the loss plateaus above the
    minimum — visible in ``reports/figures``.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 50,
        batch_size: int = 1,
        decay: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.decay = decay
        self.seed = seed

        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGDRegressor":
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n, d = X.shape
        Xb = add_bias(X)

        rng = np.random.default_rng(self.seed)
        weights = np.zeros(d + 1)
        self.loss_history_ = []

        for epoch in range(self.n_epochs):
            # Reshuffling each epoch matters: a fixed order lets the model see
            # the same correlated run of samples at the same point every pass.
            order = rng.permutation(n)
            rate = self.learning_rate / (1.0 + self.decay * epoch)

            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                Xi, yi = Xb[batch], y[batch]
                residual = Xi @ weights - yi
                gradient = (2.0 / len(batch)) * (Xi.T @ residual)
                weights -= rate * gradient

            self.loss_history_.append(float(np.mean((Xb @ weights - y) ** 2)))

        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("call fit before predict")
        return np.asarray(X, float) @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from mlkit.metrics import r2_score

        return r2_score(y, self.predict(X))
