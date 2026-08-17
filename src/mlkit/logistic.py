"""Logistic regression, binary and multiclass.

Despite the name it is a *classifier*, and the thing worth understanding is why
you cannot simply run linear regression on 0/1 labels: the output is unbounded,
squared error on a probability is not convex in the weights, and predictions like
-0.3 or 1.4 are meaningless.

The sigmoid solves the range problem and cross-entropy solves the optimisation
one. The pairing is not arbitrary — together they produce a gradient of exactly
``X^T (p - y)``, identical in form to linear regression's. That algebraic
coincidence is the whole reason the method is tractable.
"""

from __future__ import annotations

import numpy as np

from mlkit.preprocessing import add_bias, one_hot

__all__ = ["LogisticRegression", "SoftmaxRegression", "sigmoid", "softmax"]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    ``1 / (1 + exp(-z))`` overflows for large negative ``z``, giving a warning and
    then a NaN that propagates silently through training. The branch below
    rewrites the negative case as ``exp(z) / (1 + exp(z))``, which is
    mathematically identical and never exponentiates a positive number.

    Getting this wrong is one of the most common causes of a from-scratch model
    that trains for a while and then produces NaN loss.
    """
    z = np.asarray(z, float)
    out = np.empty_like(z)
    positive = z >= 0

    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted for stability.

    Subtracting the row maximum before exponentiating leaves the result unchanged
    — the constant cancels in the ratio — while guaranteeing the largest exponent
    is ``exp(0) = 1``. Without it, logits around 1000 overflow to ``inf`` and the
    division yields NaN.
    """
    z = np.asarray(z, float)
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


class LogisticRegression:
    r"""Binary logistic regression by gradient descent.

    Minimises mean cross-entropy
    :math:`-\frac{1}{n}\sum y\log p + (1-y)\log(1-p)`, optionally with an L2
    penalty.

    The gradient works out to ``(1/n) X^T (p - y)`` — the same shape as least
    squares, with the prediction passed through a sigmoid. It is worth deriving
    once: the sigmoid's derivative ``p(1-p)`` cancels exactly against the
    denominator that cross-entropy's derivative produces. Pair a sigmoid with
    squared error instead and that cancellation is lost, leaving a gradient that
    vanishes whenever the model is confidently wrong — which is precisely when you
    need it to be large.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 1000,
        l2: float = 0.0,
        tol: float = 1e-9,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2 = l2
        self.tol = tol

        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.loss_history_: list[float] = []
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n, d = X.shape
        Xb = add_bias(X)
        weights = np.zeros(d + 1)
        self.loss_history_ = []

        for iteration in range(self.n_iterations):
            p = sigmoid(Xb @ weights)
            error = p - y

            gradient = (Xb.T @ error) / n
            if self.l2:
                # The intercept is excluded, as in Ridge: shrinking it would bias
                # predictions toward p = 0.5 rather than toward the base rate.
                penalty = (self.l2 / n) * weights
                penalty[0] = 0.0
                gradient += penalty

            weights -= self.learning_rate * gradient

            eps = 1e-15
            p_clipped = np.clip(p, eps, 1 - eps)
            loss = float(-np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)))
            self.loss_history_.append(loss)
            self.n_iter_ = iteration + 1

            if iteration > 0 and abs(self.loss_history_[-2] - loss) < self.tol:
                break

        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """The raw logit — useful for ROC curves, which need scores not labels."""
        if self.coef_ is None:
            raise RuntimeError("call fit before predict")
        return np.asarray(X, float) @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Threshold the probability.

        0.5 is a default, not a law. Shifting it trades precision against recall
        without retraining, which is the right lever when one error type costs
        more than the other.
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from mlkit.metrics import accuracy

        return accuracy(y, self.predict(X))


class SoftmaxRegression:
    """Multinomial logistic regression — softmax over K classes.

    The genuine generalisation of binary logistic regression, as opposed to
    one-vs-rest. One weight vector per class, trained jointly, so the outputs are
    a proper probability distribution summing to 1. One-vs-rest trains K
    independent binary models whose scores need ad-hoc normalisation and can all
    be confidently positive at once.

    The gradient is again ``(1/n) X^T (P - Y)`` with Y one-hot — the same
    expression a third time, which is the pattern worth noticing across this
    file.
    """

    def __init__(
        self,
        learning_rate: float = 0.5,
        n_iterations: int = 1000,
        l2: float = 0.0,
        tol: float = 1e-9,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2 = l2
        self.tol = tol

        self.weights_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.loss_history_: list[float] = []
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        X = np.asarray(X, float)
        y = np.asarray(y, dtype=int).ravel()
        n, d = X.shape

        self.n_classes_ = int(y.max()) + 1
        Y = one_hot(y, self.n_classes_)
        Xb = add_bias(X)
        W = np.zeros((d + 1, self.n_classes_))
        self.loss_history_ = []

        for iteration in range(self.n_iterations):
            P = softmax(Xb @ W)
            gradient = Xb.T @ (P - Y) / n

            if self.l2:
                penalty = (self.l2 / n) * W
                penalty[0, :] = 0.0
                gradient += penalty

            W -= self.learning_rate * gradient

            loss = float(-np.mean(np.log(np.clip(P[np.arange(n), y], 1e-15, 1.0))))
            self.loss_history_.append(loss)
            self.n_iter_ = iteration + 1

            if iteration > 0 and abs(self.loss_history_[-2] - loss) < self.tol:
                break

        self.weights_ = W
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("call fit before predict")
        return softmax(add_bias(np.asarray(X, float)) @ self.weights_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from mlkit.metrics import accuracy

        return accuracy(y, self.predict(X))
