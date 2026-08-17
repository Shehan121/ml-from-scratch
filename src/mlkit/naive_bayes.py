"""Gaussian naive Bayes - and what the word "naive" is doing."""

from __future__ import annotations

import numpy as np

__all__ = ["GaussianNB"]


class GaussianNB:
    """Bayes' rule with an independence assumption and Gaussian likelihoods.

    Predicts argmax over classes of `P(c) * prod_j P(x_j | c)`.

    **"Naive" names the assumption that features are conditionally independent
    given the class.** That is essentially never true - petal length and petal
    width are obviously correlated - and the classifier works well anyway. The
    reason is that it only has to get the *argmax* right, not the probabilities:
    correlated features make the estimates badly overconfident while usually
    leaving their ordering intact. So its accuracy is often decent and its
    probability outputs should not be trusted.

    Two implementation points that are more than details:

    * **Everything is done in log space.** Multiplying d likelihoods each around
      0.01 underflows to exactly 0.0 for even modest d, at which point every
      class ties at zero and the prediction is meaningless. Summing logs instead
      of multiplying probabilities is what makes the method usable at all.
    * **Variance smoothing.** A feature that is constant within a class has zero
      variance, and the Gaussian density then divides by zero. Adding a small
      epsilon proportional to the overall variance keeps it finite.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self.log_prior_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        X = np.asarray(X, float)
        y = np.asarray(y, dtype=int).ravel()
        self.classes_ = np.unique(y)
        n_features = X.shape[1]

        self.log_prior_ = np.zeros(len(self.classes_))
        self.means_ = np.zeros((len(self.classes_), n_features))
        self.variances_ = np.zeros((len(self.classes_), n_features))

        epsilon = self.var_smoothing * X.var(axis=0).max()

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            # The prior is the observed class frequency, stored as a log so the
            # posterior is a sum throughout.
            self.log_prior_[i] = np.log(len(Xc) / len(X))
            self.means_[i] = Xc.mean(axis=0)
            self.variances_[i] = Xc.var(axis=0) + epsilon

        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """log P(c) + sum_j log P(x_j | c), for every class."""
        X = np.asarray(X, float)
        out = np.zeros((len(X), len(self.classes_)))

        for i in range(len(self.classes_)):
            var = self.variances_[i]
            # log of the Gaussian density, written out rather than exponentiated
            # and logged again.
            normaliser = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            deviation = -0.5 * np.sum(((X - self.means_[i]) ** 2) / var, axis=1)
            out[:, i] = self.log_prior_[i] + normaliser + deviation

        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("call fit before predict")
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Normalise the log-likelihoods with the log-sum-exp trick.

        Subtracting the row maximum before exponentiating is the same stabilising
        move as in softmax: without it, log-likelihoods around -800 (routine for
        30 features) exponentiate to 0.0 and the normalisation is 0/0.
        """
        jll = self._joint_log_likelihood(X)
        shifted = jll - jll.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from mlkit.metrics import accuracy

        return accuracy(y, self.predict(X))
