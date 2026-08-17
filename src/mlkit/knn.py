"""k-nearest neighbours - the classifier with no training step."""

from __future__ import annotations

import numpy as np

__all__ = ["KNeighborsClassifier"]


class KNeighborsClassifier:
    """Classify by majority vote among the k closest training points.

    A *lazy* learner: `fit` only stores the data. All the work happens at
    prediction time, which inverts the usual cost profile - training is O(1) and
    every prediction is O(n*d). That makes it excellent for prototyping and
    unusable for high-throughput serving.

    Two properties worth internalising:

    * **It needs scaled features.** Distance sums squared differences across
      dimensions, so a feature measured in thousands drowns out one measured in
      fractions. Unlike a tree, k-NN is not invariant to rescaling - the
      experiments in `reports/` measure how much accuracy this costs.
    * **k controls the bias-variance trade-off directly.** k=1 fits the training
      set perfectly and is highly sensitive to noise; large k smooths the
      boundary until it underfits. There is no other knob.
    """

    def __init__(self, k: int = 5, weights: str = "uniform") -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.k = k
        self.weights = weights
        self.X_: np.ndarray | None = None
        self.y_: np.ndarray | None = None
        self.n_classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNeighborsClassifier":
        self.X_ = np.asarray(X, float)
        self.y_ = np.asarray(y, dtype=int).ravel()
        self.n_classes_ = int(self.y_.max()) + 1
        return self

    def _distances(self, X: np.ndarray) -> np.ndarray:
        """Pairwise Euclidean distances, vectorised.

        Uses the identity ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, which turns the
        whole computation into one matrix product. A Python double loop over
        samples is the obvious implementation and is roughly two orders of
        magnitude slower - `reports/` measures the gap.

        The clip guards against tiny negative values from floating-point
        cancellation, which would otherwise make sqrt return NaN for points that
        are nearly identical.
        """
        X = np.asarray(X, float)
        squared = (
            (X**2).sum(axis=1)[:, None]
            + (self.X_**2).sum(axis=1)[None, :]
            - 2.0 * X @ self.X_.T
        )
        return np.sqrt(np.clip(squared, 0.0, None))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_ is None:
            raise RuntimeError("call fit before predict")
        distances = self._distances(X)

        # argpartition finds the k smallest in O(n) per row rather than sorting
        # all n in O(n log n). At k=5 and n=10,000 that is a real saving.
        k = min(self.k, len(self.X_))
        neighbours = np.argpartition(distances, k - 1, axis=1)[:, :k]

        proba = np.zeros((len(distances), self.n_classes_))
        for i, idx in enumerate(neighbours):
            labels = self.y_[idx]
            if self.weights == "uniform":
                weights = np.ones(len(idx))
            else:
                # Inverse distance, so closer neighbours count for more. The
                # epsilon prevents division by zero when a test point coincides
                # exactly with a training point.
                weights = 1.0 / (distances[i, idx] + 1e-12)
            np.add.at(proba[i], labels, weights)

        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from mlkit.metrics import accuracy

        return accuracy(y, self.predict(X))
