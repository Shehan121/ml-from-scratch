"""k-means clustering, with k-means++ initialisation."""

from __future__ import annotations

import numpy as np

__all__ = ["KMeans"]


class KMeans:
    """Lloyd's algorithm: assign points to the nearest centroid, move each
    centroid to the mean of its points, repeat.

    Unsupervised, so there are no labels - it partitions by geometry alone. The
    objective is *inertia*, the total squared distance from each point to its
    centroid, and every iteration is guaranteed not to increase it. That
    guarantees convergence but only to a **local** minimum, which is why
    initialisation matters so much.

    `k-means++` seeds centroids far apart by choosing each new one with
    probability proportional to its squared distance from the nearest existing
    centroid. Compared with uniform random seeding it both converges faster and
    lands on better optima - `reports/` measures the difference in final inertia
    across many seeds.

    `n_init` restarts and keeps the best run, because even k-means++ is
    randomised and one unlucky seed can produce a visibly wrong partition.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if init not in {"k-means++", "random"}:
            raise ValueError("init must be 'k-means++' or 'random'")
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = np.inf
        self.n_iter_: int = 0

    # ---- initialisation --------------------------------------------------

    def _init_random(self, X: np.ndarray, rng) -> np.ndarray:
        idx = rng.choice(len(X), size=self.n_clusters, replace=False)
        return X[idx].copy()

    def _init_plus_plus(self, X: np.ndarray, rng) -> np.ndarray:
        centroids = [X[rng.integers(len(X))]]
        for _ in range(1, self.n_clusters):
            squared = np.min(
                ((X[:, None, :] - np.array(centroids)[None, :, :]) ** 2).sum(axis=2), axis=1
            )
            total = squared.sum()
            if total == 0:
                # Every point already coincides with a centroid; pick at random
                # rather than dividing by zero.
                centroids.append(X[rng.integers(len(X))])
                continue
            centroids.append(X[rng.choice(len(X), p=squared / total)])
        return np.array(centroids)

    # ---- fitting ---------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, float)
        if self.n_clusters > len(X):
            raise ValueError("n_clusters cannot exceed the number of samples")

        best_inertia = np.inf
        best = None

        for run in range(self.n_init):
            rng = np.random.default_rng(self.seed + run)
            centroids = (
                self._init_plus_plus(X, rng) if self.init == "k-means++" else self._init_random(X, rng)
            )
            labels, centroids, inertia, iterations = self._lloyd(X, centroids)

            if inertia < best_inertia:
                best_inertia = inertia
                best = (labels, centroids, iterations)

        self.labels_, self.centroids_, self.n_iter_ = best
        self.inertia_ = float(best_inertia)
        return self

    def _lloyd(self, X: np.ndarray, centroids: np.ndarray):
        for iteration in range(self.max_iter):
            distances = self._distances(X, centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()
            for c in range(self.n_clusters):
                members = X[labels == c]
                if len(members):
                    new_centroids[c] = members.mean(axis=0)
                # An empty cluster keeps its old position. Recomputing a mean over
                # zero points gives NaN, which would poison every later distance.

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break

        distances = self._distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        inertia = float((distances[np.arange(len(X)), labels] ** 2).sum())
        return labels, centroids, inertia, iteration + 1

    @staticmethod
    def _distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        squared = (
            (X**2).sum(axis=1)[:, None]
            + (centroids**2).sum(axis=1)[None, :]
            - 2.0 * X @ centroids.T
        )
        return np.sqrt(np.clip(squared, 0.0, None))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("call fit before predict")
        return np.argmin(self._distances(np.asarray(X, float), self.centroids_), axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
