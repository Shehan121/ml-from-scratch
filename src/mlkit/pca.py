"""Principal component analysis via SVD."""

from __future__ import annotations

import numpy as np

__all__ = ["PCA"]


class PCA:
    """Project onto the directions of greatest variance.

    Computed by **SVD of the centred data**, not by eigendecomposition of the
    covariance matrix. The two are mathematically equivalent; the SVD route is
    numerically better, because forming `X^T X` squares the condition number and
    can lose half the available precision. It also avoids building a d x d matrix,
    which matters when d is large.

    Centring is mandatory and easy to forget. PCA finds directions of maximum
    variance *about the mean*; skip the centring and the first component points at
    the mean itself, which carries no information about the spread.

    Scaling is a judgement call rather than a rule. On features with different
    units, variance is not comparable across them and the largest-unit feature
    dominates - so standardise first. On features already in the same units
    (pixel intensities) standardising can amplify near-constant noise dimensions.
    """

    def __init__(self, n_components: int | None = None) -> None:
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, float)
        n_samples, n_features = X.shape

        self.mean_ = X.mean(axis=0)
        centred = X - self.mean_

        # full_matrices=False keeps Vt at (min(n,d), d) rather than (d, d).
        U, S, Vt = np.linalg.svd(centred, full_matrices=False)

        # Sign convention: the SVD is unique only up to a sign flip per component,
        # so different libraries can return components pointing opposite ways.
        # Forcing the largest-magnitude entry of each component positive makes the
        # output deterministic and comparable against scikit-learn.
        max_abs = np.argmax(np.abs(Vt), axis=1)
        signs = np.sign(Vt[np.arange(len(Vt)), max_abs])
        Vt = Vt * signs[:, None]

        # Variance along each component. The (n-1) divisor is the unbiased
        # estimator, matching np.var(ddof=1) and scikit-learn.
        variance = (S**2) / (n_samples - 1)
        total = variance.sum()

        k = self.n_components or min(n_samples, n_features)
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]
        self.explained_variance_ = variance[:k]
        self.explained_variance_ratio_ = variance[:k] / total
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("call fit before transform")
        return (np.asarray(X, float) - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Back to the original space - lossy unless every component was kept.

        The reconstruction error is exactly the variance in the discarded
        components, which is what makes PCA a principled compression rather than
        an arbitrary projection.
        """
        if self.components_ is None:
            raise RuntimeError("call fit before inverse_transform")
        return np.asarray(Z, float) @ self.components_ + self.mean_

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Mean squared error from projecting and reconstructing."""
        return float(np.mean((np.asarray(X, float) - self.inverse_transform(self.transform(X))) ** 2))
