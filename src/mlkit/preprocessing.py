"""Data preparation, and the leakage traps built into getting it wrong.

The most consequential idea in this file is that a scaler is **fitted on training
data only**. Fitting on the full dataset lets test-set statistics influence the
training features, which inflates every score that follows. It is the most common
way a model looks better than it is, and it is invisible unless you look for it —
``reports/`` measures the size of the inflation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["StandardScaler", "MinMaxScaler", "train_test_split", "one_hot", "KFold", "add_bias"]


class StandardScaler:
    """Centre to zero mean and scale to unit variance.

    Required by anything distance-based (k-NN, k-means) or gradient-based
    (logistic regression, neural nets). Without it a feature measured in
    thousands dominates one measured in fractions purely because of its units,
    and gradient descent zig-zags down a badly conditioned surface.

    Tree models do not need it — they split on thresholds, and any monotonic
    rescaling gives the same splits.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # A constant feature has zero variance; dividing by it gives NaN and
        # silently poisons every downstream computation. Substituting 1 leaves
        # such a column centred at zero, which is the correct no-information state.
        self.scale_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("call fit before transform")
        return (np.asarray(X, float) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, float) * self.scale_ + self.mean_


class MinMaxScaler:
    """Squash each feature into [0, 1].

    Preferable to standardisation when a bounded range is needed — pixel
    intensities, or the inputs to a saturating activation. Far more sensitive to
    outliers, though: one extreme value compresses everything else into a narrow
    band near zero.
    """

    def __init__(self) -> None:
        self.min_: np.ndarray | None = None
        self.range_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        X = np.asarray(X, float)
        self.min_ = X.min(axis=0)
        span = X.max(axis=0) - self.min_
        self.range_ = np.where(span == 0, 1.0, span)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("call fit before transform")
        return (np.asarray(X, float) - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    seed: int = 0,
    stratify: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train and test sets.

    ``stratify`` preserves each class's proportion in both halves. It matters
    more than it looks: on a small or imbalanced dataset a random split can leave
    a class almost absent from training, and the resulting score says more about
    the split than the model.
    """
    X, y = np.asarray(X), np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(X)

    if not stratify:
        indices = rng.permutation(n)
        cut = int(round(n * (1 - test_size)))
        train, test = indices[:cut], indices[cut:]
    else:
        train_parts, test_parts = [], []
        for label in np.unique(y):
            group = np.flatnonzero(y == label)
            group = rng.permutation(group)
            cut = int(round(len(group) * (1 - test_size)))
            train_parts.append(group[:cut])
            test_parts.append(group[cut:])
        train = rng.permutation(np.concatenate(train_parts))
        test = rng.permutation(np.concatenate(test_parts))

    return X[train], X[test], y[train], y[test]


def one_hot(y: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Integer labels to one-hot rows.

    Needed because a softmax output layer produces a probability per class, and
    the cross-entropy loss compares it against a distribution. Feeding raw
    integers would imply class 2 is "twice" class 1, which is meaningless for
    nominal labels.
    """
    y = np.asarray(y, dtype=int)
    n_classes = n_classes or int(y.max()) + 1
    encoded = np.zeros((len(y), n_classes))
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones so the intercept is just another weight.

    Lets the closed-form solution and the gradient update treat the bias
    identically instead of carrying a special case for it.
    """
    X = np.asarray(X, float)
    return np.hstack([np.ones((len(X), 1)), X])


class KFold:
    """K-fold cross-validation splits.

    Every sample is used for validation exactly once, so the score is averaged
    over k models instead of resting on one arbitrary split. That matters most
    when data is scarce, which is exactly when a single split is least reliable.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, seed: int = 0) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed

    def split(self, X: np.ndarray):
        n = len(X)
        indices = np.arange(n)
        if self.shuffle:
            indices = np.random.default_rng(self.seed).permutation(indices)

        # Distribute the remainder so fold sizes differ by at most one, rather
        # than dumping it all into the final fold.
        sizes = np.full(self.n_splits, n // self.n_splits)
        sizes[: n % self.n_splits] += 1

        start = 0
        for size in sizes:
            validation = indices[start : start + size]
            train = np.concatenate([indices[:start], indices[start + size :]])
            yield train, validation
            start += size
