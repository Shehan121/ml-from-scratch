"""A CART decision tree, and the impurity measure that drives it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DecisionTreeClassifier", "gini", "entropy"]


def gini(y: np.ndarray, n_classes: int) -> float:
    """Gini impurity: 1 - sum(p^2). Zero when a node is pure.

    Interpretable as the probability of misclassifying a randomly drawn element
    if you labelled it by drawing from the node's own class distribution.
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=n_classes)
    p = counts / len(y)
    return float(1.0 - np.sum(p**2))


def entropy(y: np.ndarray, n_classes: int) -> float:
    """Shannon entropy: -sum(p log2 p). Zero when pure, log2(K) when uniform.

    Behaves very similarly to Gini in practice - the two rarely disagree about the
    best split - but costs a logarithm per class. CART uses Gini by default for
    that reason, not because it is theoretically preferable.
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=n_classes)
    p = counts[counts > 0] / len(y)
    return float(-np.sum(p * np.log2(p)))


@dataclass
class _Node:
    """Either a leaf (`prediction` set) or a split (`feature`/`threshold` set)."""

    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    prediction: int | None = None
    proba: np.ndarray | None = None
    n_samples: int = 0
    impurity: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.prediction is not None


class DecisionTreeClassifier:
    """Binary-split decision tree, grown greedily.

    At each node it tries every (feature, threshold) pair and keeps the one with
    the largest **weighted** impurity decrease. Weighting by child size is
    essential: an unweighted comparison prefers splits that peel off one pure
    sample at a time, which builds a maximally deep and useless tree.

    The greed is worth naming. Choosing the locally best split at each step does
    not give the globally optimal tree - finding that is NP-complete - so CART is
    a heuristic that happens to work well. Unlike Kruskal's algorithm, greedy here
    carries no optimality proof.

    Two properties that follow from splitting on thresholds:

    * **Scale invariance.** Any monotonic rescaling of a feature produces the same
      splits, so no StandardScaler is needed. `tests/` asserts this.
    * **Axis-aligned boundaries only.** A diagonal boundary must be approximated
      by a staircase, which is why a tree needs surprising depth for data a linear
      model separates with one line.

    Left with an unbounded depth it will fit the training set perfectly and
    generalise badly - `reports/` measures that overfitting against `max_depth`.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
    ) -> None:
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion

        self.root_: _Node | None = None
        self.n_classes_: int = 0
        self.n_features_: int = 0

    @property
    def _impurity(self):
        return gini if self.criterion == "gini" else entropy

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = np.asarray(X, float)
        y = np.asarray(y, dtype=int).ravel()
        self.n_classes_ = int(y.max()) + 1
        self.n_features_ = X.shape[1]
        self.root_ = self._grow(X, y, depth=0)
        return self

    def _leaf(self, y: np.ndarray) -> _Node:
        counts = np.bincount(y, minlength=self.n_classes_)
        return _Node(
            prediction=int(np.argmax(counts)),
            proba=counts / len(y),
            n_samples=len(y),
            impurity=self._impurity(y, self.n_classes_),
        )

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        node_impurity = self._impurity(y, self.n_classes_)

        # Stop on a pure node, a depth cap, or too few samples to split.
        if (
            node_impurity == 0.0
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < self.min_samples_split
        ):
            return self._leaf(y)

        feature, threshold, gain = self._best_split(X, y, node_impurity)
        if feature is None or gain <= 0:
            return self._leaf(y)

        mask = X[:, feature] <= threshold
        return _Node(
            feature=feature,
            threshold=threshold,
            left=self._grow(X[mask], y[mask], depth + 1),
            right=self._grow(X[~mask], y[~mask], depth + 1),
            n_samples=len(y),
            impurity=node_impurity,
        )

    def _best_split(self, X: np.ndarray, y: np.ndarray, parent_impurity: float):
        best = (None, None, 0.0)
        n = len(y)

        for feature in range(self.n_features_):
            values = X[:, feature]
            # Candidate thresholds are midpoints between consecutive distinct
            # values. Using the observed values themselves would place the
            # boundary exactly on a data point, and any value between two
            # observations produces an identical partition anyway.
            unique = np.unique(values)
            if len(unique) < 2:
                continue
            thresholds = (unique[:-1] + unique[1:]) / 2.0

            for threshold in thresholds:
                mask = values <= threshold
                left, right = y[mask], y[~mask]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue

                weighted = (
                    len(left) / n * self._impurity(left, self.n_classes_)
                    + len(right) / n * self._impurity(right, self.n_classes_)
                )
                gain = parent_impurity - weighted
                if gain > best[2]:
                    best = (feature, float(threshold), gain)

        return best

    def _walk(self, x: np.ndarray) -> _Node:
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("call fit before predict")
        return np.array([self._walk(x).prediction for x in np.asarray(X, float)])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("call fit before predict")
        return np.array([self._walk(x).proba for x in np.asarray(X, float)])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from mlkit.metrics import accuracy

        return accuracy(y, self.predict(X))

    def depth(self) -> int:
        def go(node: _Node) -> int:
            return 0 if node.is_leaf else 1 + max(go(node.left), go(node.right))

        return go(self.root_) if self.root_ else 0

    def n_leaves(self) -> int:
        def go(node: _Node) -> int:
            return 1 if node.is_leaf else go(node.left) + go(node.right)

        return go(self.root_) if self.root_ else 0
