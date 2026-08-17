"""Evaluation metrics, implemented from the definitions.

Writing these out rather than importing them is the fastest way to stop confusing
precision with recall, and to find out that "accuracy" is close to meaningless on
imbalanced data. Every function here is verified against its scikit-learn
counterpart in ``tests/test_metrics.py``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "accuracy",
    "confusion_matrix",
    "precision",
    "recall",
    "f1",
    "classification_report",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "log_loss",
    "roc_curve",
    "roc_auc",
]


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction correct.

    The default metric, and the one that lies most often. On a dataset that is
    99% one class, predicting that class always scores 0.99 while being useless.
    ``reports/`` demonstrates this rather than just warning about it.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Counts indexed ``[true, predicted]``.

    Row/column order is the thing to get right and easy to invert: entry
    ``[i, j]`` is "actually class i, predicted class j", so the diagonal is
    correct predictions and everything off it is a specific kind of mistake.
    """
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1

    matrix = np.zeros((n_classes, n_classes), dtype=int)
    # np.add.at handles repeated indices correctly, which fancy-index += does not.
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def _per_class(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    true_positive = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)   # column totals
    actual = cm.sum(axis=1).astype(float)      # row totals
    return true_positive, predicted, actual


def precision(y_true, y_pred, average: str = "macro") -> float | np.ndarray:
    """Of what we flagged, how much was right — TP / (TP + FP).

    The metric that matters when a false positive is expensive: blocking a
    legitimate login, or quarantining a clean file.
    """
    tp, predicted, actual = _per_class(y_true, y_pred)
    # A class that was never predicted has an undefined precision; scikit-learn
    # reports 0 for it, and matching that is what makes the tests comparable.
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where(predicted > 0, tp / predicted, 0.0)
    return _reduce(per_class, tp, predicted, actual, average)


def recall(y_true, y_pred, average: str = "macro") -> float | np.ndarray:
    """Of what was really there, how much did we catch — TP / (TP + FN).

    The metric that matters when a miss is expensive: an undetected intrusion,
    or a missed diagnosis.
    """
    tp, _, actual = _per_class(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where(actual > 0, tp / actual, 0.0)
    return _reduce(per_class, tp, actual, actual, average)


def f1(y_true, y_pred, average: str = "macro") -> float | np.ndarray:
    """Harmonic mean of precision and recall.

    Harmonic, not arithmetic, and that is the point: it punishes imbalance
    between the two. Precision 1.0 with recall 0.0 gives an arithmetic mean of
    0.5 but an F1 of 0.0, which is the honest score for a classifier that finds
    nothing.
    """
    p = np.atleast_1d(precision(y_true, y_pred, average="none"))
    r = np.atleast_1d(recall(y_true, y_pred, average="none"))
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)

    if average == "none":
        return per_class
    if average == "macro":
        return float(np.mean(per_class))
    if average == "micro":
        # Micro-averaged F1 equals accuracy in single-label classification.
        return accuracy(y_true, y_pred)
    if average == "weighted":
        _, _, support = _per_class(y_true, y_pred)
        return float(np.average(per_class, weights=support))
    raise ValueError(f"unknown average: {average}")


def _reduce(
    per_class: np.ndarray,
    tp: np.ndarray,
    denominator: np.ndarray,
    support: np.ndarray,
    average: str,
):
    """Collapse per-class scores into one number.

    ``denominator`` differs per metric (predicted totals for precision, actual
    totals for recall) and is what micro-averaging sums over. ``support`` is
    always the *true* class counts, because a weighted average must weight by how
    much of each class actually exists — weighting precision by how often the
    model happened to predict each class would let a trigger-happy classifier
    inflate its own weights.
    """
    if average == "none":
        return per_class
    if average == "macro":
        # Unweighted mean: every class counts equally regardless of size, which
        # is why macro scores drop sharply when a rare class is handled badly.
        return float(np.mean(per_class))
    if average == "micro":
        return float(tp.sum() / denominator.sum()) if denominator.sum() else 0.0
    if average == "weighted":
        return float(np.average(per_class, weights=support)) if support.sum() else 0.0
    raise ValueError(f"unknown average: {average}")


def classification_report(y_true, y_pred, labels: list[str] | None = None) -> str:
    """A readable per-class summary, in the spirit of sklearn's version."""
    p = np.atleast_1d(precision(y_true, y_pred, average="none"))
    r = np.atleast_1d(recall(y_true, y_pred, average="none"))
    f = np.atleast_1d(f1(y_true, y_pred, average="none"))
    _, _, support = _per_class(y_true, y_pred)

    names = labels or [str(i) for i in range(len(p))]
    lines = [f"{'class':<14}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}"]
    for i, name in enumerate(names):
        lines.append(f"{name:<14}{p[i]:>10.3f}{r[i]:>9.3f}{f[i]:>8.3f}{int(support[i]):>9}")
    lines.append("")
    lines.append(f"{'accuracy':<14}{'':>10}{'':>9}{accuracy(y_true, y_pred):>8.3f}{int(support.sum()):>9}")
    lines.append(f"{'macro avg':<14}{np.mean(p):>10.3f}{np.mean(r):>9.3f}{np.mean(f):>8.3f}{int(support.sum()):>9}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------


def mean_squared_error(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred) -> float:
    """In the units of the target, which is what makes it interpretable."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred) -> float:
    """Less sensitive to outliers than MSE, because errors are not squared."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    """Fraction of variance explained — 1 minus (residual SS / total SS).

    Can be **negative**, which surprises people: a model worse than predicting
    the mean scores below zero. That makes it a comparison against a baseline
    rather than a bounded percentage.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# --------------------------------------------------------------------------
# Probabilistic
# --------------------------------------------------------------------------


def log_loss(y_true, probabilities, eps: float = 1e-15) -> float:
    """Cross-entropy. Punishes confident wrong answers far harder than accuracy.

    The clipping is not cosmetic: a predicted probability of exactly 0 for the
    true class gives ``log(0) = -inf`` and the whole score becomes ``inf``.
    Every practical implementation clips, and knowing why is the point.
    """
    y_true = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, float), eps, 1 - eps)

    if p.ndim == 1:  # binary, probability of class 1
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    # Multiclass: pick out the probability assigned to the true class.
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(p[np.arange(len(y_true)), y_true])))


def roc_curve(y_true, scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False-positive and true-positive rates at every threshold.

    Built by sorting scores descending and sweeping the threshold, which is O(n
    log n) — the naive approach of looping over candidate thresholds and
    rescanning is O(n^2) for the same answer.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, float)

    order = np.argsort(-scores)
    y_sorted = y_true[order]
    thresholds = scores[order]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)

    positives, negatives = y_true.sum(), len(y_true) - y_true.sum()
    tpr = tps / positives if positives else np.zeros_like(tps, dtype=float)
    fpr = fps / negatives if negatives else np.zeros_like(fps, dtype=float)

    # Start at (0, 0) so the curve is anchored.
    return np.r_[0, fpr], np.r_[0, tpr], np.r_[np.inf, thresholds]


def roc_auc(y_true, scores) -> float:
    """Area under the ROC curve, by the trapezoid rule.

    Interpretation worth holding on to: AUC is the probability that a randomly
    chosen positive is scored above a randomly chosen negative. 0.5 is coin
    flipping; below 0.5 means the model is anti-correlated and inverting it
    would help.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.trapezoid(tpr, fpr))
