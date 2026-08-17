"""Every metric is checked against scikit-learn's implementation."""

import numpy as np
import pytest
from sklearn import metrics as skm

from mlkit import metrics as m


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=300)
    y_pred = y_true.copy()
    flip = rng.choice(300, size=90, replace=False)   # inject 30% error
    y_pred[flip] = rng.integers(0, 3, size=90)
    return y_true, y_pred


@pytest.fixture
def binary_scores():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=400)
    # Scores correlated with the label, so AUC is meaningfully above 0.5.
    scores = y * 0.7 + rng.normal(0, 0.5, size=400)
    return y, scores


def test_accuracy(labels):
    y_true, y_pred = labels
    assert m.accuracy(y_true, y_pred) == pytest.approx(skm.accuracy_score(y_true, y_pred))


def test_confusion_matrix(labels):
    y_true, y_pred = labels
    assert np.array_equal(m.confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred))


@pytest.mark.parametrize("average", ["macro", "micro", "weighted"])
def test_precision_recall_f1(labels, average):
    y_true, y_pred = labels
    kw = dict(average=average, zero_division=0)
    assert m.precision(y_true, y_pred, average) == pytest.approx(skm.precision_score(y_true, y_pred, **kw))
    assert m.recall(y_true, y_pred, average) == pytest.approx(skm.recall_score(y_true, y_pred, **kw))
    assert m.f1(y_true, y_pred, average) == pytest.approx(skm.f1_score(y_true, y_pred, **kw))


def test_per_class_scores(labels):
    y_true, y_pred = labels
    assert np.allclose(
        m.f1(y_true, y_pred, "none"), skm.f1_score(y_true, y_pred, average=None, zero_division=0)
    )


def test_regression_metrics():
    rng = np.random.default_rng(2)
    y_true = rng.normal(size=200) * 10
    y_pred = y_true + rng.normal(size=200)

    assert m.mean_squared_error(y_true, y_pred) == pytest.approx(skm.mean_squared_error(y_true, y_pred))
    assert m.mean_absolute_error(y_true, y_pred) == pytest.approx(skm.mean_absolute_error(y_true, y_pred))
    assert m.r2_score(y_true, y_pred) == pytest.approx(skm.r2_score(y_true, y_pred))
    assert m.root_mean_squared_error(y_true, y_pred) == pytest.approx(
        np.sqrt(skm.mean_squared_error(y_true, y_pred))
    )


def test_r2_can_be_negative():
    """A model worse than the mean scores below zero - not a bug."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([4.0, 3.0, 2.0, 1.0])
    assert m.r2_score(y_true, y_pred) < 0


def test_log_loss_binary_and_multiclass(binary_scores):
    y, scores = binary_scores
    p = 1 / (1 + np.exp(-scores))
    assert m.log_loss(y, p) == pytest.approx(skm.log_loss(y, p))

    rng = np.random.default_rng(3)
    y_multi = rng.integers(0, 4, size=200)
    logits = rng.normal(size=(200, 4))
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    assert m.log_loss(y_multi, proba) == pytest.approx(skm.log_loss(y_multi, proba))


def test_log_loss_clipping_prevents_infinity():
    """A confident wrong prediction must give a large finite loss, not inf."""
    assert np.isfinite(m.log_loss(np.array([1]), np.array([0.0])))


def test_roc_auc(binary_scores):
    y, scores = binary_scores
    assert m.roc_auc(y, scores) == pytest.approx(skm.roc_auc_score(y, scores), abs=1e-3)


def test_roc_auc_of_perfect_and_random():
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    assert m.roc_auc(y, y.astype(float)) == pytest.approx(1.0)
    # Inverted scores give AUC 0 - the model is perfectly anti-correlated.
    assert m.roc_auc(y, -y.astype(float)) == pytest.approx(0.0)


def test_f1_punishes_imbalance_between_precision_and_recall():
    """Precision 1.0 with recall ~0 must not score 0.5."""
    y_true = np.r_[np.ones(100), np.zeros(100)].astype(int)
    y_pred = np.zeros(200, dtype=int)
    y_pred[0] = 1                      # one correct positive, 99 missed
    per_class = m.f1(y_true, y_pred, "none")
    assert per_class[1] < 0.03


def test_accuracy_is_misleading_on_imbalanced_data():
    """The point of having more than one metric."""
    y_true = np.r_[np.ones(5), np.zeros(995)].astype(int)
    y_pred = np.zeros(1000, dtype=int)          # predict the majority always
    assert m.accuracy(y_true, y_pred) == pytest.approx(0.995)
    assert m.recall(y_true, y_pred, "macro") == pytest.approx(0.5)
    assert m.f1(y_true, y_pred, "none")[1] == 0.0
