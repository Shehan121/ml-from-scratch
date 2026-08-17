import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn import datasets  # noqa: E402

from mlkit.preprocessing import StandardScaler, train_test_split  # noqa: E402


@pytest.fixture(scope="session")
def iris():
    d = datasets.load_iris()
    return d.data, d.target


@pytest.fixture(scope="session")
def breast_cancer():
    """Binary, 30 features - the realistic classification case."""
    d = datasets.load_breast_cancer()
    return d.data, d.target


@pytest.fixture(scope="session")
def diabetes():
    """Regression, 10 features, already standardised by sklearn."""
    d = datasets.load_diabetes()
    return d.data, d.target


@pytest.fixture(scope="session")
def digits():
    d = datasets.load_digits()
    return d.data, d.target


@pytest.fixture
def scaled_split():
    """A properly leak-free split: the scaler is fitted on train only."""

    def make(X, y, test_size=0.25, seed=0):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, seed=seed, stratify=True)
        scaler = StandardScaler().fit(Xtr)
        return scaler.transform(Xtr), scaler.transform(Xte), ytr, yte

    return make


@pytest.fixture
def tiny_regression():
    """A small exactly-linear problem, so solvers must recover known weights."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    true_w = np.array([2.0, -3.0, 0.5])
    y = X @ true_w + 1.5
    return X, y, true_w, 1.5
