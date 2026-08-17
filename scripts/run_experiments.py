"""Run every experiment and write the results to ``reports/``.

    python scripts/run_experiments.py

scikit-learn appears here only to load the bundled datasets and to provide the
reference numbers the from-scratch implementations are compared against. Every
model being measured is from :mod:`mlkit`.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn import datasets  # noqa: E402
from sklearn import linear_model as sklinear  # noqa: E402
from sklearn import naive_bayes as sknb  # noqa: E402
from sklearn import neighbors as skneighbors  # noqa: E402
from sklearn import tree as sktree  # noqa: E402

from mlkit.adversarial import accuracy_under_attack  # noqa: E402
from mlkit.gradcheck import check_gradients  # noqa: E402
from mlkit.knn import KNeighborsClassifier  # noqa: E402
from mlkit.linear import LinearRegression, Ridge  # noqa: E402
from mlkit.logistic import LogisticRegression, SoftmaxRegression  # noqa: E402
from mlkit.metrics import accuracy, f1, precision, r2_score, recall, roc_auc  # noqa: E402
from mlkit.naive_bayes import GaussianNB  # noqa: E402
from mlkit.neural_net import MLPClassifier  # noqa: E402
from mlkit.pca import PCA  # noqa: E402
from mlkit.preprocessing import KFold, MinMaxScaler, StandardScaler, train_test_split  # noqa: E402
from mlkit.tree import DecisionTreeClassifier  # noqa: E402

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def write_csv(name: str, rows: list[dict], fields: list[str]) -> None:
    with (REPORTS / name).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote reports/{name}  ({len(rows)} rows)")


def split_and_scale(X, y, test_size=0.25, seed=0):
    """The leak-free path: fit the scaler on train only."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, seed=seed, stratify=True)
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte), ytr, yte


# --------------------------------------------------------------------------


def experiment_vs_sklearn() -> list[dict]:
    """Every from-scratch model against its scikit-learn counterpart."""
    rows: list[dict] = []

    cancer = datasets.load_breast_cancer()
    Xtr, Xte, ytr, yte = split_and_scale(cancer.data, cancer.target)

    pairs = [
        ("Logistic regression", "breast_cancer",
         LogisticRegression(learning_rate=0.5, n_iterations=3000),
         sklinear.LogisticRegression(max_iter=5000)),
        ("k-NN (k=5)", "breast_cancer",
         KNeighborsClassifier(k=5), skneighbors.KNeighborsClassifier(n_neighbors=5)),
        ("Decision tree (depth 4)", "breast_cancer",
         DecisionTreeClassifier(max_depth=4), sktree.DecisionTreeClassifier(max_depth=4, random_state=0)),
        ("Gaussian naive Bayes", "breast_cancer", GaussianNB(), sknb.GaussianNB()),
    ]

    for name, dataset, mine, theirs in pairs:
        t0 = time.perf_counter()
        mine.fit(Xtr, ytr)
        mine_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        theirs.fit(Xtr, ytr)
        their_time = time.perf_counter() - t0

        rows.append({
            "model": name,
            "dataset": dataset,
            "mine_accuracy": round(accuracy(yte, mine.predict(Xte)), 4),
            "sklearn_accuracy": round(accuracy(yte, theirs.predict(Xte)), 4),
            "difference": round(accuracy(yte, mine.predict(Xte)) - accuracy(yte, theirs.predict(Xte)), 4),
            "mine_fit_seconds": round(mine_time, 5),
            "sklearn_fit_seconds": round(their_time, 5),
        })

    digits = datasets.load_digits()
    Xtr, Xte, ytr, yte = split_and_scale(digits.data, digits.target)
    mine = SoftmaxRegression(learning_rate=0.5, n_iterations=800).fit(Xtr, ytr)
    theirs = sklinear.LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    rows.append({
        "model": "Softmax regression", "dataset": "digits",
        "mine_accuracy": round(mine.score(Xte, yte), 4),
        "sklearn_accuracy": round(theirs.score(Xte, yte), 4),
        "difference": round(mine.score(Xte, yte) - theirs.score(Xte, yte), 4),
        "mine_fit_seconds": "", "sklearn_fit_seconds": "",
    })

    diabetes = datasets.load_diabetes()
    mine_lr = LinearRegression(solver="normal").fit(diabetes.data, diabetes.target)
    their_lr = sklinear.LinearRegression().fit(diabetes.data, diabetes.target)
    rows.append({
        "model": "Linear regression (R2)", "dataset": "diabetes",
        "mine_accuracy": round(r2_score(diabetes.target, mine_lr.predict(diabetes.data)), 6),
        "sklearn_accuracy": round(r2_score(diabetes.target, their_lr.predict(diabetes.data)), 6),
        "difference": round(
            r2_score(diabetes.target, mine_lr.predict(diabetes.data))
            - r2_score(diabetes.target, their_lr.predict(diabetes.data)), 9),
        "mine_fit_seconds": "", "sklearn_fit_seconds": "",
    })
    return rows


def experiment_solver_agreement() -> list[dict]:
    """Do the three least-squares solvers reach the same weights?"""
    from mlkit.linear import SGDRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    true_w = np.array([2.0, -3.0, 0.5, 1.25, -0.75])
    y = X @ true_w + 1.5 + rng.normal(0, 0.1, size=400)

    exact = LinearRegression(solver="normal").fit(X, y)
    rows = [{
        "solver": "normal equation", "iterations": exact.n_iter_,
        "max_weight_error": 0.0, "r2": round(exact.score(X, y), 6),
    }]

    for iters in (100, 500, 2000, 10000):
        gd = LinearRegression(solver="gradient", learning_rate=0.05, n_iterations=iters).fit(X, y)
        rows.append({
            "solver": f"batch gradient ({iters} iters)", "iterations": gd.n_iter_,
            "max_weight_error": round(float(np.max(np.abs(gd.coef_ - exact.coef_))), 8),
            "r2": round(gd.score(X, y), 6),
        })

    sgd = SGDRegressor(learning_rate=0.02, n_epochs=200, batch_size=16).fit(X, y)
    rows.append({
        "solver": "stochastic gradient (200 epochs)", "iterations": 200,
        "max_weight_error": round(float(np.max(np.abs(sgd.coef_ - exact.coef_))), 8),
        "r2": round(sgd.score(X, y), 6),
    })
    return rows


def experiment_learning_rate() -> list[dict]:
    """The learning rate is the hyperparameter that decides everything."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = X @ np.array([1.5, -2.0, 0.5, 1.0]) + 0.5

    rows: list[dict] = []
    for rate in (0.0001, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0):
        model = LinearRegression(solver="gradient", learning_rate=rate, n_iterations=300).fit(X, y)
        history = model.loss_history_
        diverged = not np.isfinite(history[-1]) or history[-1] > history[0]
        rows.append({
            "learning_rate": rate,
            "final_loss": f"{history[-1]:.6e}" if np.isfinite(history[-1]) else "inf",
            "iterations_used": model.n_iter_,
            "diverged": diverged,
        })
    return rows


def experiment_data_leakage() -> list[dict]:
    """Scaling before the split versus after — how much does the leak inflate?"""
    cancer = datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target
    rows: list[dict] = []

    for seed in range(20):
        # Wrong: the scaler sees the whole dataset, including the test rows.
        leaked = StandardScaler().fit_transform(X)
        Xtr_l, Xte_l, ytr_l, yte_l = train_test_split(leaked, y, seed=seed, stratify=True)
        leaked_score = KNeighborsClassifier(k=5).fit(Xtr_l, ytr_l).score(Xte_l, yte_l)

        # Correct: fit on train only.
        Xtr, Xte, ytr, yte = train_test_split(X, y, seed=seed, stratify=True)
        scaler = StandardScaler().fit(Xtr)
        clean_score = KNeighborsClassifier(k=5).fit(scaler.transform(Xtr), ytr).score(
            scaler.transform(Xte), yte
        )

        rows.append({
            "seed": seed,
            "leaked_accuracy": round(leaked_score, 5),
            "clean_accuracy": round(clean_score, 5),
            "inflation": round(leaked_score - clean_score, 5),
        })
    return rows


def experiment_knn_k() -> list[dict]:
    """k as the bias-variance dial, with cross-validation."""
    cancer = datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target
    Xtr, Xte, ytr, yte = split_and_scale(X, y)

    rows: list[dict] = []
    for k in (1, 3, 5, 9, 15, 25, 51, 101):
        train_score = KNeighborsClassifier(k=k).fit(Xtr, ytr).score(Xtr, ytr)
        test_score = KNeighborsClassifier(k=k).fit(Xtr, ytr).score(Xte, yte)

        cv_scores = []
        for train_idx, val_idx in KFold(5, seed=0).split(Xtr):
            model = KNeighborsClassifier(k=min(k, len(train_idx))).fit(Xtr[train_idx], ytr[train_idx])
            cv_scores.append(model.score(Xtr[val_idx], ytr[val_idx]))

        rows.append({
            "k": k,
            "train_accuracy": round(train_score, 4),
            "test_accuracy": round(test_score, 4),
            "cv_accuracy": round(float(np.mean(cv_scores)), 4),
            "cv_std": round(float(np.std(cv_scores)), 4),
        })
    return rows


def experiment_tree_depth() -> list[dict]:
    """Overfitting, made visible as the train/test gap widens with depth."""
    cancer = datasets.load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(cancer.data, cancer.target, seed=0, stratify=True)

    rows: list[dict] = []
    for depth in (1, 2, 3, 4, 5, 7, 10, None):
        model = DecisionTreeClassifier(max_depth=depth).fit(Xtr, ytr)
        train_score = model.score(Xtr, ytr)
        test_score = model.score(Xte, yte)
        rows.append({
            "max_depth": depth if depth is not None else "unbounded",
            "actual_depth": model.depth(),
            "n_leaves": model.n_leaves(),
            "train_accuracy": round(train_score, 4),
            "test_accuracy": round(test_score, 4),
            "gap": round(train_score - test_score, 4),
        })
    return rows


def experiment_optimizers() -> list[dict]:
    """SGD versus Momentum versus Adam on the same network and seed."""
    digits = datasets.load_digits()
    Xtr, Xte, ytr, yte = split_and_scale(digits.data, digits.target)

    rows: list[dict] = []
    settings = [("sgd", 0.1), ("sgd", 0.5), ("momentum", 0.05), ("adam", 0.001)]

    for name, rate in settings:
        model = MLPClassifier(
            hidden=(64,), learning_rate=rate, n_epochs=60, batch_size=32,
            optimizer=name, seed=0,
        )
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        elapsed = time.perf_counter() - t0

        history = model.loss_history_
        # Epochs to reach a *fixed* loss, which actually separates the optimisers.
        # An earlier version measured epochs to get within 10% of each run's own
        # final loss, which returned 56-58 for every optimiser - the loss keeps
        # falling, so a moving target is always reached near the end. A relative
        # target cannot compare convergence speed.
        def epochs_to(target: float) -> int | str:
            return next((i + 1 for i, v in enumerate(history) if v <= target), "not reached")

        rows.append({
            "optimizer": name,
            "learning_rate": rate,
            "final_loss": round(history[-1], 5),
            "epochs_to_loss_0.5": epochs_to(0.5),
            "epochs_to_loss_0.1": epochs_to(0.1),
            "epochs_to_loss_0.01": epochs_to(0.01),
            "test_accuracy": round(model.score(Xte, yte), 4),
            "fit_seconds": round(elapsed, 3),
        })

        # Per-epoch curves for the figures.
        write_csv(
            f"loss_{name}_{str(rate).replace('.', 'p')}.csv",
            [{"epoch": i + 1, "loss": f"{v:.6f}"} for i, v in enumerate(history)],
            ["epoch", "loss"],
        )
    return rows


def experiment_activations() -> list[dict]:
    """ReLU versus sigmoid versus tanh, at increasing depth.

    The vanishing-gradient claim, measured: sigmoid's derivative peaks at 0.25, so
    a deep sigmoid stack should degrade in a way ReLU does not.
    """
    digits = datasets.load_digits()
    Xtr, Xte, ytr, yte = split_and_scale(digits.data, digits.target)

    rows: list[dict] = []
    architectures = [(32,), (32, 32), (32, 32, 32), (32, 32, 32, 32)]

    # Two learning rates, because the comparison is not activation-independent:
    # at lr=0.5 the deep ReLU stacks diverge outright, which is itself the result.
    for rate in (0.5, 0.05):
        for activation in ("relu", "sigmoid", "tanh"):
            for hidden in architectures:
                row = {
                    "activation": activation,
                    "learning_rate": rate,
                    "depth": len(hidden),
                    "hidden": "x".join(str(h) for h in hidden),
                }
                try:
                    model = MLPClassifier(
                        hidden=hidden, activation=activation, learning_rate=rate,
                        n_epochs=60, batch_size=32, seed=0,
                    ).fit(Xtr, ytr)
                except FloatingPointError:
                    row.update({"final_loss": "diverged", "train_accuracy": "",
                                "test_accuracy": "", "note": "weights overflowed"})
                    rows.append(row)
                    continue

                row.update({
                    "final_loss": round(model.loss_history_[-1], 5),
                    "train_accuracy": round(model.score(Xtr, ytr), 4),
                    "test_accuracy": round(model.score(Xte, yte), 4),
                    "note": "",
                })
                rows.append(row)
    return rows


def experiment_gradcheck() -> list[dict]:
    """The gradient check, recorded as evidence rather than only asserted."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 4))
    y = rng.integers(0, 3, size=8)

    rows: list[dict] = []
    for activation in ("relu", "sigmoid", "tanh"):
        for hidden in [(5,), (6, 4), (8, 6, 4)]:
            model = MLPClassifier(hidden=hidden, activation=activation, seed=0)
            model._build(4, 3)
            errors = check_gradients(model, X, y)
            rows.append({
                "activation": activation,
                "hidden": "x".join(str(h) for h in hidden),
                "n_params_checked": len(errors),
                "worst_relative_error": f"{max(errors.values()):.3e}",
                "verdict": "PASS" if max(errors.values()) < 1e-6 else "FAIL",
            })
    return rows


def experiment_pca() -> list[dict]:
    """How many components does it take to keep the digits recognisable?"""
    digits = datasets.load_digits()
    X = digits.data

    rows: list[dict] = []
    pca_full = PCA().fit(X)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)

    for k in (1, 2, 5, 10, 20, 30, 40, 64):
        pca = PCA(n_components=k).fit(X)
        Xtr, Xte, ytr, yte = split_and_scale(pca.transform(X), digits.target)
        rows.append({
            "n_components": k,
            "variance_explained": round(float(cumulative[k - 1]), 4),
            "reconstruction_mse": round(pca.reconstruction_error(X), 4),
            "knn_accuracy": round(KNeighborsClassifier(k=5).fit(Xtr, ytr).score(Xte, yte), 4),
        })
    return rows


def experiment_adversarial() -> list[dict]:
    """Accuracy against perturbation budget — the AI-security experiment."""
    digits = datasets.load_digits()
    X = MinMaxScaler().fit_transform(digits.data)
    Xtr, Xte, ytr, yte = train_test_split(X, digits.target, test_size=0.3, seed=0, stratify=True)

    model = MLPClassifier(hidden=(64,), learning_rate=0.5, n_epochs=80, batch_size=32, seed=0)
    model.fit(Xtr, ytr)

    epsilons = [0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
    rows: list[dict] = []

    for attack in ("fgsm", "pgd"):
        for epsilon, acc in accuracy_under_attack(model, Xte, yte, epsilons, attack=attack, clip=(0, 1)):
            rows.append({"attack": attack, "epsilon": epsilon, "accuracy": round(acc, 4)})
    return rows


def experiment_imbalance() -> list[dict]:
    """Why accuracy alone is not a metric.

    An artificially imbalanced problem is built from digits: class 0 against all
    others, subsampled so positives are rare.
    """
    digits = datasets.load_digits()
    X, y = digits.data, (digits.target == 0).astype(int)

    rng = np.random.default_rng(0)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    keep = np.r_[rng.choice(positives, 12, replace=False), negatives]
    X, y = X[keep], y[keep]

    Xtr, Xte, ytr, yte = split_and_scale(X, y, test_size=0.3)

    rows: list[dict] = []
    # A baseline that ignores the input entirely.
    majority = np.zeros(len(yte), dtype=int)
    rows.append({
        "model": "always predict majority",
        "accuracy": round(accuracy(yte, majority), 4),
        "precision_positive": round(float(precision(yte, majority, "none")[1]), 4),
        "recall_positive": round(float(recall(yte, majority, "none")[1]), 4),
        "f1_positive": round(float(f1(yte, majority, "none")[1]), 4),
        "roc_auc": "",
    })

    model = LogisticRegression(learning_rate=0.5, n_iterations=2000).fit(Xtr, ytr)
    predictions = model.predict(Xte)
    rows.append({
        "model": "logistic regression",
        "accuracy": round(accuracy(yte, predictions), 4),
        "precision_positive": round(float(precision(yte, predictions, "none")[1]), 4),
        "recall_positive": round(float(recall(yte, predictions, "none")[1]), 4),
        "f1_positive": round(float(f1(yte, predictions, "none")[1]), 4),
        "roc_auc": round(roc_auc(yte, model.decision_function(Xte)), 4),
    })
    return rows


# --------------------------------------------------------------------------


def main() -> None:
    print("Running experiments\n")

    print("from-scratch versus scikit-learn")
    vs = experiment_vs_sklearn()
    write_csv("vs_sklearn.csv", vs,
              ["model", "dataset", "mine_accuracy", "sklearn_accuracy", "difference",
               "mine_fit_seconds", "sklearn_fit_seconds"])

    print("solver agreement")
    write_csv("solver_agreement.csv", experiment_solver_agreement(),
              ["solver", "iterations", "max_weight_error", "r2"])

    print("learning rate sweep")
    write_csv("learning_rate.csv", experiment_learning_rate(),
              ["learning_rate", "final_loss", "iterations_used", "diverged"])

    print("data leakage")
    leak = experiment_data_leakage()
    write_csv("data_leakage.csv", leak, ["seed", "leaked_accuracy", "clean_accuracy", "inflation"])

    print("k-NN choice of k")
    write_csv("knn_k.csv", experiment_knn_k(),
              ["k", "train_accuracy", "test_accuracy", "cv_accuracy", "cv_std"])

    print("tree depth")
    write_csv("tree_depth.csv", experiment_tree_depth(),
              ["max_depth", "actual_depth", "n_leaves", "train_accuracy", "test_accuracy", "gap"])

    print("optimisers")
    write_csv("optimizers.csv", experiment_optimizers(),
              ["optimizer", "learning_rate", "final_loss", "epochs_to_loss_0.5",
               "epochs_to_loss_0.1", "epochs_to_loss_0.01", "test_accuracy", "fit_seconds"])

    print("activations by depth")
    write_csv("activations.csv", experiment_activations(),
              ["activation", "learning_rate", "depth", "hidden", "final_loss",
               "train_accuracy", "test_accuracy", "note"])

    print("gradient check")
    write_csv("gradcheck.csv", experiment_gradcheck(),
              ["activation", "hidden", "n_params_checked", "worst_relative_error", "verdict"])

    print("PCA")
    write_csv("pca.csv", experiment_pca(),
              ["n_components", "variance_explained", "reconstruction_mse", "knn_accuracy"])

    print("adversarial robustness")
    write_csv("adversarial.csv", experiment_adversarial(), ["attack", "epsilon", "accuracy"])

    print("class imbalance")
    write_csv("imbalance.csv", experiment_imbalance(),
              ["model", "accuracy", "precision_positive", "recall_positive", "f1_positive", "roc_auc"])

    inflation = [r["inflation"] for r in leak]
    summary = {
        "max_abs_difference_vs_sklearn": max(abs(r["difference"]) for r in vs),
        "leakage_mean_inflation": round(float(np.mean(inflation)), 5),
        "leakage_max_inflation": round(float(np.max(inflation)), 5),
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n  wrote reports/summary.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
