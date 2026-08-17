"""Render the figures used in the README from the experiment CSVs.

    python scripts/make_figures.py

Charting rules followed here: at most four series per panel, each labelled at its
right-hand end so identity never depends on colour alone; log scales where the
quantity spans orders of magnitude; a single hue ramp for magnitude and a fixed
categorical order for identity.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Fixed categorical order, validated for colour-vision-deficiency separation.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#e6e5e1"
BAD = "#d03b3b"
GOOD = "#0ca30c"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 9.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.labelsize": 9.5, "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2.0, "lines.markersize": 4.5,
    })


def load(name: str) -> list[dict]:
    with (REPORTS / name).open() as fh:
        return list(csv.DictReader(fh))


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"  {name}")


def label_end(ax, xs, ys, text, colour, dx=6) -> None:
    ax.annotate(text, xy=(xs[-1], ys[-1]), xytext=(dx, 0), textcoords="offset points",
                fontsize=8.5, color=colour, va="center", fontweight="semibold")


# --------------------------------------------------------------------------


def fig_vs_sklearn() -> None:
    """Paired accuracy: mine against the reference."""
    rows = [r for r in load("vs_sklearn.csv") if r["model"] != "Linear regression (R2)"]
    names = [r["model"] for r in rows]
    mine = [float(r["mine_accuracy"]) for r in rows]
    theirs = [float(r["sklearn_accuracy"]) for r in rows]

    y = np.arange(len(names))
    height = 0.36

    fig, ax = plt.subplots(figsize=(9, 3.9))
    ax.barh(y - height / 2, mine, height, color=SERIES[0], label="from scratch")
    ax.barh(y + height / 2, theirs, height, color=MUTED, label="scikit-learn")

    for i, (a, b) in enumerate(zip(mine, theirs)):
        difference = a - b
        note = "identical" if abs(difference) < 1e-9 else f"{difference:+.4f}"
        ax.text(max(a, b) + 0.008, i, note, va="center", fontsize=8.5,
                color=GOOD if abs(difference) < 1e-9 else INK_2)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0.85, 1.06)
    ax.set_xlabel("test accuracy")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.set_title("From-scratch implementations against scikit-learn")
    save(fig, "01_vs_sklearn.png")


def fig_learning_rate() -> None:
    """The learning rate's usable window, and the cliff at its edge."""
    rows = load("learning_rate.csv")
    rates = [float(r["learning_rate"]) for r in rows]
    losses = [float(r["final_loss"]) if r["final_loss"] != "inf" else np.inf for r in rows]
    diverged = [r["diverged"] == "True" for r in rows]

    plotted = [(r, l, d) for r, l, d in zip(rates, losses, diverged) if np.isfinite(l)]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))

    ok = [(r, l) for r, l, d in plotted if not d]
    bad = [(r, l) for r, l, d in plotted if d]
    ax.plot([r for r, _ in ok], [l for _, l in ok], marker="o", color=SERIES[0], zorder=3)
    if bad:
        ax.scatter([r for r, _ in bad], [l for _, l in bad], color=BAD, s=55, zorder=4, marker="X")

    ax.axvspan(0.9, 2.2, color=BAD, alpha=0.07, zorder=0)
    ax.annotate("diverges\n(loss → 10⁶⁴ and beyond)", xy=(1.0, 1e30), fontsize=8.8,
                color=BAD, ha="left", va="center")
    best = min(plotted, key=lambda t: t[1])
    ax.annotate(f"fastest: lr={best[0]}\nconverged in "
                f"{[r['iterations_used'] for r in rows if float(r['learning_rate']) == best[0]][0]} iters",
                xy=(best[0], best[1]), xytext=(-30, 40), textcoords="offset points",
                fontsize=8.8, color=SERIES[0],
                arrowprops=dict(arrowstyle="-", color=SERIES[0], lw=0.9))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("final loss after 300 iterations")
    ax.set_title("A factor of two in the learning rate separates convergence from divergence")
    save(fig, "02_learning_rate.png")


def fig_bias_variance() -> None:
    """k-NN and tree depth, side by side — the same trade-off from two directions."""
    knn = load("knn_k.csv")
    tree = load("tree_depth.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ks = [int(r["k"]) for r in knn]
    ax = axes[0]
    ax.plot(ks, [float(r["train_accuracy"]) for r in knn], marker="o", color=SERIES[0])
    ax.plot(ks, [float(r["test_accuracy"]) for r in knn], marker="s", color=SERIES[1])
    ax.plot(ks, [float(r["cv_accuracy"]) for r in knn], marker="^", color=SERIES[2], ls=(0, (4, 2)))
    label_end(ax, ks, [float(r["train_accuracy"]) for r in knn], "train", SERIES[0])
    label_end(ax, ks, [float(r["test_accuracy"]) for r in knn], "test", SERIES[1])
    label_end(ax, ks, [float(r["cv_accuracy"]) for r in knn], "5-fold CV", SERIES[2])
    ax.annotate("k=1 memorises\nthe training set", xy=(1, 1.0), xytext=(14, -26),
                textcoords="offset points", fontsize=8.5, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("k (neighbours)")
    ax.set_ylabel("accuracy")
    ax.set_xlim(right=ks[-1] * 3.2)
    ax.set_title("k-NN: more neighbours, less variance")

    depths = [r["max_depth"] for r in tree]
    x = np.arange(len(depths))
    ax = axes[1]
    train = [float(r["train_accuracy"]) for r in tree]
    test = [float(r["test_accuracy"]) for r in tree]
    ax.plot(x, train, marker="o", color=SERIES[0])
    ax.plot(x, test, marker="s", color=SERIES[1])
    ax.fill_between(x, test, train, color=SERIES[1], alpha=0.08)

    peak = int(np.argmax(test))
    ax.scatter([peak], [test[peak]], s=70, facecolor="none", edgecolor=GOOD, linewidth=1.8, zorder=5)
    ax.annotate(f"best test accuracy\nat depth {depths[peak]}", xy=(peak, test[peak]),
                xytext=(-18, -42), textcoords="offset points", fontsize=8.5, color=GOOD,
                arrowprops=dict(arrowstyle="-", color=GOOD, lw=0.9))
    ax.annotate("gap = overfitting", xy=(len(x) - 1.5, (train[-1] + test[-1]) / 2),
                xytext=(-88, 0), textcoords="offset points", fontsize=8.5, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels(depths, fontsize=8)
    ax.set_xlabel("max tree depth")
    ax.set_title("Decision tree: deeper fits training data, not reality")
    axes[1].legend(["train", "test"], loc="lower right")

    fig.suptitle("The bias–variance trade-off, measured on breast-cancer data",
                 fontsize=12.5, fontweight="bold", color=INK, y=1.04)
    save(fig, "03_bias_variance.png")


def fig_optimizers() -> None:
    """Per-epoch loss curves for each optimiser."""
    runs = [
        ("loss_sgd_0p1.csv", "SGD (lr 0.1)", SERIES[0]),
        ("loss_sgd_0p5.csv", "SGD (lr 0.5)", SERIES[1]),
        ("loss_momentum_0p05.csv", "Momentum (lr 0.05)", SERIES[2]),
        ("loss_adam_0p001.csv", "Adam (lr 0.001)", SERIES[3]),
    ]

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for filename, label, colour in runs:
        rows = load(filename)
        epochs = [int(r["epoch"]) for r in rows]
        losses = [float(r["loss"]) for r in rows]
        ax.plot(epochs, losses, color=colour, label=label)
        label_end(ax, epochs, losses, label.split(" (")[0], colour)

    ax.axhline(0.01, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.annotate("loss = 0.01", xy=(2, 0.0115), fontsize=8, color=MUTED, style="italic")

    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (log scale)")
    ax.set_xlim(right=76)
    ax.set_title("Optimisers on the same 64-unit network and seed (digits)")
    save(fig, "04_optimizers.png")


def fig_activations() -> None:
    """Vanishing gradients: accuracy against depth, per activation."""
    rows = [r for r in load("activations.csv") if r["learning_rate"] == "0.05"]

    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    for i, activation in enumerate(("relu", "tanh", "sigmoid")):
        subset = sorted((r for r in rows if r["activation"] == activation),
                        key=lambda r: int(r["depth"]))
        depths = [int(r["depth"]) for r in subset]
        accuracy = [float(r["test_accuracy"]) for r in subset]
        ax.plot(depths, accuracy, marker="o", color=SERIES[i], label=activation)
        label_end(ax, depths, accuracy, activation, SERIES[i])

    sigmoid = sorted((r for r in rows if r["activation"] == "sigmoid"), key=lambda r: int(r["depth"]))
    drop = float(sigmoid[0]["test_accuracy"]) - float(sigmoid[-1]["test_accuracy"])
    ax.annotate(f"sigmoid loses {drop * 100:.0f} points\nfrom depth 1 to 4",
                xy=(4, float(sigmoid[-1]["test_accuracy"])), xytext=(-124, 18),
                textcoords="offset points", fontsize=8.8, color=SERIES[2],
                arrowprops=dict(arrowstyle="-", color=SERIES[2], lw=0.9))

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("number of hidden layers (32 units each)")
    ax.set_ylabel("test accuracy")
    ax.set_xlim(0.85, 4.6)
    ax.set_title("Vanishing gradients: sigmoid degrades with depth, ReLU and tanh do not")
    save(fig, "05_activations.png")


def fig_pca() -> None:
    """Variance retained and downstream accuracy against component count."""
    rows = load("pca.csv")
    ks = [int(r["n_components"]) for r in rows]
    variance = [float(r["variance_explained"]) for r in rows]
    accuracy = [float(r["knn_accuracy"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(ks, variance, marker="o", color=SERIES[0])
    ax.axhline(0.95, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.annotate("95% of variance", xy=(1.2, 0.965), fontsize=8, color=MUTED, style="italic")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("components kept")
    ax.set_ylabel("cumulative variance explained")
    ax.set_title("Variance is concentrated in few components")

    ax = axes[1]
    ax.plot(ks, accuracy, marker="s", color=SERIES[1])
    peak = int(np.argmax(accuracy))
    ax.scatter([ks[peak]], [accuracy[peak]], s=80, facecolor="none",
               edgecolor=GOOD, linewidth=1.8, zorder=5)
    ax.annotate(f"peak at {ks[peak]} components\n({accuracy[peak]:.4f}) — better than\nall 64 "
                f"({accuracy[-1]:.4f})",
                xy=(ks[peak], accuracy[peak]), xytext=(-40, -62), textcoords="offset points",
                fontsize=8.5, color=GOOD, arrowprops=dict(arrowstyle="-", color=GOOD, lw=0.9))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("components kept")
    ax.set_ylabel("k-NN test accuracy")
    ax.set_title("Discarding components *improves* accuracy")

    fig.suptitle("PCA on handwritten digits (64 pixel features)",
                 fontsize=12.5, fontweight="bold", color=INK, y=1.04)
    save(fig, "06_pca.png")


def fig_adversarial() -> None:
    """The robustness curve — the AI-security result."""
    rows = load("adversarial.csv")

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for i, attack in enumerate(("fgsm", "pgd")):
        subset = sorted((r for r in rows if r["attack"] == attack),
                        key=lambda r: float(r["epsilon"]))
        eps = [float(r["epsilon"]) for r in subset]
        acc = [float(r["accuracy"]) for r in subset]
        ax.plot(eps, acc, marker="o" if attack == "fgsm" else "s",
                color=SERIES[i], label=attack.upper())
        # Both curves end at ~0, so the end labels must be offset vertically or
        # they overlap illegibly.
        ax.annotate(attack.upper(), xy=(eps[-1], acc[-1]), xytext=(8, 8 if i == 0 else -10),
                    textcoords="offset points", fontsize=8.5, color=SERIES[i],
                    va="center", fontweight="semibold")

    clean = [float(r["accuracy"]) for r in rows if r["epsilon"] == "0.0"][0]
    ax.axhline(clean, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.annotate(f"clean accuracy {clean:.1%}", xy=(0.005, clean + 0.03),
                fontsize=8.5, color=MUTED, style="italic")
    ax.axhline(0.1, color=BAD, lw=1.0, ls=(0, (2, 2)))
    ax.annotate("chance (10 classes)", xy=(0.005, 0.125), fontsize=8.5, color=BAD, style="italic")

    fgsm = sorted((r for r in rows if r["attack"] == "fgsm"), key=lambda r: float(r["epsilon"]))
    # Annotate the measured point nearest to half of clean accuracy, and state its
    # actual value rather than claiming a threshold the data does not show.
    target = clean / 2
    nearest = min(fgsm, key=lambda r: abs(float(r["accuracy"]) - target))
    ax.annotate(f"ε={nearest['epsilon']} → {float(nearest['accuracy']):.1%}\n"
                f"about half of clean accuracy,\nfrom a {float(nearest['epsilon']):.0%} "
                f"change per pixel",
                xy=(float(nearest["epsilon"]), float(nearest["accuracy"])), xytext=(30, 34),
                textcoords="offset points", fontsize=8.8, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))

    ax.set_xlabel("ε  (maximum change per pixel, inputs scaled to [0, 1])")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlim(right=0.34)
    ax.set_title("A 97.8%-accurate network collapses under imperceptible perturbation")
    save(fig, "07_adversarial.png")


def fig_solver_agreement() -> None:
    """Three solvers, one objective — how closely do they agree?"""
    rows = [r for r in load("solver_agreement.csv") if float(r["max_weight_error"]) > 0]
    names = [r["solver"].replace(" (", "\n(") for r in rows]
    errors = [float(r["max_weight_error"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.barh(np.arange(len(names)), errors, color=SERIES[0], height=0.55)
    for i, error in enumerate(errors):
        ax.text(error * 1.15, i, f"{error:.2e}", va="center", fontsize=8.5, color=INK_2)

    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(right=max(errors) * 6)
    ax.set_xlabel("largest weight difference from the closed-form solution")
    ax.set_title("Iterative solvers converge to the exact least-squares answer")
    save(fig, "08_solver_agreement.png")


def main() -> None:
    style()
    print(f"Writing figures to reports/figures")
    fig_vs_sklearn()
    fig_learning_rate()
    fig_bias_variance()
    fig_optimizers()
    fig_activations()
    fig_pca()
    fig_adversarial()
    fig_solver_agreement()
    print("Done.")


if __name__ == "__main__":
    main()
