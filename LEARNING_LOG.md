# Learning log

What building these algorithms actually taught me — including three bugs in my own
code, one hypothesis that did not survive measurement, and a diagnostic that gave
me a confident wrong answer.

Every number is from `reports/`, reproducible with
`python scripts/run_experiments.py`.

---

## 1. My ReLU was silently swallowing NaN

The best bug in the project, because the mechanism was invisible.

A three-layer ReLU network on digits collapsed to **10.0% accuracy** — exactly
chance for ten classes — with a final loss of 2.3086, which is `ln(10)`. The
network was predicting a uniform distribution.

`dead_fraction()` reported **100% of units dead in all three layers**. That is
textbook dying ReLU, so I had my answer.

It was the wrong answer. Tracing the loss per epoch showed something else:

| epoch | loss | max&nbsp;\|W\| |
|---|---:|---:|
| 5 | 0.0008 | 1.25 |
| 6 | 0.0000 | 1.26 |
| **7** | **10.93** | **6.30** |
| 8 | 2.3051 | **NaN** |

Training was going *well* until epoch 7, when one bad mini-batch produced a huge
gradient, the weights blew up 5×, and by epoch 8 they were NaN. Not dying ReLU —
numerical divergence.

So why did `dead_fraction()` say 100% dead? Because I had written ReLU as:

```python
return np.where(x > 0, x, 0.0)      # the bug
```

`NaN > 0` is `False`, so `np.where` substituted a clean `0.0` for every NaN. **My
ReLU was laundering NaN into valid-looking zeros.** The first layer's output
became all zeros, downstream layers saw perfectly reasonable input, the logits
were small finite numbers, and the loss settled at `ln(10)`. A guard watching the
loss for non-finite values could never have fired.

Two fixes:

```python
return np.maximum(x, 0.0)           # propagates NaN
```

and a divergence check on the **parameters**, not just the loss — because a finite
loss is not evidence of a healthy network when an activation can mask non-finite
weights upstream.

**What I took from it:** a diagnostic that returns a plausible wrong answer is
worse than no diagnostic. `dead_fraction()` now ships beside `nan_fraction()`, and
its docstring says to read them together. And two functions that agree on every
finite input can differ in exactly the case you care about.

---

## 2. My gradient checker was itself subtly wrong

I built finite-difference checking specifically so I would not have to trust
hand-derived backprop. Then the checker gave this:

```
layer0.W  6.95e-08   OK
layer0.b  1.86e-09   OK
layer1.W  3.77e-05   ...
layer1.b  2.54e-04   FAIL
```

First layer perfect, later layers off by 1e-4. The pattern was the clue — a wrong
derivation would not be correct for layer 0 and wrong for layer 1.

The cause: I captured each layer's analytical gradient lazily, just before
checking it. But `numerical_gradient` calls the loss repeatedly, and every call
runs a full forward **and backward** pass, overwriting every layer's gradient
buffer. By the time the sweep reached layer 1, its `dW` held values from the last
*perturbed* pass of layer 0.

Snapshotting all analytical gradients before touching anything gives:

| architecture | worst relative error |
|---|---:|
| 5 | 6.9e-08 |
| 6×4 | 8.9e-10 |
| 8×6×4 | 4.5e-09 |

**What I took from it:** 2.5e-04 is the dangerous kind of wrong — small enough to
look like a tolerance issue you could widen, large enough to be a real bug. The
tool built to prevent false confidence was granting it.

---

## 3. The data leakage I set out to measure was not there

I built an experiment to show that scaling before splitting inflates test scores.
Twenty seeds, k-NN on breast-cancer data, `StandardScaler` fitted on everything
versus on train only.

**Mean inflation: 0.00000.** The leaked version scored higher on 2 seeds, *lower*
on 2, and identical on 16. Maximum difference in either direction: 0.7 percentage
points, which is one or two samples out of 143.

The hypothesis was wrong for a specific reason. `StandardScaler` leaks exactly two
numbers per feature — a mean and a standard deviation — estimated from 569 samples.
Adding the 143 test rows to that estimate barely moves it, so the transform is
almost the same either way.

Leakage bites when the fitted quantity depends *strongly* on the specific rows:
target encoding, feature selection by correlation with the label, oversampling
before splitting, or imputing from the full dataset. Scaling is the textbook
example, and it is close to the mildest one.

**What I took from it:** I kept this experiment and reported the null result. The
rule "fit the scaler on training data only" is still correct — it costs nothing and
it is a leak in principle — but I had absorbed it as "otherwise your scores are
inflated" without ever checking the magnitude. The reasoning was right and my sense
of scale was wrong.

---

## 4. Deriving the gradient once explains three algorithms

Linear regression, logistic regression and softmax regression each reduce to the
same gradient:

```
linear    (1/n) X^T (Xw - y)
logistic  (1/n) X^T (p  - y)      p = sigmoid(Xw)
softmax   (1/n) X^T (P  - Y)      P = softmax(XW), Y one-hot
```

`X^T (prediction - target)` in all three. It appears a fourth time as the softmax
cross-entropy gradient inside the neural network, where it is `(p - y)/n`.

That is not a coincidence. Each output activation is paired with the log-likelihood
loss for its distribution, and the activation's derivative cancels exactly against
the term the loss derivative produces. Pair a sigmoid with *squared error* instead
and the cancellation is lost, leaving a gradient with an extra `p(1-p)` factor that
vanishes when the model is confidently wrong — precisely when you need it largest.

**What I took from it:** cross-entropy is not "the loss for classification" by
convention. It is the loss that makes the gradient well-behaved, and the sigmoid /
softmax pairing is the reason. Three algorithms I had learned separately turned out
to be one idea with different output layers.

---

## 5. Vanishing gradients, measured

The claim is that sigmoid's derivative peaks at 0.25, so gradients shrink by at
least 4× per layer on the way back. Test accuracy on digits, 32 units per layer,
lr = 0.05:

| hidden layers | ReLU | tanh | sigmoid |
|---|---:|---:|---:|
| 1 | 0.9800 | 0.9822 | 0.9800 |
| 2 | 0.9800 | 0.9733 | 0.9578 |
| 3 | 0.9756 | 0.9800 | 0.9044 |
| 4 | 0.9667 | 0.9756 | **0.6578** |

Sigmoid loses **34 percentage points** from depth 1 to depth 4. ReLU and tanh lose
1–2. The theoretical claim showed up as a clean monotonic collapse in exactly the
predicted direction.

**What I took from it:** I had filed "ReLU fixed vanishing gradients" as history.
Seeing sigmoid fall to 65.8% at four layers — a depth that is nothing by modern
standards — made it concrete why training deep networks was impractical before
ReLU. It is not a small constant-factor effect.

---

## 6. Adam was the slowest optimiser here

Epochs to reach a training loss of 0.01, same 64-unit network and seed:

| optimiser | rate | epochs to 0.01 | test accuracy |
|---|---|---:|---:|
| Momentum | 0.05 | **8** | 0.9822 |
| SGD | 0.5 | 9 | 0.9889 |
| SGD | 0.1 | 40 | 0.9867 |
| Adam | 0.001 | 45 | 0.9889 |

Adam is the default choice in most tutorials, and here it took **5.6× more epochs
than momentum**. Not because Adam is bad — because its conventional default rate of
0.001 is conservative, and this problem is small and well conditioned. Adam's
per-parameter scaling earns its keep when gradient magnitudes differ wildly across
layers, which is not the case in a two-layer network on 64 clean features.

I also had to fix the metric before this table meant anything. My first version
measured "epochs to get within 10% of each run's own final loss" and returned
56–58 for every optimiser — the loss keeps falling, so a moving target is always
reached near the end. A relative target cannot compare convergence speed; a fixed
one can.

**What I took from it:** "use Adam" is a reasonable default and not a finding.
Also, a badly designed metric produces numbers that look like data.

---

## 7. Throwing away 10% of the variance made the model better

PCA on digits, then k-NN on the projection:

| components | variance kept | k-NN accuracy |
|---|---:|---:|
| 10 | 73.8% | 0.9778 |
| **20** | **89.4%** | **0.9867** |
| 30 | 95.9% | 0.9822 |
| 64 (all) | 100% | 0.9556 |

Twenty components beat all sixty-four by **3.1 points**, while discarding 10.6% of
the variance. Using every component is *worse* than using less than a third of
them.

Two effects, both working the same way. The discarded components hold mostly noise,
so dropping them removes noise rather than signal. And k-NN degrades in high
dimensions regardless — distances concentrate, so "nearest" becomes less meaningful
as dimensions grow.

**What I took from it:** I had understood PCA as lossy compression you accept for
speed or memory. It is also a denoiser, and the compression can *pay for itself* in
accuracy. The reconstruction error falling monotonically while downstream accuracy
peaks in the middle is the whole point.

---

## 8. A useless classifier scoring 99.2%

An imbalanced problem — digit 0 against the rest, positives subsampled to 12:

| model | accuracy | precision | recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| always predict majority | **0.9918** | 0.000 | 0.000 | 0.000 | — |
| logistic regression | 0.9959 | 1.000 | 0.500 | 0.667 | 1.000 |

A model that ignores its input entirely scores **99.18% accuracy**. The real model
beats it by 0.41 points on accuracy — a difference that looks like noise — while
going from finding *nothing* to finding half the positives at perfect precision.

The AUC of 1.000 is the interesting part: the ranking is perfect, so every positive
is scored above every negative. Recall is only 0.5 because the **threshold** is
wrong, not the model. Moving it from 0.5 downward would recover the missed
positives at no cost in precision.

**What I took from it:** accuracy on imbalanced data is not a weak metric, it is an
actively misleading one. And a low recall with a high AUC is a threshold problem,
not a model problem — which means the fix is free.

---

## 9. Three solvers, one answer

The least-squares objective solved three ways, on the same data, compared against
the closed form:

| solver | largest weight difference |
|---|---:|
| normal equation | 0 (exact) |
| batch gradient, 500 iterations | 4.3e-05 |
| batch gradient, 10,000 iterations | 4.3e-05 |
| stochastic gradient, 200 epochs | 3.1e-04 |

Batch gradient descent stops improving after 500 iterations — it hits my
convergence tolerance and exits, so 10,000 iterations gives an identical answer to
500. SGD lands slightly further out, which is the noise it trades for cheaper
steps.

This is the check I found most useful in practice: **agreement between independent
solvers is a correctness test that needs no reference implementation.** If the
closed form and gradient descent disagree, one of them has a bug, and I know that
before comparing against scikit-learn.

Related, the learning rate sweep on the same problem:

| learning rate | outcome |
|---|---|
| 0.0001 | 6.36 after 300 iterations — barely moved |
| 0.5 | converged in **9** iterations |
| 1.0 | diverged to 1.3e+64 |
| 2.0 | overflowed to infinity |

**A factor of two — 0.5 to 1.0 — separates the fastest convergence in the sweep
from total divergence.** There is no gentle degradation at the edge.

---

## 10. Numerical stability is most of the work

Four places where the mathematically correct formula is the wrong code:

| naive form | problem | fix |
|---|---|---|
| `1/(1+exp(-z))` | overflows for large negative z | branch on the sign of z |
| `exp(z)/sum(exp(z))` | overflows for logits ~1000 | subtract the row max first |
| `prod(P(x_j\|c))` | underflows to 0.0 for ~30 features | sum logs instead |
| `inv(X.T @ X) @ X.T @ y` | squares the condition number; singular if collinear | `lstsq` (SVD) |

Each is a one-line difference that changes nothing algebraically. Gaussian naive
Bayes on 64 features is the starkest: multiplying 64 densities of ~0.01 underflows
to exactly zero, every class ties at zero, and the prediction becomes meaningless —
so the log-space version is not an optimisation, it is the only version that works
at all.

The softmax shift is my favourite because the correction is provably free: the
constant cancels in the ratio, so subtracting the maximum changes the output not at
all while guaranteeing the largest exponent is `exp(0) = 1`.

**What I took from it:** I expected implementing ML from scratch to be about
deriving gradients. A large share of the actual work was floating-point behaviour,
and the bugs it produces are quiet — a NaN that propagates, or a probability that
silently becomes zero.

---

## Summary of corrected expectations

| I assumed | Measurement showed |
|---|---|
| 100% dead ReLU means dying ReLU | It meant NaN weights; my ReLU was converting NaN to 0.0 |
| A gradient checker validates the maths | Mine was wrong for every layer after the first |
| Scaling before splitting inflates scores | Mean inflation 0.00000 over 20 seeds |
| Adam converges fastest | 5.6× slower than momentum here, at its default rate |
| Keeping all components is safest | 20 of 64 beat all 64 by 3.1 points |
| 99% accuracy means a good model | A model ignoring its input scored 99.18% |
| Deep sigmoid networks are just slower | 34-point accuracy collapse from 1 to 4 layers |
| High test accuracy means a robust model | 97.8% → 53.1% under a 10%-per-pixel perturbation |
