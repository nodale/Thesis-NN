# Loss comparison: `simple_mamba_out6` vs `simple_gml_mamba_out10`

## TL;DR

The two runs are **not directly comparable on raw loss values** because they use
different loss functions:

- `simple_mamba_out6` → **6-dim regression head trained with MSE** on all cycles.
- `simple_gml_mamba_out10` → **10-dim "Gaussian" head** (5 means + 5 log-variances)
  trained with **Gaussian negative log-likelihood (GML)** from cycle 1 onward.
  Cycle 0 is an MSE-scale warmup.

Comparing the plain MSE magnitudes ("out6 has smaller numbers") is meaningless —
the two heads emit different quantities and are optimised against fundamentally
different objectives. The good downstream estimation from `out10` is consistent
with what a GML head buys you: it learns a full predictive distribution, so the
model is rewarded for producing well-calibrated variance instead of just point
estimates.

## Data inspected

`generations/train_losses/cycle_{0..3}.npy` in each folder, each an
`np.float32` array of shape `(32,)`. Values inside a cycle are monotonic
(likely already sorted / evenly sub-sampled by the trainer, not raw epoch
history), so treat them as a per-cycle *loss profile* rather than the exact
training curve.

| run                          | cycle | min          | max          | mean         |
|------------------------------|-------|-------------:|-------------:|-------------:|
| `simple_mamba_out6`          | 0     | 1.51e-06     | 2.10e-05     | 1.16e-05     |
| `simple_mamba_out6`          | 1     | 7.64e-07     | 1.60e-05     | 8.80e-06     |
| `simple_mamba_out6`          | 2     | 9.56e-07     | 1.32e-05     | 7.44e-06     |
| `simple_mamba_out6`          | 3     | 8.14e-07     | 1.22e-05     | 6.39e-06     |
| `simple_gml_mamba_out10`     | 0     | 6.73e-07     | 9.13e-06     | 5.57e-06     |
| `simple_gml_mamba_out10`     | 1     | -34.71       | -0.066       | -6.39        |
| `simple_gml_mamba_out10`     | 2     | -599.9       | -4.58        | -240.8       |
| `simple_gml_mamba_out10`     | 3     | -813.4       | -29.89       | -462.1       |

## What the numbers actually say

### `simple_mamba_out6` (MSE)
- Numbers are tiny (1e-6 – 2e-5). That is a scale artefact — the regression
  targets are already in a small numerical range, so a well-fit MSE will look
  "microscopic" without saying anything about physical estimation quality.
- Cycle-to-cycle mean drops from 1.16e-05 → 6.39e-06. Training is progressing,
  but there is no distributional signal — the model has no way to express
  "I'm unsure here", and every dimension is treated with the same implicit
  homoscedastic weight. If some outputs are noisy or heavy-tailed, MSE will
  either be dominated by them or (if they're normalised down) ignore them.

### `simple_gml_mamba_out10` (Gaussian NLL)
- Cycle 0 sits on the same MSE-scale as the `out6` run (they even start very
  close: 5.6e-06 vs 1.2e-05). This confirms cycle 0 is an MSE warmup on the
  mean head before the variance head is activated.
- From cycle 1 on, the loss is Gaussian NLL and therefore **negative and
  unbounded below** — more negative means higher predictive log-likelihood.
- The mean loss goes -6.4 → -240.8 → -462.1 across cycles 1-3. That monotone
  descent means the model is putting sharper (lower-variance) Gaussians on the
  right values, i.e. it is *calibrated*, not just point-accurate.
- Very negative NLL is exactly what you'd expect from a run that "estimates
  really well" — because the GML head is being scored on how confident it is
  in the truth, not just how close its point estimate is.

## Why `out6` gives "shit estimation" despite tiny numbers

1. **No uncertainty modelling.** A pure MSE head cannot say "this observation
   is noisy" — it will overconfidently interpolate through outliers and
   under-weight regions with legitimately higher noise. Downstream consumers
   that need a variance (Kalman-style fusion, gating, etc.) get nothing useful.
2. **Loss magnitude ≠ estimation quality.** The MSE values are small because
   the targets are small, not because the model is good. Look at the paired
   estimation plots in `viz/00_current_champion_estimation.png` and
   `viz/05_run1_champion00*_gen00*_estimation.png` for the real quality signal.
3. **Different output dimensionality.** `out6` predicts 6 quantities directly;
   `out10` predicts 5 means + 5 log-variances. If `out6` is missing a channel
   the estimator downstream needs, that alone can cause the poor behaviour
   independently of the loss choice.

## Recommended follow-ups

- Don't compare MSE and GML numerically. Instead evaluate both runs with the
  *same* downstream metric — e.g. per-cycle RMSE or ATE from
  `04_cycle_ate.png` — plus a calibration metric (NLL / CRPS) for the GML run.
- Sanity-check whether the arrays in `train_losses/` are truly per-step or
  already sorted. If they are sorted, plot the raw epoch history from the
  `.log` files or add the raw curve to `generations/train_losses/`.
- If `out6` is genuinely the target-shape you need, at minimum add a
  log-variance head and re-train with GML so it competes on the same footing.

## Plot

See `loss_comparison.png` next to this file:

- top-left: `out6` MSE curves for all four cycles (log-y).
- top-right: cycle-0 MSE-scale comparison of both runs (they start on the same
  order of magnitude).
- bottom-left: `out10` cycles 1-3 in NLL space, showing how much more negative
  (better) it gets each cycle.
- bottom-right: per-cycle mean loss for both runs on twin axes — a visual
  reminder that MSE and NLL live on incompatible scales.
