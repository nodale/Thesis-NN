# Training-Mode Ablation: attention / out_dim=10

**Date:** 2026-07-27
**Fixed config:** `models.architecture.block_type=attention`, `models.out_dim=10`, all other config defaults (`config/config.yaml` + `config/models/rollout.yaml`).
**Data:** train = `patient_one_data.zarr` (5000 episodes), eval = `patient_two_data.zarr` (50 episodes).
**GPU:** 0 (`CUDA_VISIBLE_DEVICES=0`), seed=0.
**Runs:** `ablation_runs/2026-07-27_11-13-35_train_mode/`
**Eval:** `evaluate.batched_evaluator`, autoregressive rollout metrics.

## Scenarios

1. **rollout only** — `training.mode=rollout epochs=5` (scheduled-sampling ramp over all 5 epochs).
2. **rollout(5) → gml_rollout_pure(5)** — two-phase: phase 1 `training.mode=rollout epochs=5`; phase 2 resumes from phase-1 checkpoint with a new `gml_rollout_pure` mode (added to `train/trainer.py`) running `training.mode=gml_rollout_pure epochs=5`, its own full 5-epoch scheduled-sampling ramp, pure Gaussian-NLL loss throughout.
3. **rollout(5) → gml_rollout(1 @ 0%)** — single run, `training.mode=gml_rollout epochs=6`. This mode's existing behavior in `train/trainer.py` runs plain rollout for `epochs-1` steps then switches to the Gaussian-NLL loss for exactly the last epoch with scheduled-sampling forced to 0%.

## Results

| Scenario | ATE RMSE | Mean Error | Max Error | Endpoint Error | KITTI Drift | Drift % | Drift (m/km) |
|---|---|---|---|---|---|---|---|
| **rollout only** | **0.0952** | **0.0910** | **0.1357** | **0.0986** | **24.43** | **2.18%** | **21.79** |
| rollout(5) → gml_rollout(1 @ 0%) | 1.4123 | 1.2672 | 2.3812 | 2.3513 | 184.07 | 53.20% | 532.01 |
| rollout(5) → gml_rollout_pure(5) | 16.4671 | 14.2932 | 28.4975 | 28.4975 | 1762.63 | 633.75% | 6337.49 |

(Trajectory length constant at 4.749 across all runs — same eval set.)

## Conclusion

Plain `rollout` training wins by a wide margin on every autoregressive-rollout metric. Introducing the GML (Gaussian-NLL) loss phase consistently hurts rollout accuracy, and the damage scales with how much of training is spent in that phase:

- 1 epoch of GML (at 0% scheduled sampling) already degrades ATE RMSE by ~15x (0.095 → 1.41).
- 5 epochs of GML (with its own full scheduled-sampling ramp) causes the model to effectively diverge under autoregressive rollout (ATE RMSE 16.5, drift 634%).

Likely explanation: the GML objective optimizes for calibrated mean/variance (Gaussian NLL) rather than tight point-prediction accuracy. Under closed-loop rollout, small mean-prediction degradations compound step-over-step, and this effect is amplified the longer training spends under the GML loss instead of plain MSE.

**Recommendation:** avoid `gml_rollout` / `gml_rollout_pure` phases if closed-loop rollout accuracy is the deployment metric; `rollout`-only training is the strongest baseline of the three tested here.

## Caveats

- Single seed (seed=0) per scenario — no variance estimate across seeds.
- `gml_rollout`'s internal warmup ramp is coupled to `cfg.epochs` (see `train/trainer.py`), so scenario 3's rollout phase (epochs 0-4 of a 6-epoch run) uses a slightly different schedule ramp than scenario 1's standalone 5-epoch run (ramp denominator differs: `epochs-1` = 5 vs. 4).
- Checkpoint resume between phase 1 and phase 2 of scenario 2 only transfers shape-compatible weights (a bugfix was added to `train/trainer.py` for this) — the final `out_proj` layer in phase 2 is therefore trained from scratch (mu+var head), while the rest of the network is warm-started from the phase-1 rollout checkpoint.
