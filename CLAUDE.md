# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A drone state-estimation neural network thesis project. The model (`JeuralJetwork` in `include/mama.py`) is a Mamba-based sequence-to-sequence network that predicts drone position/velocity deltas from a rolling history of sensor observations. Training data is flight log episodes stored in Zarr format. The trained model is deployed online inside IsaacLab simulation via `include/QuickEstimator.py`.

## Commands

All scripts must be run from the project root so that package imports resolve correctly.

**Training (single run):**
```bash
python train/trainer.py
```

**Training with Hydra multirun sweep:**
```bash
python train/trainer.py --multirun models=rollout,replica training.lr=1e-3,2e-4
```

**Hyperparameter search (Optuna):**
```bash
python tune/optimiser.py
```

**Evaluate the latest sweep across all runs:**
```bash
python evaluate/evaluator.py       # one trajectory at a time, with 3-D plot
python evaluate/batched_evaluator.py  # batched over multiple episodes, parallel
```

**Compare two Zarr datasets:**
```bash
python tools/compare_datasets.py
```

**Convert a PX4 .ulg flight log to a `.pt` tensor:**
```bash
python data/converter.py   # edit hardcoded path at the top first
```

**Smoke-test the model architecture:**
```bash
python model/network.py
```

**Smoke-test the online estimator (loads latest multirun checkpoint):**
```bash
python -m deploy.estimator
python -m deploy.estimator --run multirun/<date>/<time>/0 --envs 4
```

There is no dedicated test runner; `test/` contains standalone scripts run directly with `python`.

## Architecture

### Data flow
1. Raw PX4 `.ulg` flight logs → `flight_log_converter.py` → `.pt` tensors or Zarr `episodes` arrays.
2. Zarr dataset shape: `(num_episodes, seq_len, 23)` stored at `root["episodes"]`.
3. Feature layout (23 dims, normalised): `pos/3` (3), `lin_vel/0.8` (3), `quat_frd` (4), `ang_vel/0.5` (3), `acc/25` (3), `thrust/9.81` (4), `setpoint/3` (3).
4. `include/central_denormaliser.py` inverts this normalisation for metric reporting.

### Model (`model/network.py`)
`JeuralJetwork`: `[B, input_len, n_dim]` → `[B, output_len, out_dim]`.

Pipeline: `in_proj` (Linear+Norm) → encoder Mamba blocks → `time_projector` (time adapter) → decoder Mamba blocks → `out_proj`.

Key choices controlled by config:
- `block_type`: `simple` (Mamba+residual), `advanced` (Mamba+SwiGLU+LayerScale+DropPath), `cls` (CLS-token attention), `simple_cls`.
- `time_adapter`: `linear`, `pool`, `conv`, `attn`, `mlp`, `last_token`. `last_token` takes the final hidden state and projects it to `output_len` steps.
- `mamba_type`: `mamba`, `mamba2`, `mamba3`.

The model predicts **state deltas** (not absolute states). Training adds `pred[..., :pred_dim]` to the current state to get the next-step prediction. `out_dim` is doubled for probabilistic (GML) training modes.

### Training (`train/trainer.py`)
Configured via Hydra (`hydra-cfgs/config.yaml` + `hydra-cfgs/models/*.yaml`). Key training modes (`training.mode`):
- `rollout`: multi-step rollout loss, teacher-forced with scheduled sampling.
- `rollout_horz`: rollout with horizon weighting (later steps weighted higher).
- `gml_rollout`: Gaussian mixture loss after a warm-up phase.
- `standard`: single-step MSE.

Uses AMP (`float16`), `torch.compile(mode="reduce-overhead")`, AdamW, and DataLoader with `fork` multiprocessing.

Checkpoints saved to Hydra's `run_dir` as `.pth` + `loss_history.npy`. Evaluator scripts auto-discover the latest run by scanning `multirun/`.

### Datasets (`data/dataset.py`)
- `QuickDataset2`: random-sampled windows from the Zarr episodes array; used for training.
- `QuickDatasetStraight`: sequential windows from a single episode; used for evaluation.
- `FlightLog`: wraps a pre-loaded tensor for real flight log evaluation.

### Evaluation (`evaluate/evaluator.py`, `evaluate/batched_evaluator.py`)
Both scripts auto-discover the latest Hydra multirun, reconstruct `JeuralJetwork` from the saved `.hydra/config.yaml`, and run autoregressive rollout (feeding the model's own position prediction back as input, not ground-truth). Metrics in `include/metrics.py`: ATE RMSE, drift %, KITTI translation drift, endpoint error.

### Online inference (`deploy/estimator.py`)
`QuickEstimator` is the deployment wrapper for IsaacLab. It maintains a per-environment rolling history buffer and calls `model.forward()` once per control tick across all `num_envs` in a single batched call. Load via `QuickEstimator.from_run(run_dir, num_envs, device)` or `QuickEstimator.from_latest_multirun(multirun_root, ...)`.

### Hyperparameter optimisation (`tune/optimiser.py`)
Optuna TPE + Hyperband pruner. Objective minimises `ATE_RMSE + 0.1*endpoint_error + 0.01*drift_m_per_km`. Best config saved to `best_config.yaml`.

## Key configuration

`config/config.yaml` controls all global hyperparameters. `config/models/rollout.yaml` (and `replica.yaml`) define model architecture. Dataset paths are hardcoded in the YAML — update `dataset.path` and `evaluation.path` when moving between machines.

`torch.compile` is enabled by default; checkpoints save the unwrapped state dict via `model._orig_mod.state_dict()`. Loading strips the `_orig_mod.` prefix automatically in evaluator scripts.
