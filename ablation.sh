#!/usr/bin/env bash
# Ablation study runner.
#
# Add one config (a short name + hydra overrides) per entry in CONFIGS below.
# Up to MAX_PARALLEL configs train simultaneously (separate `python -m
# train.trainer` processes, each pinned to its own hydra.sweep.dir so they
# never collide). Once all training finishes, each config is evaluated in
# turn and results are collected under one results folder.
set -euo pipefail

cd "$(dirname "$0")"

CONFIGS=(
  "baseline_simple_mamba       	models.architecture.block_type=simple models.architecture.mamba_type=mamba training.mode=rollout"
  "simple_mamba2                models.architecture.block_type=simple models.architecture.mamba_type=mamba2 training.mode=rollout"
  "advanced_mamba3              models.architecture.block_type=advanced models.architecture.mamba_type=mamba3 training.mode=rollout"
  "simple_cls_block             models.architecture.block_type=simple_cls training.mode=rollout"
  "cls_block                    models.architecture.block_type=cls training.mode=rollout"
  "advanced_deep                models.architecture.block_type=advanced models.architecture.mamba_type=mamba models.architecture.n_encoder_layers=4 models.architecture.n_decoder_layers=2 training.mode=rollout"
  "simple_mamba_gml_rollout     models.architecture.block_type=simple models.architecture.mamba_type=mamba training.mode=gml_rollout"
  "simple_mamba_short_rollout   models.architecture.block_type=simple models.architecture.mamba_type=mamba training.mode=rollout training.rollout_steps=16"
  "simple_mamba_long_rollout    models.architecture.block_type=simple models.architecture.mamba_type=mamba training.mode=rollout training.rollout_steps=48"
  "simple_mamba_out6            models.architecture.block_type=simple models.architecture.mamba_type=mamba training.mode=rollout models.out_dim=6"
  "simple_mamba_out10           models.architecture.block_type=simple models.architecture.mamba_type=mamba training.mode=rollout models.out_dim=10"
  "simple_mamba_out13           models.architecture.block_type=simple models.architecture.mamba_type=mamba training.mode=rollout models.out_dim=13"
)

MAX_PARALLEL="${MAX_PARALLEL:-4}"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
RUNS_DIR="ablation_runs/${RUN_ID}"
RESULTS_DIR="ablation_results/${RUN_ID}"
mkdir -p "$RUNS_DIR" "$RESULTS_DIR"

# --- train all configs, MAX_PARALLEL at a time ---
pids=()
for entry in "${CONFIGS[@]}"; do
  name="${entry%% *}"
  cfg="${entry#* }"

  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    wait -n
  done

  echo "=== Training config: ${name} (${cfg}) ==="
  python -m train.trainer --multirun "hydra.sweep.dir=${RUNS_DIR}/${name}" ${cfg} \
    > "${RESULTS_DIR}/${name}.train.log" 2>&1 &
  pids+=($!)
done

wait "${pids[@]}"
echo "All training finished."

# --- evaluate each config in turn ---
for entry in "${CONFIGS[@]}"; do
  name="${entry%% *}"

  echo "=== Evaluating: ${name} ==="
  SWEEP_DIR="${RUNS_DIR}/${name}" EVAL_OUTPUT="${RESULTS_DIR}/${name}.json" \
    python -m evaluate.batched_evaluator
done

echo "All ablation results saved under ${RESULTS_DIR}/"
