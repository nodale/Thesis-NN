#!/usr/bin/env bash
# Ablation study runner.
#
# Add one config (a short name + hydra overrides) per entry in CONFIGS below.
# Up to MAX_PARALLEL configs train simultaneously (separate `python -m
# train.trainer` processes, each pinned to its own hydra.sweep.dir so they
# never collide). Once all training finishes, each config is evaluated in
# turn and results are collected under one results folder.
set -euo pipefail
GPU_ID=1

cd "$(dirname "$0")"

CONFIGS=(
  "cls_gml_1e0d_out10		  models.architecture.block_type=cls  training.mode=gml_rollout models.out_dim=10 models.architecture.n_encoder_layers=1 models.architecture.n_decoder_layers=0"
  "cls_1e0d_out10		          models.architecture.block_type=cls  training.mode=rollout models.out_dim=10 models.architecture.n_encoder_layers=1 models.architecture.n_decoder_layers=0"
)

MAX_PARALLEL="${MAX_PARALLEL:-6}"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
RUNS_DIR="ablation_runs/${RUN_ID}"
RESULTS_DIR="ablation_results_1000_all/${RUN_ID}"
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
  CUDA_VISIBLE_DEVICES=$GPU_ID python -m train.trainer --multirun "hydra.sweep.dir=${RUNS_DIR}/${name}" ${cfg} \
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
  CUDA_VISIBLE_DEVICES=$GPU_ID python -m evaluate.batched_evaluator
done

echo "All ablation results saved under ${RESULTS_DIR}/"
