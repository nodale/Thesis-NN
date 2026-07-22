#!/usr/bin/env bash
# Ablation study evaluator — evaluate an already-trained ablation run.
#
# Usage: ./ablation_eval_only.sh <RUN_ID>
#   <RUN_ID> must match an existing ablation_runs/<RUN_ID>/ directory
#   produced by a previous ablation.sh training run.
#
# Same CONFIGS list as ablation.sh (must match the names used for training).
# Each config is evaluated in turn and results are stored under
# ablation_results/<RUN_ID>/.
set -euo pipefail

cd "$(dirname "$0")"

CONFIGS=(
  "baseline_simple_mamba       	models.architecture.block_type=simple models.architecture.mamba_type=mamba training.mode=rollout"
  "simple_mamba2                models.architecture.block_type=simple models.architecture.mamba_type=mamba2 training.mode=rollout"
  "advanced_mamba3              models.architecture.block_type=advanced models.architecture.mamba_type=mamba3 training.mode=rollout"
  "simple_cls_block             models.architecture.block_type=simple_cls training.mode=rollout"
  "cls_block                    models.architecture.block_type=cls training.mode=rollout"
  "advanced_deep                models.architecture.block_type=advanced models.architecture.mamba_type=mamba models.architecture.n_encoder_layers=4 models.architecture.n_decoder_layers=2 training.mode=rollout"
)

RUN_ID="${1:-}"
if [ -z "$RUN_ID" ]; then
  echo "Usage: $0 <RUN_ID>" >&2
  echo "Available runs:" >&2
  ls -1 ablation_runs 2>/dev/null >&2
  exit 1
fi

RUNS_DIR="ablation_runs/${RUN_ID}"
if [ ! -d "$RUNS_DIR" ]; then
  echo "Error: ${RUNS_DIR} does not exist." >&2
  exit 1
fi

RESULTS_DIR="ablation_results/${RUN_ID}"
mkdir -p "$RESULTS_DIR"

# --- evaluate each config in turn, storing results as JSON ---
for entry in "${CONFIGS[@]}"; do
  name="${entry%% *}"

  echo "=== Evaluating: ${name} ==="
  SWEEP_DIR="${RUNS_DIR}/${name}" EVAL_OUTPUT="${RESULTS_DIR}/${name}.json" \
    python -m evaluate.batched_evaluator \
    > "${RESULTS_DIR}/${name}.eval.log" 2>&1
done

echo "Evaluation done. Results saved under ${RESULTS_DIR}/"

# --- collect all per-config JSON results into one markdown table ---
SUMMARY="${RESULTS_DIR}/summary.md"

METRIC_KEYS=(
  trajectory_length ate_rmse mean_error max_error
  endpoint_error kitti_translation_drift drift_percent drift_m_per_km
)

{
  header="| name | overrides |"
  sep="|---|---|"
  for k in "${METRIC_KEYS[@]}"; do
    header+=" ${k} |"
    sep+="---|"
  done
  echo "$header"
  echo "$sep"

  for entry in "${CONFIGS[@]}"; do
    name="${entry%% *}"
    json="${RESULTS_DIR}/${name}.json"
    [ -f "$json" ] || continue

    jq -r --arg name "$name" '
      .runs | to_entries[] |
      [$name, .value.overrides,
       .value.metrics.trajectory_length, .value.metrics.ate_rmse,
       .value.metrics.mean_error, .value.metrics.max_error,
       .value.metrics.endpoint_error, .value.metrics.kitti_translation_drift,
       .value.metrics.drift_percent, .value.metrics.drift_m_per_km
      ] | "| " + join(" | ") + " |"
    ' "$json"
  done
} > "$SUMMARY"

echo "Summary table saved to ${SUMMARY}"
