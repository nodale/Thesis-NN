#!/usr/bin/env bash
# Ablation study evaluator — evaluate an already-trained ablation run.
#
# Usage: ./ablation_eval_only.sh <RUN_ID>
#   <RUN_ID> must match an existing ablation_runs/<RUN_ID>/ directory
#   produced by a previous ablation.sh training run.
#
# Evaluates every subdirectory under ablation_runs/<RUN_ID>/ and stores
# results under ablation_results/<RUN_ID>/.
set -euo pipefail

cd "$(dirname "$0")"

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

RESULTS_DIR="ablation_results_1000_all/${RUN_ID}"
mkdir -p "$RESULTS_DIR"

# --- evaluate each config directory ---
for config_dir in "${RUNS_DIR}"/*/; do
  [ -d "$config_dir" ] || continue

  name="$(basename "$config_dir")"

  echo "=== Evaluating: ${name} ==="

  SWEEP_DIR="$config_dir" \
  EVAL_OUTPUT="${RESULTS_DIR}/${name}.json" \
    python -m evaluate.batched_evaluator \
    > "${RESULTS_DIR}/${name}.eval.log" 2>&1
done

echo "Evaluation done. Results saved under ${RESULTS_DIR}/"

## --- collect all per-config JSON results into one markdown table ---
#SUMMARY="${RESULTS_DIR}/summary.md"
#
#METRIC_KEYS=(
#  trajectory_length
#  ate_rmse
#  mean_error
#  max_error
#  endpoint_error
#  kitti_translation_drift
#  drift_percent
#  drift_m_per_km
#)
#
#{
#  header="| name | overrides |"
#  sep="|---|---|"
#
#  for k in "${METRIC_KEYS[@]}"; do
#    header+=" ${k} |"
#    sep+="---|"
#  done
#
#  echo "$header"
#  echo "$sep"
#
#  for json in "${RESULTS_DIR}"/*.json; do
#    [ -f "$json" ] || continue
#
#    name="$(basename "$json" .json)"
#
#    jq -r --arg name "$name" '
#      .runs | to_entries[] |
#      [$name, .value.overrides,
#       .value.metrics.trajectory_length,
#       .value.metrics.ate_rmse,
#       .value.metrics.mean_error,
#       .value.metrics.max_error,
#       .value.metrics.endpoint_error,
#       .value.metrics.kitti_translation_drift,
#       .value.metrics.drift_percent,
#       .value.metrics.drift_m_per_km
#      ] | "| " + join(" | ") + " |"
#    ' "$json"
#  done
#} > "$SUMMARY"
#
#echo "Summary table saved to ${SUMMARY}"
