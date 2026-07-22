#!/usr/bin/env bash
# Ablation study runner.
#
# Add one config (a string of hydra overrides) per entry in CONFIGS below.
# Each config is run with `python -m train.trainer --multirun <config>`,
# using hydra's joblib launcher (n_jobs=4, set in config/config.yaml) so up
# to 4 processes run at a time within that config's sweep.
# (invoked as `-m` so the `model`/`data` package imports resolve correctly)
set -euo pipefail

cd "$(dirname "$0")"

CONFIGS=(
  "baseline_rollout          models=rollout training.mode=rollout"
  "baseline_standard         models=rollout training.mode=standard"
  "replica_rollout           models=replica training.mode=rollout"
  # "replica_lr_sweep        models=replica training.lr=1e-3,2e-4"
)

RESULTS_DIR="ablation_results/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$RESULTS_DIR"

for entry in "${CONFIGS[@]}"; do
  name="${entry%% *}"
  cfg="${entry#* }"

  echo "=== Running ablation config: ${name} (${cfg}) ==="
  python -m train.trainer --multirun ${cfg}

  echo "=== Evaluating: ${name} ==="
  EVAL_OUTPUT="${RESULTS_DIR}/${name}.json" python -m evaluate.batched_evaluator
done

echo "All ablation results saved under ${RESULTS_DIR}/"
