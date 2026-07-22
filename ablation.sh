#!/usr/bin/env bash
# Ablation study runner.
#
# Add one config (a string of hydra overrides) per entry in CONFIGS below.
# Each config is run with `python train/trainer.py --multirun <config>`,
# using hydra's joblib launcher (n_jobs=4, set in config/config.yaml) so up
# to 4 processes run at a time within that config's sweep.
set -euo pipefail

cd "$(dirname "$0")"

CONFIGS=(
  "models=rollout training.mode=rollout"
  "models=rollout training.mode=standard"
  "models=replica training.mode=rollout"
  # "models=replica training.lr=1e-3,2e-4"
)

for cfg in "${CONFIGS[@]}"; do
  echo "=== Running ablation config: ${cfg} ==="
  python train/trainer.py --multirun ${cfg}
done
