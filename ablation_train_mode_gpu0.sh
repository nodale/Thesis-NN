#!/usr/bin/env bash
# Training-mode ablation for the attention/out10 architecture (GPU 0).
#
# Compares three training-mode strategies, all on block_type=attention,
# out_dim=10, rest of config default:
#   1. rollout only            - rollout, 5 epochs (+ scheduled-sampling ramp)
#   2. rollout(5) + gml(5)     - rollout 5 epochs, then a second run resumes
#                                 from that checkpoint and runs gml_rollout_pure
#                                 for 5 more epochs (+ its own ramp)
#   3. rollout(5) + gml(1@0%)  - single gml_rollout run, epochs=6: the existing
#                                 gml_rollout mode already does rollout for all
#                                 but the last epoch, then 1 gml epoch at 0%
#                                 scheduled sampling.
set -euo pipefail
GPU_ID=0
ARCH_OVERRIDES="models.architecture.block_type=attention models.out_dim=10"
DATA_OVERRIDES="dataset.path=/media/egghead/Scratch/joey/simulation_data/patient_one_data.zarr evaluation.path=/media/egghead/Scratch/joey/simulation_data/patient_two_data.zarr"

cd "$(dirname "$0")"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
RUNS_DIR="$(pwd)/ablation_runs/${RUN_ID}_train_mode"
RESULTS_DIR="ablation_train_mode/${RUN_ID}"
mkdir -p "$RUNS_DIR" "$RESULTS_DIR"

# --- scenario 1: rollout only ---
echo "=== Training: attn_out10_rollout_only ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python -m train.trainer --multirun \
  "hydra.sweep.dir=${RUNS_DIR}/attn_out10_rollout_only" \
  ${ARCH_OVERRIDES} ${DATA_OVERRIDES} training.mode=rollout epochs=5 seed=0 \
  > "${RESULTS_DIR}/attn_out10_rollout_only.train.log" 2>&1

# --- scenario 3: rollout(5) + gml(1 @ 0% schedule), single run ---
echo "=== Training: attn_out10_rollout5_gml1 ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python -m train.trainer --multirun \
  "hydra.sweep.dir=${RUNS_DIR}/attn_out10_rollout5_gml1" \
  ${ARCH_OVERRIDES} ${DATA_OVERRIDES} training.mode=gml_rollout epochs=6 seed=0 \
  > "${RESULTS_DIR}/attn_out10_rollout5_gml1.train.log" 2>&1

# --- scenario 2: rollout(5) then gml_rollout_pure(5), two sequential runs ---
echo "=== Training: attn_out10_rollout5_gml5 (phase 1: rollout) ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python -m train.trainer --multirun \
  "hydra.sweep.dir=${RUNS_DIR}/attn_out10_rollout5_gml5_phase1" \
  ${ARCH_OVERRIDES} ${DATA_OVERRIDES} training.mode=rollout epochs=5 seed=0 \
  > "${RESULTS_DIR}/attn_out10_rollout5_gml5_phase1.train.log" 2>&1

PHASE1_CKPT="${RUNS_DIR}/attn_out10_rollout5_gml5_phase1/0/hydra-test.pth"
if [ ! -f "$PHASE1_CKPT" ]; then
  echo "ERROR: expected phase-1 checkpoint not found at ${PHASE1_CKPT}" >&2
  exit 1
fi

echo "=== Training: attn_out10_rollout5_gml5 (phase 2: gml_rollout_pure, resumed) ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python -m train.trainer --multirun \
  "hydra.sweep.dir=${RUNS_DIR}/attn_out10_rollout5_gml5_phase2" \
  ${ARCH_OVERRIDES} ${DATA_OVERRIDES} training.mode=gml_rollout_pure epochs=5 seed=0 \
  checkpoint.load=true "checkpoint.path=${PHASE1_CKPT}" \
  > "${RESULTS_DIR}/attn_out10_rollout5_gml5_phase2.train.log" 2>&1

echo "All training finished."

# --- evaluate each final scenario ---
declare -A EVAL_DIRS=(
  [attn_out10_rollout_only]="${RUNS_DIR}/attn_out10_rollout_only"
  [attn_out10_rollout5_gml1]="${RUNS_DIR}/attn_out10_rollout5_gml1"
  [attn_out10_rollout5_gml5]="${RUNS_DIR}/attn_out10_rollout5_gml5_phase2"
)

for name in "${!EVAL_DIRS[@]}"; do
  echo "=== Evaluating: ${name} ==="
  SWEEP_DIR="${EVAL_DIRS[$name]}" EVAL_OUTPUT="${RESULTS_DIR}/${name}.json" \
  CUDA_VISIBLE_DEVICES=$GPU_ID python -m evaluate.batched_evaluator
done

echo "All ablation results saved under ${RESULTS_DIR}/"
