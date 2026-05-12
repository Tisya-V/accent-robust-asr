#!/bin/bash
#SBATCH --job-name=exp1_eval
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

# Experiment 1 evaluation: MiniMDM token accuracy on test set
#
# Usage:
# sbatch src/experiments/scripts/slurm/eval_exp1.sh [checkpoint_path]

source scripts/slurm_env.sh

cd "${PROJECT_ROOT}"

set -e

echo "==============================="
echo "Experiment 1: Token Accuracy Evaluation"
echo "==============================="
echo ""

echo "Checking GPU/SLURM setup..."
nvidia-smi | grep "GPU\|Memory"
echo ""

CHECKPOINT="${1:-results/experiment1_stage1/low_perturb_medium_masking/checkpoint.pt}"
DATA_ROOT="${TEST_DATA_DIR:-/vol/gpudata/tsv22-dev_test/data/processed/test}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo "Test data: $DATA_ROOT"
echo ""

echo "Starting evaluation..."
python -u -m src.experiments.exp1_text_correction.eval \
    --checkpoint "$CHECKPOINT" \
    --device cuda \
    --data_root "$DATA_ROOT"

echo ""
echo "✅ Evaluation finished."
