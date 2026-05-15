#!/bin/bash
#SBATCH --job-name=exp1_eval_integration
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

# Experiment 1 evaluation: MiniMDM integrated into Whisfusion decoding
#
# Usage:
# sbatch src/experiments/scripts/slurm/eval_exp1_whisfusion_integration.sh \
#   [corrector_checkpoint] [whisfusion_checkpoint] [mask_ratio_schedule]

source scripts/slurm_env.sh

cd "${PROJECT_ROOT}"

set -e

echo "==============================="
echo "Experiment 1: Whisfusion Integration Evaluation"
echo "==============================="
echo ""

echo "Checking GPU/SLURM setup..."
nvidia-smi | grep "GPU\|Memory"
echo ""

CORRECTOR_CHECKPOINT="${1:-results/experiment1_stage1/low_perturb_medium_masking/checkpoint.pt}"
WHISFUSION_CHECKPOINT="${2:-models/whisfusion_finetuned/stage2_decoder/whisfusion_stage2_decoder.pt}"
MASK_RATIO_SCHEDULE="${3:-0.9,0.7,0.5,0.3}"
DATA_ROOT="${TEST_DATA_DIR:-/vol/gpudata/tsv22-dev_test/data/processed/test}"

if [ ! -f "$CORRECTOR_CHECKPOINT" ]; then
    echo "ERROR: Corrector checkpoint not found: $CORRECTOR_CHECKPOINT"
    exit 1
fi

if [ ! -f "$WHISFUSION_CHECKPOINT" ]; then
    echo "ERROR: Whisfusion checkpoint not found: $WHISFUSION_CHECKPOINT"
    exit 1
fi

echo "Corrector checkpoint: $CORRECTOR_CHECKPOINT"
echo "Whisfusion checkpoint: $WHISFUSION_CHECKPOINT"
echo "Mask ratio schedule: $MASK_RATIO_SCHEDULE"
echo "Test data: $DATA_ROOT"
echo ""

echo "Starting evaluation..."
python -u -m src.experiments.exp1_text_correction.eval_whisfusion_integration \
    --corrector_checkpoint "$CORRECTOR_CHECKPOINT" \
    --whisfusion_checkpoint "$WHISFUSION_CHECKPOINT" \
    --device cuda \
    --data_root "$DATA_ROOT" \
    --mask_ratio_schedule "$MASK_RATIO_SCHEDULE"

echo ""
echo "✅ Evaluation finished."
