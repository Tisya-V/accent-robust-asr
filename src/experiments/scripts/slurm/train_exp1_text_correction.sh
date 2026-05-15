#!/bin/bash
#SBATCH --job-name=exp1_text_correction
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

# Experiment 1: Text-level diffusion with phoneme perturbations
#
# Usage:
# 1. chmod +x src/experiments/scripts/train_exp1_text_correction.sh
# 2. sbatch src/experiments/scripts/train_exp1_text_correction.sh
# OR
# 3. ./src/experiments/scripts/train_exp1_text_correction.sh (for interactive submission)

# Source centralized environment configuration
source scripts/slurm_env.sh

cd "${PROJECT_ROOT}"

export NCCL_P2P_DISABLE=1

set -e

echo "==============================="
echo "Experiment 1: Text-level Diffusion"
echo "==============================="
echo ""

echo "Checking GPU/SLURM setup..."
nvidia-smi
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python - <<'PY'
import os, torch
print("DEBUG torch.cuda.is_available =", torch.cuda.is_available())
print("DEBUG torch.cuda.device_count =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"DEBUG cuda[{i}] =", torch.cuda.get_device_name(i))
PY

echo ""
echo "Staging data to local node storage ($TMPDIR)..."
mkdir -p "$TMPDIR/train" "$TMPDIR/dev"

echo "Copying train data..."
rsync -a --inplace --info=progress2 "${TRAIN_DATA_DIR}"/ "$TMPDIR/train/"

echo "Copying dev data..."
rsync -a --inplace --info=progress2 "${DEV_DATA_DIR}"/ "$TMPDIR/dev/"

echo "✓ Data staging complete"

echo ""
echo "Starting Experiment 1 training..."
echo ""

# Use config from command line arg, or default to phoneme_perturb_low
CONFIG="${1:-src/experiments/exp1_text_correction/configs/low_perturb_medium_masking.json}"

python -u -m src.experiments.exp1_text_correction.train \
    --config "$CONFIG" \
    --device cuda \
    --train_data_dir "$TMPDIR/train" \
    --val_data_dir "$TMPDIR/dev"

echo ""
echo "✅ Experiment 1 training finished."
