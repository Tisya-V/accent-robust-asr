#!/bin/bash
#SBATCH --job-name=exp1_text_correction
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

# Experiment 1: Text-level diffusion with phoneme perturbations
#
# Usage:
# 1. chmod +x src/experiments/scripts/train_exp1_text_correction.sh
# 2. sbatch src/experiments/scripts/train_exp1_text_correction.sh
# OR
# 3. ./src/experiments/scripts/train_exp1_text_correction.sh (for interactive submission)

export HF_HOME=/vol/bitbucket/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/vol/bitbucket/$USER/.cache/huggingface/transformers
export XDG_CACHE_HOME=/vol/bitbucket/$USER/.cache
export MPLCONFIGDIR=/vol/bitbucket/$USER/.cache/matplotlib

export NCCL_P2P_DISABLE=1

export PATH=/vol/bitbucket/$USER/accent-robust-asr/.venv/bin/:$PATH
source activate

source /vol/cuda/12.4.0/setup.sh

cd /vol/bitbucket/$USER/accent-robust-asr/

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
echo "Starting Experiment 1 training..."
echo ""

python -u -m src.experiments.exp1_text_correction.train \
    --config src/experiments/exp1_text_correction/configs/phoneme_perturb_low.json  \
    --device cuda

echo ""
echo "✅ Experiment 1 training finished."
