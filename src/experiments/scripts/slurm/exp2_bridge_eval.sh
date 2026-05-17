#!/bin/bash
#SBATCH --job-name=exp2_bridge_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp2_bridge_eval_%j.out
#SBATCH --error=logs/exp2_bridge_eval_%j.err

# Evaluate E2 Latent Diffusion Bridge model on SLURM
#
# Usage on SLURM cluster:
# 1. chmod +x src/experiments/scripts/slurm/exp2_bridge_eval.sh
# 2. sbatch src/experiments/scripts/slurm/exp2_bridge_eval.sh

set -e

# Source centralized environment configuration
source scripts/slurm_env.sh

cd "${PROJECT_ROOT}"

# Create real-time log file
RUNTIME_LOG="logs/exp2_bridge_eval_runtime_${SLURM_JOB_ID}.log"
mkdir -p logs
exec > >(tee -a "$RUNTIME_LOG")
exec 2>&1

echo "=========================================="
echo "E2 Latent Diffusion Bridge Evaluation Job"
echo "Real-time log: $RUNTIME_LOG"
echo "Track with: tail -f $RUNTIME_LOG"
echo "=========================================="
echo ""

echo "Checking GPU setup..."
nvidia-smi
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_GPUS=$SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python - <<'PY'
import os, torch
print("DEBUG CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("DEBUG torch.cuda.is_available =", torch.cuda.is_available())
print("DEBUG torch.cuda.device_count =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"DEBUG cuda[{i}] =", torch.cuda.get_device_name(i))
PY

echo -e "\n\n==============================\n\n"
echo "Starting Bridge Evaluation..."

python -m src.experiments.exp2_latent_diffusion_bridge.eval \
    --bridge_ckpt models/bridge/checkpoint_best.pt \
    --decoder whisper \
    --output_dir results/bridge_eval \
    --n_steps 20

echo "✅ Bridge evaluation complete."
