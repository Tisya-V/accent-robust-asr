#!/bin/bash
#SBATCH --job-name=exp2_bridge_train
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/exp2_bridge_train_%j.out
#SBATCH --error=logs/exp2_bridge_train_%j.out

# Train E2 Latent Diffusion Bridge model on SLURM
#
# Usage on SLURM cluster:
# 1. chmod +x src/experiments/scripts/slurm/exp2_bridge_train.sh
# 2. sbatch src/experiments/scripts/slurm/exp2_bridge_train.sh

set -e

# Source centralized environment configuration
source scripts/slurm_env.sh

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "E2 Latent Diffusion Bridge Training Job"
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
echo "Starting Bridge Training..."

python -m src.experiments.exp2_latent_diffusion_bridge.train \
    --mapping_train_path src/experiments/exp2_latent_diffusion_bridge/data/mapping_train.json \
    --mapping_val_path src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev.json \
    --out_dir models/bridge \
    --n_epochs 50 \
    --batch_size 24 \
    --lr 1e-4 \
    --weight_decay 1e-3 \
    --sigma_max 0.5 \
    --num_workers 4 \
    --profile

echo "✅ Bridge training complete."
