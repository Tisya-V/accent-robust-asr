#!/bin/bash
#SBATCH --job-name=whisfusion_stage1_adapter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=24GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/whisfusion_stage1_adapter_%j.out
#SBATCH --error=logs/whisfusion_stage1_adapter_%j.err

# Starts Stage 1 training - trains only the cross-attention adapter
#
# Usage on HPC cluster with SLURM:
# 1. chmod +x src/training/slurm_jobs/03_train_stage1_adapter.sh
# 2. sbatch src/training/slurm_jobs/03_train_stage1_adapter.sh

set -e

# Source centralized environment configuration
source scripts/env.sh

cd "${PROJECT_ROOT}"

# Create real-time log file
RUNTIME_LOG="logs/whisfusion_stage1_adapter_runtime_${SLURM_JOB_ID}.log"
mkdir -p logs
exec > >(tee -a "$RUNTIME_LOG")
exec 2>&1

echo "=========================================="
echo "Whisfusion Stage 1 Training Job Started"
echo "Real-time log: $RUNTIME_LOG"
echo "Track with: tail -f $RUNTIME_LOG"
echo "=========================================="
echo ""

echo "Checking GPU setup..."
nvidia-smi
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
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
echo "Starting Stage 1: Adapter Training..."

fabric run src/training/train_stage1_adapter.py \
    --strategy=ddp \
    --devices=1 \
    --train_data_dir "${DATA_DIR}/processed/train/" \
    --val_data_dir   "${DATA_DIR}/processed/dev/" \
    --pretrain_path  "${MODELS_DIR}/smdm/mdm_safetensors/mdm-170M-100e18-rsl-0.01.safetensors" \
    --out_dir        "${MODELS_DIR}/whisfusion_finetuned/stage1_adapter/ft-Diff_LLaMA_170M-1777965131" \
    --resume \
    --num_devices    1 \
    --batch_size     64 \
    --gradient_accumulation_steps 6 \
    --learning_rate  1e-4 \
    --lr_scaling     linear \
    --lr_max         3e-4 \
    --scheduler_type cosine_epoch \
    --warmup_ratio   0.02 \
    --epochs         80 \
    --patience       5 \
    --weight_decay   0.01 \
    --clip_grad_norm 0.5 \
    --num_workers    4

echo "✅ Stage 1 training script finished."
