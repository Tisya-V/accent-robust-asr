#!/bin/bash
#SBATCH --job-name=whisfusion_ft_hov_train_stage1
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

# Starts Stage 1 training - trains only the cross-attention adapter
#
# Usage:
# 1. chmod +x scripts/03_train_stage1_adapter.sh
# 2. ./scripts/03_train_stage1_adapter.sh


export HF_HOME=/vol/bitbucket/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/vol/bitbucket/$USER/.cache/huggingface/transformers
export XDG_CACHE_HOME=/vol/bitbucket/$USER/.cache
export MPLCONFIGDIR=/vol/bitbucket/$USER/.cache/matplotlib
ifconfig -a | grep -E 'UP|inet ' | grep -v lo

export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=enoinp0
# export NCCL_DEBUG=WARN


export PATH=/vol/bitbucket/$USER/accent-robust-asr/.venv/bin/:$PATH
source activate

source /vol/cuda/12.4.0/setup.sh

cd /vol/bitbucket/$USER/accent-robust-asr/

set -e

echo "Checking GPU/SLURM setup..."
nvidia-smi
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python - <<'PY'
import os, torch
print("DEBUG CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("DEBUG torch.cuda.is_available =", torch.cuda.is_available())
print("DEBUG torch.cuda.device_count =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"DEBUG cuda[{i}] =", torch.cuda.get_device_name(i))
PY

# srun bash -c 'echo "host=$(hostname) rank=$SLURM_PROCID localid=$SLURM_LOCALID cuda=$CUDA_VISIBLE_DEVICES"'
echo -e "\n\n==============================\n\n"
echo "Starting Stage 1: Adapter Training..."

fabric run models/whisfusion/src/training/train_stage1_adapter.py \
    --strategy=ddp \
    --devices=3 \
    --train_data_dir data/processed/train_ho_vietnamese \
    --val_data_dir   data/processed/dev_ho_vietnamese \
    --pretrain_path  models/smdm/mdm_safetensors/mdm-170M-100e18-rsl-0.01.safetensors \
    --out_dir        models/whisfusion_ft_hov/stage1_adapter \
    --num_devices    3 \
    --batch_size     48 \
    --gradient_accumulation_steps 4 \
    --learning_rate  1e-4 \
    --lr_scaling     linear \
    --lr_max         3e-4 \
    --scheduler_type cosine_epoch \
    --warmup_ratio   0.02 \
    --epochs         80 \
    --patience       5 \
    --weight_decay   0.01 \
    --clip_grad_norm 0.5 \
    --num_workers    12

echo "✅ Stage 1 training script finished."