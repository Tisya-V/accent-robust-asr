#!/bin/bash
#PBS -N exp2_bridge_train
#PBS -l select=1:ngpus=1:ncpus=2:mem=24gb
#PBS -l walltime=6:00:00
#PBS -o logs/exp2_bridge_train.out
#PBS -e logs/exp2_bridge_train.err
#PBS -j oe

# Train E2 Latent Diffusion Bridge model
#
# Usage on RDS HPC:
# 1. chmod +x src/experiments/scripts/pbs/exp2_bridge_train.sh
# 2. qsub src/experiments/scripts/pbs/exp2_bridge_train.sh

set -e

# Source centralized environment configuration
source ${PBS_O_WORKDIR}/scripts/pbs_env.sh

cd "${PROJECT_ROOT}"

# Create real-time log file
RUNTIME_LOG="logs/exp2_bridge_train_runtime_${PBS_JOBID}.log"
mkdir -p logs
exec > >(tee -a "$RUNTIME_LOG")
exec 2>&1

echo "=========================================="
echo "E2 Latent Diffusion Bridge Training Job"
echo "Real-time log: $RUNTIME_LOG"
echo "Track with: tail -f $RUNTIME_LOG"
echo "=========================================="
echo ""

echo "Checking GPU setup..."
nvidia-smi
echo "PBS_JOBID=$PBS_JOBID"
echo "PBS_O_WORKDIR=$PBS_O_WORKDIR"
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
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --sigma_max 0.5 \
    --num_workers 4

echo "✅ Bridge training complete."
