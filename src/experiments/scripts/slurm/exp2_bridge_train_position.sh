#!/bin/bash
#SBATCH --job-name=exp2_bridge_position
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --requeue

set -e

source scripts/slurm_env.sh
cd "${PROJECT_ROOT}"

RUNTIME_LOG="logs/%x_%j.log"
mkdir -p logs
exec > >(tee -a "$RUNTIME_LOG") 2>&1

echo "=========================================="
echo "E2 Bridge — position alignment"
echo "Job:  ${SLURM_JOB_ID}"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Track: tail -f ${RUNTIME_LOG}"
echo "=========================================="

python -m src.experiments.exp2_latent_diffusion_bridge.train \
    --mapping_train_path src/experiments/exp2_latent_diffusion_bridge/data/mapping_train_v2.json \
    --mapping_val_path   src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev_v2.json \
    --alignment    position \
    --cond_acc        \
    --parameterization cfm \
    --out_dir      models/bridge_position_cfm \
    --d_model 768 \
    --dim_feedforward 3072 \
    --n_heads 12 \
    --n_epochs     15 \
    --batch_size   64 \
    --lr           1e-4 \
    --weight_decay 1e-4 \
    --sigma_max    0.0 \
    --num_workers  6 \
    --patience     5 \
    --tail_weight  0.0 \
    --notes        "cfm parameterization with cond, position alignment, sigma_max=0.0, tail_weight=0.0 , d_model=768"

echo "Done at $(date)"
