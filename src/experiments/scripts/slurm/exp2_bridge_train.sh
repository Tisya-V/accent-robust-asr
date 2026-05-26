#!/bin/bash
#SBATCH --job-name=exp2_bridge_train_pos_e
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --requeue

# set -e

# source scripts/slurm_env.sh
# cd "${PROJECT_ROOT}"

# RUNTIME_LOG="logs/%x_%j.log"
# mkdir -p logs
# exec > >(tee -a "$RUNTIME_LOG") 2>&1

# echo "=========================================="
# echo "E2 Bridge Training — DTW"
# echo "Job:  ${SLURM_JOB_ID}"
# echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
# echo "Log:  ${RUNTIME_LOG}"
# echo "=========================================="

# echo "DTW alignment..."
# python -m src.experiments.exp2_latent_diffusion_bridge.train \
#     --alignment    dtw \
#     --out_dir      models/bridge_dtw \
#     --n_epochs     50 \
#     --batch_size   64 \
#     --lr           1e-4 \
#     --weight_decay 1e-4 \
#     --sigma_max    1.5 \
#     --num_workers  6 \
#     --patience     10 \
#     --notes        "x0-prediction, DTW alignment (interpolated tail)"

# echo "Done at $(date)"



# ===================================================================================



set -e

source scripts/slurm_env.sh
cd "${PROJECT_ROOT}"

RUNTIME_LOG="logs/%x_%j.log"
mkdir -p logs
exec > >(tee -a "$RUNTIME_LOG") 2>&1

echo "=========================================="
echo "E2 Bridge Training — position"
echo "Job:  ${SLURM_JOB_ID}"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Log:  ${RUNTIME_LOG}"
echo "=========================================="

echo "Position alignment..."
python -m src.experiments.exp2_latent_diffusion_bridge.train \
    --alignment    position \
    --out_dir      models/bridge_position_e \
    --n_epochs     50 \
    --batch_size   64 \
    --lr           1e-4 \
    --weight_decay 1e-4 \
    --sigma_max    1.5 \
    --num_workers  6 \
    --patience     10 \
    --notes        "e-prediction, position alignment (interpolated tail)"

echo "Done at $(date)"
