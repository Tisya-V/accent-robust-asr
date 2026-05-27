#!/bin/bash
#SBATCH --job-name=teng_predictor
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
echo "T_eng Predictor"
echo "Job:  ${SLURM_JOB_ID}"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Track: tail -f ${RUNTIME_LOG}"
echo "=========================================="

python -m src.experiments.exp2_latent_diffusion_bridge.train_teng_predictor \
    --out_dir      models/teng_predictor \
    --subset_frac  0.5 \
    --n_epochs     50 \
    --batch_size   512 \
    --lr           1e-3 \
    --weight_decay 1e-4

echo "Done at $(date)"
