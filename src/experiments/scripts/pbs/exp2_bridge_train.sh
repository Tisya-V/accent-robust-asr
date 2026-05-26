#!/bin/bash
#PBS -N exp2_bridge_train
#PBS -l select=1:ngpus=1:ncpus=8:mem=48gb
#PBS -l walltime=16:00:00
#PBS -o logs/exp2_bridge_train.out
#PBS -e logs/exp2_bridge_train.err
#PBS -j oe

set -e

source "${PBS_O_WORKDIR}/scripts/pbs_env.sh"
cd "${PROJECT_ROOT}"

RUNTIME_LOG="logs/exp2_bridge_train_${PBS_JOBID}.log"
mkdir -p logs
exec > >(tee -a "$RUNTIME_LOG") 2>&1

echo "=========================================="
echo "E2 Bridge Training — position + DTW"
echo "Job:  ${PBS_JOBID}"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Log:  ${RUNTIME_LOG}"
echo "Track: tail -f ${RUNTIME_LOG}"
echo "=========================================="

echo "[1/2] Position alignment..."
python -m src.experiments.exp2_latent_diffusion_bridge.train \
    --alignment    position \
    --out_dir      models/bridge_position \
    --n_epochs     50 \
    --batch_size   32 \
    --lr           1e-4 \
    --weight_decay 1e-4 \
    --sigma_max    0.5 \
    --num_workers  6 \
    --patience     10 \
    --notes        "x0-prediction, position alignment"

echo "[2/2] DTW alignment..."
python -m src.experiments.exp2_latent_diffusion_bridge.train \
    --alignment    dtw \
    --out_dir      models/bridge_dtw \
    --n_epochs     50 \
    --batch_size   32 \
    --lr           1e-4 \
    --weight_decay 1e-4 \
    --sigma_max    0.5 \
    --num_workers  6 \
    --patience     10 \
    --notes        "x0-prediction, DTW alignment"

echo "Done at $(date)"
