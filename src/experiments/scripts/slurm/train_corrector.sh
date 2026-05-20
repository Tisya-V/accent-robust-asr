#!/usr/bin/env bash
# Train position-based latent corrector (Exp2 feasibility).
# Runs on local GPU or SLURM cluster.
#
# Local:  bash src/experiments/scripts/05_train_corrector_position.sh
# SLURM:  sbatch src/experiments/scripts/05_train_corrector_position.sh

#SBATCH --job-name=corrector_position
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/corrector_position_%j.log
#SBATCH --error=logs/corrector_position_%j.log

source scripts/slurm_env.sh
cd "${PROJECT_ROOT}"

echo "[Job] Starting at $(date)"
echo "[Job] Host: $(hostname)"
echo "[Job] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no GPU')"

mkdir -p logs

python -m src.experiments.exp2_transformer_bridge.train \
    --mapping_train_path src/experiments/exp2_latent_diffusion_bridge/data/mapping_train.json \
    --mapping_val_path src/experiments/exp2_latent_diffusion_bridge/data/mapping_dev.json \
    --out_dir models/corrector_position \
    --n_epochs 12 \
    --batch_size 128 \
    --lr 3e-4 \
    --d_model 256 \
    --n_layers 4 \
    --dim_feedforward 1024 \
    --patience 5 \
    --num_workers 6\
    --notes "position-based feasibility, no DTW, lr=3e-4, mse all positions"

echo "[Job] Training done at $(date)"

python -m src.experiments.exp2_transformer_bridge.eval \
    --ckpt models/corrector_position/checkpoint_best.pt \
    --out_dir results/corrector_position_eval

echo "[Job] Eval done at $(date)"