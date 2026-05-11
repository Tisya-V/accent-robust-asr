#!/bin/bash
#SBATCH --job-name=whisper_ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24GB
#SBATCH --time=03:00:00
#SBATCH --output=logs/whisper_ft_%j.out
#SBATCH --error=logs/whisper_ft_%j.err

# Fine-tunes Whisper-small on L2-ARCTIC (all L1s)
#
# Usage on HPC cluster with SLURM:
# 1. chmod +x src/training/slurm_jobs/06_finetune_whisper.sh
# 2. sbatch src/training/slurm_jobs/06_finetune_whisper.sh

set -e

# Source centralized environment configuration
source scripts/env.sh

cd "${PROJECT_ROOT}"

# Create real-time log file (so user can tail it while job runs)
RUNTIME_LOG="logs/whisper_ft_runtime_${SLURM_JOB_ID}.log"
mkdir -p logs
exec > >(tee -a "$RUNTIME_LOG")
exec 2>&1

echo "=========================================="
echo "Whisper Fine-Tuning Job Started"
echo "Real-time log: $RUNTIME_LOG"
echo "Track with: tail -f $RUNTIME_LOG"
echo "=========================================="
echo ""

echo "Checking GPU setup..."
nvidia-smi
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo -e "\n\n==============================\n\n"
echo "Starting Whisper Fine-Tuning (all L1s)..."

python -u -m src.training.finetune_whisper \
    --output_dir models/whisper_finetuned_all_l1s \
    --epochs 10 \
    --batch_size 16

echo "✅ Whisper fine-tuning completed."
