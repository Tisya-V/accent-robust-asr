#!/bin/bash
#SBATCH --job-name=eval_whisper_ft_hoc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_whisper_ft_hoc_%j.out
#SBATCH --error=logs/eval_whisper_ft_hoc_%j.err

# Whisper fine-tuned model evaluation
#
# Usage on HPC cluster with SLURM:
# 1. chmod +x src/training/slurm_jobs/05b_eval_whisper.sh
# 2. sbatch src/training/slurm_jobs/05b_eval_whisper.sh

set -e

# Source centralized environment configuration
source scripts/env.sh

cd "${PROJECT_ROOT}"

# Create real-time log file
RUNTIME_LOG="logs/eval_whisper_runtime_${SLURM_JOB_ID}.log"
mkdir -p logs
exec > >(tee -a "$RUNTIME_LOG")
exec 2>&1

echo "=========================================="
echo "Whisper Evaluation Job Started"
echo "Real-time log: $RUNTIME_LOG"
echo "Track with: tail -f $RUNTIME_LOG"
echo "=========================================="
echo ""

echo "=== Job started ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "hostname: $(hostname)"
nvidia-smi
echo "python --version: $(python --version)"
which python
echo "====================="

python -u -m src.training.evaluation.eval_whisper --models "baseline,whisper_finetuned"

echo "✅ Evaluation completed."
