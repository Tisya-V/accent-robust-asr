#!/bin/bash
#SBATCH --job-name=eval_whisfusion_ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_whisfusion_ft_%j.out
#SBATCH --error=logs/eval_whisfusion_ft_%j.err

# Whisfusion model evaluation
#
# Usage on HPC cluster with SLURM:
# 1. chmod +x src/training/slurm_jobs/05a_eval_whisfusion.sh
# 2. sbatch src/training/slurm_jobs/05a_eval_whisfusion.sh

set -e

# Source centralized environment configuration
source scripts/env.sh

cd "${PROJECT_ROOT}"

# Create real-time log file
RUNTIME_LOG="logs/eval_whisfusion_runtime_${SLURM_JOB_ID}.log"
mkdir -p logs
exec > >(tee -a "$RUNTIME_LOG")
exec 2>&1

echo "=========================================="
echo "Whisfusion Evaluation Job Started"
echo "Real-time log: $RUNTIME_LOG"
echo "Track with: tail -f $RUNTIME_LOG"
echo "=========================================="
echo ""

nvidia-smi

python -u -m src.training.evaluation.eval_whisfusion --model whisfusion_ft

echo "✅ Evaluation completed."
