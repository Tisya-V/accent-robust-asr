#!/bin/bash
#PBS -N eval_whisfusion_ft
#PBS -l select=1:ngpus=1:ncpus=4:mem=24gb
#PBS -l walltime=02:00:00
#PBS -o logs/eval_whisfusion_ft.out
#PBS -e logs/eval_whisfusion_ft.err
#PBS -j oe

# Whisfusion model evaluation
#
# Usage on RDS HPC:
# 1. chmod +x src/training/pbs_jobs/eval_whisfusion.sh
# 2. qsub src/training/pbs_jobs/eval_whisfusion.sh

set -e

# Source centralized environment configuration
source ${PBS_O_WORKDIR}/scripts/env.sh

cd "${PROJECT_ROOT}"

# Create real-time log file
RUNTIME_LOG="logs/eval_whisfusion_runtime_${PBS_JOBID}.log"
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

python -u -m src.training.evaluation.eval_whisfusion --model whisfusion_finetuned

echo "✅ Evaluation completed."