#!/bin/bash
#PBS -N e1_eval_decoder_internals
#PBS -l select=1:ngpus=1:ncpus=4:mem=32gb
#PBS -l walltime=00:30:00
#PBS -o logs/e1_eval_decoder_internals.out
#PBS -e logs/e1_eval_decoder_internals.err
#PBS -j oe

# Evaluate Whisfusion and save decoder internals for E1 analysis
#
# Usage on RDS HPC:
# 1. chmod +x src/analysis/scripts/e1_eval_with_decoder_internals.sh
# 2. qsub src/analysis/scripts/e1_eval_with_decoder_internals.sh

set -e

# Source centralized environment configuration
source ${PBS_O_WORKDIR}/scripts/env.sh

cd "${PROJECT_ROOT}"

# Create real-time log file
RUNTIME_LOG="logs/e1_eval_decoder_internals_${PBS_JOBID}.log"
mkdir -p logs results/e1_decoder_internals
exec > >(tee -a "$RUNTIME_LOG")
exec 2>&1

echo "=========================================="
echo "E1 Whisfusion Evaluation (with Internals)"
echo "Real-time log: $RUNTIME_LOG"
echo "Track with: tail -f $RUNTIME_LOG"
echo "=========================================="
echo ""

nvidia-smi
echo ""

python -u -m src.training.evaluation.eval_whisfusion \
  --save_decoder_internals \
  --internals_dir results/e1_decoder_internals \
  --model whisfusion_finetuned

echo ""
echo "✅ Evaluation with decoder internals completed."
echo "Internals saved to: results/e1_decoder_internals/"
