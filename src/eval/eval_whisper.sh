#!/bin/bash
#PBS -N eval_whisper_ft_hoc
#PBS -l select=1:ngpus=1:ncpus=4:mem=32gb
#PBS -l walltime=06:00:00
#PBS -o logs/eval_whisper_ft_hoc.out
#PBS -e logs/eval_whisper_ft_hoc.err
#PBS -j oe

# Whisper fine-tuned model evaluation
#
# Usage on RDS HPC:
# 1. chmod +x src/eval/eval_whisper.sh
# 2. qsub src/eval/eval_whisper.sh

set -e

# Source centralized environment configuration
source ${PBS_O_WORKDIR}/scripts/env.sh

cd "${PROJECT_ROOT}"

echo "=== Job started ==="
echo "PBS_JOBID: $PBS_JOBID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "hostname: $(hostname)"
nvidia-smi
echo "python --version: $(python --version)"
which python
echo "====================="

python -u -m src.eval.eval_whisper --models "whisper_ft_hoc"

echo "✅ Evaluation completed."