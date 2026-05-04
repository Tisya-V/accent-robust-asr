#!/bin/bash
#PBS -N eval_whisfusion_ft
#PBS -l select=1:ngpus=1:ncpus=4:mem=32gb
#PBS -l walltime=06:00:00
#PBS -o logs/eval_whisfusion_ft.out
#PBS -e logs/eval_whisfusion_ft.err
#PBS -j oe

# Whisfusion model evaluation
#
# Usage on RDS HPC:
# 1. chmod +x src/eval/eval_whisfusion.sh
# 2. qsub src/eval/eval_whisfusion.sh

set -e

# Source centralized environment configuration
source ${PBS_O_WORKDIR}/scripts/env.sh

cd "${PROJECT_ROOT}"

nvidia-smi

python -u -m src.eval.eval_whisfusion --model whisfusion_ft

echo "✅ Evaluation completed."