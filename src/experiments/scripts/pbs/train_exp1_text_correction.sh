#!/bin/bash
#PBS -N exp1_text_correction
#PBS -l select=1:ngpus=1:ncpus=4:mem=16GB
#PBS -l walltime=00:30:00
#PBS -o logs/exp1_text_correction_${PBS_JOBID}.out
#PBS -e logs/exp1_text_correction_${PBS_JOBID}.out

# Experiment 1: Text-level diffusion with phoneme perturbations (RDS PBS version)
#
# Usage:
# 1. qsub src/experiments/scripts/pbs/train_exp1_text_correction.sh
# OR with config file argument:
# 2. qsub -v CONFIG=configs/my_config.json src/experiments/scripts/pbs/train_exp1_text_correction.sh

source /rds/general/user/tsv22/home/accent-robust-asr/scripts/env.sh

cd "$PROJECT_ROOT"

set -e

echo "==============================="
echo "Experiment 1: Text-level Diffusion"
echo "==============================="
echo ""

echo "Checking GPU/PBS setup..."
nvidia-smi
echo "PBS_JOBID=$PBS_JOBID"
python - <<'PY'
import os, torch
print("DEBUG torch.cuda.is_available =", torch.cuda.is_available())
print("DEBUG torch.cuda.device_count =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"DEBUG cuda[{i}] =", torch.cuda.get_device_name(i))
PY

echo ""
echo "Starting Experiment 1 training..."
echo ""

# Use config from environment var, or default to phoneme_perturb_low
CONFIG="${CONFIG:-src/experiments/exp1_text_correction/configs/phoneme_perturb_low.json}"

python -u -m src.experiments.exp1_text_correction.train \
    --config "$CONFIG" \
    --device cuda

echo ""
echo "✅ Experiment 1 training finished."
