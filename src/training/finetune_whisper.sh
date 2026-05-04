#!/bin/bash
#PBS -N whisper_ft
#PBS -l select=1:ngpus=1:ncpus=4:mem=24gb
#PBS -l walltime=03:00:00
#PBS -o logs/whisper_ft.out
#PBS -e logs/whisper_ft.err
#PBS -j oe

# Fine-tunes Whisper-small on L2-ARCTIC (all L1s)
#
# Usage on RDS HPC:
# 1. chmod +x src/training/finetune_whisper.sh
# 2. qsub src/training/finetune_whisper.sh

set -e

# Source centralized environment configuration
source ${PBS_O_WORKDIR}/scripts/env.sh

cd "${PROJECT_ROOT}"


echo "Checking GPU setup..."
nvidia-smi
echo "PBS_JOBID=$PBS_JOBID"
echo "PBS_O_WORKDIR=$PBS_O_WORKDIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo -e "\n\n==============================\n\n"
echo "Starting Whisper Fine-Tuning (all L1s)..."

python -u -m src.training.finetune_whisper \
    --output_dir models/whisper_finetuned_all_l1s \
    --epochs 10 \
    --batch_size 16

echo "✅ Whisper fine-tuning completed."