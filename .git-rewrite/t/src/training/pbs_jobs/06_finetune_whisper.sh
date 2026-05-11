#!/bin/bash
#PBS -N whisper_ft_hoc
#PBS -l select=1:ngpus=1:ncpus=4:mem=16gb
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

# Create real-time log file (so user can tail it while job runs)
RUNTIME_LOG="logs/whisper_ft_runtime_${PBS_JOBID}.log"
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
echo "PBS_JOBID=$PBS_JOBID"
echo "PBS_O_WORKDIR=$PBS_O_WORKDIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo -e "\n\n==============================\n\n"
echo "Starting Whisper Fine-Tuning (all L1s)..."

python -u -m src.training.finetune_whisper \
    --output_dir models/whisper_finetuned_hoc \
    --epochs 10 \
    --batch_size 16 \
    --held_out_l1 Chinese

echo "✅ Whisper fine-tuning completed."