#!/bin/bash
#PBS -N preprocess_data
#PBS -l select=1:ngpus=1:ncpus=4:mem=24gb
#PBS -l walltime=04:00:00
#PBS -o logs/preprocess_data.out
#PBS -e logs/preprocess_data.err
#PBS -j oe

# Preprocesses L2-ARCTIC data into raw/ and processed/ directories for Whisfusion
#
# Usage on RDS HPC:
# 1. chmod +x src/training/scripts/01_preprocess_data.sh
# 2. qsub src/training/scripts/01_preprocess_data.sh

set -e

# Source centralized environment configuration
source ${PBS_O_WORKDIR}/scripts/env.sh

cd "${PROJECT_ROOT}"

# Create real-time log file
RUNTIME_LOG="logs/preprocess_data_runtime_${PBS_JOBID}.log"
mkdir -p logs
exec > >(tee -a "$RUNTIME_LOG")
exec 2>&1

echo "=========================================="
echo "Data Preprocessing Job Started"
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
echo "Starting data preprocessing for Whisfusion..."

python -u -m src.utils.preprocess_data \
    --raw_output_dir data/raw \
    --processed_output_dir data/processed \
    --whisper_model openai/whisper-small

echo "✅ Data preprocessing completed."
echo "Raw data: data/raw/{train,dev,test}/"
echo "Processed data: data/processed/{train,dev,test}/"
