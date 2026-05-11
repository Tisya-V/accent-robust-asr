#!/bin/bash
#PBS -N extract_encoder_states
#PBS -l select=1:ngpus=1:mem=4gb
#PBS -l walltime=00:40:00
#PBS -o logs/extract_encoder_states.out
#PBS -e logs/extract_encoder_states.err
#PBS -j oe

set -e

# Source centralized environment configuration
source ${PBS_O_WORKDIR}/scripts/env.sh

cd "${PROJECT_ROOT}"

# Create real-time log file
RUNTIME_LOG="logs/extract_encoder_states_runtime_${PBS_JOBID}.log"
mkdir -p logs
exec > >(tee -a "$RUNTIME_LOG")
exec 2>&1

echo "=========================================="
echo "Encoder State Extraction Job Started"
echo "Real-time log: $RUNTIME_LOG"
echo "Track with: tail -f $RUNTIME_LOG"
echo "=========================================="
echo ""

echo "[extract_encoder_states] Starting feature extraction..."
python -m src.analysis.extract_encoder_states \
    --output_dir "${EPHEMERAL_DATA_DIR:-/rds/general/user/tsv22/ephemeral}"

echo "✅ [extract_encoder_states] Done!"
