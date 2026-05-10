#!/bin/bash
#PBS -N postprocess_phones
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=00:20:00
#PBS -o logs/postprocess_phone_segments.out
#PBS -e logs/postprocess_phone_segments.err
#PBS -j oe

set -e

# Source centralized environment configuration
source ${PBS_O_WORKDIR}/scripts/env.sh

cd "${PROJECT_ROOT}"

# Create real-time log file
RUNTIME_LOG="logs/postprocess_phone_segments_runtime_${PBS_JOBID}.log"
mkdir -p logs
exec > >(tee -a "$RUNTIME_LOG")
exec 2>&1

echo "=========================================="
echo "Phone Segment Post-Processing Job Started"
echo "Real-time log: $RUNTIME_LOG"
echo "Track with: tail -f $RUNTIME_LOG"
echo "=========================================="
echo ""

echo "[postprocess_phone_segments] Starting phone extraction..."
python -m src.analysis.postprocess_phone_segments

echo "✅ [postprocess_phone_segments] Done!"
