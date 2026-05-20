#!/bin/bash
#PBS -N precompute_dtw
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -l walltime=00:30:00
#PBS -o logs/precompute_dtw.out
#PBS -e logs/precompute_dtw.err
#PBS -j oe

# Precompute DTW alignment paths for all bridge training pairs.
# CPU-only — no GPU requested.
#
# Usage:
#   qsub src/experiments/exp2_latent_diffusion_bridge/pbs_precompute_dtw.sh

set -e

source "${PBS_O_WORKDIR}/scripts/pbs_env.sh"
cd "${PROJECT_ROOT}"

RUNTIME_LOG="logs/precompute_dtw_${PBS_JOBID}.log"
mkdir -p logs
exec > >(tee -a "${RUNTIME_LOG}") 2>&1

echo "=========================================="
echo "DTW Precompute Job"
echo "Job ID:   ${PBS_JOBID}"
echo "Host:     $(hostname)"
echo "CPUs:     $(nproc)"
echo "Log:      ${RUNTIME_LOG}"
echo "Track:    tail -f ${RUNTIME_LOG}"
echo "=========================================="

python -u -m src.experiments.exp2_latent_diffusion_bridge.precompute_dtw \
    --cache_dir data/bridge_dtw_cache \
    --workers 8

echo "Done at $(date)"
