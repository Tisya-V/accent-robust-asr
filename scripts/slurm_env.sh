#!/bin/bash
# Environment for SLURM cluster (gpudata)
# Source this file in all SLURM job scripts: source scripts/slurm_env.sh

set -e

# ============================================================================
# PROJECT PATHS
# ============================================================================
export PROJECT_ROOT="/vol/gpudata/tsv22-fyp/accent-robust-asr"
export MODELS_DIR="${PROJECT_ROOT}/models"
export RESULTS_DIR="${PROJECT_ROOT}/results"
export HPSEARCH_DIR="${PROJECT_ROOT}/hpsearch"
export LOGS_DIR="${PROJECT_ROOT}/logs"
export DATA_DIR="${PROJECT_ROOT}/data"

# Processed data — split across three gpudata projects
export TRAIN_DATA_DIR="/vol/gpudata/tsv22-train/data/processed/train"
export DEV_DATA_DIR="/vol/gpudata/tsv22-dev_test/data/processed/dev"
export TEST_DATA_DIR="/vol/gpudata/tsv22-dev_test/data/processed/test"

# Raw data paths (point to gpudata copy; fall back to relative for local runs)
export L2ARCTIC_DIR="${DATA_DIR}/l2_arctic"
export CMU_ARCTIC_DIR="${DATA_DIR}/cmu_arctic"

# ============================================================================
# CACHE & CACHE PATHS (persistent in gpudata quota)
# ============================================================================
export HOME_CACHE="${PROJECT_ROOT}/.cache"
export HF_HOME="${HOME_CACHE}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export XDG_CACHE_HOME="${HOME_CACHE}"
export MPLCONFIGDIR="${HOME_CACHE}/matplotlib"
export NLTK_DATA="${PROJECT_ROOT}/nltk_data"

# ============================================================================
# CUDA (SLURM cluster style, no module system)
# ============================================================================
source /vol/cuda/12.4.0/setup.sh

# ============================================================================
# PYTHON & VENV
# ============================================================================
export VENV_DIR="${PROJECT_ROOT}/.venv"
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
fi

# ============================================================================
# CUDA & DISTRIBUTED TRAINING
# ============================================================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export NCCL_P2P_DISABLE=1

# ============================================================================
# DIRECTORIES
# ============================================================================
mkdir -p "${LOGS_DIR}" "${HOME_CACHE}"

# ============================================================================
# DEBUGGING / INFO
# ============================================================================
if [ "${VERBOSE_ENV}" = "1" ]; then
    echo "====== SLURM Environment Configuration ======"
    echo "PROJECT_ROOT:    ${PROJECT_ROOT}"
    echo "TRAIN_DATA_DIR:  ${TRAIN_DATA_DIR}"
    echo "DEV_DATA_DIR:    ${DEV_DATA_DIR}"
    echo "TEST_DATA_DIR:   ${TEST_DATA_DIR}"
    echo "MODELS_DIR:      ${MODELS_DIR}"
    echo "HF_HOME:         ${HF_HOME}"
    echo "NLTK_DATA:       ${NLTK_DATA}"
    echo "VENV_DIR:        ${VENV_DIR}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "=============================================="
fi
