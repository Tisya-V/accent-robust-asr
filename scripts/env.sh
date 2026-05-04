#!/bin/bash
# Centralized environment configuration for RDS HPC
# Source this file in all shell scripts: source scripts/env.sh

set -e

# ============================================================================
# PROJECT PATHS
# ============================================================================
export PROJECT_ROOT="/rds/general/user/tsv22/home/accent-robust-asr"
export DATA_DIR="${PROJECT_ROOT}/data"
export MODELS_DIR="${PROJECT_ROOT}/models"
export RESULTS_DIR="${PROJECT_ROOT}/results"
export HPSEARCH_DIR="${PROJECT_ROOT}/hpsearch"
export LOGS_DIR="${PROJECT_ROOT}/logs"

# Data subdirectories
export RAW_DATA_DIR="${DATA_DIR}/raw"
export L2ARCTIC_DIR="${DATA_DIR}/l2_arctic"
export CMU_ARCTIC_DIR="${DATA_DIR}/cmu_arctic"
export PROCESSED_DATA_DIR="${DATA_DIR}/processed"

# ============================================================================
# CACHE & CACHE PATHS (in home directory)
# ============================================================================
export HOME_CACHE="${HOME}/.cache"
export HF_HOME="${HOME_CACHE}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export XDG_CACHE_HOME="${HOME_CACHE}"
export MPLCONFIGDIR="${HOME_CACHE}/matplotlib"
export NLTK_DATA="${PROJECT_ROOT}/nltk_data"

# ============================================================================
# MODULES
# ============================================================================
# Load required modules for RDS HPC
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.4.0

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
# Uncomment if needed:
# export NCCL_SOCKET_IFNAME=enoinp0
# export NCCL_DEBUG=WARN

# ============================================================================
# DIRECTORIES
# ============================================================================
mkdir -p "${LOGS_DIR}"
mkdir -p "${DATA_DIR}"
mkdir -p "${RAW_DATA_DIR}"
mkdir -p "${PROCESSED_DATA_DIR}"
mkdir -p "${HOME_CACHE}"

# ============================================================================
# DEBUGGING / INFO
# ============================================================================
if [ "${VERBOSE_ENV}" = "1" ]; then
    echo "====== Environment Configuration ======"
    echo "PROJECT_ROOT: ${PROJECT_ROOT}"
    echo "DATA_DIR: ${DATA_DIR}"
    echo "MODELS_DIR: ${MODELS_DIR}"
    echo "HF_HOME: ${HF_HOME}"
    echo "NLTK_DATA: ${NLTK_DATA}"
    echo "VENV_DIR: ${VENV_DIR}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "========================================="
fi
