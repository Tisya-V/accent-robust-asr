#!/bin/bash
#PBS -N download_pretrained
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=01:00:00
#PBS -o logs/download_pretrained.out
#PBS -e logs/download_pretrained.err
#PBS -j oe

# Downloads required pre-trained models for Whisfusion project
#
# Usage on RDS HPC:
# 1. chmod +x src/training/scripts/02_download_pretrained_model.sh
# 2. qsub src/training/scripts/02_download_pretrained_model.sh

set -e

# Source centralized environment configuration
source ${PBS_O_WORKDIR}/scripts/env.sh

cd "${PROJECT_ROOT}"

# Configuration
REPO_ID="nieshen/SMDM"
FILENAME="mdm_safetensors/mdm-170M-100e18-rsl-0.01.safetensors"
TARGET_DIR="${MODELS_DIR}/smdm"

echo "============================================================"
echo "Downloading pre-trained models..."
echo "============================================================"

mkdir -p "$TARGET_DIR"

echo "Downloading ${FILENAME} from Hugging Face Hub..."
echo "Repo: ${REPO_ID}"
echo "Target: ${TARGET_DIR}/"

hf download \
   "$REPO_ID" \
   "$FILENAME" \
   --repo-type model \
   --local-dir "$TARGET_DIR"

echo -e "\n============================================================"
echo "✅ Pre-trained model downloaded successfully!"
echo "============================================================"