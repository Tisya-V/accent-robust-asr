#!/bin/bash
# Downloads required pre-trained models for Whisfusion project
#
# Usage on RDS HPC:
# 1. chmod +x src/training/scripts/02_download_pretrained_model.sh
# 2. source scripts/env.sh && ./src/training/scripts/02_download_pretrained_model.sh

set -e

# Source centralized environment configuration
source scripts/env.sh

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

if ! command -v huggingface-cli &> /dev/null; then
   echo "❌ huggingface-cli not found. Please install with: pip install huggingface_hub"
   exit 1
fi

huggingface-cli download \
   "$REPO_ID" \
   "$FILENAME" \
   --repo-type model \
   --local-dir "$TARGET_DIR"

echo -e "\n============================================================"
echo "✅ Pre-trained model downloaded successfully!"
echo "============================================================"