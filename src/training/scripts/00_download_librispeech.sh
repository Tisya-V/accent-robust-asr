#!/bin/bash
# Downloads and prepares LibriSpeech partitions for Whisfusion project
#
# Usage:
# 1. chmod +x scripts/00_download_librispeech.sh
# 2. ./scripts/00_download_librispeech.sh

set -e

# Configuration
TARGET_DIR="data/raw"

PARTITIONS=(
    "train-clean-100"
    "train-clean-360"
    "train-other-500"
    "dev-clean"
    "dev-other"
    "test-clean"
    "test-other"
)

echo "============================================================"
echo "Starting LibriSpeech dataset download..."
echo "Target directory: ./${TARGET_DIR}/"
echo "============================================================"

mkdir -p "$TARGET_DIR"

for part in "${PARTITIONS[@]}"; do
    URL="https://www.openslr.org/resources/12/${part}.tar.gz"
    TAR_FILE="${part}.tar.gz"
    
    echo -e "\n--- Processing partition: ${part} ---"
    
    echo "Downloading ${URL}..."
    wget -c "$URL" -O "${TARGET_DIR}/${TAR_FILE}"
    
    echo "Unpacking ${TAR_FILE}..."
    tar -xzvf "${TARGET_DIR}/${TAR_FILE}" -C "$TARGET_DIR"
    
    echo "Cleaning up ${TAR_FILE}..."
    rm "${TARGET_DIR}/${TAR_FILE}"
    
    echo "-> Successfully processed ${part}"
done

echo -e "\n============================================================"
echo "✅ All LibriSpeech partitions downloaded and extracted successfully!"
echo "============================================================"