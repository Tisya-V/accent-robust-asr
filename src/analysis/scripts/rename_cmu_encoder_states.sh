#!/bin/bash
#PBS -N rename_cmu_encoder_states
#PBS -l select=1:ncpus=1:mem=1gb
#PBS -l walltime=00:05:00
#PBS -o logs/rename_cmu_encoder_states.out
#PBS -e logs/rename_cmu_encoder_states.err
#PBS -j oe

# Source centralized environment configuration
source ${PBS_O_WORKDIR}/scripts/env.sh

cd "${PROJECT_ROOT}"

mkdir -p logs

echo "=========================================="
echo "Renaming CMU Encoder State Files"
echo "Stripping speaker prefix: CLB_arctic_a0509.pt → arctic_a0509.pt"
echo "=========================================="
echo "EPHEMERAL=$EPHEMERAL"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo ""

RENAMED=0
SKIPPED=0

for split in train dev test; do
    for speaker in BDL CLB RMS SLT; do
        dir="$EPHEMERAL/accent-robust-asr/probing/encoder_states/$split/$speaker"

        if [ ! -d "$dir" ]; then
            echo "[rename_cmu_encoder_states] Skipping $split/$speaker (not found)"
            continue
        fi

        echo "[rename_cmu_encoder_states] Processing $split/$speaker..."

        find "$dir" -maxdepth 1 -name "${speaker}_*.pt" -type f -print0 | while IFS= read -r -d '' file; do
            filename=$(basename "$file")
            newname="${filename#${speaker}_}"
            newpath="$dir/$newname"

            if [ "$filename" != "$newname" ]; then
                if mv "$file" "$newpath"; then
                    echo "  ✓ $filename → $newname"
                else
                    echo "  ✗ Failed: $filename"
                fi
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "[rename_cmu_encoder_states] Done!"
echo "  Renamed: $RENAMED files"
echo "=========================================="
