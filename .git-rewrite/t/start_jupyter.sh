#!/bin/bash
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.4.0
source /rds/general/user/tsv22/home/accent-robust-asr/.venv/bin/activate
exec python -m ipykernel_launcher "$@"