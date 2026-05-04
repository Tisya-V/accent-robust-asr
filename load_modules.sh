#!/bin/bash -l
source /etc/profile.d/modules.sh
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.4.0
echo -e "Modules loaded:"
module list