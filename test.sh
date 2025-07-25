#!/bin/bash

#SBATCH --job-name=kdeng
#SBATCH --output=log/test-%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_reasoning_env
# conda activate phi

set -a && source .env && set +a
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"

python test-vlm.py