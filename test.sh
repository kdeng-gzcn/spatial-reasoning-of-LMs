#!/bin/bash

#SBATCH --job-name=kdeng
#SBATCH --output=log/test-qwen-32B-%j.out
#SBATCH --gpus=1
#SBATCH --time=12:00:00  

nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_reasoning_env
set -a && source .env && set +a
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"

python test-vlm.py