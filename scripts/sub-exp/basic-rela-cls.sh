#!/bin/bash

#SBATCH --job-name=kdeng
#SBATCH --output=log/basic-rela-%j.out
#SBATCH --gpus=4
#SBATCH --time=24:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_reasoning_env
# conda activate phi

set -a && source .env && set +a
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"

# vlm_id="Qwen/Qwen2.5-VL-7B-Instruct"
vlm_id="meta-llama/Llama-4-Scout-17B-16E-Instruct"
# vlm_id="gpt-4o"
if [[ "$vlm_id" == */* ]]; then
    dir_vlm="${vlm_id##*/}"
else
    dir_vlm="$vlm_id"
fi

python /home/u5u/kdeng.u5u/spatial-reasoning-of-LMs/sub-exp/basic-rela-cls.py \
    --vlm_id "${vlm_id}" \
    --data_dir /home/u5u/kdeng.u5u/spatial-reasoning-of-LMs/demo/basic-rela.jsonl \
    --result_dir /home/u5u/kdeng.u5u/spatial-reasoning-of-LMs/result/basic-rela-cls/${dir_vlm} \