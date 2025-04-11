#!/bin/bash
source ~/spatial-reasoning-language-models/miniconda3/bin/activate spatial_reasoning_env

source .env
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"
export HF_HOME=~/spatial-reasoning-language-models/spatial-reasoning-of-LMs/.cache/huggingface

# VLM_ID=microsoft/Phi-3.5-vision-instruct
VLM_ID=Qwen/Qwen2.5-VL-7B-Instruct
# VLM_ID=gpt-4-turbo
# SPLIT="tx"
# PROMPT_TYPE=zero-shot
# RESULT_DIR=result/vlm_only_big_table/${VLM_ID}/${PROMPT_TYPE}

# python experiments/experiment_1_individual_vlm.py \
#     --vlm_id ${VLM_ID} \
#     --data_dir "benchmark/RGBD_7_Scenes_Rebuilt" \
#     --result_dir ${RESULT_DIR} \
#     --split ${SPLIT} \
#     --is_shuffle \
#     --prompt_type ${PROMPT_TYPE}

for PROMPT_TYPE in "zero-shot" "add-info-zero-shot" "CoT-zero-shot" "VoT-zero-shot"; do
    for SPLIT in "theta" "phi" "psi" "tx" "ty" "tz"; do
        echo "Running experiment with prompt_type=${PROMPT_TYPE} and split=${SPLIT}"
        RESULT_DIR=result/vlm_only_big_table/${VLM_ID}/${PROMPT_TYPE}/${SPLIT}
        mkdir -p ${RESULT_DIR}
        python experiments/experiment_1_individual_vlm.py \
            --vlm_id ${VLM_ID} \
            --data_dir "benchmark/RGBD_7_Scenes_Rebuilt" \
            --result_dir ${RESULT_DIR} \
            --split ${SPLIT} \
            --is_shuffle \
            --prompt_type ${PROMPT_TYPE}
    done
done
