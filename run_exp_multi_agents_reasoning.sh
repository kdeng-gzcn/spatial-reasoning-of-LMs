#!/bin/bash
source ~/spatial-reasoning-language-models/miniconda3/bin/activate spatial_reasoning_env

source .env
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"

# RESULT_DIR="result/exp_ma_scale_up"
PROMPT_TYPE="add-info-zero-shot"
DATA_DIR="benchmark/RGBD_7_Scenes_Rebuilt"
VLM_IMAGE_INPUT_TYPE="pair"
MAX_LEN_OF_CONV=10

for VLM_ID in Qwen/Qwen2.5-VL-7B-Instruct gpt-4o; do
    for LLM_ID in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; do
        for SPLIT in theta phi psi tx ty tz; do
            echo "Running with VLM_ID: ${VLM_ID}, LLM_ID: ${LLM_ID} on split: ${SPLIT}"
            RESULT_DIR=result/ma_big_table/$(basename "${VLM_ID}")/$(basename "${LLM_ID}")/${SPLIT}
            python experiments/experiment_2_multi_agents.py \
                --vlm_id "${VLM_ID}" \
                --llm_id "${LLM_ID}" \
                --data_dir "${DATA_DIR}" \
                --result_dir "${RESULT_DIR}" \
                --vlm_image_input_type "${VLM_IMAGE_INPUT_TYPE}" \
                --split "${SPLIT}" \
                --is_shuffle \
                --prompt_type "${PROMPT_TYPE}" \
                --max_len_of_conv "${MAX_LEN_OF_CONV}" \
                --is_vlm_keep_hisroty \
                --is_remove_trap_var
        done
    done
done