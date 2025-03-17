#!/bin/bash

module load GCC/12.3.0 # Load GCC for pandas
export LD_LIBRARY_PATH=/bask/apps/live/EL8-ice/software/GCCcore/12.3.0/lib64:$LD_LIBRARY_PATH
export HF_HOME=/bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM/cache/ # Transformers Cache Dir (Necessary)

source /bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/etc/profile.d/conda.sh
conda activate VLM

# VLM_ID=microsoft/Phi-3.5-vision-instruct
VLM_ID=Qwen/Qwen2.5-VL-7B-Instruct
# LLM_ID=meta-llama/Meta-Llama-3-8B-Instruct
LLM_ID=Qwen/Qwen2.5-7B-Instruct
RESULT_DIR=result/exp_multi_agents_reasoning
PROMPT_TYPE=zero-shot

python experiments/experiment_2_multi_agents.py \
    --vlm_id ${VLM_ID} \
    --llm_id ${LLM_ID} \
    --data_dir "benchmark/Rebuild_7_Scenes_1739853799" \
    --result_dir ${RESULT_DIR} \
    --vlm_image_input_type "pair" \
    --split "phi" \
    --is_shuffle \
    --prompt_type ${PROMPT_TYPE} \
    --max_len_of_conv 10
