#!/bin/bash

# module purge
# module load baskerville

module load GCC/12.3.0
export LD_LIBRARY_PATH=/bask/apps/live/EL8-ice/software/GCCcore/12.3.0/lib64:$LD_LIBRARY_PATH
export HF_HOME=/bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM/cache/

source /bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/etc/profile.d/conda.sh
conda activate VLM

# VLM_ID=microsoft/Phi-3.5-vision-instruct
VLM_ID=Qwen/Qwen2.5-VL-7B-Instruct
RESULT_DIR=result/exp_vlm_only_reasoning_test_history
PROMPT_TYPE=VoT-zero-shot

python experiments/experiment_1_individual_vlm.py \
    --vlm_id ${VLM_ID} \
    --data_dir "benchmark/Rebuild_7_Scenes_1739853799" \
    --result_dir ${RESULT_DIR} \
    --split "phi" \
    --is_shuffle \
    --prompt_type ${PROMPT_TYPE}
