#!/bin/bash
source ~/spatial-reasoning-language-models/miniconda3/bin/activate spatial_reasoning_env
source .env
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"
export HF_HOME=~/spatial-reasoning-language-models/spatial-reasoning-of-LMs/.cache/huggingface

# VLM_ID=microsoft/Phi-3.5-vision-instruct
# VLM_ID=Qwen/Qwen2.5-VL-7B-Instruct
# VLM_ID=gpt-4o
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

# for PROMPT_TYPE in "zero-shot" "add-info-zero-shot" "CoT-zero-shot" "VoT-zero-shot"; do
#     for SPLIT in "theta" "phi" "psi" "tx" "ty" "tz"; do
#         echo "Running experiment with prompt_type=${PROMPT_TYPE} and split=${SPLIT}"
#         RESULT_DIR=result/vlm_only_big_table/${VLM_ID}/${PROMPT_TYPE}/${SPLIT}
#         python experiments/experiment_1_individual_vlm.py \
#             --vlm_id ${VLM_ID} \
#             --data_dir "benchmark/RGBD_7_Scenes_Rebuilt" \
#             --result_dir ${RESULT_DIR} \
#             --split ${SPLIT} \
#             --is_shuffle \
#             --prompt_type ${PROMPT_TYPE}
#     done
# done

# for dataset in 7-scenes scannet scannetpp; do
#     for vlm_id in gpt-4o; do
#         for prompt_type in zero-shot add-info-zero-shot CoT-zero-shot VoT-zero-shot; do
#             for split in "theta" "phi" "psi" "tx" "ty" "tz"; do
#                 echo "Running experiment with dataset=${dataset}, vlm_id=${vlm_id}, prompt_type=${prompt_type}, and split=${split}"
#                 RESULT_DIR=result/single-dof-camera-motion-cls/v1/${dataset}/${vlm_id}/${prompt_type}/${split}
#                 python eval_camera_motion_cls/eval_vlm_only.py \
#                     --vlm_id ${vlm_id} \
#                     --data_dir benchmark/single-dof-camera-motion-${dataset} \
#                     --result_dir ${RESULT_DIR} \
#                     --split ${split} \
#                     --is_shuffle \
#                     --prompt_type ${prompt_type} &
#             done
#         done
#     done
# done
# wait
# echo "All experiments completed."

for dataset in scannet scannetpp; do
    for vlm_id in gpt-4o; do
        for prompt_type in zero-shot add-info-zero-shot CoT-zero-shot VoT-zero-shot; do
            for split in theta; do
                echo "Running experiment with dataset=${dataset}, vlm_id=${vlm_id}, prompt_type=${prompt_type}, and split=${split}"
                RESULT_DIR=result/single-dof-camera-motion-cls/v1/${dataset}/${vlm_id}/${prompt_type}/${split}
                python eval_camera_motion_cls/eval_vlm_only.py \
                    --vlm_id ${vlm_id} \
                    --data_dir benchmark/single-dof-camera-motion-${dataset} \
                    --result_dir ${RESULT_DIR} \
                    --split ${split} \
                    --is_shuffle \
                    --prompt_type ${prompt_type} &
            done
        done
    done
done

wait
echo "All experiments completed."