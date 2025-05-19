#!/bin/bash
source ~/spatial-reasoning-language-models/miniconda3/bin/activate spatial_reasoning_env
source .env
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"
export HF_HOME=~/spatial-reasoning-language-models/spatial-reasoning-of-LMs/.cache/huggingface

for dataset in 7-scenes scannet; do
    for model_id in gpt-4o; do
        for min_angle in 15 30 45 60; do
            echo "Running experiment with dataset=${dataset}, model_id=${model_id}, and min_angle=${min_angle}"
            data_dir=benchmark/obj-centered-view-shift-${dataset}/min-angle-${min_angle}-deg
            result_dir=result/obj-centered-view-shift-cls/v2-left-right-camera-movement/${dataset}/${model_id}/min-angle-${min_angle}-deg
            python eval_view_shift_cls/relative_pose_vlm_v1.py \
                --data_dir "$data_dir" \
                --result_dir "$result_dir" \
                --model_id "$model_id" &
        done
    done
done

wait
echo "All experiments completed."