#!/bin/bash

source ~/spatial-reasoning-language-models/miniconda3/bin/activate spatial_reasoning_env

for min_angle in 15 30 45 60; do
    echo "testing relative pose for min_angle: $min_angle"
    for model_id in gpt-4o; do
        echo "testing relative pose for model_id: $model_id"
        data_dir="benchmark/relative-pose-7-scenes-v1/min-angle-${min_angle}-deg"
        result_dir="result/task-relative-pose/${model_id}/min-angle-${min_angle}-deg"
        python relative_pose_vlm_v1.py \
            --data_dir "$data_dir" \
            --result_dir "$result_dir" \
            --model_id "$model_id"
    done
done