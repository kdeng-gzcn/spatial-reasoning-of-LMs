#!/bin/bash

source ~/spatial-reasoning-language-models/miniconda3/bin/activate spatial_reasoning_env

for min_angle in 15 30 45 60; do
    echo "testing SIFT on relative pose for min_angle: $min_angle"
    data_dir="benchmark/relative-pose-7-scenes-v1/min-angle-${min_angle}-deg"
    result_dir="result/sift_relative_pose_v1/min-angle-${min_angle}-deg"
    python sift_relative_pose_v1.py \
        --yaml_file config/config_sift.yaml \
        --data_dir "$data_dir" \
        --result_dir "$result_dir"
done