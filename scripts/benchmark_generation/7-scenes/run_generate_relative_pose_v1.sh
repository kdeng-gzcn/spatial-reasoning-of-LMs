#!/bin/bash

source ~/spatial-reasoning-language-models/miniconda3/bin/activate spatial_reasoning_env

for min_angle in 60; do
    echo "Generating relative pose for min_angle: $min_angle"
    output_dir="benchmark/relative-pose-7-scenes-v1/min-angle-${min_angle}-deg"
    python generate_relative_pose_7_scenes_v1.py \
        --yaml_file config/config_relative_pose_v1.yaml \
        --dataset_dir data/rgb-d-dataset-7-scenes \
        --output_dir "$output_dir" \
        --min_angle "$min_angle"
done
