#!/bin/bash
source ~/spatial-reasoning-language-models/miniconda3/bin/activate spatial_reasoning_env
source .env

for min_angle in 15 30 45 60; do
    echo "Generating relative pose for min_angle: $min_angle"
    output_dir="benchmark/relative-pose-scannet/min-angle-${min_angle}-deg"
    python benchmark_generation/scannet/generate_relative_pose.py \
        --yaml_file config/benchmark_generation/scannet/relative_pose/config.yaml \
        --dataset_dir data/scannet-v2/scans_test \
        --output_dir "$output_dir" \
        --min_angle "$min_angle"
done