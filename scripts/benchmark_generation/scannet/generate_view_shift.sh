#!/bin/bash
source ~/spatial-reasoning-language-models/miniconda3/bin/activate spatial_reasoning_env
source .env

for min_angle in 15 30 45 60; do
    echo "Generating view shift for min_angle: $min_angle"
    output_dir="benchmark/obj-centered-view-shift-scannet-v2/min-angle-${min_angle}-deg"
    python benchmark_generation/scannet/generate_view_shift.py \
        --yaml_file config/benchmark_generation/scannet/obj-centered-view-shift-cls/config.yaml \
        --dataset_dir data/scannet-v2/scans_test \
        --output_dir "$output_dir" \
        --min_angle "$min_angle" &
done

wait
echo "All relative pose generations completed."