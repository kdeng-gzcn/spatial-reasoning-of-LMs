#!/bin/bash
source ~/spatial-reasoning-language-models/miniconda3/bin/activate spatial_reasoning_env
source .env

python benchmark_generation/scannet/generate_spatial_reasoning.py \
    --yaml_file config/benchmark_generation/scannet/spatial_reasoning/config.yaml \
    --dataset_dir data/scannet-v2/scans_test \
    --output_dir benchmark/spatial-reasoning-scannet \