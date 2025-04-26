#!/bin/bash
source ~/spatial-reasoning-language-models/miniconda3/bin/activate spatial_reasoning_env
source .env

python benchmark_generation/scannetpp/generate_spatial_reasoning.py \
    --yaml_file config/benchmark_generation/scannetpp/spatial_reasoning/config.yaml \
    --dataset_dir data/scannetpp/data \
    --output_dir benchmark/spatial-reasoning-scannetpp \