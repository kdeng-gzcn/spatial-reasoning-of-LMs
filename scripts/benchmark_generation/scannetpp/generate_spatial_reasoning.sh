#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_reasoning_env
set -a && source .env && set +a

python benchmark-generation/scannetpp/generate_camera_motion.py \
    --yaml_file config/benchmark_generation/scannetpp/single-dof-camera-motion-cls/config.yaml \
    --dataset_dir ~/data/scannetpp/data \
    --output_dir ~/benchmark/single-dof-camera-motion-scannetpp \