#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_reasoning_env
set -a && source .env && set +a

python /home/u5u/kdeng.u5u/spatial-reasoning-of-LMs/benchmark-generation/scannet/generate_camera_motion.py \
    --yaml_file /home/u5u/kdeng.u5u/spatial-reasoning-of-LMs/config/benchmark_generation/scannet/single-dof-camera-motion-cls/config.yaml \
    --dataset_dir ~/data/scannet-v2/scans_test \
    --output_dir ~/benchmark/single-dof-camera-motion-scannet \