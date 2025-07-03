#!/bin/bash

#SBATCH --job-name=kdeng
#SBATCH --output=log/c2-scannet-%j.out
#SBATCH --gpus=0
#SBATCH --time=12:00:00 

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_reasoning_env
set -a && source .env && set +a

for min_angle in 30 45; do
    echo "Generating view shift for min_angle: $min_angle"
    output_dir="/home/u5u/kdeng.u5u/benchmark/obj-centered-view-shift-scannet-v2/min-angle-${min_angle}-deg"
    python /home/u5u/kdeng.u5u/spatial-reasoning-of-LMs/benchmark-generation/scannet/generate_view_shift.py \
        --yaml_file config/benchmark_generation/scannet/obj-centered-view-shift-cls/config.yaml \
        --dataset_dir ~/data/scannet-v2/scans_test \
        --output_dir "$output_dir" \
        --min_angle "$min_angle" &
done

wait
echo "All relative pose generations completed."