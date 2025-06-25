#!/bin/bash

#SBATCH --job-name=sift
#SBATCH --output=sift.out
#SBATCH --gpus=1
#SBATCH --time=5:00:00 

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_reasoning_env
set -a && source .env && set +a

model=sift

for dataset in 7-scenes scannet; do
    for min_angle in 15 30 45 60; do
        for task_split in translation rotation; do
            echo "Running experiment with dataset=${dataset}, model is sift, and min_angle=${min_angle}"

            data_dir=~/benchmark/obj-centered-view-shift-${dataset}/min-angle-${min_angle}-deg
            result_dir=result/final-table/obj-centered-cls/${task_split}/${dataset}/${model}/min-angle-${min_angle}-deg

            python eval-obj-centered-cls/sift-cls.py \
                --yaml_file config/eval_view_shift/config_sift.yaml \
                --data_dir "$data_dir" \
                --result_dir "$result_dir" \
                --min_angle "$min_angle" \
                --dataset "$dataset" \
                --split "$task_split" &
        done
    done
done

wait
echo "All experiments completed."