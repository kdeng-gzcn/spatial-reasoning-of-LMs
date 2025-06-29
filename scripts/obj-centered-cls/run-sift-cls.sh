#!/bin/bash

#SBATCH --job-name=kdeng
#SBATCH --output=log/sift-%j.out
#SBATCH --gpus=0
#SBATCH --time=12:00:00 

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_reasoning_env
set -a && source .env && set +a

model=sift

for dataset in 7-scenes scannet; do
    for task_split in translation rotation; do
        for min_angle in 15 30 45 60; do
            data_dir=~/benchmark/obj-centered-view-shift-${dataset}/min-angle-${min_angle}-deg
            result_dir=result/final-table-cv-methods/obj-centered-cls/${dataset}/min-angle-${min_angle}-deg/${model}/${task_split}

            python eval-obj-centered-cls/sift-cls.py \
                --yaml_file config/eval/sift.yaml \
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