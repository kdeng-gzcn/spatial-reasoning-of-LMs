#!/bin/bash

#SBATCH --job-name=kdeng
#SBATCH --output=log/c2-%j.out
#SBATCH --gpus=0
#SBATCH --time=12:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_reasoning_env
set -a && source .env && set +a
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"

MAX_JOBS=3

for dataset in 7-scenes scannet; do
    for model_id in claude-sonnet-4-20250514; do
        for min_angle in 15 30 45 60; do
            for task_split in rotation translation; do
                if [[ "$model_id" == */* ]]; then
                    dir_vlm="${model_id##*/}"
                else
                    dir_vlm="$model_id"
                fi

                data_dir=~/benchmark/obj-centered-view-shift-${dataset}/min-angle-${min_angle}-deg
                result_dir=result/final-table-w-trap/obj-centered-cls/${task_split}/${dataset}/${dir_vlm}/min-angle-${min_angle}-deg

                python eval-obj-centered-cls/vlm-only-cls.py \
                    --data_dir "$data_dir" \
                    --result_dir "$result_dir" \
                    --dataset "$dataset" \
                    --min_angle "$min_angle" \
                    --model_id "$model_id" \
                    --is_trap \
                    --is_shuffle \
                    --split "$task_split" \

                # while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                #     sleep 1
                # done
                
            done
        done
    done
done

wait
echo "All experiments completed."