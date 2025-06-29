#!/bin/bash

#SBATCH --job-name=kdeng
#SBATCH --output=log/c2-%j.out
#SBATCH --gpus=1
#SBATCH --time=12:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_reasoning_env
set -a && source .env && set +a
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"

MAX_JOBS=3

for dataset in 7-scenes scannet; do
    for model_id in gpt-4o Qwen/Qwen2.5-VL-7B-Instruct; do
        for min_angle in 15 30 45 60; do
            for task_split in translation; do
                echo "Running experiment with dataset=${dataset}, model_id=${model_id}, and min_angle=${min_angle}"

                if [[ "$model_id" == */* ]]; then
                    dir_vlm="${model_id##*/}"
                else
                    dir_vlm="$model_id"
                fi

                data_dir=~/benchmark/obj-centered-view-shift-${dataset}/min-angle-${min_angle}-deg
                result_dir=result/final-table-with-trap-opt/obj-centered-cls/${task_split}/${dataset}/${dir_vlm}/min-angle-${min_angle}-deg

                python eval-obj-centered-cls/vlm-only-cls.py \
                    --data_dir "$data_dir" \
                    --result_dir "$result_dir" \
                    --model_id "$model_id" \
                    --min_angle "$min_angle" \
                    --dataset "$dataset" \
                    --split "$task_split" &

                while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                    sleep 1
                done
                
            done
        done
    done
done

wait
echo "All experiments completed."