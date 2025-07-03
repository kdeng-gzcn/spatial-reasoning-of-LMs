#!/bin/bash

#SBATCH --job-name=kdeng
#SBATCH --output=log/c1-%j.out
#SBATCH --gpus=0
#SBATCH --time=12:00:00  

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_reasoning_env
set -a && source .env && set +a
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"

MAX_JOBS=3

for dataset in 7-scenes scannet scannetpp; do
    for vlm_id in claude-sonnet-4-20250514; do
        for prompt_type in zero-shot dataset-prior-hint CoT-hint VoT-hint; do
            for split in theta phi psi tx ty tz; do
                if [[ "$vlm_id" == */* ]]; then
                    dir_vlm="${vlm_id##*/}"
                else
                    dir_vlm="$vlm_id"
                fi

                DATA_DIR=~/benchmark/single-dof-camera-motion-${dataset}/${split}_significant
                RESULT_DIR=result/final-table-w-trap/single-dof-cls/${dataset}/${dir_vlm}/${prompt_type}/${split}

                python eval-single-dof-cls/vlm-only-cls.py \
                    --data_dir ${DATA_DIR} \
                    --result_dir ${RESULT_DIR} \
                    --dataset ${dataset} \
                    --split ${split} \
                    --vlm_id ${vlm_id} \
                    --is_trap \
                    --is_shuffle \
                    --prompt_type ${prompt_type} &

                # while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                #     sleep 1
                # done

            done
        done
    done
done

wait
echo "All experiments completed."