#!/bin/bash

#SBATCH --job-name=kdeng
#SBATCH --output=log/c1-%j.out
#SBATCH --gpus=1
#SBATCH --time=12:00:00  

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_reasoning_env
set -a && source .env && set +a
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"

MAX_JOBS=3

for dataset in 7-scenes scannet scannetpp; do
    for vlm_id in gpt-4o Qwen/Qwen2.5-VL-7B-Instruct; do
        for prompt_type in zero-shot dataset-prior-hint CoT-hint VoT-hint; do
            for split in theta phi psi tx ty tz; do
                DATA_DIR=~/benchmark/single-dof-camera-motion-${dataset}/${split}_significant

                if [[ "$vlm_id" == */* ]]; then
                    dir_vlm="${vlm_id##*/}"
                else
                    dir_vlm="$vlm_id"
                fi

                RESULT_DIR=result/final-table-with-trap-opt/single-dof-cls/${dataset}/${dir_vlm}/${prompt_type}/${split}

                python eval-single-dof-cls/vlm-only-cls.py \
                    --vlm_id ${vlm_id} \
                    --data_dir ${DATA_DIR} \
                    --result_dir ${RESULT_DIR} \
                    --split ${split} \
                    --prompt_type ${prompt_type} \
                    --dataset ${dataset} &

                while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                    sleep 1
                done

            done
        done
    done
done

wait
echo "All experiments completed."