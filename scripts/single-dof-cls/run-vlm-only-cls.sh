#!/bin/bash

#SBATCH --job-name=kdeng
#SBATCH --output=log/spaceqwen-c1-wo-trap-%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00  

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_reasoning_env
set -a && source .env && set +a
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"

MAX_JOBS=2

for dataset in 7-scenes scannet scannetpp; do
    for vlm_id in remyxai/SpaceQwen2.5-VL-3B-Instruct; do
        for prompt_type in zero-shot dataset-prior-hint CoT-hint VoT-hint; do
            for split in theta phi psi tx ty tz; do
                if [[ "$vlm_id" == */* ]]; then
                    dir_vlm="${vlm_id##*/}"
                else
                    dir_vlm="$vlm_id"
                fi

                DATA_DIR=~/benchmark/single-dof-camera-motion-${dataset}/${split}_significant
                RESULT_DIR=result/final-table-wo-trap/single-dof-cls/${dataset}/${dir_vlm}/${prompt_type}/${split} # remember the change the result dir

                python eval-single-dof-cls/vlm-only-cls.py \
                    --data_dir ${DATA_DIR} \
                    --result_dir ${RESULT_DIR} \
                    --dataset ${dataset} \
                    --split ${split} \
                    --vlm_id ${vlm_id} \
                    --is_shuffle \
                    --prompt_type ${prompt_type} &

                while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                    sleep 1
                done

            done
        done
    done
done

wait
echo "All experiments completed."