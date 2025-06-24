#!/usr/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_reasoning_env
set -a && source .env && set +a
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"

for dataset in scannetpp; do
    for vlm_id in gpt-4o; do
        for prompt_type in zero-shot dataset-prior-hint CoT-hint VoT-hint; do
            for split in theta phi psi tx ty tz; do
                DATA_DIR=~/benchmark/single-dof-camera-motion-${dataset}/${split}_significant
                RESULT_DIR=result/final-table/single-dof-cls/${dataset}/${vlm_id}/${prompt_type}/${split}

                python eval-single-dof-cls/vlm-only-cls.py \
                    --vlm_id ${vlm_id} \
                    --data_dir ${DATA_DIR} \
                    --result_dir ${RESULT_DIR} \
                    --split ${split} \
                    --prompt_type ${prompt_type} \
                    --dataset ${dataset} &
            done
        done
    done
done

wait
echo "All experiments completed."