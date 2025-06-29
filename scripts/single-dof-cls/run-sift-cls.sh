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
    for split in theta phi psi tx ty tz; do
        data_dir=~/benchmark/single-dof-camera-motion-${dataset}/${split}_significant
        result_dir=result/final-table-cv-methods/single-dof-cls/${dataset}/${model}/${split}

        python eval-single-dof-cls/sift-cls.py \
            --yaml_file config/eval/sift.yaml \
            --data_dir "$data_dir" \
            --result_dir "$result_dir" \
            --dataset "$dataset" \
            --split "$split" &
    done
done

wait
echo "All experiments completed."