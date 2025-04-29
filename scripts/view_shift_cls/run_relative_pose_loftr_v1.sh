#!/bin/bash
source ~/spatial-reasoning-language-models/miniconda3/bin/activate spatial_reasoning_env
source .env

for dataset in 7-scenes scannet; do
    for min_angle in 15 30 45 60; do
        echo "Running experiment with dataset=${dataset}, model is loftr, and min_angle=${min_angle}"
        data_dir=benchmark/obj-centered-view-shift-${dataset}/min-angle-${min_angle}-deg
        result_dir=result/obj-centered-view-shift-cls/v1/${dataset}/loftr/min-angle-${min_angle}-deg
        python eval_view_shift_cls/relative_pose_loftr_v1.py \
            --yaml_file config/eval_view_shift/config_loftr.yaml \
            --data_dir "$data_dir" \
            --result_dir "$result_dir" &
    done
done

wait 
echo "All experiments completed."