#!/usr/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_reasoning_env
source .env
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"

for dataset in 7-scenes; do
    for vlm_id in gpt-4o; do
        for llm_id in gpt-4o-text-only; do
            for min_angle in 60; do
                echo "Running experiment with dataset=${dataset}, model_id=${model_id}, and min_angle=${min_angle}"
                data_dir=benchmark/obj-centered-view-shift-${dataset}/min-angle-${min_angle}-deg
                result_dir=result/demo/view-shift-multi-agent/${dataset}/${model_id}/min-angle-${min_angle}-deg
                python eval_view_shift_cls/multi-agent-cls.py \
                    --yaml_file config/eval_view_shift/multi_agent/multi_agents_options.yaml \
                    --vlm_id "$vln_id" \
                    --llm_id "$llm_id" \
                    --data_dir "$data_dir" \
                    --result_dir "$result_dir" &
            done
        done
    done
done

wait
echo "All experiments completed."