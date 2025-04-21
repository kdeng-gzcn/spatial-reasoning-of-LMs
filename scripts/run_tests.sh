#!/bin/bash
source ~/spatial-reasoning-language-models/miniconda3/bin/activate spatial_reasoning_env

source .env
huggingface-cli login --token "${HUGGINGFACE_TOKEN}"

python tests/test_unproject_distance_angle.py
