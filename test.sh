#!/bin/bash
module load GCC/12.3.0
export LD_LIBRARY_PATH=/bask/apps/live/EL8-ice/software/GCCcore/12.3.0/lib64:$LD_LIBRARY_PATH
export HF_HOME=/bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM/cache/ # Transformers Cache Dir (Necessary)

source /bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/etc/profile.d/conda.sh
conda activate VLM

python tests/test_qwen_instruct.py
