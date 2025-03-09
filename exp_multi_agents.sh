# # renew env 
# module purge
# module load baskerville

# # load cuda (nvcc...)
# module load bask-apps/live
# module load oneAPI-bundle/2024.2.0-CUDA-12.1.1

# Activate Env
source /bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/etc/profile.d/conda.sh
conda activate VLM

# Transformers Cache Dir (Necessary)
export HF_HOME=/bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM/cache/

# Test
# /bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/envs/VLM/bin/python /bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM/Tests/text_VLM.py

# python + .py + VLM Phi3 + LLM llama3 + 7Scenes Dataset Subset
# python /bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM/Experiments/Experiment1_OneTurnConversation.py \
# python Experiments/Experiment1_OneTurnConversation.py \
#     --VLM "microsoft/Phi-3.5-vision-instruct" \
#     --LLM "meta-llama/Meta-Llama-3-8B-Instruct" \
#     --data_path "./data/Rebuild_7_Scenes_1200_1738445186" \
#     --result_path "./Result/Pair Conversation Experiment phi/" \
#     --mode "pair" \
#     --subset "phi"

python Experiments/Experiment2_MultiTurnConversation.py \
    --VLM "microsoft/Phi-3.5-vision-instruct" \
    --LLM "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data_path "./benchmark/Rebuild_7_Scenes_1739853799" \
    --result_path "./Result/Pair VLM Exp on phi newbenchmark/" \
    --mode "pair" \
    --subset "phi" \
    --max_len_conv 10
