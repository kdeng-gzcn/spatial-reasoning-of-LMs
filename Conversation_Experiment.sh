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

# multi-turn conversation with start point
# python + .py + VLM llava-Next + LLM llama3
# /bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/envs/VLM/bin/python /bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM/Experiments/Experiment1_OneTurnConversation.py --test helloworld --VLM "llava-hf/llava-v1.6-mistral-7b-hf" --LLM "meta-llama/Meta-Llama-3-8B-Instruct" --data_path "./data/images_conversation" --mode "single"

# python + .py + VLM idefics2 + LLM llama3
# /bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/envs/VLM/bin/python /bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM/Experiments/Experiment1_OneTurnConversation.py --info "This is Conversation Experiments with VLM: idefics2, LLM: llama3, using pairwise images for VLN" --VLM "HuggingFaceM4/idefics2-8b" --LLM "meta-llama/Meta-Llama-3-8B-Instruct" --data_path "./data/images_conversation" --mode "pair"

# # python + .py + VLM Phi3 + LLM llama3
# /bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/envs/VLM/bin/python /bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM/Experiments/Experiment1_OneTurnConversation.py --info "This is Conversation Experiments with VLM: Phi3.5, LLM: llama3, using pairwise images for VLN" --VLM "microsoft/Phi-3.5-vision-instruct" --LLM "meta-llama/Meta-Llama-3-8B-Instruct" --data_path "./data/images_conversation" --mode "pair"

# python + .py + VLM Phi3 + LLM llama3 + 7Scenes Dataset
# /bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/envs/VLM/bin/python /bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM/Experiments/Experiment1_OneTurnConversation.py --VLM "microsoft/Phi-3.5-vision-instruct" --LLM "meta-llama/Meta-Llama-3-8B-Instruct" --data_path "./data/Rebuild_7_Scenes_1200" --mode "pair"

# Test
# /bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/envs/VLM/bin/python /bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM/Tests/text_VLM.py

# /bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/envs/VLM/bin/python /bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM/Tests/test_Phi.py

# python + .py + VLM Phi3 + LLM llama3 + 7Scenes Dataset Subset
/bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/envs/VLM/bin/python \
    /bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM/Experiments/Experiment1_OneTurnConversation.py \
    --VLM "microsoft/Phi-3.5-vision-instruct" \
    --LLM "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data_path "./data/Rebuild_7_Scenes_1200_1738445186" \
    --result_path "./Result/Pair Conversation Experiment phi/" \
    --mode "pair" \
    --subset "phi"
