# # renew env 
# module purge
# module load baskerville

module load GCC/12.3.0
export LD_LIBRARY_PATH=/bask/apps/live/EL8-ice/software/GCCcore/12.3.0/lib64:$LD_LIBRARY_PATH

# Activate Env
source /bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/etc/profile.d/conda.sh
conda activate VLM

# Transformers Cache Dir (Necessary)
export HF_HOME=/bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM/cache/

VLM_ID=microsoft/Phi-3.5-vision-instruct
# VLM_ID=Qwen/Qwen2.5-VL-7B-Instruct
RESULT_DIR=result/exp_individual_vlm_phi
PROMPT_TYPE=VoT-zero-shot

# Test
python experiments/experiment_1_individual_vlm.py \
    --vlm_id ${VLM_ID} \
    --data_path "benchmark/Rebuild_7_Scenes_1739853799" \
    --result_path ${RESULT_DIR} \
    --split "phi" \
    --is_shuffle \
    --prompt ${PROMPT_TYPE}
