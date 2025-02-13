# # renew env 
# module purge
# module load baskerville

# Activate Env
source /bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/etc/profile.d/conda.sh
conda activate VLM

# Transformers Cache Dir (Necessary)
export HF_HOME=/bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM/cache/

# Test
python Experiments/Experiment0_IndividualVLM.py \
    --VLM "microsoft/Phi-3.5-vision-instruct" \
    --data_path "./data/Rebuild_7_Scenes_1200_1738445186" \
    --result_path "./Result/Individual VLM Experiment phi/" \
    --subset "phi"
