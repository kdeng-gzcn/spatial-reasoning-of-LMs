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

# Test
python Experiments/Experiment0_IndividualVLM.py \
    --VLM "microsoft/Phi-3.5-vision-instruct" \
    --data_path "./benchmark/Rebuild_7_Scenes_1739853799" \
    --result_path "./Result/Individual VLM Experiment phi newbenchmark/" \
    --subset "phi"
