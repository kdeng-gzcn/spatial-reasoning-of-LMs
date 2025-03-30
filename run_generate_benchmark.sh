module purge
module load baskerville

# config for pandas
module load GCC/12.3.0
export LD_LIBRARY_PATH=/bask/apps/live/EL8-ice/software/GCCcore/12.3.0/lib64:$LD_LIBRARY_PATH

source /bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/etc/profile.d/conda.sh
conda activate VLM

python generate_benchmark.py \
    --config_path "configs/generate_benchmark.yaml" \
    --dataset_dir "benchmark/RGBD_7_Scenes_Rebuilt" \
    --output_dir "result/benchmark/RGBD_7_Scenes_Rebuilt" \
