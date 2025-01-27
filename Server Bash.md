# PowerShell

```bash
ssh -m hmac-sha2-256 slke8951@147.188.172.65
```

# Interactive GPU

```bash
srun --account jlxi8926-auto-sum --qos epsrc --gpus 1 --time 4:00:00 --export=USER,HOME,PATH,TERM --pty /bin/bash
```

> if torch.bfloat16:
>
> ​	Llama3-8B + Phi3.5-4B = 16GB + 8GB $\in [0, 40]$ .
>
> ​	Llama3-8B + llava1.6-7B = 16GB + 14GB $\in [0, 40]$ .

# Env

```bash
# renew env 
module purge
module load baskerville

# load cuda (nvcc...)
module load bask-apps/live
module load oneAPI-bundle/2024.2.0-CUDA-12.1.1
module load GCC/12.3.0 # for pandas

strings /lib64/libstdc++.so.6 | grep GLIBCXX # for checking pandas's pkg
strings /bask/apps/live/EL8-ice/software/GCCcore/12.3.0/lib64/libstdc++.so.6 | grep GLIBCXX

export LD_LIBRARY_PATH=/bask/apps/live/EL8-ice/software/GCCcore/12.3.0/lib64:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

```

# Server Account

```bash
# usrname
slke8951
DKgsg04180913#
One Time Password

# root fir
/bask/homes/s/slke8951

# actual root dir
/bask/projects/j/jlxi8926-auto-sum/kdeng/
cd /bask/projects/j/jlxi8926-auto-sum/kdeng/

# temp dir
cd /bask/projects/j/jlxi8926-auto-sum/kdeng/SpatialVLM

# bin dir
cd /bask/projects/j/jlxi8926-auto-sum/kdeng/doppelgangers/data/doppelgangers_dataset/doppelgangers

# compile
sed -i 's/\r$//' ???.sh
```

# Basic Server Command for Linux

```shell
ls
pwd # print wd
rm -r doppelgangers # 删除文件夹
rm -rf doppelgangers

# quota -s
du -sh ~ # 已经使用空间
du -sh * # 查看每个文件夹对应使用空间
df -h ~ # rate for entire system
du -sh ~/.cache/* # cache root
du -sh /bask/projects/j/jlxi8926-auto-sum/kdeng/cache/* # cache my dir
rm -rf /bask/projects/j/jlxi8926-auto-sum/kdeng/cache/*
rm -rf ~/.cache/huggingface/*
rm -rf ~/.cache/pip/*

# sinfo
# sinfo -Nel

conda config --show channels
conda config --add channels conda-forge
conda config --remove channels conda-forge
conda config --add channels conda-forge --prepend

# 恢复默认channels
conda config --remove-key channels
conda config --add channels defaults

# 临时尝试channels
conda install <package_name> -c conda-forge

conda config --set show_channel_urls yes # 安装包时显示来源
conda config --set channel_priority flexible # 规则是按照优先级安装

python3 demo.py
python3 --version

# vscode server problem
rm -rf ~/.vscode-server # on baserville
# rerun vscode remote ssh
```



# Job

```shell
#!/bin/bash
#SBATCH --account=jlxi8926-auto-sum
#SBATCH --qos=epsrc
#SBATCH --job-name=_yourname_
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=8G

source /bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/etc/profile.d/conda.sh
conda activate doppelgangers

export LD_LIBRARY_PATH=/bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/envs/doppelgangers/lib:$LD_LIBRARY_PATH

python test.py doppelgangers/configs/training_configs/doppelgangers_classifier_noflip.yaml \
  --pretrained weights/doppelgangers_classifier_loftr.pt
```



```shell
#!/bin/bash
#SBATCH --account=jlxi8926-auto-sum
#SBATCH --qos=epsrc
#SBATCH --job-name=check_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=2G

# Load any necessary modules (optional)
# module load cuda

# Check GPU details
nvidia-smi
```



```shell
sed -i 's/\r$//' check_gpu.sh
sbatch check_gpu.sh

cat slurm-<job_id>.out
cat slurm-<job_id>.stats
```



# Pytorch

```shell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```



# Cache Link

```bash
mv ~/.cache /bask/projects/j/jlxi8926-auto-sum/kdeng/.cache
ln -s /bask/projects/j/jlxi8926-auto-sum/kdeng/.cache ~/.cache
```















# Archive

```bash
# SSH
ssh -o MACs=hmac-sha2-256 slke8951@147.188.172.65

# Conda Env Variables (Maybe useless)
export LD_LIBRARY_PATH=/bask/projects/j/jlxi8926-auto-sum/kdeng/anaconda3/envs/VLM/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
```

