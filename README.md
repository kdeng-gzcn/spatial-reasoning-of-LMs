# Evaluation of Spatial Reasoning
## Task: Single-DoF Camera Motion Classification
This benchmark includes tasks like judging how camera moves. The camera does not move randomly, for each task, the camera only moves in one DoF, which leads the view a slight difference from source view to target view.
> - [] sdfadsf
## Task: Object-Centered View Shift Classification
This benchmark is a more common case in computer vision or robot navigation. The benchmark are also consist of tasks of judging camera movement, but more focus on leftward/rightward translation and rotation of the camera. In this task, we constrain that, there must be some objects centered in both views, and the angles made by source camera, object, target camera should be large to make views a big difference.
## Env
#### SpaceLLaVA
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.45 --force-reinstall --no-cache-dir
```
#### microsoft/Phi-3.5-vision-instruct
```bash
flash_attn==2.5.8
numpy==1.24.4
Pillow==10.3.0
Requests==2.31.0
torch==2.3.0
torchvision==0.18.0
transformers==4.43.0
accelerate==0.30.0
```
#### Qwen/Qwen2.5-VL-7B-Instruct
```bash
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]==0.0.8
```

## Usage
### Generate benchmark based on 3D dataset
```bash
bash xxx
```
### Evaluate Vision Language Models' performance on customed benchmark
- task1
```bash
bash xxx
```
- task2
```bash
bash xxx
```

run demo on pair images:
```bash
bash scripts/run_tests.sh
```

## Datset
### RGBD-7-Scenes
### ScanNet-V2
this script is to download dataset
```bash
bash download_scannet_v2.sh
```

this script for reading the dataset
```bash
git clone https://github.com/ScanNet/ScanNet.git
```

```bash
bash data/scannet-v2/reader.sh 
```

### ScanNet++
this script is to download dataset


