# Evaluation of Spatial Reasoning
## Rebuild
### Config
- [ ] use yacs for all config
- [ ] specify how many nodes should i make? what is the structure of the nodes?
### Dataset
- [ ] metadata, translate one into image path, otherwise, i will keep using hard-code prefix for result dataframe

current metadata:

{
    "scene": "scene0714_00",
    "pair": "000875-001007",
    "tx": 1.627007,
    "ty": -0.725143,
    "tz": 1.67707,
    "theta": 1.2503,
    "phi": -49.181585,
    "psi": -28.36509,
    "tx_text": "right",
    "ty_text": "up",
    "tz_text": "forward",
    "theta_text": "upward",
    "phi_text": "leftward",
    "psi_text": "counterclockwise",
    "distance": 290.646965,
    "angle": 63.09769
}

expected metadata:

{
    "data_path": "/root/spatial-reasoning-of-LMs/scene0714_00/000875-001007",
    "tx": 1.627007,
    "ty": -0.725143,
    "tz": 1.67707,
    "theta": 1.2503,
    "phi": -49.181585,
    "psi": -28.36509,
    "tx_text": "right",
    "ty_text": "up",
    "tz_text": "forward",
    "theta_text": "upward",
    "phi_text": "leftward",
    "psi_text": "counterclockwise",
    "distance": 290.646965,
    "angle": 63.09769
}

### Prompt 
- [ ] make a general template for all dataset subset
- [ ] find a banalance between dataset subset prior, the reasoning skils and config
- [ ] make a nicer candidate generater (shuffle, trap, labelencoder)
### Pipeline
- [x] multi agents
### Parser
- [ ] parser function for extracting answer
- [ ] consider how prompt design?
### Result Saver
- [ ] save result with jsonlines in a stream way for each sample
- [ ] make a nice inference_result dataframe, chat_history dataframe for multi-agents, consider how prompt design?
- [ ] filter out the errors.jsonlines
### Data Analysis
- [ ] for final result (list of dict), do data analysis and record basic metrics
## Task: Single-DoF Camera Motion Classification
This benchmark includes tasks like judging how camera moves. The camera does not move randomly, for each task, the camera only moves in one DoF, which leads the view a slight difference from source view to target view.
> - [ ] make new dataset with larger threshold (now, the view changes are slight)
## Task: Object-Centered View Shift Classification
This benchmark is a more common case in computer vision or robot navigation. The benchmark are also consist of tasks of judging camera movement, but more focus on leftward/rightward translation and rotation of the camera. In this task, we constrain that, there must be some objects centered in both views, and the angles made by source camera, object, target camera should be large to make views a big difference.
> - [ ] filter out overlapped dataset, make it cleaner
## Strategy
### VLM-Only
An intuitive way to evaluate is to provide a task description prompt to VLM, and ask VLM to do a multi-choice classification, and see if VLM can find the correct answer.

1. zero-shot

2. with dataset prior

3. CoT

4. VoT

> - [ ] try to figure out hou many choices as candidates is the best
### Multi-Agent (LLM + VLM)
Based on our ablation study, we find that if the caption of images is given (including the depth information is the best), there are might be improvements. Also, we hope LLM here to come up with a better reasonning.

1. with dataset prior
> - [ ] refine the pipeline, and the prompt
## Ablation Study
Once we find that, even SOTA VLM fails in these easy spatial reasoning tasks. (inferring the camera motion based on pair of images), we did an error analysis and ablation study on demo dataset to find out the underlying problem.
### Error Analysis
We define the error type for failed cases. And manually find a distribution of error types. (This is conducted on demo dataset, obj-centered task, 37 samples)
### Refine with caption
We also conducted experiments with different level description (image caption), and see how LLM performs on it, and we compare with VLM's performance. The results reveal that, if the LLM with detailed caption of image, they could do better than VLM. (This is conducted on demo dataset, obj-centered task, 37 samples) But the prompt should at least provide depth information, i.e. obj A is occluded by obj B.
### Choice Preference
We also find that, language models have a strong preference in choosing one of the options, e.g. gpt-do prefer rightward answer.
# Usage
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

## Generate benchmark based on 3D dataset
```bash
bash xxx
```
## Evaluate Vision Language Models' performance on customed benchmark
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

## Datset Download
#### RGBD-7-Scenes
#### ScanNet-V2
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
#### ScanNet++
this script is to download dataset

