# Evaluation of Spatial Reasoning
## Task: Single-DoF Camera Motion Classification

## Task: Object-Centered View Shift Classification

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
