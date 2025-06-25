from pathlib import Path
import json
from tqdm import tqdm
from pathlib import Path
import logging
import argparse
from typing import Any
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### load modules
from src.dataset.utils import load_dataset
from src.models.utils import load_model
from src.logging.logging_config import setup_logging

### load config
from config.default import cfg

### load modules
from src.prompt_generator import PromptGenerator
from src.pipeline import SpatialReasoningPipeline


def _load_model(model_id: str):
    """
    Load the model based on the model ID.
    
    """
    model = load_model(model_id)
    model._load_weight()
    return model


def _get_benchmark_name(data_dir: str) -> str:
    """
    Extract the benchmark name from the data directory.
    """
    data_dir = Path(data_dir)
    if data_dir.is_dir():
        return data_dir.parent.name
    else:
        raise ValueError(f"Invalid data directory: {data_dir}")

def _load_dataloader(data_dir: str, cfg: Any):
    """
    Load the dataset and create a DataLoader.
    """
    dataset = load_dataset(_get_benchmark_name(data_dir), data_root_dir=data_dir, cfg=cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    dataloader_tqdm = tqdm(dataloader, desc="Processing", total=len(dataloader) if hasattr(dataloader, '__len__') else None)
    return dataloader_tqdm


data_dir = "/home/u5u/kdeng.u5u/benchmark/single-dof-camera-motion-scannet/theta_significant"
dataloader = _load_dataloader(data_dir, cfg)

# Load the model
vlm_id = "Qwen/Qwen2.5-VL-7B-Instruct"
vlm = _load_model(vlm_id)

# Load the prompt generator
prompt_generator = PromptGenerator(cfg)

# Load the pipeline
pipe = SpatialReasoningPipeline(cfg, prompt_generator=prompt_generator)

batch = next(iter(dataloader))
item = next(iter(batch))  # get the first item in the batch

src_img, tgt_img = item["source_image"], item["target_image"]
metadata = item["metadata"]
images = (src_img, tgt_img)

vlm._clear_history()  # clear the history of VLM for each pair of images

prompt = "What do you see?"
    
vlm_answer = vlm.pipeline(images, prompt)

print(f"VLM Answer: \n\n{vlm_answer}")