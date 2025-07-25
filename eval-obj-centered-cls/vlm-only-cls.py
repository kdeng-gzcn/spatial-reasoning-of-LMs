import json
from tqdm import tqdm
import re
from pathlib import Path
import numpy as np
import logging
import argparse
from typing import Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

### load modules
from src.dataset.utils import load_dataset
from src.models.utils import load_model
from src.logging.logging_config import setup_logging

# from config.eval_view_shift.vlm_relative_pose_prompt_v1 import task_prompt, short_answer_dict, detailed_answer_dict
# from config.eval_view_shift.vlm_view_shift_left_right_prompt_v2 import task_prompt, short_answer_dict, detailed_answer_dict

### load config
from config.default import cfg

### load modules
from src.prompt_generator import PromptGenerator
from src.pipeline import SpatialReasoningPipeline

# set seed
import random
import torch
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="obj-centered-cls VLM-Only")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model ID for VLM on the experiment"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--result_dir", 
        type=str,
        required=True,
        help="Directory to save the results"
    )
    parser.add_argument(
        "--min_angle",
        type=str,
        required=True,
        help="Minimum angle for the task, e.g., '0.0' for translation, '0.1' for phi"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, e.g., '7-scenes', 'scannet', 'scannetpp'"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Task name for the experiment, default is 'view-shift-cls'"
    )
    parser.add_argument(
        "--is_trap",
        action='store_true',
        help="Whether to add trap option in the prompt"
    )
    parser.add_argument(
        "--is_shuffle",
        action='store_true',
        help="Whether to shuffle the options in the prompt"
    )
    return parser.parse_args()


def _merge_cfg(args):
    """
    Merge the command line arguments into the configuration.
    """
    cfg.set_new_allowed(True)  # allow new keys to be set
    # cfg.merge_from_other_cfg(CN(vars(args)))
    # TODO: use CN to merge args
    cfg.EXPERIMENT.DATA_DIR = args.data_dir
    benchmark_name = _get_benchmark_name(args.data_dir)
    cfg.EXPERIMENT.TASK_NAME = _parse_benchmark_name(benchmark_name)
    cfg.EXPERIMENT.RESULT_DIR = args.result_dir 
    cfg.MODEL.VLM.ID = args.model_id
    cfg.EXPERIMENT.DATASET = args.dataset
    cfg.EXPERIMENT.MIN_ANGLE = float(args.min_angle)
    cfg.EXPERIMENT.TASK_SPLIT = args.split
    cfg.STRATEGY.IS_TRAP = args.is_trap
    cfg.STRATEGY.IS_SHUFFLE = args.is_shuffle
    return cfg


def _get_benchmark_name(data_dir: str) -> str:
    """
    Extract the benchmark name from the data directory.
    """
    data_dir = Path(data_dir)
    if data_dir.is_dir():
        return data_dir.parent.name
    else:
        raise ValueError(f"Invalid data directory: {data_dir}")


def _parse_benchmark_name(benchmark_name: str) -> str:
    """
    Parse the benchmark name to get the task name and split.
    """
    parts = benchmark_name.split('-')
    if len(parts) < 2:
        raise ValueError(f"Invalid benchmark name format: {benchmark_name}")
    
    task_name = '-'.join(parts[:2]) + '-cls'
    
    return task_name


def main(args):
    # Set up logger
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create result dir for later result saver and data analysis
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results will be saved to %s", result_dir)

    cfg = _merge_cfg(args)

    ###------------global-config------------###
    # Print the configuration
    logger.info("Configuration: %s", cfg)
    ###------------global-config------------###

    model = load_model(args.model_id)
    model._load_weight()

    # TODO: data_dir -> task dataset
    dataset = load_dataset(_get_benchmark_name(args.data_dir), data_root_dir=args.data_dir, cfg=cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    dataloader_tqdm = tqdm(dataloader, desc="Processing", total=len(dataloader) if hasattr(dataloader, '__len__') else None)

    prompt_generator = PromptGenerator(cfg)

    pipe = SpatialReasoningPipeline(cfg, prompt_generator=prompt_generator)

    i = 0
    for batch in dataloader_tqdm:
        item = next(iter(batch)) 
        # i += 1
        # if i > 3:
        #     break
        try:
            src_img, tgt_img = item["source_image"], item["target_image"]
            metadata = item["metadata"]
            images = (src_img, tgt_img)

            model._clear_history()  # clear the history of VLM for each pair of images
            
            pipe.run_vlm_only(
                images=images,
                metadata=metadata,
                vlm=model,
            )
        except Exception as e:
            logger.error(f"Debug: Error processing batch: {e}")
            continue

    try:
        model.print_total_tokens_usage()
    except:
        logger.warning("Model does not support token usage tracking.")

    ### Data Analysis Part
    import pandas as pd
    import seaborn as sns

    df = pd.read_json(result_dir / "inference.jsonl", lines=True)
    y_true = df["label"]
    y_pred = df["pred"]
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    metrics_path = result_dir / "metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)  # ensure the directory exists
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
  
    # Compute confusion matrix and save as CSV
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(df["pred"].unique())
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_csv_path = result_dir / "metrics" / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    # Plot and save heatmap
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix Heatmap')
    heatmap_path = result_dir / "metrics" / "confusion_matrix_heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    logger.info("Processing completed.")
    logger.info("Results saved to %s", args.result_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)
