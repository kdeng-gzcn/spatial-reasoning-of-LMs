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
    parser = argparse.ArgumentParser(description="single-dof-cls VLM-Only")
    parser.add_argument(
        "--vlm_id",
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
        "--prompt_type",
        type=str,
        required=True,
        help="[zero-shot, dataset-prior, dataset-prior-CoT, dataset-prior-VoT]"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, e.g., 'seven-scenes', 'scannet', 'scannetpp'"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Task name for the experiment, default is 'view-shift-cls'"
    )
    return parser.parse_args()


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
    cfg.MODEL.VLM.ID = args.vlm_id
    cfg.EXPERIMENT.DATASET = args.dataset
    cfg.STRATEGY.VLM_ONLY.PROMPT_TYPE = args.prompt_type
    cfg.EXPERIMENT.TASK_SPLIT = args.split
    return cfg


def _create_result_dir(result_dir: str) -> Path:
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def _load_model(model_id: str):
    """
    Load the model based on the model ID.
    """
    model = load_model(model_id)
    model._load_weight()
    return model


def _load_dataloader(data_dir: str, cfg: Any):
    """
    Load the dataset and create a DataLoader.
    """
    dataset = load_dataset(_get_benchmark_name(data_dir), data_root_dir=data_dir, cfg=cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    dataloader_tqdm = tqdm(dataloader, desc="Processing", total=len(dataloader) if hasattr(dataloader, '__len__') else None)
    return dataloader_tqdm


def _print_cost(model):
    """
    Print the cost of the model.
    """
    try:
        model.print_total_tokens_usage()
    except Exception as e:
        logging.warning("Model does not support token usage tracking: %s", e)


def _load_inference_result(result_dir: Path) -> pd.DataFrame:
    """
    Load the inference results from the result directory.
    """
    inference_file = result_dir / "inference.jsonl"
    if not inference_file.exists():
        raise FileNotFoundError(f"Inference file not found: {inference_file}")
    
    df = pd.read_json(inference_file, lines=True)
    return df


def _save_general_metrics(df: pd.DataFrame, result_dir: Path):
    """
    Save general metrics to the result directory.
    """
    metrics = {
        "accuracy": accuracy_score(df["label"], df["pred"]),
        "precision": precision_score(df["label"], df["pred"], average='weighted', zero_division=0),
        "recall": recall_score(df["label"], df["pred"], average='weighted', zero_division=0),
        "f1_score": f1_score(df["label"], df["pred"], average='weighted', zero_division=0),
    }
    
    metrics_path = result_dir / "metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)  # ensure the directory exists
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


def _save_confusion_matrix(df: pd.DataFrame, result_dir: Path):
    """
    Save confusion matrix to the result directory.
    """
    cm = confusion_matrix(df["label"], df["pred"])
    labels = sorted(df["pred"].unique())
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    cm_csv_path = result_dir / "metrics" / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    # Plot and save heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix Heatmap')
    heatmap_path = result_dir / "metrics" / "confusion_matrix_heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()


def inference(dataloader_tqdm, vlm, pipe, **kwargs) -> None:
    """
    Run inference on the dataloader using the VLM and pipeline.
    """
    for batch in dataloader_tqdm:
        item = next(iter(batch))  # get the first item in the batch

        src_img, tgt_img = item["source_image"], item["target_image"]
        metadata = item["metadata"]
        images = (src_img, tgt_img)

        vlm._clear_history()  # clear the history of VLM for each pair of images
        
        pipe.run_vlm_only(
            images=images,
            metadata=metadata,
            vlm=vlm,
        )


def main(args):
    # Set up logger
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create result dir for later result saver and data analysis
    result_dir = _create_result_dir(args.result_dir)
    logger.info("Results will be saved to %s", result_dir)

    # Merge command line arguments into the configuration
    cfg = _merge_cfg(args)
    logger.info("Configuration: %s", cfg)

    # Load dataloader
    dataloader_tqdm = _load_dataloader(args.data_dir, cfg)

    # Load the model
    vlm = _load_model(args.vlm_id)

    # Load the prompt generator
    prompt_generator = PromptGenerator(cfg)

    # Load the pipeline
    pipe = SpatialReasoningPipeline(cfg, prompt_generator=prompt_generator)

    # Run the inference
    logger.info("Starting inference...")
    inference(dataloader_tqdm, vlm, pipe)

    # try to print cost
    _print_cost(vlm)

    ### Data Analysis Part
    df = _load_inference_result(result_dir)
    _save_general_metrics(df, result_dir)
    _save_confusion_matrix(df, result_dir)

    logger.info("Processing completed.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
