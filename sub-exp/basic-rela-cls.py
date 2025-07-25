import json
import jsonlines
from tqdm import tqdm
from pathlib import Path
import logging
import argparse
from typing import Any
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

### load modules
from src.dataset.utils import load_dataset
from src.models.utils import load_model
from src.logging.logging_config import setup_logging
from src.prompt_generator import PromptGenerator
from src.pipeline import SpatialReasoningPipeline

### load config
from config.default import cfg

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
    parser = argparse.ArgumentParser(description="sub-exp-1")
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
    return parser.parse_args()


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


def _load_dataloader(data_dir: str):
    """
    Load the dataset and create a DataLoader.
    """
    with jsonlines.open(data_dir) as reader:
        dataset = [obj for obj in reader]
    dataloader_tqdm = tqdm(dataset, desc="Processing", total=len(dataset))
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
        "accuracy": accuracy_score(df["cor_idx_shuf"], df["pred"]),
        "precision": precision_score(df["cor_idx_shuf"], df["pred"], average='weighted', zero_division=0),
        "recall": recall_score(df["cor_idx_shuf"], df["pred"], average='weighted', zero_division=0),
        "f1_score": f1_score(df["cor_idx_shuf"], df["pred"], average='weighted', zero_division=0),
        "total_num": int(df.shape[0]),
        "valid_num": int(df["is_parse"].sum()),
        "valid_ratio": float(df["is_parse"].sum() / len(df)),
        "valid_acc": accuracy_score(df[df["is_parse"]]["cor_idx_shuf"], df[df["is_parse"]]["pred"]),
        "f1_score": f1_score(df[df["is_parse"]]["cor_idx_shuf"], df[df["is_parse"]]["pred"], average='weighted', zero_division=0),
    }
    
    metrics_path = result_dir / "metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)  # ensure the directory exists
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


# def _save_confusion_matrix(df: pd.DataFrame, result_dir: Path):
#     """
#     Save confusion matrix to the result directory.
#     """
#     cm = confusion_matrix(df["label"], df["pred"])
#     labels = sorted(df["pred"].unique())
#     cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
#     cm_csv_path = result_dir / "metrics" / "confusion_matrix.csv"
#     cm_df.to_csv(cm_csv_path)

#     # Plot and save heatmap
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.title('Confusion Matrix Heatmap')
#     heatmap_path = result_dir / "metrics" / "confusion_matrix_heatmap.png"
#     plt.tight_layout()
#     plt.savefig(heatmap_path)
#     plt.close()


def inference(dataloader_tqdm, vlm, result_dir, **kwargs) -> None:
    """
    Run inference on the dataloader using the VLM and pipeline.
    """
    func = transforms.ToTensor()
    for data in dataloader_tqdm:
        image = Image.open(data["img_path"])
        image = func(image)
        vlm._clear_history()
        response = vlm.pipe_one_img(image, data["prompt_shuf"])
        # find ans inside<ans></ans> tags from output
        ans = re.search(r'<ans>(.*?)</ans>', response, re.DOTALL)
        if ans:
            ans = ans.group(1)
        try:
            pred = int(ans)
            is_parse = True
        except:
            pred = random.randint(0, len(data["cap"]) - 1) 
            print(f"❌ Error in prediction: {ans}, using random index {pred}.")
            is_parse = False

        is_true = int(data["cor_idx_shuf"]) == pred

        data["vlm_id"] = vlm.model_name
        data["pred"] = pred
        data["is_correct"] = is_true
        data["is_parse"] = is_parse

        with jsonlines.open(result_dir / "inference.jsonl", mode='a') as writer:
            writer.write(data)
        


def main(args):
    # Set up logger
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create result dir for later result saver and data analysis
    result_dir = _create_result_dir(args.result_dir)
    logger.info("Step0: Results will be saved to %s", result_dir)

    # Load dataloader
    dataloader_tqdm = _load_dataloader(args.data_dir)

    # Load the model
    vlm = _load_model(args.vlm_id)

    # Run the inference
    logger.info("Starting inference...")
    inference(dataloader_tqdm, vlm, result_dir)

    # try to print cost
    _print_cost(vlm)

    ### Data Analysis Part
    df = _load_inference_result(result_dir)
    _save_general_metrics(df, result_dir)
    # _save_confusion_matrix(df, result_dir)

    logger.info("Processing completed.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
