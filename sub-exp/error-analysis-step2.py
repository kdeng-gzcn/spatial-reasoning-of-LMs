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
import re

### load modules
from src.models.utils import load_model
from src.logging.logging_config import setup_logging

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
    parser = argparse.ArgumentParser(description="sub-exp-error-analysis-step2")
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
        "--prompt_mode", 
        type=int,
        required=True,
        help="Prompt mode to use for the experiment. 0 for default, 1 for custom."
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


def _parse_answer(answer: str) -> dict:
    try:
        pat_rsn = r"<obj>(.*?)(?:</obj>|<ans>|$)"
        match_rsn = re.search(pat_rsn, answer, re.DOTALL)
        

        pat_ans = r"<ans>\s*(\d+)\s*(?:</ans>|$)"
        match_ans = re.search(pat_ans, answer)

        if match_rsn and match_ans:
            rsn = match_rsn.groups()[0]
            ans = match_ans.groups()[0]
            return {
                "rsn": rsn,
                "ans": ans,
                "is_parse": True,
            }
        else:
            rsn = match_rsn.groups()[0] if match_rsn else None
            ans = match_ans.groups()[0] if match_ans else None
            return {
                "rsn": rsn,
                "ans": ans,
                "is_parse": False,
            }
    except Exception as e:
        print(f"Error in parsing answer: {e}")


def inference(dataloader_tqdm, vlm, result_dir, **kwargs) -> None:
    """
    Run inference on the dataloader using the VLM and pipeline.
    """
    mode = kwargs.get("mode")
    func = transforms.ToTensor()
    for data in dataloader_tqdm:
        src_img = func(Image.open(data["src_img_path"]))
        tgt_img = func(Image.open(data["tgt_img_path"]))

        vlm._clear_history()
        response = vlm.pipeline([src_img, tgt_img], data[f"prompt{mode}"])
        output = _parse_answer(response)
        if output["ans"]:
            pred = output["ans"]
        else:
            print(f"❌❌ Warning: No answer found in response: {response[:20]}, using 'error' as default.")
            pred = 'error'
        
        is_true = True if pred in data["label"] else False

        row = {}
        row["src_img_path"] = data["src_img_path"]
        row["tgt_img_path"] = data["tgt_img_path"]
        row["dof"] = data["dof"]
        row["sign"] = data["sign"]
        row["ref_obj"] = data["ref_obj"]
        row["label"] = data["label"]
        row["vlm_id"] = vlm.model_name
        row["prompt_mode"] = mode
        row["prompt"] = data[f"prompt{mode}"]
        row["answer"] = response
        row["obj_desc"] = output["obj"]
        row["pred"] = pred
        row["is_correct"] = is_true
        row["is_parse"] = output["is_parse"]

        with jsonlines.open(result_dir / "inference.jsonl", mode='a') as writer:
            writer.write(row)


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
        "accuracy": accuracy_score(df["cor_idx"], df["pred"]),
        "precision": precision_score(df["cor_idx"], df["pred"], average='weighted', zero_division=0),
        "recall": recall_score(df["cor_idx"], df["pred"], average='weighted', zero_division=0),
        "f1_score": f1_score(df["cor_idx"], df["pred"], average='weighted', zero_division=0),
        "total_num": int(df.shape[0]),
        "valid_num": int(df["is_parse"].sum()),
        "valid_ratio": float(df["is_parse"].sum() / len(df)),
        "valid_acc": accuracy_score(df[df["is_parse"]]["cor_idx"], df[df["is_parse"]]["pred"]),
        "f1_score": f1_score(df[df["is_parse"]]["cor_idx"], df[df["is_parse"]]["pred"], average='weighted', zero_division=0),
    }
    
    metrics_path = result_dir / "metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)  # ensure the directory exists
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        

def main(args):
    # Set up logger
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create result dir for later result saver and data analysis
    logger.info("🚀🚀 Step0: Results will be saved to %s", args.result_dir)
    result_dir = _create_result_dir(args.result_dir)

    # Load dataloader
    dataloader_tqdm = _load_dataloader(args.data_dir)

    # Load the model
    vlm = _load_model(args.vlm_id)

    # Run the inference
    logger.info("🚀🚀 Step1: Running inference with VLM %s", vlm.model_name)
    inference(dataloader_tqdm, vlm, result_dir, mode=args.prompt_mode)

    # try to print cost
    _print_cost(vlm)

    ### Data Analysis Part
    logger.info("🚀🚀 Step2: Starting data analysis")
    df = _load_inference_result(result_dir)
    df = df[df["prompt_mode"] == args.prompt_mode]
    quick_acc = df["is_correct"].mean()
    logger.info("⚠️⚠️ Quick accuracy: %.2f%%", quick_acc * 100)

    logger.info("🚀🚀 Step3: Data analysis completed. Results saved.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
