"""
This is the script conducting vlm+llm performance on large-view-change spatial reasoning tasks.

focus on camera traslation.
"""
import json
import jsonlines
from tqdm import tqdm
import re
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import logging
import argparse
from typing import Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.dataset.utils import load_dataset
from src.models.utils import load_model
from src.logging.logging_config import setup_logging

from config.eval_view_shift.multi_agent.multi_agent_obj_centered_prompt import (
    task_prompt_image_caption,
    spatial_reasoning_prompt_image_caption,
    short_answer_dict,
    detailed_answer_dict,
)

from yacs.config import CfgNode as CN
from config.config import cfg
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
    parser = argparse.ArgumentParser(description="SIFT Relative Pose Estimation")
    parser.add_argument(
        "--yaml_file",
        type=str,
        default=True,
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="tx",
        choices=["tx", "phi"],
        help="Split of the dataset to use (e.g., 'tx', 'phi')"
    )
    parser.add_argument(
        "--llm_id",
        type=str,
        required=True,
        help="Model ID for LLM on the experiment"
    )
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


# def save_results(cfg: dict, results: list, result_dir: str, full_history: list) -> None: 
#     result_dir = Path(result_dir)
#     result_dir.mkdir(parents=True, exist_ok=True)
#     logging.info("Saving results to %s", result_dir)

#     with open(result_dir / "cfg.json", "w") as f:
#         json.dump(cfg, f, indent=4)

#     with jsonlines.open(result_dir / f"view_shift_results.jsonl", mode='w') as writer:
#         for result in results:
#             writer.write(result)

#     df = pd.DataFrame(results)
#     df.to_csv(result_dir / f"view_shift_results.csv", index=False) 
#     logging.info("Results saved to %s", result_dir / f"view_shift_results.csv") 

#     df = pd.DataFrame(full_history)
#     df.to_csv(result_dir / f"full_history.csv", index=False)
#     logging.info("Full history saved to %s", result_dir / f"full_history.csv")

#     # compute metrics and save results
#     y_true = df["label"].values
#     y_pred = df["pred"].values
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
#     recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
#     f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
#     metrics = {
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1_score": f1,
#     }
#     # save metrics to json
#     with open(result_dir / "metrics.json", "w") as f:
#         json.dump(metrics, f, indent=4)
#     # save confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     labels = list(np.unique(y_pred))
#     cm_df = pd.DataFrame(cm, index=labels, columns=labels)
#     cm_df.to_csv(result_dir / "confusion_matrix.csv", index=True)
#     logging.info("Metrics saved to %s", result_dir / "metrics.json")

def main(args):
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create result dir
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results will be saved to %s", result_dir)

    # Load yaml file and merge with yacs
    cfg.set_new_allowed(True)  # allow new keys to be set
    logger.info("Configuration loaded from %s", args.yaml_file)
    cfg.merge_from_file(args.yaml_file)

    cfg.merge_from_other_cfg(CN(vars(args)))

    ###------------global-config------------###
    # Print the configuration
    logger.info("Configuration: %s", cfg)
    ###------------global-config------------###

    # Load models
    llm = load_model(args.llm_id)
    llm._load_weight()
    vlm = load_model(args.vlm_id)
    vlm._load_weight()

    # Load dataset
    dataset = load_dataset(Path(args.data_dir).parent.name, data_root_dir=args.data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    dataloader_tqdm = tqdm(dataloader, desc="Processing", total=len(dataloader) if hasattr(dataloader, '__len__') else None)

    # Load prompt generator
    prompt_generator = PromptGenerator(cfg)

    # Load pipeline
    pipe = SpatialReasoningPipeline(cfg, prompt_generator)

    i = 0
    for batch in dataloader_tqdm:
        item = next(iter(batch))  # get the only item from the batch
        i += 1
        # if i > 1:
        #     break

        try:
            ### prepare data
            src_img, tgt_img = item["source_image"], item["target_image"]
            metadata = item["metadata"]
            images = (src_img, tgt_img)

            llm._clear_history()  # clear the history of VLM for each pair of images
            vlm._clear_history()  # clear the history of LLM for each pair of images

            pipe.run_multi_agents(
                images=images,
                metadata=metadata,
                vlm=vlm,
                llm=llm,
            )
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            break # it should not have any error.

    try:
        vlm.print_total_tokens_usage()
        llm.print_total_tokens_usage()
    except:
        logger.warning("Model does not support token usage tracking.")

    ### Data analysis
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

    # save_results(cfg, structure_result, args.result_dir, full_history)
    logger.info("Processing completed.")
    logger.info("Results saved to %s", args.result_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)
