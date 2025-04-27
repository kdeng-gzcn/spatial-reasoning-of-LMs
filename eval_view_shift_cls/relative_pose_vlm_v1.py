import json
import jsonlines
from tqdm import tqdm
import re
from pathlib import Path
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
from config.vlm_relative_pose_prompt_v1 import task_prompt, short_answer_dict, detailed_answer_dict

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
    # parser.add_argument(
    #     "--yaml_file",
    #     type=str,
    #     required=True,
    #     help="Path to the YAML file"
    # )
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
    return parser.parse_args()


def _shuffle_dict(dict: dict) -> dict:
    keys = list(dict.keys())
    np.random.shuffle(keys)
    new_dict = {i: dict[keys[i]] for i in range(len(keys))}
    return new_dict


def generate_prompt() -> Tuple[str, dict]:
    option_map = _shuffle_dict(short_answer_dict)  # short dict
    task_prompt_fill = task_prompt.format(
        opt1=detailed_answer_dict[option_map[0]], 
        opt2=detailed_answer_dict[option_map[1]],
        opt3=detailed_answer_dict[option_map[2]],
        opt4=detailed_answer_dict[option_map[3]],
    )
    return task_prompt_fill, option_map


def parse_answer(text: str, option_map: dict) -> dict:
    rsn_match = re.search(r"<rsn>\s*(.*?)(?:\s*</rsn>|\s*<ans>|$)", text, re.DOTALL)
    rsn = rsn_match.group(1) if rsn_match else "None"

    ans_match = re.search(r"<ans>.*?(\d+).*?(?:</ans>|$)", text, re.IGNORECASE)
    ans = int(ans_match.group(1)) if ans_match else None
    if ans is None or ans not in [0, 1, 2, 3]: # avoid NoneType error
        logging.warning("Answer Option Not Extracted")
        ans = next(key for key, value in option_map.items() if value == "unable to judge")
    return {
        "rsn": rsn,
        "ans": ans,
        "ans_text": option_map[ans],
    }


def save_results(cfg: dict, results: list, result_dir: str) -> None: 
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Saving results to %s", result_dir)

    with open(result_dir / "cfg.json", "w") as f:
        json.dump(cfg, f, indent=4)

    with jsonlines.open(result_dir / f"relative_pose_results.jsonl", mode='w') as writer:
        for result in results:
            writer.write(result)

    df = pd.DataFrame(results)
    df.to_csv(result_dir / f"relative_pose_results.csv", index=False) 
    logging.info("Results saved to %s", result_dir / f"relative_pose_results.csv") 

    # compute metrics and save results
    y_true = df["label"].values
    y_pred = df["pred"].values
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
    # save metrics to json
    with open(result_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    # save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = list(np.unique(y_pred))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(result_dir / "confusion_matrix.csv", index=True)
    logging.info("Metrics saved to %s", result_dir / "metrics.json")


def main(args):
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Using model ID: %s", args.model_id)
    logger.info("Using data directory: %s", args.data_dir)
    logger.info("Using result directory: %s", args.result_dir)

    cfg = {
        "model_id": args.model_id,
        "data_dir": args.data_dir,
        "result_dir": args.result_dir,
    }

    model = load_model(args.model_id)
    model._load_weight()
    dataset = load_dataset("relative-pose-7-scenes", data_root_dir=args.data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    dataloader_tqdm = tqdm(dataloader, desc="Processing", total=len(dataloader) if hasattr(dataloader, '__len__') else None)

    structure_result = []
    i = 0
    for batch in dataloader_tqdm:
        for item in batch:
            # i += 1
            # if i > 3:
            #     break
            try:
                src_img, tgt_img = item["source_image"], item["target_image"]
                images = (src_img, tgt_img)

                model._clear_history()  # clear the history of VLM for each pair of images

                prompt, option_map = generate_prompt()
                answer = model.pipeline(images, prompt)  # __call__
                pred = parse_answer(answer, option_map)  # parse the answer
                
                structure_result.append({
                    "scene": item["metadata"]["scene"],
                    "seq": item["metadata"]["seq"],
                    "pair": item["metadata"]["pair"],
                    "label": item["metadata"]["phi_text"],
                    "pred": pred["ans_text"],
                    "label_val": np.abs(item["metadata"]["phi"]),
                })

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

    try:
        model.print_total_tokens_usage()
    except:
        pass
    save_results(cfg, structure_result, args.result_dir)
    logger.info("Processing completed.")
    logger.info("Results saved to %s", args.result_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)
