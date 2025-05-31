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


# def _shuffle_dict(dict: dict) -> dict:
#     keys = list(dict.keys())
#     np.random.shuffle(keys)
#     new_dict = {i: dict[keys[i]] for i in range(len(keys))}
#     return new_dict


# def generate_prompt(image_caption: str) -> Tuple[str, dict]:
#     option_map = _shuffle_dict(short_answer_dict)  # short dict
#     task_prompt_fill = spatial_reasoning_prompt_image_caption.format(
#         vlm_answers=image_caption,
#         opt1=detailed_answer_dict[option_map[0]], 
#         opt2=detailed_answer_dict[option_map[1]],
#         opt3=detailed_answer_dict[option_map[2]],
#         opt4=detailed_answer_dict[option_map[3]],
#     )
#     return task_prompt_fill, option_map


# def parse_answer(text: str, option_map: dict) -> dict:
#     ques_match = re.search(r"<ques>\s*(.*?)(?:\s*</ques>|$)", text, re.DOTALL)
#     ques = ques_match.group(1) if ques_match else None
#     if ques is not None:
#         return {
#             "ques": ques,
#         }

#     rsn_match = re.search(r"<rsn>\s*(.*?)(?:\s*</rsn>|\s*<ans>|$)", text, re.DOTALL)
#     rsn = rsn_match.group(1) if rsn_match else "None"

#     ans_match = re.search(r"<ans>.*?(\d+).*?(?:</ans>|$)", text, re.IGNORECASE)
#     ans = int(ans_match.group(1)) if ans_match else None
#     if ans is None or ans not in [0, 1, 2, 3]: # avoid NoneType error
#         logging.warning("Answer Option Not Extracted")
#         ans = next(key for key, value in option_map.items() if value == "unable to judge")

#     return {
#         "rsn": rsn,
#         "ans": ans,
#         "ans_text": option_map[ans],
#     }


# def _full_history_append(full_history: list, metadata: dict, idx: int, 
#                              speaker: str, receiver: str, content: str) -> None:
#         full_history.append(
#             {
#                 "scene": metadata["scene"],
#                 "seq": metadata["seq"],
#                 "pair": metadata["pair"],
#                 "label_dof": metadata["significance"],
#                 "label": metadata["significance_text"],
#                 "label_val": metadata["significance_value"],
#                 "idx": idx,
#                 "speaker": speaker,
#                 "receiver": receiver,
#                 "content": content,
#             }
#         )


def save_results(cfg: dict, results: list, result_dir: str, full_history: list) -> None: 
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Saving results to %s", result_dir)

    with open(result_dir / "cfg.json", "w") as f:
        json.dump(cfg, f, indent=4)

    with jsonlines.open(result_dir / f"view_shift_results.jsonl", mode='w') as writer:
        for result in results:
            writer.write(result)

    df = pd.DataFrame(results)
    df.to_csv(result_dir / f"view_shift_results.csv", index=False) 
    logging.info("Results saved to %s", result_dir / f"view_shift_results.csv") 

    df = pd.DataFrame(full_history)
    df.to_csv(result_dir / f"full_history.csv", index=False)
    logging.info("Full history saved to %s", result_dir / f"full_history.csv")

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

    # load yaml file
    with open(args.yaml_file, 'r') as f:
        yaml_cfg = yaml.safe_load(f)

    # Load configuration
    cfg = {arg: getattr(args, arg) for arg in vars(args)}
    cfg = {**cfg, **yaml_cfg["exp_config"]}  # merge with yaml config

    # logger info for k and v in cfg
    for k, v in cfg.items():
        logger.info(f"{k}: {v}")

    llm = load_model(args.llm_id)
    llm._load_weight()
    vlm = load_model(args.vlm_id)
    vlm._load_weight()

    dataset = load_dataset(Path(args.data_dir).parent.name, data_root_dir=args.data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    dataloader_tqdm = tqdm(dataloader, desc="Processing", total=len(dataloader) if hasattr(dataloader, '__len__') else None)

    structure_result = []
    full_history = []
    i = 0
    for batch in dataloader_tqdm:
        for item in batch:
            i += 1
            if i > 3:
                break
            try:
                ### prepare data
                src_img, tgt_img = item["source_image"], item["target_image"]
                metadata = item["metadata"]
                images = (src_img, tgt_img)

                llm._clear_history()  # clear the history of VLM for each pair of images
                vlm._clear_history()  # clear the history of LLM for each pair of images

                ### vlm inference: image caption
                vlm_caption_for_images = vlm.pipeline(images, task_prompt)  # __call__

                for idx in range(cfg["max_len_of_conv"]): # for loop for max length or stop condition
                    if idx:
                        # if pred["pred text"] != "ask more questions":
                        #     break
                        if "ques" not in pred:
                            break

                        ### parse llm question for new round
                        llm_questions_to_vlm = pred["ques"]
                        # if not self.is_vlm_keep_hisroty:
                        #     llm_questions_to_vlm = self.spatial_question_prompter(llm_questions_to_vlm)
                        _full_history_append(full_history, metadata, idx+1, "LLM", "VLM", llm_questions_to_vlm)

                        ### vlm inference: spatial understanding
                        vlm_answers_to_llm = vlm.pipeline(images, llm_questions_to_vlm)
                        _full_history_append(full_history, metadata, idx+1, "VLM", "LLM", vlm_answers_to_llm)

                    ### llm inference: spatial reasoning
                    # vlm_answers_to_llm, opt_map = self.spatial_reasoning_prompter(
                    #     self.VLM.pipeline(images, llm_questions_to_vlm)
                    # )
                    # self._full_history_append(full_history, metadata, idx+1, "VLM", "LLM", vlm_answers_to_llm)
                    # if not self.is_vlm_keep_hisroty:
                    #     self.VLM._clear_history() # for lower cost on memory, clear up history in VLM

                    ### llm inference: spatial reasoning
                    spatial_reasoning_prompt, option_map = generate_prompt(vlm_caption_for_images)
                    llm_reasoning = llm.pipeline(spatial_reasoning_prompt)
                    pred = parse_answer(llm_reasoning, option_map)  # parse the answer
                    _full_history_append(full_history, metadata, idx+1, "LLM", "User or VLM", llm_reasoning)

                    ### collect result
                    if Path(args.data_dir).parent.name == "obj-centered-view-shift-7-scenes":
                        metadata_prefix = {
                            "scene": item["metadata"]["scene"],
                            "seq": item["metadata"]["seq"],
                        }
                    elif Path(args.data_dir).parent.name == "obj-centered-view-shift-scannet":
                        metadata_prefix = {
                            "scene": item["metadata"]["scene"],
                        }
                    else:
                        metadata_prefix = {}
                        logger.error(f"Invalid dataset: {Path(args.data_dir).parent.name}.")
                    
                    structure_result.append({
                        **metadata_prefix,
                        "pair": item["metadata"]["pair"],
                        # "label": item["metadata"]["phi_text"],
                        "label": item["metadata"]["tx_text"], # v2
                        "pred": pred["ans_text"],
                        # "label_val": np.abs(item["metadata"]["phi"]),
                        "label_val": item["metadata"]["tx"], # v2
                    })

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

    try:
        model.print_total_tokens_usage()
    except:
        logger.warning("Model does not support token usage tracking.")

    save_results(cfg, structure_result, args.result_dir, full_history)
    logger.info("Processing completed.")
    logger.info("Results saved to %s", args.result_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)
