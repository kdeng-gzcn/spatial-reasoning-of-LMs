import cv2
import json
import jsonlines
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import argparse
from tqdm import tqdm
from typing import Tuple
import torch

import kornia as K
import kornia.feature as KF
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.dataset.utils import load_dataset
from src.logging.logging_config import setup_logging
from config.camera_intrinsic import K as K_intrinsic

def parse_args():
    parser = argparse.ArgumentParser(description="SIFT Relative Pose Estimation")
    parser.add_argument(
        "--yaml_file",
        type=str,
        required=True,
        help="Path to the YAML file"
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


def pose2prediction(Rmat: np.ndarray, t: np.ndarray) -> dict:
    theta = np.degrees(np.arctan2(Rmat[2, 1], Rmat[2, 2]))
    phi = np.degrees(np.arcsin(-Rmat[2, 0]))
    psi = np.degrees(np.arctan2(Rmat[1, 0], Rmat[0, 0]))  

    pred = {
        "tx": t[0], "ty": t[1], "tz": t[2],
        "theta": theta, "phi": phi, "psi": psi,
        "tx_text": "right" if t[0] > 0 else "left",
        "ty_text": "down" if t[1] > 0 else "up",
        "tz_text": "forward" if t[2] > 0 else "backward",
        "theta_text": "upward" if theta > 0 else "downward",
        "phi_text": "rightward" if phi > 0 else "leftward",
        "psi_text": "clockwise" if psi > 0 else "counterclockwise",
    }

    return pred


def save_results(results: list, result_dir: str) -> None:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Saving results to %s", result_dir)

    with jsonlines.open(result_dir / "relative_pose_results.jsonl", mode='w') as writer:
        for result in results:
            writer.write(result)

    df = pd.DataFrame(results)
    df.to_csv(result_dir / "relative_pose_results.csv", index=False) 
    logging.info("Results saved to %s", result_dir / "relative_pose_results.csv") 

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

    logger.info("Loaded config from %s", args.yaml_file)
    logger.info("Using data directory: %s", args.data_dir)
    logger.info("Using result directory: %s", args.result_dir)

    cfg = yaml.safe_load(Path(args.yaml_file).read_text())
    dataset = load_dataset("relative-pose-7-scenes", data_root_dir=args.data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    dataloader_tqdm = tqdm(dataloader, desc="Processing", total=len(dataloader) if hasattr(dataloader, '__len__') else None)

    matcher = KF.LoFTR(pretrained="indoor_new")

    structure_result = []
    for batch in dataloader_tqdm:
        for item in batch:
            try:
                src_img, tgt_img = item["source_image"], item["target_image"]

                src_img = src_img.float() / 255.0 # transform uint8 into rgb32
                tgt_img = tgt_img.float() / 255.0

                src_img = src_img.unsqueeze(0)
                tgt_img = tgt_img.unsqueeze(0)

                src_img = K.geometry.resize(src_img, (480, 640), antialias=True)
                tgt_img = K.geometry.resize(tgt_img, (480, 640), antialias=True)

                input_dict = {
                    "image0": K.color.rgb_to_grayscale(tgt_img), # image0 is target image
                    "image1": K.color.rgb_to_grayscale(src_img),
                }

                with torch.inference_mode():
                    correspondences = matcher(input_dict)

                mkpts0 = correspondences["keypoints0"].cpu().numpy()
                mkpts1 = correspondences["keypoints1"].cpu().numpy()

                E, _ = cv2.findEssentialMat(
                    mkpts0, mkpts1, 
                    cameraMatrix=K_intrinsic, 
                    method=cv2.RANSAC, 
                    threshold=1.0, 
                    prob=0.999
                )
                
                _, Rmat, t, _ = cv2.recoverPose(E, mkpts0, mkpts1, K_intrinsic)

                pred = pose2prediction(Rmat, t.squeeze())
                
                structure_result.append({
                    "scene": item["metadata"]["scene"],
                    "seq": item["metadata"]["seq"],
                    "pair": item["metadata"]["pair"],
                    "label": item["metadata"]["phi_text"],
                    "pred": pred["phi_text"],
                    "label_val": np.abs(item["metadata"]["phi"]),
                    "pred_val": np.abs(pred["phi"]),
                })

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                structure_result.append({
                    "scene": item["metadata"]["scene"],
                    "seq": item["metadata"]["seq"],
                    "pair": item["metadata"]["pair"],
                    "label": item["metadata"]["phi_text"],
                    "pred": "error",
                    "label_val": np.abs(item["metadata"]["phi"]),
                    "pred_val": None,
                })
                continue
            
    save_results(structure_result, args.result_dir)
    logger.info("Processing completed.")
    logger.info("Results saved to %s", args.result_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)
