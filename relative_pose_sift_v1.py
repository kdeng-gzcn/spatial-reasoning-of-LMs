import cv2
import json
import jsonlines
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import argparse
from typing import Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.dataset.utils import load_dataset
from src.logging.logging_config import setup_logging
from config.camera_intrinsic import K

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

def extract_keypoints_and_descriptors(sift, image: np.ndarray) -> Tuple[list, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray, None)


def match_descriptors(cfg: dict, des1, des2) -> list:
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    return [m for m, n in matches if m.distance < cfg["lowes_match_ratio"] * n.distance]


def estimate_pose(cfg: dict, kp1, kp2, matches) -> Tuple[np.ndarray, np.ndarray]:
    if len(matches) < cfg["min_match_count"]:
        logging.warning("Not enough good matches.")
        return None, None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    E_mat, _ = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, Rmat, t, _ = cv2.recoverPose(E_mat, src_pts, dst_pts, K)
    return Rmat, t.squeeze()


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

    sift = cv2.SIFT_create()
    structure_result = []
    for batch in dataloader_tqdm:
        for item in batch:
            try:
                src_img, tgt_img = item["source_image"], item["target_image"]
                src_img = src_img.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
                tgt_img = tgt_img.permute(1, 2, 0).cpu().numpy()

                kp1, des1 = extract_keypoints_and_descriptors(sift, tgt_img)
                kp2, des2 = extract_keypoints_and_descriptors(sift, src_img)

                if kp1 is None or kp2 is None:
                    logger.warning("No keypoints found.")
                    continue
                if des1 is None or des2 is None:
                    logger.warning("No descriptors found.")
                    continue

                matches = match_descriptors(cfg, des1, des2)
                Rmat, t = estimate_pose(cfg, kp1, kp2, matches)

                pred = pose2prediction(Rmat, t)
                
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
