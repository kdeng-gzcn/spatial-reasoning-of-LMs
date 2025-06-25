import cv2
import json
import jsonlines
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import argparse
from typing import Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.dataset.utils import load_dataset
from src.logging.logging_config import setup_logging

from config.default import cfg


label_map = {
    "single-dof-cls": {
        "phi": "phi_text",
        "theta": "theta_text",
        "psi": "psi_text",
        "tx": "tx_text",
        "ty": "ty_text",
        "tz": "tz_text",
    },
    "obj-centered-cls": {
        "translation": "tx_text",
        "rotation": "phi_text",
    },
}


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
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, e.g., 'seven-scenes', 'scannet', 'scannetpp'"
    )
    parser.add_argument(
        "--min_angle",
        type=str,
        required=True,
        help="Minimum angle for the task, e.g., '0.0' for translation, '0.1' for phi"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Task name for the experiment, ['rotation', 'translation']"
    )
    return parser.parse_args()


def _merge_cfg(args):
    """
    Merge the command line arguments into the configuration.
    """
    cfg.set_new_allowed(True)  # allow new keys to be set
    # cfg.merge_from_other_cfg(CN(vars(args)))
    cfg.merge_from_file(args.yaml_file) # TODO: merge with yaml file
    cfg.EXPERIMENT.DATA_DIR = args.data_dir
    benchmark_name = _get_benchmark_name(args.data_dir)
    cfg.EXPERIMENT.TASK_NAME = _parse_benchmark_name(benchmark_name)
    cfg.EXPERIMENT.RESULT_DIR = args.result_dir 
    cfg.EXPERIMENT.DATASET = args.dataset
    cfg.EXPERIMENT.MIN_ANGLE = float(args.min_angle)
    cfg.EXPERIMENT.TASK_SPLIT = args.split
    cfg.MODEL.CV_METHOD.ID = "SIFT"
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
    

def extract_keypoints_and_descriptors(sift, image: np.ndarray) -> Tuple[list, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray, None)


def match_descriptors(cfg: dict, des1, des2) -> list:
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    return [m for m, n in matches if m.distance < cfg["lowes_match_ratio"] * n.distance]


def estimate_pose(cfg: dict, kp1, kp2, matches, K) -> Tuple[np.ndarray, np.ndarray]:
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


def save_metrics(results: list, result_dir: str) -> None:
    result_dir = Path(result_dir)

    ### Data Analysis
    df = pd.DataFrame(results)

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

    # # compute metrics and save results
    # y_true = df["label"].values
    # y_pred = df["pred"].values

    # accuracy = accuracy_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    # recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    # f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # metrics = {
    #     "accuracy": accuracy,
    #     "precision": precision,
    #     "recall": recall,
    #     "f1_score": f1,
    # }

    # # save metrics to json
    # metrics_dir = result_dir / "metrics"
    # metrics_dir.mkdir(parents=True, exist_ok=True)

    # with open(metrics_dir / "metrics.json", "w") as f:
    #     json.dump(metrics, f, indent=4)

    # Save confusion matrix
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


def main(args):
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

    # load dataloader
    dataset = load_dataset(Path(args.data_dir).parent.name, data_root_dir=args.data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    dataloader_tqdm = tqdm(dataloader, desc="Processing", total=len(dataloader) if hasattr(dataloader, '__len__') else None)

    # init model
    sift = cv2.SIFT_create()

    structure_result = []
    for batch in dataloader_tqdm:
        item = next(iter(batch))

        try:
            src_img, tgt_img = item["source_image"], item["target_image"]
            src_img = src_img.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
            tgt_img = tgt_img.permute(1, 2, 0).cpu().numpy()

            if Path(args.data_dir).parent.name == "obj-centered-view-shift-7-scenes":
                K = np.loadtxt("config/eval_view_shift/intrinsic-7-scene.txt", delimiter=",")

            if Path(args.data_dir).parent.name == "obj-centered-view-shift-scannet":
                K = np.loadtxt(Path("data/scannet-v2/scans_test") / item["metadata"]["scene"] / "intrinsic" / "intrinsic_color.txt")
                K = K[:3, :3]

            kp1, des1 = extract_keypoints_and_descriptors(sift, tgt_img)
            kp2, des2 = extract_keypoints_and_descriptors(sift, src_img)

            matches = match_descriptors(cfg, des1, des2)
            Rmat, t = estimate_pose(cfg, kp1, kp2, matches, K)

            pred = pose2prediction(Rmat, t)

            dof_text = label_map[cfg.EXPERIMENT.TASK_NAME][cfg.EXPERIMENT.TASK_SPLIT]
            dof = dof_text.split("_")[0]  # e.g., "tx", "ty", "tz", "phi", "theta", "psi"

            row = {
                **item["metadata"],
                "pred_val": np.abs(pred[dof]),
                "label_val": item["metadata"][dof],
                "pred": pred[dof_text],
                "label": item["metadata"][dof_text],
                "is_correct": pred[dof_text] == item["metadata"][dof_text],
                "is_valid": True,
            }
            structure_result.append(row)

            with jsonlines.open(result_dir / "inference.jsonl", mode='a') as writer:
                writer.write(row)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")

            dof_text = label_map[cfg.EXPERIMENT.TASK_NAME][cfg.EXPERIMENT.TASK_SPLIT]
            dof = dof_text.split("_")[0]  # e.g., "tx", "ty", "tz", "phi", "theta", "psi"

            row = {
                **item["metadata"],
                "pred_val": None,
                "label_val": item["metadata"][dof],
                "pred": "error",
                "label": item["metadata"][dof_text],
                "is_correct": False,
                "is_valid": False,
            }
            structure_result.append(row)

            with jsonlines.open(result_dir / "inference.jsonl", mode='a') as writer:
                writer.write(row)

            continue
    
    # TODO
    save_metrics(structure_result, args.result_dir)

    logger.info("Processing completed.")
    logger.info("Results saved to %s", args.result_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)
