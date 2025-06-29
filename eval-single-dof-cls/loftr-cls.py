import cv2
import json
import jsonlines
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    parser = argparse.ArgumentParser(description="LoFTR Relative Pose Estimation")
    parser.add_argument(
        "--yaml_file",
        type=str,
        required=True,
        help="Path to the LoFTR YAML file"
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
        help="Dataset name, e.g., 'seven-scenes', 'scannet'"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Task name for the experiment, ['theta', 'phi', 'psi', 'tx', 'ty', 'tz']"
    )
    return parser.parse_args()


def _merge_cfg(args):
    """
    Merge the command line arguments into the configuration.
    """
    cfg.set_new_allowed(True)  # allow new keys to be set
    cfg.merge_from_file(args.yaml_file)
    cfg.EXPERIMENT.DATA_DIR = args.data_dir
    benchmark_name = _get_benchmark_name(args.data_dir)
    cfg.EXPERIMENT.TASK_NAME = _parse_benchmark_name(benchmark_name)
    cfg.EXPERIMENT.RESULT_DIR = args.result_dir 
    cfg.EXPERIMENT.DATASET = args.dataset
    cfg.EXPERIMENT.TASK_SPLIT = args.split
    cfg.MODEL.CV_METHOD.ID = "LoFTR"
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


def get_intrinsic_matrix(data_dir: str, **kwargs) -> np.ndarray:
    """
    Get the intrinsic matrix based on the benchmark name.
    """
    benchmark_name = _get_benchmark_name(data_dir)
    if benchmark_name == "single-dof-camera-motion-7-scenes":
        return np.loadtxt("/home/u5u/kdeng.u5u/spatial-reasoning-of-LMs/config/eval/intrinsic-7-scenes.txt", delimiter=",")
    
    elif benchmark_name == "single-dof-camera-motion-scannet":
        item = kwargs.get("item") # item["metadata"]["scene"]
        return np.loadtxt(Path("/home/u5u/kdeng.u5u/data/scannet-v2/scans_test") / item["metadata"]["scene"] / "intrinsic" / "intrinsic_color.txt")[:3, :3]
    
    else:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")


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

    dataset = load_dataset(Path(args.data_dir).parent.name, data_root_dir=args.data_dir, cfg=cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    dataloader_tqdm = tqdm(dataloader, desc="Processing", total=len(dataloader) if hasattr(dataloader, '__len__') else None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = KF.LoFTR(pretrained="indoor_new").to(device)

    structure_result = []
    for batch in dataloader_tqdm:
        item = next(iter(batch))
        try:
            src_img, tgt_img = item["source_image"], item["target_image"]

            src_img = src_img.float() / 255.0 # transform uint8 into rgb32
            tgt_img = tgt_img.float() / 255.0

            src_img = src_img.unsqueeze(0).to(device)
            tgt_img = tgt_img.unsqueeze(0).to(device)

            src_img = K.geometry.resize(src_img, (480, 640), antialias=True)
            tgt_img = K.geometry.resize(tgt_img, (480, 640), antialias=True)

            K_intrinsic = get_intrinsic_matrix(args.data_dir, item=item)

            input_dict = {
                "image0": K.color.rgb_to_grayscale(tgt_img), # image0 is target image
                "image1": K.color.rgb_to_grayscale(src_img),
            }

            with torch.inference_mode():
                with torch.amp.autocast("cuda"):
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
            
    save_metrics(structure_result, args.result_dir)

    logger.info("Processing completed.")
    logger.info("Results saved to %s", args.result_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)
