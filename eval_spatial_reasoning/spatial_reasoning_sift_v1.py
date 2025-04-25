import cv2
import json
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from torch.utils.data import DataLoader
from src.dataset.utils import load_dataset
from src.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Constants
MIN_MATCH_COUNT = 5
LOWE_RATIO = 0.7

# Initialize detector
sift = cv2.SIFT_create()

K  = np.array([[585, 0, 320],
               [0, 585, 240],
               [0,  0,  1]])

def extract_keypoints_and_descriptors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray, None)


def match_descriptors(des1, des2):
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    return [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]


def estimate_pose(kp1, kp2, matches):
    if len(matches) < MIN_MATCH_COUNT:
        logging.warning("Not enough good matches.")
        return None, None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    E_mat, _ = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R_mat, t, _ = cv2.recoverPose(E_mat, src_pts, dst_pts, K)
    return R_mat, t

# Load dataset
dataset = load_dataset("7 Scenes", data_root_dir="benchmark/RGBD_7_Scenes_Rebuilt", split="all")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

structure_result = []

# Process
for batch in dataloader:
    for item in batch:
        try:
            src_img, tgt_img = item["source_image"], item["target_image"]
            src_img = src_img.permute(1, 2, 0).cpu().numpy() # Convert to HWC format
            tgt_img = tgt_img.permute(1, 2, 0).cpu().numpy()
            kp1, des1 = extract_keypoints_and_descriptors(tgt_img) # tgt the image1
            kp2, des2 = extract_keypoints_and_descriptors(src_img) # src the image2
            if kp1 is None or kp2 is None:
                logging.warning("No keypoints found.")
                continue
            if des1 is None or des2 is None:
                logging.warning("No descriptors found.")
                continue
            matches = match_descriptors(des1, des2)
            Rmat, t = estimate_pose(kp1, kp2, matches) # relative pose from tgt to src, which means the translation is the center of tgt under src coordinate system
            t = t.squeeze()
            
            theta = np.degrees(np.arctan2(Rmat[2, 1], Rmat[2, 2]))
            phi = np.degrees(np.arcsin(-Rmat[2, 0]))
            psi = np.degrees(np.arctan2(Rmat[1, 0], Rmat[0, 0]))   

            pred_map = {
                "tx_text": "right" if t[0] > 0 else "left",
                "ty_text": "down" if t[1] > 0 else "up",
                "tz_text": "forward" if t[2] > 0 else "backward",
                "theta_text": "upward" if theta > 0 else "downward",
                "phi_text": "rightward" if phi > 0 else "leftward",
                "psi_text": "clockwise" if psi > 0 else "counterclockwise",
            }

            mapping = {
                "tx": t[0],
                "ty": t[1],
                "tz": t[2],
                "theta": theta,
                "phi": phi,
                "psi": psi,
            }

            pred = pred_map[f"{item['metadata']['significance']}_text"]
            pred_val = mapping[f"{item['metadata']['significance']}"]

            label = item["metadata"]["significance_text"]

            # Convert label to match pred_map
            if label == "head towards right":
                label = "clockwise"
            elif label == "head towards left":
                label = "counterclockwise"

            label_val = item["metadata"]["significance_value"]

            logging.info(f"Sig. DoF: {item['metadata']['significance']}")
            logging.info(f"Pred: {pred_val}")
            logging.info(f"Label: {label_val}")

            if pred == label:
                logging.info("Correct prediction")
            else:
                logging.info("Incorrect prediction")

            structure_result.append(
                {
                    "scene": item["metadata"]["scene"],
                    "seq": item["metadata"]["seq"],
                    "pair": item["metadata"]["pair"],
                    "label_dof": item["metadata"]["significance"],
                    "label": label,
                    "label_val": label_val,
                    "pred": pred,
                    "pred_val": pred_val,
                }
            )

        except Exception as e:
            logging.error(f"Error: {e}")

# Save results
df = pd.DataFrame(structure_result)
dir = Path("result/baseline")
dir.mkdir(parents=True, exist_ok=True)
df.to_csv(dir / "sift.csv", index=False)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_pred = df["pred"].tolist()
y_true = df["label"].tolist()
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

stat = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
}
with open(dir / "sift_stat.json", "w") as f:
    json.dump(stat, f, indent=4)

cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_pred))
cm_df.to_csv(dir / "sift_confusion_matrix.csv")

logging.info(f"Accuracy: {accuracy}")
logging.info(f"Precision: {precision}")
logging.info(f"Recall: {recall}")
logging.info(f"F1 Score: {f1}")
