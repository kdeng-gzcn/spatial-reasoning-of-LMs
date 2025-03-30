import argparse
import os
from pathlib import Path
import re
import shutil
import json
import time

import numpy as np
import pandas as pd
import jsonlines
import tqdm
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Generate benchmark dataset for RGBD 7 Scenes.")
    parser.add_argument("--config_path", type=str, default="config_benchmark.yaml", help="Path to the configuration file.")
    parser.add_argument("--dataset_dir", type=str, default="data/RGBD_7_Scenes", help="Path to the dataset directory.")
    parser.add_argument("--output_dir", type=str, default=f"benchmark/RGBD_7_Scenes_Rebuilt_{int(time.time())}", help="Path to the output directory.")
    return parser.parse_args()

args = parse_args()

CONFIG = yaml.safe_load(open(args.config_path, "r"))
DATASET_DIR = Path(args.dataset_dir)
REBUILT_DATA_DIR = Path(args.output_dir)
REBUILT_DATA_DIR.mkdir(parents=True, exist_ok=True)

max_interval = CONFIG["max_interval"]
threshold_relative_pose = CONFIG["threshold_relative_pose"]

num_pos_count = {
    "tx": 0,
    "ty": 0,
    "tz": 0,
    "theta": 0,
    "phi": 0,
    "psi": 0,
}

num_neg_count = {
    "tx": 0,
    "ty": 0,
    "tz": 0,
    "theta": 0,
    "phi": 0,
    "psi": 0,
}

global_csv_data = []
global_json_data = []

def _create_bar(desc: str, unit: str = "pair") -> tqdm.tqdm:
    return tqdm.tqdm(total=None, desc=desc, unit=unit)

bar_dict = {
    "tx_pos": _create_bar("tx (pos)"),
    "tx_neg": _create_bar("tx (neg)"),
    "ty_pos": _create_bar("ty (pos)"),
    "ty_neg": _create_bar("ty (neg)"),  
    "tz_pos": _create_bar("tz (pos)"),
    "tz_neg": _create_bar("tz (neg)"),
    "theta_pos": _create_bar("theta (pos)"),
    "theta_neg": _create_bar("theta (neg)"),
    "phi_pos": _create_bar("phi (pos)"),
    "phi_neg": _create_bar("phi (neg)"),
    "psi_pos": _create_bar("psi (pos)"),
    "psi_neg": _create_bar("psi (neg)"),
}

# Initialize a total progress bar
progress_bar = _create_bar("Total Progress")

scene_bar = tqdm.tqdm(
    total=len([f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]), 
    desc="Scene",
)

# Iterate through each scene folder
for scene in os.listdir(DATASET_DIR):
    scene_path = os.path.join(DATASET_DIR, scene)
    scene_path = DATASET_DIR / scene
    if not os.path.isdir(scene_path):
        continue

    scene_bar.set_postfix(scene=scene, refresh=True)
    scene_bar.update(1)

    seq_bar = tqdm.tqdm(
        total=len([f for f in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, f))]),
        desc=f"Seq",
        leave=False,
    )

    # Iterate through each sequence folder in the scene
    for seq in os.listdir(scene_path):
        seq_path = os.path.join(scene_path, seq)
        if not os.path.isdir(seq_path):
            continue

        seq_bar.set_postfix(seq=seq, refresh=True)
        seq_bar.update(1)

        color_images = [f for f in os.listdir(seq_path) if f.endswith('.color.png')]

        frame_numbers = []
        for img in color_images:
            match = re.match(r'frame-(\d{6})\.color\.png', img)
            if match:
                frame_numbers.append(int(match.group(1)))

        frame_numbers.sort()

        # Initialize start index for frame pair generation
        i = 0
        pair_bar = tqdm.tqdm(desc=f"ij Searching", leave=False)
        while i < len(frame_numbers) - 1:
            frame1_num = frame_numbers[i]
            found_pair = False
            
            # Try to find a valid pair starting from frame1_num
            j = i + 1
            while j < len(frame_numbers):
                frame2_num = frame_numbers[j]
                pair_bar.set_postfix(i=f"{i}/{len(frame_numbers)}", j=f"{j}/{len(frame_numbers)}", refresh=True)

                if (j - i) > max_interval:
                    i += 1
                    break

                """
                
                do math work on pose transformation
                
                """

                # load pose matrix for pair
                pose_W2S = np.loadtxt(os.path.join(seq_path, f"frame-{frame1_num:06d}.pose.txt"))
                pose_W2T = np.loadtxt(os.path.join(seq_path, f"frame-{frame2_num:06d}.pose.txt"))

                # P_T2S = P_W2S^-1 @ P_W2T
                pose_S2T = np.linalg.inv(pose_W2S) @ pose_W2T

                t = pose_S2T[:3, 3:].squeeze()
                Rmat = pose_S2T[:3, :3]

                # transform rad to degree
                theta = np.degrees(np.arctan2(Rmat[2, 1], Rmat[2, 2]))
                phi = np.degrees(np.arcsin(-Rmat[2, 0]))
                psi = np.degrees(np.arctan2(Rmat[1, 0], Rmat[0, 0]))

                relative_pose = {
                    "tx": t[0],
                    "ty": t[1],
                    "tz": t[2],
                    "theta": theta,
                    "phi": phi,
                    "psi": psi,
                }

                """
                
                judge if we keep this pair
                
                """

                # Initialize a variable to track if a valid pair is found
                valid_pair = False

                # Iterate through the keys of relative_pose to check conditions
                for df in relative_pose.keys():
                    # Check if the current relative_pose is greater than the upper threshold
                    if np.abs(relative_pose[df]) > threshold_relative_pose[df][1]:
                        # Test other keys to ensure they are below their lower thresholds
                        if all(
                            np.abs(relative_pose[other_key]) < threshold_relative_pose[other_key][0]
                            for other_key in relative_pose.keys() if other_key != df
                        ):
                            valid_pair = True
                            break

                if not valid_pair:
                    j += 1
                    if j >= len(frame_numbers):
                        i += 1
                    continue

                # If we reach here, it means we have a valid pair for a dimension that hasn't reached 500 yet
                if relative_pose[df] > 0:
                    num_pos_count[df] += 1
                    progress_bar.update(1)
                    bar_dict[f"{df}_pos"].update(1)
                    
                else:
                    num_neg_count[df] += 1
                    progress_bar.update(1)
                    bar_dict[f"{df}_neg"].update(1)

                """
                
                if we keep, save the metadata
                
                """

                pair_path = os.path.join(REBUILT_DATA_DIR, f"{df}_significant", f"{scene}", f"{seq}", f"{frame1_num:06d}-{frame2_num:06d}")

                if os.path.exists(pair_path):
                    continue

                os.makedirs(pair_path)

                source_path = os.path.join(pair_path, "source")
                os.makedirs(source_path)

                target_path = os.path.join(pair_path, "target")
                os.makedirs(target_path)

                # Construct the filenames for the selected pair
                frame1_file_path = os.path.join(seq_path, f"frame-{frame1_num:06d}.color.png")
                frame2_file_path = os.path.join(seq_path, f"frame-{frame2_num:06d}.color.png")

                frame1_depth_path = os.path.join(seq_path, f"frame-{frame1_num:06d}.depth.png")
                frame2_depth_path = os.path.join(seq_path, f"frame-{frame2_num:06d}.depth.png")

                frame1_pose_path = os.path.join(seq_path, f"frame-{frame1_num:06d}.pose.txt")
                frame2_pose_path = os.path.join(seq_path, f"frame-{frame2_num:06d}.pose.txt")

                try:
                    shutil.copy(frame1_file_path, source_path)
                    shutil.copy(frame2_file_path, target_path)

                    shutil.copy(frame1_depth_path, source_path)
                    shutil.copy(frame2_depth_path, target_path)

                    shutil.copy(frame1_pose_path, source_path)
                    shutil.copy(frame2_pose_path, target_path)

                    # print(f"File copied successfully from {frame1_file_path} to {source_path}")
                    # print(f"File copied successfully from {frame2_file_path} to {target_path}")

                except Exception as e:
                    print(f"Copy Fails: {e}")

                info = {
                    "scene": scene,
                    "seq": seq,
                    "pair": f"{frame1_num:06d}-{frame2_num:06d}",
                    "tx": np.round(t[0], 6),
                    "ty": np.round(t[1], 6),
                    "tz": np.round(t[2], 6),
                    "theta": np.round(theta, 6),
                    "phi": np.round(phi, 6),
                    "psi": np.round(psi, 6),
                    "tx_text": "right" if t[0] > 0 else "left",
                    "ty_text": "down" if t[1] > 0 else "up",
                    "tz_text": "forward" if t[2] > 0 else "backward",
                    "theta_text": "upward" if theta > 0 else "downward",
                    "phi_text": "rightward" if phi > 0 else "leftward",
                    "psi_text": "clockwise" if psi > 0 else "counterclockwise",
                    "significance": df,
                }

                info["significance_text"] = info[f"{df}_text"]
                info["significance_value"] = np.abs(info[f"{df}"])

                """
                
                metadata
                
                """

                pair_csv_path = os.path.join(pair_path, "metadata.csv")
                pair_df = pd.DataFrame([info])
                pair_df.to_csv(pair_csv_path, index=False)

                pair_json_path = os.path.join(pair_path, "metadata.json")

                with open(pair_json_path, "w") as f:
                    json.dump(info, f, indent=4)

                global_csv_data.append(info)
                global_json_data.append(info)

                # Move to the next pair, starting from frame2_num
                i = j
                found_pair = True
                break

# Close the progress bar
progress_bar.close()
scene_bar.close()
seq_bar.close()
pair_bar.close()
for bar in bar_dict.values():
    bar.close()

# csv
global_csv_path = os.path.join(REBUILT_DATA_DIR, "global_metadata.csv")
global_csv_df = pd.DataFrame(global_csv_data)
global_csv_df.to_csv(global_csv_path, index=False)

# json
global_json_path = os.path.join(REBUILT_DATA_DIR, "global_metadata.json")
with open(global_json_path, "w") as f:
    json.dump(global_json_data, f, indent=4)

# jsonlines
global_jsonl_path = os.path.join(REBUILT_DATA_DIR, "global_metadata.jsonl")
with jsonlines.open(global_jsonl_path, mode="w") as writer:
    for item in global_json_data:
        writer.write(item)

print("length of dataset:")
print(len(global_csv_df))

print("Done!")
