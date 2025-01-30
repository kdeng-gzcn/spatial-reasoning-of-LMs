import numpy as np
import pandas as pd
import re
import random
import shutil
import json
import jsonlines
import tqdm

import os

orgin_data_dir = "./data/RGBD_7_Scenes/"

csv = pd.DataFrame()

frame_threshold = [25, 150]

num_framepairs_each_dimension = 200

total_pairs = num_framepairs_each_dimension * 6

rebuild_data_dir = f"./data/Rebuild_7_Scenes_{total_pairs}/"

num_framepairs_count = {
    "tx": 0,
    "ty": 0,
    "tz": 0,
    "theta": 0,
    "phi": 0,
    "psi": 0,
}

threshold_relative_pose = {
            "tx": [0.2, 0.5],
            "ty": [0.1, 0.3],
            "tz": [0.2, 0.5],
            "theta": [5, 15],
            "phi": [5, 15],
            "psi": [5, 15],
        }

global_csv_data = []
global_json_data = []

def all_dimensions_complete():
    return all(count >= num_framepairs_each_dimension for count in num_framepairs_count.values())

# Initialize a total progress bar
progress_bar = tqdm.tqdm(total=total_pairs, desc="Total Progress", unit="pair")
bar_tx = tqdm.tqdm(total=num_framepairs_each_dimension, desc="tx", unit="pair")
bar_ty = tqdm.tqdm(total=num_framepairs_each_dimension, desc="ty", unit="pair")
bar_tz = tqdm.tqdm(total=num_framepairs_each_dimension, desc="tz", unit="pair")
bar_theta = tqdm.tqdm(total=num_framepairs_each_dimension, desc="theta", unit="pair")
bar_phi = tqdm.tqdm(total=num_framepairs_each_dimension, desc="phi", unit="pair")
bar_psi = tqdm.tqdm(total=num_framepairs_each_dimension, desc="psi", unit="pair")

bar_dict = {
    "tx": bar_tx,
    "ty": bar_ty,
    "tz": bar_tz,
    "theta": bar_theta,
    "phi": bar_phi,
    "psi": bar_psi,
}

# Iterate through each scene folder
while not all_dimensions_complete():

    for scene in os.listdir(orgin_data_dir):
        scene_path = os.path.join(orgin_data_dir, scene)
        if not os.path.isdir(scene_path):
            continue

        # Iterate through each sequence folder in the scene
        for seq in os.listdir(scene_path):
            seq_path = os.path.join(scene_path, seq)
            if not os.path.isdir(seq_path):
                continue

            """
            
            finding a pair of images
            
            """

            color_images = [f for f in os.listdir(seq_path) if f.endswith('.color.png')]

            # Extract frame numbers from the filenames
            frame_numbers = []
            for img in color_images:
                match = re.match(r'frame-(\d{6})\.color\.png', img)
                if match:
                    frame_numbers.append(int(match.group(1)))
            
            # Sort frame numbers, 000000-000999
            frame_numbers.sort()
            
            # Randomly select a pair of frames with a difference between 25 and 100
            found_pair = False
            max_attempts = 1000 # for randomly pick a pair
            
            for _ in range(max_attempts):

                # Randomly select two frames
                frame1_num, frame2_num = sorted(random.sample(frame_numbers, 2))
                diff = frame2_num - frame1_num
                
                # Check if the difference is within the desired range
                if frame_threshold[0] <= diff <= frame_threshold[1]:

                    # Print or process the selected pair
                    found_pair = True

                    break
            
            if not found_pair:

                print(f"No valid pairs found in {seq_path} after {max_attempts} attempts")

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

                # print(f"No valid significant pose pair found in {seq_path}")

                continue

            # Check if the significant dimension has already reached the required number of pairs
            if num_framepairs_count[df] >= num_framepairs_each_dimension:

                continue

            # If we reach here, it means we have a valid pair for a dimension that hasn't reached 500 yet
            num_framepairs_count[df] += 1

            # Update the total progress bar
            progress_bar.update(1)

            # Update the specific df bar
            bar_dict[df].update(1)

            """
            
            if we keep, save the metadata
            
            """

            pair_path = os.path.join(rebuild_data_dir, f"{df}_Significant", f"{scene}", f"{seq}", f"{frame1_num:06d}-{frame2_num:06d}")

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

            try:

                shutil.copy(frame1_file_path, source_path)
                shutil.copy(frame2_file_path, target_path)

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
                "psi_text": "head towards right" if psi > 0 else "head towards left",
                "significance": df,
            }

            info["significance_text"] = info[f"{df}_text"]

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

        # Check if all dimensions have reached the required number of pairs
        if all_dimensions_complete():
            break

    # Check if all dimensions have reached the required number of pairs
    if all_dimensions_complete():
        break

# Close the progress bar
progress_bar.close()

# csv
global_csv_path = os.path.join(rebuild_data_dir, "global_metadata.csv")
global_csv_df = pd.DataFrame(global_csv_data)
global_csv_df.to_csv(global_csv_path, index=False)

# json
global_json_path = os.path.join(rebuild_data_dir, "global_metadata.json")
with open(global_json_path, "w") as f:
    json.dump(global_json_data, f, indent=4)

# jsonlines
global_jsonl_path = os.path.join(rebuild_data_dir, "global_metadata.jsonl")
with jsonlines.open(global_jsonl_path, mode="w") as writer:
    for item in global_json_data:
        writer.write(item)

print("length of dataset:")
print(len(global_csv_df))

print("Done!")
