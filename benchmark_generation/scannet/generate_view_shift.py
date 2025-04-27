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

import cv2
from PIL import Image

import logging

from src.logging.logging_config import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Generate relative pose for 7 scenes")
    parser.add_argument(
        "--yaml_file",
        type=str,
        required=True,
        help="Path to the YAML file containing camera intrinsics and extrinsics"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Dir for the dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output JSON file"
    )
    parser.add_argument(
        "--min_angle",
        type=float,
        required=True, # TODO: make it optional
        help="Minimum angle between two frames to be considered a valid pair"
    )
    return parser.parse_args()


def _get_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """
    (x, y, z)_1
    (x, y, z)_2
    (x, y, z)_3

    output:
        angle ABC
    """
    BA = A - B
    BC = C - B

    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)

    if norm_BA < 1e-8 or norm_BC < 1e-8:
        # print("Warning: One of the vectors has near-zero magnitude.")
        return 0.0

    BA = BA / norm_BA
    BC = BC / norm_BC

    angle = np.arccos(np.dot(BA, BC)) * 180 / np.pi
    return angle


def _unproject(pose, K_inv, pixelx, pixely, depth) -> np.ndarray:
    """
    (x, y) -> (x, y, z)_wrd
    """
    point3d_camera_coords = depth * K_inv @ np.array([pixelx, pixely, 1])
    point3D_world = pose[:3, :3] @ point3d_camera_coords + pose[:3, 3] # vec_wrd = P^cam_wrd * vec_cam
    return point3D_world


def _reproject(pose, K, point3D) -> np.ndarray:
    """
    (x, y, z)_wrd -> (x, y)
    """
    P = K @ np.linalg.inv(pose)[:3]
    projection = P[:3, :3] @ point3D + P[:3, 3]
    projection[:2] /= projection[2:]
    return projection[:2]  


def _get_distance_and_angle(view_dict: dict, idx_src_frame: str, idx_tgt_frame: str,  K, K_inv) -> tuple:
    src_center_x, src_center_y = view_dict[idx_src_frame]["image"].shape[1] // 2, view_dict[idx_src_frame]["image"].shape[0] // 2
    tgt_center_x, tgt_center_y = view_dict[idx_tgt_frame]["image"].shape[1] // 2, view_dict[idx_tgt_frame]["image"].shape[0] // 2

    src_depth = view_dict[idx_src_frame]["depth_of_center"]
    tgt_depth = view_dict[idx_tgt_frame]["depth_of_center"]

    src_pose = view_dict[idx_src_frame]["pose"]
    tgt_pose = view_dict[idx_tgt_frame]["pose"]
    
    # Extract camera centers for later use
    src_center_world = src_pose[:3, 3]
    tgt_center_world = tgt_pose[:3, 3]
    
    # Unproject central pixels to 3D world coordinates
    src_point_world = _unproject(src_pose, K_inv, src_center_x, src_center_y, src_depth)
    angle_src = _get_angle(src_center_world, src_point_world, tgt_center_world)
    if angle_src < 3:
        return 0.0, 0.0
    tgt_point_world = _unproject(tgt_pose, K_inv, tgt_center_x, tgt_center_y, tgt_depth)
    angle_tgt = _get_angle(src_center_world, tgt_point_world, tgt_center_world)
    if angle_tgt < 3:
        return 0.0, 0.0

    # Reproject 3D points to 2D in the other view
    src_to_tgt_proj = _reproject(tgt_pose, K, src_point_world)
    tgt_to_src_proj = _reproject(src_pose, K, tgt_point_world)
    
    # Calculate distances in 2D
    distance_src = np.linalg.norm(np.array((src_center_x, src_center_y)) - tgt_to_src_proj)
    distance_tgt = np.linalg.norm(np.array((tgt_center_x, tgt_center_y)) - src_to_tgt_proj)
    
    # Return averages
    return (distance_src + distance_tgt) / 2, (angle_src + angle_tgt) / 2


def main(args):
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load the YAML file
    with open(args.yaml_file, 'r') as file:
        cfg = yaml.safe_load(file)

    # Create a new directory for the output
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.info(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"Output directory already exists: {output_dir}")

    # Load the dataset directory
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error(f"Dataset directory {dataset_dir} does not exist.")
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")
    if not dataset_dir.is_dir():
        logger.error(f"Dataset path {dataset_dir} is not a directory.")
        raise NotADirectoryError(f"Dataset path {dataset_dir} is not a directory.")
    logger.info(f"Successfully loaded dataset directory: {dataset_dir}")

    global_metadata = []
    # Initialize tqdm progress bar
    scene_bar = tqdm.tqdm(total=sum(1 for _ in dataset_dir.iterdir() if _.is_dir()), unit="scene")
    for scene_dir in dataset_dir.iterdir():
        if not scene_dir.is_dir():
            continue
        scene_bar.set_description(f"Processing scene: {scene_dir.name}")
        scene_bar.update(1)
        logger.info(f"Processing scene: {scene_dir.name}")

        color_dir = scene_dir / "color"
        depth_dir = scene_dir / "depth"
        pose_dir = scene_dir / "pose"
        intrinsic_dir = scene_dir / "intrinsic"

        K_path = intrinsic_dir / "intrinsic_color.txt"
        K = np.loadtxt(K_path)
        K = K[:3, :3]  # Use only the intrinsic matrix
        K_inv = np.linalg.inv(K)

        img_indices = [file.stem for file in color_dir.iterdir()]
        img_indices.sort(key=int)
        logger.info(f"Number of images in {scene_dir.name}: {len(img_indices)}")

        # read images and depth and pose here, to make a big dict
        view_dict = {}
        for view_id in img_indices:
            view_dict[view_id] = {}
            view_dict[view_id]["image_path"] = str(color_dir / f"{view_id}.jpg")
            view_dict[view_id]["depth_path"] = str(depth_dir / f"{view_id}.png")
            view_dict[view_id]["pose_path"] = str(pose_dir / f"{view_id}.txt")
            view_dict[view_id]["image"] = cv2.imread(view_dict[view_id]["image_path"])
            view_dict[view_id]["depth"] = cv2.imread(view_dict[view_id]["depth_path"], cv2.IMREAD_UNCHANGED)
            view_dict[view_id]["pose"] = np.loadtxt(view_dict[view_id]["pose_path"])
            view_dict[view_id]["depth"] = cv2.resize(view_dict[view_id]["depth"], (view_dict[view_id]["image"].shape[1], view_dict[view_id]["image"].shape[0]), interpolation=cv2.INTER_NEAREST)
            view_dict[view_id]["depth"] = view_dict[view_id]["depth"] / 1000.0  # Convert to meters
            view_dict[view_id]["depth_of_center"] = view_dict[view_id]["depth"][view_dict[view_id]["image"].shape[0] // 2, view_dict[view_id]["image"].shape[1] // 2]

        logger.info(f"Loaded {len(img_indices)} images for scene {scene_dir.name} in a dict")

        i = 0
        while i < len(img_indices) - 1:
            idx_src_frame = img_indices[i]
            is_satisfied_pair = False
            is_out_of_range = False
            for j in range(i + 1, len(img_indices)):
                if (j - i) < cfg["min_frame_interval"]:
                    continue
                idx_tgt_frame = img_indices[j]
                if (j - i) > cfg["max_frame_interval"]:
                    logger.warning(
                        f"Frame interval not satisfied, we find all neighbour frame for {scene_dir.name} "
                        f"frame-{int(idx_src_frame):06d}, moving to next source frame."
                    )
                    is_out_of_range = True
                    i += 25
                    break

                distance, angle = _get_distance_and_angle(view_dict, idx_src_frame, idx_tgt_frame, K, K_inv)

                # test the condition
                if distance < cfg["max_distance"] and angle > args.min_angle:
                    i = j
                    logger.info(
                        f"Found a valid pair: {scene_dir.name} "
                        f"frame-{int(idx_src_frame):06d} and frame-{int(idx_tgt_frame):06d} "
                        f"with distance: {distance:.4f} and angle: {angle:.4f}"
                    )
                    is_satisfied_pair = True
                    break

            if is_satisfied_pair:
                # idx_src_frame = f"{int(idx_src_frame):06d}"
                # idx_tgt_frame = f"{int(idx_tgt_frame):06d}"
                # Create the output directory for the pair
                pair_dir = output_dir / scene_dir.name / f"{int(idx_src_frame):06d}-{int(idx_tgt_frame):06d}"
                # pair_dir.mkdir(parents=True, exist_ok=True)
                src_dir = pair_dir / "source"
                tgt_dir = pair_dir / "target"
                src_dir.mkdir(parents=True, exist_ok=True)
                tgt_dir.mkdir(parents=True, exist_ok=True)
                # Copy the images and poses to the output directory
                shutil.copy(view_dict[idx_src_frame]["image_path"], src_dir / f"{idx_src_frame}.jpg")
                shutil.copy(view_dict[idx_src_frame]["depth_path"], src_dir / f"{idx_src_frame}.png")
                shutil.copy(view_dict[idx_src_frame]["pose_path"], src_dir / f"{idx_src_frame}.txt")
                shutil.copy(view_dict[idx_tgt_frame]["image_path"], tgt_dir / f"{idx_tgt_frame}.jpg")
                shutil.copy(view_dict[idx_tgt_frame]["depth_path"], src_dir / f"{idx_tgt_frame}.png")
                shutil.copy(view_dict[idx_tgt_frame]["pose_path"], src_dir / f"{idx_tgt_frame}.txt")

                pose_src2world = view_dict[idx_src_frame]["pose"]
                pose_tgt2world = view_dict[idx_tgt_frame]["pose"]

                pose_tgt2src = np.linalg.inv(pose_src2world) @ pose_tgt2world

                t = pose_tgt2src[:3, 3]
                Rmat = pose_tgt2src[:3, :3]

                theta = np.degrees(np.arctan2(Rmat[2, 1], Rmat[2, 2]))
                phi = np.degrees(np.arcsin(-Rmat[2, 0]))
                psi = np.degrees(np.arctan2(Rmat[1, 0], Rmat[0, 0]))

                metadata = {
                    "scene": scene_dir.name,
                    "pair": f"{int(idx_src_frame):06d}-{int(idx_tgt_frame):06d}",
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
                    "distance": np.round(distance, 6),
                    "angle": np.round(angle, 6),
                }

                global_metadata.append(metadata)

                # save in csv with pandas and json file
                csv_file = pair_dir / "metadata.csv"
                json_file = pair_dir / "metadata.json"

                df = pd.DataFrame([metadata])
                df.to_csv(csv_file, index=False)

                with open(json_file, "w") as f:
                    json.dump(metadata, f, indent=4)

                logger.info(f"Saved metadata for {scene_dir.name} pair-{int(idx_src_frame):06d}-{int(idx_tgt_frame):06d}")
            else:                    
                if not is_out_of_range:
                    logger.warning(
                        f"Frame interval exceeded, we are in the tail for {scene_dir.name} "
                        f"frame-{int(idx_src_frame):06d}, moving to next source frame."
                    )
                    i += 25
                    
    logger.info("Processing completed.")
    scene_bar.close()

    # Save the global metadata to a CSV file
    global_metadata_csv = output_dir / "global_metadata.csv"
    df = pd.DataFrame(global_metadata)
    df.to_csv(global_metadata_csv, index=False)
    logger.info(f"Global metadata saved to {global_metadata_csv}")

    # Save the global metadata to a JSONL file
    global_metadata_jsonl = output_dir / "global_metadata.jsonl"
    with jsonlines.open(global_metadata_jsonl, mode='w') as writer:
        for item in global_metadata:
            writer.write(item)
    logger.info(f"Global metadata saved to {global_metadata_jsonl}")

    # Save the config of this script
    config_file = output_dir / "config.yaml"
    config = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "min_frame_interval": cfg["min_frame_interval"],
        "max_frame_interval": cfg["max_frame_interval"],
        "max_distance": cfg["max_distance"],
        "min_angle": args.min_angle,
    }
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    logger.info(f"Config saved to {config_file}")
    
    # print the length of new dataset
    logger.info(f"Total number of pairs: {len(global_metadata)}")

if __name__ == "__main__":
    args = parse_args()
    main(args)