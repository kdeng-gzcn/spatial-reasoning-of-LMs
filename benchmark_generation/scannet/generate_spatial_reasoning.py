import argparse
from typing import Tuple, List
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

from PIL import Image

import logging

from src.logging.logging_config import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Generate spatial reasoning benchmark")
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
    return parser.parse_args()


def _get_relative_pose(src_img: dict, tgt_img: dict) -> dict:
    """
    Get the relative pose between two images.
    """
    # Load the poses
    pose_src2world = np.loadtxt(src_img["pose_path"])
    pose_tgt2world = np.loadtxt(tgt_img["pose_path"])

    # Compute the relative pose
    pose_tgt2src = np.linalg.inv(pose_src2world) @ pose_tgt2world

    t = pose_tgt2src[:3, 3:].squeeze()
    Rmat = pose_tgt2src[:3, :3]

    theta = np.degrees(np.arctan2(Rmat[2, 1], Rmat[2, 2]))
    phi = np.degrees(np.arcsin(-Rmat[2, 0]))
    psi = np.degrees(np.arctan2(Rmat[1, 0], Rmat[0, 0]))

    return {
        "tx": t[0],
        "ty": t[1],
        "tz": t[2],
        "theta": theta,
        "phi": phi,
        "psi": psi,
    }


def _judge_pair_validity(cfg: dict, relative_pose: dict) -> Tuple[bool, str]:
    is_valid_pair = False
    for dof in relative_pose.keys():
        # Check if the current relative_pose is greater than the upper threshold
        if np.abs(relative_pose[dof]) > cfg["threshold_of_relative_pose"][dof][1]:
            # Test other keys to ensure they are below their lower thresholds
            if all(
                np.abs(relative_pose[other_key]) < cfg["threshold_of_relative_pose"][other_key][0]
                for other_key in relative_pose.keys() if other_key != dof
            ):
                is_valid_pair = True
                break

    return is_valid_pair, dof


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

        img_indices = [file.stem for file in color_dir.iterdir()]
        img_indices.sort(key=int)
        logger.info(f"Number of images in {scene_dir.name}: {len(img_indices)}")

        i = 0
        while i < len(img_indices) - 1:
            idx_src_frame = img_indices[i]
            is_satisfied_pair = False
            is_out_of_range = False
            for j in range(i + 1, len(img_indices)):
                # if (j - i) < cfg["min_frame_interval"]:
                #     continue
                idx_tgt_frame = img_indices[j]
                if (j - i) > cfg["max_frame_interval"]:
                    logger.warning(
                        f"Frame interval not satisfied for {scene_dir.name}, "
                        f"frame-{idx_src_frame}, moving to next source frame."
                    )
                    is_out_of_range = True
                    i += 25
                    break

                src_img = {
                    "image_path": color_dir / f"{idx_src_frame}.jpg",
                    "depth_path": depth_dir / f"{idx_src_frame}.png",
                    "pose_path": pose_dir / f"{idx_src_frame}.txt",
                }

                tgt_img = {
                    "image_path": color_dir / f"{idx_tgt_frame}.jpg",
                    "depth_path": depth_dir / f"{idx_tgt_frame}.png",
                    "pose_path": pose_dir / f"{idx_tgt_frame}.txt",
                }

                # condition for spatial reasoning
                relative_pose = _get_relative_pose(src_img, tgt_img)
                is_satisfied_pair, dof = _judge_pair_validity(cfg, relative_pose)

                if is_satisfied_pair:
                    i = j
                    logger.info(
                        f"Found a valid pair: {scene_dir.name} "
                        f"frame-{idx_src_frame} and frame-{idx_tgt_frame} "
                        f"with {dof}: {relative_pose[dof]:.4f}"
                    )
                    is_satisfied_pair = True
                    break

            if is_satisfied_pair:
                idx_src_frame = f"{int(idx_src_frame):06d}"
                idx_tgt_frame = f"{int(idx_tgt_frame):06d}"
                # Create the output directory for the pair
                pair_dir = output_dir / f"{dof}_significant" / scene_dir.name / f"{idx_src_frame}-{idx_tgt_frame}"
                # pair_dir.mkdir(parents=True, exist_ok=True)
                src_dir = pair_dir / "source"
                tgt_dir = pair_dir / "target"
                src_dir.mkdir(parents=True, exist_ok=True)
                tgt_dir.mkdir(parents=True, exist_ok=True)
                # Copy the images and poses to the output directory
                shutil.copy(src_img["image_path"], src_dir / f"{idx_src_frame}.jpg")
                shutil.copy(src_img["depth_path"], src_dir / f"{idx_src_frame}.png")
                shutil.copy(src_img["pose_path"], src_dir / f"{idx_src_frame}.txt")
                shutil.copy(tgt_img["image_path"], tgt_dir / f"{idx_tgt_frame}.jpg")
                shutil.copy(tgt_img["depth_path"], tgt_dir / f"{idx_tgt_frame}.png")
                shutil.copy(tgt_img["pose_path"], tgt_dir / f"{idx_tgt_frame}.txt")

                metadata = {
                    "scene": scene_dir.name,
                    "pair": f"{idx_src_frame}-{idx_tgt_frame}",
                    "tx": np.round(relative_pose["tx"], 6),
                    "ty": np.round(relative_pose["ty"], 6),
                    "tz": np.round(relative_pose["tz"], 6),
                    "theta": np.round(relative_pose["theta"], 6),
                    "phi": np.round(relative_pose["phi"], 6),
                    "psi": np.round(relative_pose["psi"], 6),
                    "tx_text": "right" if relative_pose["tx"] > 0 else "left",
                    "ty_text": "down" if relative_pose["ty"] > 0 else "up",
                    "tz_text": "forward" if relative_pose["tz"] > 0 else "backward",
                    "theta_text": "upward" if relative_pose["theta"] > 0 else "downward",
                    "phi_text": "rightward" if relative_pose["phi"] > 0 else "leftward",
                    "psi_text": "clockwise" if relative_pose["psi"] > 0 else "counterclockwise",
                }

                global_metadata.append(metadata)

                # save in csv with pandas and json file
                csv_file = pair_dir / "metadata.csv"
                json_file = pair_dir / "metadata.json"

                df = pd.DataFrame([metadata])
                df.to_csv(csv_file, index=False)

                with open(json_file, "w") as f:
                    json.dump(metadata, f, indent=4)

                logger.info(f"Saved metadata for {scene_dir.name} pair-{idx_src_frame}-{idx_tgt_frame}")
            else:
                # common case is out of range, no need to debug

                # if we are going to the tail but not out of range:                
                if not is_out_of_range:
                    logger.warning(
                        f"Frame interval exceeded for {scene_dir.name} "
                        f"frame-{idx_src_frame}, moving to next source frame."
                    )
                    i += 25
                    
    logger.info("Processing completed.")
    scene_bar.close()

    # # Save the global metadata to a JSON file
    # global_metadata_file = output_dir / "global_metadata.json"
    # with open(global_metadata_file, "w") as f:
    #     json.dump(global_metadata, f, indent=4)
    # logger.info(f"Global metadata saved to {global_metadata_file}")

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