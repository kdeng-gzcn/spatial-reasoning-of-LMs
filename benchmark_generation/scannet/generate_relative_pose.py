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


def _get_distance_and_angle(src_img: dict, tgt_img: dict, K, K_inv) -> tuple:
    im0 = np.array(Image.open(src_img["image_path"]))
    im100 = np.array(Image.open(tgt_img["image_path"]))
    depth0 = np.array(Image.open(src_img["depth_path"]))
    depth100 = np.array(Image.open(tgt_img["depth_path"]))
    pose0 = np.loadtxt(src_img["pose_path"])
    pose100 = np.loadtxt(tgt_img["pose_path"])

    # unproject, from 2D to 3D World Coordinate
    central_pixel_0_x = im0.shape[1] // 2
    central_pixel_0_y = im0.shape[0] // 2
    central_pixel_0_depth = depth0[central_pixel_0_y, central_pixel_0_x] / 1000
    centeral_pixel_3D_point_0_world = _unproject(pose0, K_inv, central_pixel_0_x, central_pixel_0_y, central_pixel_0_depth)

    # unproject, from 2D to 3D World Coordinate
    central_pixel_100_x = im100.shape[1] // 2
    central_pixel_100_y = im100.shape[0] // 2
    central_pixel_100_depth = depth100[central_pixel_100_y, central_pixel_100_x] / 1000

    centeral_pixel_3D_point_100_world = _unproject(pose100, K_inv,central_pixel_100_x, central_pixel_100_y, central_pixel_100_depth)

    # reproject, from 3D World Coordinate to 2D
    reprojection_0_to_100 = _reproject(pose100, K, centeral_pixel_3D_point_0_world)
    reprojection_100_to_0 = _reproject(pose0, K, centeral_pixel_3D_point_100_world)

    center0_world = pose0[:3, 3]
    center100_world = pose100[:3, 3]

    angle_point0 = _get_angle(
        center0_world, 
        centeral_pixel_3D_point_0_world, 
        center100_world
    )
    angle_point100 = _get_angle(center0_world, centeral_pixel_3D_point_100_world, center100_world)
    distance_0 = np.linalg.norm(np.array((central_pixel_0_x, central_pixel_0_y)) - reprojection_100_to_0)
    distance_100 = np.linalg.norm(np.array((central_pixel_100_x, central_pixel_100_y)) - reprojection_0_to_100)

    return np.mean([distance_0, distance_100]), np.mean([angle_point0, angle_point100])


def _get_distance_and_angle(src_img: dict, tgt_img: dict, K, K_inv) -> tuple:
    # Load only the necessary parts of the images/depth maps
    src_im = np.array(Image.open(src_img["image_path"]))
    tgt_im = np.array(Image.open(tgt_img["image_path"]))
    
    # Extract central pixels once
    src_center_x, src_center_y = src_im.shape[1] // 2, src_im.shape[0] // 2
    tgt_center_x, tgt_center_y = tgt_im.shape[1] // 2, tgt_im.shape[0] // 2
    
    # Load and resize depth maps to match color image size
    with Image.open(src_img["depth_path"]) as depth_img:
        depth_img = np.array(depth_img)
        depth_img = cv2.resize(depth_img, (src_im.shape[1], src_im.shape[0]), interpolation=cv2.INTER_NEAREST)
        src_depth = depth_img[src_center_y, src_center_x] / 1000.0 
    
    with Image.open(tgt_img["depth_path"]) as depth_img:
        depth_img = np.array(depth_img)
        depth_img = cv2.resize(depth_img, (tgt_im.shape[1], tgt_im.shape[0]), interpolation=cv2.INTER_NEAREST)
        tgt_depth = depth_img[tgt_center_y, tgt_center_x] / 1000.0
    
    # Load pose matrices
    src_pose = np.loadtxt(src_img["pose_path"])
    tgt_pose = np.loadtxt(tgt_img["pose_path"])
    
    # Extract camera centers for later use
    src_center_world = src_pose[:3, 3]
    tgt_center_world = tgt_pose[:3, 3]
    
    # Unproject central pixels to 3D world coordinates
    src_point_world = _unproject(src_pose, K_inv, src_center_x, src_center_y, src_depth)
    tgt_point_world = _unproject(tgt_pose, K_inv, tgt_center_x, tgt_center_y, tgt_depth)
    
    # Reproject 3D points to 2D in the other view
    src_to_tgt_proj = _reproject(tgt_pose, K, src_point_world)
    tgt_to_src_proj = _reproject(src_pose, K, tgt_point_world)
    
    # Calculate distances in 2D
    distance_src = np.linalg.norm(np.array((src_center_x, src_center_y)) - tgt_to_src_proj)
    distance_tgt = np.linalg.norm(np.array((tgt_center_x, tgt_center_y)) - src_to_tgt_proj)
    
    # Calculate angles
    angle_src = _get_angle(src_center_world, src_point_world, tgt_center_world)
    angle_tgt = _get_angle(src_center_world, tgt_point_world, tgt_center_world)
    
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
                        f"Frame interval not satisfied for {scene_dir.name}, "
                        f"frame-{idx_src_frame}, moving to next source frame."
                    )
                    is_out_of_range = True
                    i += 50
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

                distance, angle = _get_distance_and_angle(src_img, tgt_img, K, K_inv)

                # test the condition
                if distance < cfg["max_distance"] and angle > args.min_angle:
                    i = j
                    logger.info(
                        f"Found a valid pair: {scene_dir.name} "
                        f"frame-{idx_src_frame} and frame-{idx_tgt_frame} "
                        f"with distance: {distance:.4f} and angle: {angle:.4f}"
                    )
                    is_satisfied_pair = True
                    break

            if is_satisfied_pair:
                idx_src_frame = f"{int(idx_src_frame):06d}"
                idx_tgt_frame = f"{int(idx_tgt_frame):06d}"
                # Create the output directory for the pair
                pair_dir = output_dir / scene_dir.name / f"{idx_src_frame}-{idx_tgt_frame}"
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

                pose_src2world = np.loadtxt(src_img["pose_path"])
                pose_tgt2world = np.loadtxt(tgt_img["pose_path"])

                pose_tgt2src = np.linalg.inv(pose_src2world) @ pose_tgt2world

                t = pose_tgt2src[:3, 3]
                Rmat = pose_tgt2src[:3, :3]

                theta = np.degrees(np.arctan2(Rmat[2, 1], Rmat[2, 2]))
                phi = np.degrees(np.arcsin(-Rmat[2, 0]))
                psi = np.degrees(np.arctan2(Rmat[1, 0], Rmat[0, 0]))

                metadata = {
                    "scene": scene_dir.name,
                    "pair": f"{idx_src_frame}-{idx_tgt_frame}",
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

                logger.info(f"Saved metadata for {scene_dir.name} pair-{idx_src_frame}-{idx_tgt_frame}")
            else:                    
                if not is_out_of_range:
                    logger.warning(
                        f"Frame interval exceeded for {scene_dir.name} "
                        f"frame-{idx_src_frame}, moving to next source frame."
                    )
                    i += 50
                    
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