from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

def get_angle(A, B, C):

    BA = A - B
    BC = C - B
    BA = BA / np.linalg.norm(BA)
    BC = BC / np.linalg.norm(BC)

    angle = np.arccos(np.dot(BA, BC)) * 180 / np.pi

    return angle

def unproject(pose, K_inv, pixelx, pixely, depth):

    """
    
    (x, y) -> (x, y, z)_wrd

    """

    point3d_camera_coords = depth * K_inv @ np.array([pixelx, pixely, 1])
    point3D_world = pose[:3, :3] @ point3d_camera_coords + pose[:3, 3] # vec_wrd = P^cam_wrd * vec_cam

    return point3D_world

def reproject(pose, K, point3D):

    """
    
    (x, y, z)_wrd -> (x, y)
    
    """

    P = K @ np.linalg.inv(pose)[:3]
    projection = P[:3, :3] @ point3D + P[:3, 3]
    projection[:2] /= projection[2:]

    return projection[:2]  

K=np.array(
    [[585, 0, 320.0],
     [0, 585, 240.0],
     [0, 0, 1.0]]
     )

K_inv=np.linalg.inv(K)

import sys

sys.path.append("./")

from SpatialVLM.Dataset import SevenScenesImageDataset
from torch.utils.data import DataLoader

data_path = "./data/Rebuild_7_Scenes_120_1738712209"

dataset = SevenScenesImageDataset(data_path, subset="phi")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)

batch = next(iter(dataloader))

for item in batch:

    source_image = item["source_image"]
    target_image = item["target_image"]
    metadata = item["metadata"]

scene = metadata["scene"]
seq = metadata["seq"]
pair = metadata["pair"]

frame1, frame2 = pair.split('-')

import os

source_path = os.path.join(data_path, "phi_Significant", scene, seq, pair, "source")
target_path = os.path.join(data_path, "phi_Significant", scene, seq, pair, "target")

im0 = np.array(Image.open(os.path.join(source_path, f"frame-{frame1}.color.png")))

depth0 = np.array(Image.open(os.path.join(source_path, f"frame-{frame1}.depth.png")))

im100 = np.array(Image.open(os.path.join(target_path, f"frame-{frame2}.color.png")))

depth100 = np.array(Image.open(os.path.join(target_path, f"frame-{frame2}.depth.png")))

pose0 = np.loadtxt(os.path.join(source_path, f"frame-{frame1}.pose.txt"))
pose100 = np.loadtxt(os.path.join(target_path, f"frame-{frame2}.pose.txt"))

# unproject, from 2D to 3D World Coordinate

central_pixel_0_x = im0.shape[1] // 2
central_pixel_0_y = im0.shape[0] // 2
central_pixel_0_depth = depth0[central_pixel_0_y, central_pixel_0_x] / 1000.0
centeral_pixel_3D_point_0_world = unproject(pose0, K_inv, central_pixel_0_x, central_pixel_0_y, central_pixel_0_depth)

# unproject, from 2D to 3D World Coordinate

central_pixel_100_x = im100.shape[1] // 2
central_pixel_100_y = im100.shape[0] // 2
central_pixel_100_depth = depth100[central_pixel_100_y, central_pixel_100_x] / 1000.0
centeral_pixel_3D_point_100_world = unproject(pose100, K_inv,central_pixel_100_x, central_pixel_100_y, central_pixel_100_depth)

reprojection_0_to_100 = reproject(pose100, K, centeral_pixel_3D_point_0_world)
reprojection_100_to_0 = reproject(pose0, K, centeral_pixel_3D_point_100_world)

center0_world = pose0[:3, 3]
center100_world = pose100[:3, 3]

# angle_0_point0_100 = (180 / np.pi) * np.arccos(center0_world-centeral_pixel_3D_point_0_world, center100_world-centeral_pixel_3D_point_0_world)

# angle_0_point100_100 = (180 / np.pi) * np.arccos(center0_world-centeral_pixel_3D_point_100_world, center100_world-centeral_pixel_3D_point_100_world)

angle_point0 = get_angle(center0_world, centeral_pixel_3D_point_0_world, center100_world)
angle_point100 = get_angle(center0_world, centeral_pixel_3D_point_100_world, center100_world)

print(metadata)

print(angle_point0)
print(angle_point100)

# plot fig
fig = plt.figure(figsize=(20, 10))
fig.suptitle("Angles: %3.3f, %3.3f. " % (angle_point0, angle_point100), fontsize=16)
ax = plt.subplot(1, 2, 1)

distance_0 = np.linalg.norm(np.array((central_pixel_0_x, central_pixel_0_y)) - reprojection_100_to_0)

ax.set_title("Image A. Distance center reprojection: %3.3f" % (distance_0))
ax.imshow(im0)
ax.plot(central_pixel_0_x, central_pixel_0_y, 'c*', markersize=18)
ax.plot(reprojection_100_to_0[0], reprojection_100_to_0[1], 'm*', markersize=18)

ax = plt.subplot(1, 2, 2)
ax.set_title("Image B. Distance center reprojection: %3.3f" % (np.linalg.norm(np.array((central_pixel_100_x, central_pixel_100_y)) - reprojection_0_to_100)))
ax.imshow(im100)
ax.plot(central_pixel_100_x,central_pixel_100_y, 'm*', markersize=18)
ax.plot(reprojection_0_to_100[0], reprojection_0_to_100[1], 'c*', markersize=19)

plt.savefig("demo.pdf")

plt.show()
