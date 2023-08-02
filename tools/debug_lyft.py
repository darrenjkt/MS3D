import sys
sys.path.append('/MS3D')
from lyft_dataset_sdk.lyftdataset import LyftDataset
import pickle
import numpy as np
from pathlib import Path
import math
from pyquaternion import Quaternion
from pcdet.utils.transform_utils import ego_to_world   
from PIL import Image
import matplotlib.pyplot as plt

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    Args:
        points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
        view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
        normalize: Whether to normalize the remaining coordinate (along the third axis).

    Returns: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.

    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

def get_map_mask(lyftd, sd_record):
    sample = lyftd.get("sample", sd_record["sample_token"])
    scene = lyftd.get("scene", sample["scene_token"])
    log = lyftd.get("log", scene["log_token"])
    map = lyftd.get("map", log["map_token"])
    map_mask = map["mask"]
    return map_mask

def crop_image(image: np.array, x_px: int, y_px: int, axes_limit_px: int) -> np.array:
    x_min = int(x_px - axes_limit_px)
    x_max = int(x_px + axes_limit_px)
    y_min = int(y_px - axes_limit_px)
    y_max = int(y_px + axes_limit_px)

    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image

# Load the dataset/mnt/big-data/darren/data/lyft/trainval/data
# Adjust the dataroot parameter below to point to your local dataset path.
# The correct dataset path contains at least the following four folders (or similar): images, lidar, maps, v1.0.1-train
level5data = LyftDataset(data_path='/MS3D/data/lyft/trainval', json_path='/MS3D/data/lyft/trainval/data', verbose=True)
pth = '/MS3D/data/lyft/trainval/lyft_infos_train_16sweeps.pkl'
with open(pth,'rb') as f:
    lyft_infos = pickle.load(f)

found_idx = None
for idx, info in enumerate(lyft_infos):
    frame_id = info['lidar_path'].split('/')[1].split('.')[0]
    if frame_id == 'host-a004_lidar1_1233961192101151686':
        found_idx = idx

lidar_path = Path('/MS3D/data/lyft/trainval') / lyft_infos[found_idx]['lidar_path']
points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1)        
if points.shape[0] % 5 != 0:
    points = points[: points.shape[0] - (points.shape[0] % 5)]
points = points.reshape([-1, 5])[:, :3]
point_cloud_range = np.array([-75.2, -75.2, -5.0, 75.2, 75.2, 5.0])
axes_limit = point_cloud_range[3]
mask_in_range = np.logical_and(points[:, :3] > point_cloud_range[:3], points[:, :3] < point_cloud_range[3:] - 1e-3).all(axis=1)
points = points[mask_in_range]

my_sample = level5data.get('sample', '5c0efcd9608120c3d2d54d22403f88ace7a4b0a9b41a02c709344449ef8dbf55')
sd_record = level5data.get("sample_data", my_sample['data']['LIDAR_TOP'])
map_mask = get_map_mask(level5data, sd_record)


# -------------------------------------------------------------------------------
# pose = np.dot(np.linalg.inv(lyft_infos[found_idx]['car_from_global']), 
#                np.linalg.inv(lyft_infos[found_idx]['ref_from_car']))
# pixel_coords = map_mask.to_pixel_coords(pose[0,3],pose[1,3])


# scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
# mask_raster = map_mask.mask()
# cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

# # rotate
# ypr_rad = Quaternion(matrix=pose[:3,:3]).yaw_pitch_roll
# yaw_deg = -math.degrees(ypr_rad[0])

# rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))
# ego_centric_map = crop_image(
#     rotated_cropped, rotated_cropped.shape[1] / 2, rotated_cropped.shape[0] / 2, scaled_limit_px
# )

# mask_in_range = np.logical_and(points[:, :3] > point_cloud_range[:3], points[:, :3] < point_cloud_range[3:] - 1e-3).all(axis=1)
# points = points[mask_in_range]

# # Query the map value at the point coordinate
# points_bev = np.floor((points[:, :2] - point_cloud_range[0]) / map_mask.resolution).astype(int)
# points_map_feat = ego_centric_map[points_bev[:, 1], points_bev[:, 0]]

# drive_mask = np.logical_not(np.all(points_map_feat == 255, axis=1))

# fig, ax = plt.subplots(1,1, figsize=(9,9))
# ax.imshow(ego_centric_map, cmap='gray', extent=[-axes_limit, axes_limit, -axes_limit, axes_limit])
# ax.scatter(points[drive_mask, 0], points[drive_mask, 1], color='orange', s=0.2)
# ax.scatter(points[~drive_mask, 0], points[~drive_mask, 1], color='black', s=0.2)
# plt.show()

# print('loaded')
# -------------------------------------------------------------------------------

pose = level5data.get("ego_pose", sd_record["ego_pose_token"])
pixel_coords = map_mask.to_pixel_coords(pose["translation"][0], pose["translation"][1])

scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
mask_raster = map_mask.mask()

cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

ypr_rad = Quaternion(pose["rotation"]).yaw_pitch_roll
yaw_deg = -math.degrees(ypr_rad[0])

rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))

_, ax = plt.subplots(1, 1, figsize=(9, 9))
ego_centric_map = crop_image(
    rotated_cropped, rotated_cropped.shape[1] / 2, rotated_cropped.shape[0] / 2, scaled_limit_px
)

# Compute transformation matrices for lidar point cloud
cs_record = level5data.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
pose_record = level5data.get("ego_pose", sd_record["ego_pose_token"])
vehicle_from_sensor = np.eye(4)
vehicle_from_sensor[:3, :3] = Quaternion(cs_record["rotation"]).rotation_matrix
vehicle_from_sensor[:3, 3] = cs_record["translation"]

ego_yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]
rot_vehicle_flat_from_vehicle = np.dot(
    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
    Quaternion(pose_record["rotation"]).inverse.rotation_matrix,
)

vehicle_flat_from_vehicle = np.eye(4)
vehicle_flat_from_vehicle[:3, :3] = rot_vehicle_flat_from_vehicle

# Show point cloud.
points_map = view_points(
    points.T[:3, :], np.dot(vehicle_flat_from_vehicle, vehicle_from_sensor), normalize=False
)
# dists = np.sqrt(np.sum(points_map[:2, :] ** 2, axis=0))
# colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
# ax.scatter(points_map[0, :], points_map[1, :], c=colors, s=0.2)
# plt.show()

# Query the map value at the point coordinate
points_bev = np.floor((points_map[:2, :] - point_cloud_range[0]) / map_mask.resolution).astype(int)
mask_x = np.logical_and(points_bev[0, :] < ego_centric_map.shape[1], points_bev[0, :] > 0)
mask_y = np.logical_and(points_bev[1, :] < ego_centric_map.shape[0], points_bev[1, :] > 0)
points_bev = points_bev[:, np.logical_and(mask_x, mask_y)]

points_map_feat = ego_centric_map[points_bev[1, :], points_bev[0, :]]

drive_mask = np.logical_not(np.all(points_map_feat == 255, axis=1))

fig, ax = plt.subplots(1,1, figsize=(9,9))
points_map = points_map[:,np.logical_and(mask_x, mask_y)]
ax.imshow(ego_centric_map, cmap='gray', extent=[-axes_limit, axes_limit, -axes_limit, axes_limit])
ax.scatter(points_map[0, drive_mask], points_map[1, drive_mask], color='orange', s=0.2)
ax.scatter(points_map[0, ~drive_mask], points_map[1, ~drive_mask], color='black', s=0.2)
plt.show()