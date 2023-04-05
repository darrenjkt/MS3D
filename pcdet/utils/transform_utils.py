import math
import torch
from scipy.spatial.transform import Rotation
import numpy as np

try:
    from kornia.geometry.conversions import (
        convert_points_to_homogeneous,
        convert_points_from_homogeneous,
    )
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')


def project_to_image(project, points):
    """
    Project points to image
    Args:
        project [torch.tensor(..., 3, 4)]: Projection matrix
        points [torch.Tensor(..., 3)]: 3D points
    Returns:
        points_img [torch.Tensor(..., 2)]: Points in image
        points_depth [torch.Tensor(...)]: Depth of each point
    """
    # Reshape tensors to expected shape
    points = convert_points_to_homogeneous(points)
    points = points.unsqueeze(dim=-1)
    project = project.unsqueeze(dim=1)

    # Transform points to image and get depths
    points_t = project @ points
    points_t = points_t.squeeze(dim=-1)
    points_img = convert_points_from_homogeneous(points_t)
    points_depth = points_t[..., -1] - project[..., 2, 3]

    return points_img, points_depth


def normalize_coords(coords, shape):
    """
    Normalize coordinates of a grid between [-1, 1]
    Args:
        coords: (..., 3), Coordinates in grid
        shape: (3), Grid shape
    Returns:
        norm_coords: (.., 3), Normalized coordinates in grid
    """
    min_n = -1
    max_n = 1
    shape = torch.flip(shape, dims=[0])  # Reverse ordering of shape

    # Subtract 1 since pixel indexing from [0, shape - 1]
    norm_coords = coords / (shape - 1) * (max_n - min_n) + min_n
    return norm_coords


def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    """
    Converts depth map into bin indices
    Args:
        depth_map: (H, W), Depth Map
        mode: string, Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min: float, Minimum depth value
        depth_max: float, Maximum depth value
        num_bins: int, Number of depth bins
        target: bool, Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices: (H, W), Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = ((depth_map - depth_min) / bin_size)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
            (math.log(1 + depth_max) - math.log(1 + depth_min))
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = indices.type(torch.int64)
    return indices

def ego_to_world(pose, points=None, boxes=None):
    """
    Transforms points and boxes from the ego frame to world frame
    """
    points_global, boxes_global = None,None
    if points is not None:
        expand_points = np.concatenate([points[:, :3], 
                                        np.ones((points.shape[0], 1))], 
                                        axis=-1)
        points_global = np.dot(expand_points, pose.T)[:,:3]        
    
    if boxes is not None:
        r = Rotation.from_matrix(pose[:3,:3])
        ego2world_yaw = r.as_euler('xyz')[-1]
        boxes_global = boxes.copy()
        expand_centroids = np.concatenate([boxes[:, :3], 
                                           np.ones((boxes.shape[0], 1))], 
                                           axis=-1)
        centroids_global = np.dot(expand_centroids, pose.T)[:,:3]        
        boxes_global[:,:3] = centroids_global
        boxes_global[:,6] += ego2world_yaw
    
    return points_global, boxes_global

def world_to_ego(pose, points=None, boxes=None):
    """
    Transforms points and boxes from the world frame to ego frame
    """
    points_ego, boxes_ego = None,None
    if points is not None:
        expand_points_global = np.concatenate([points[:, :3], 
                                               np.ones((points.shape[0], 1))], 
                                               axis=-1)
        points_ego = np.dot(expand_points_global, np.linalg.inv(pose.T))[:,:3]
    
    if boxes is not None:
        r = Rotation.from_matrix(pose[:3,:3])
        world2ego_yaw = r.as_euler('xyz')[-1]
        boxes_ego = boxes.copy()
        expand_centroids_global = np.concatenate([boxes[:, :3], 
                                                  np.ones((boxes.shape[0], 1))], 
                                                  axis=-1)
        centroids_ego = np.dot(expand_centroids_global, np.linalg.inv(pose.T))[:,:3]       
        boxes_ego[:,:3] = centroids_ego
        boxes_ego[:,6] -= world2ego_yaw
        
    return points_ego, boxes_ego  

def make_vector(angle):    
    return np.hstack([np.cos(angle).reshape(-1,1), 
                      np.sin(angle).reshape(-1,1)])

def get_mean_rotation(angles, weights=None):
    """
    Computes weighted mean rotation in vector space due to wrap-around of angles
    """
    mean_vec = np.average(make_vector(angles), axis=0, weights=weights)
    return np.arctan2(mean_vec[1],mean_vec[0])

def get_rotation_near_weighted_mean(angles, weights=None):
    """
    Computes weighted mean rotation in vector space due to wrap-around of angles
    and returns the angle that is closest to that weighted mean
    """
    mean_vec = np.average(make_vector(angles), axis=0, weights=weights)
    mean_angle = np.arctan2(mean_vec[1],mean_vec[0])    
    return angles[np.argmin(angles - mean_angle)]

def get_abs_angle_diff(a1, a2):
    """
    Gives abs angle 
    """
    v1 = np.array([np.cos(a1), np.sin(a1)])
    v2 = np.array([np.cos(a2), np.sin(a2)])
    angle_diff = np.arccos(np.clip(np.dot(v1,v2), -1, 1))    
    return angle_diff    