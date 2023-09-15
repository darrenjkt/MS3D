import os
import pickle
import numpy as np
from ...utils import common_utils
import open3d as o3d

def get_fov_flag(points, img_shape, calib):
    IMG_H, IMG_W = img_shape
    cameramat = np.array(calib['intrinsic']).reshape((3,3))
    camera2sensorframe = np.array(calib['extrinsic']).reshape((4,4))

    pts_3d_hom = np.hstack((points, np.ones((points.shape[0],1)))).T # (4,N)
    pts_imgframe = np.dot(camera2sensorframe[:3], pts_3d_hom) # (3,4) * (4,N) = (3,N)
    image_pts = np.dot(cameramat, pts_imgframe).T # (3,3) * (3,N)

    image_pts[:,0] /= image_pts[:,2]
    image_pts[:,1] /= image_pts[:,2]
    uv = image_pts.copy()
    fov_inds =  (uv[:,0] > 0) & (uv[:,0] < IMG_W -1) & \
                (uv[:,1] > 0) & (uv[:,1] < IMG_H -1)  
    return fov_inds

def gtbox_to_corners(box):
    """
    Takes an array containing [x,y,z,l,w,h,r], and returns an [8, 3] matrix that 
    represents the [x, y, z] for each 8 corners of the box.
    
    Note: Openpcdet __getitem__ gt_boxes are in the format [x,y,z,l,w,h,r,alpha]
    where alpha is "observation angle of object, ranging [-pi..pi]"
    """
    # To return
    corner_boxes = np.zeros((8, 3))

    translation = box[0:3]
    l, w, h = box[3], box[4], box[5] 
    rotation = box[6]

    # Create a bounding box outline
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])

    return bounding_box.transpose(), rotation_matrix

def get_o3dbox(gt_box):
    # Input is a single annotated box of format [x,y,z,l,w,h]
    box_corners, r_mat = gtbox_to_corners(gt_box)
    boxpts = o3d.utility.Vector3dVector(box_corners)
    o3dbox = o3d.geometry.OrientedBoundingBox().create_from_points(boxpts)
    o3dbox.color = np.array([1,0,0])
    o3dbox.center = gt_box[0:3]
    o3dbox.R = r_mat
    return o3dbox

def convert_to_o3dpcd(points):
    if type(points) == list:
        pcds = []
        for pointcloud in points:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud[:,:3])
            pcds.append(pcd)
        return pcds
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        return pcd    
