import numpy as np
import copy
from pcdet.utils import box_utils

## ========= WAYMO EVALUATION ==========
MAP_NUSCENES_TO_WAYMO = {
            'car' : 'Vehicle',
            'truck' : 'Vehicle',
            'construction_vehi' : 'unknown',
            'bus' : 'Vehicle', 
            'trailer' : 'unknown',
            'barrier': 'unknown',
            'motorcycle': 'Cyclist', 
            'bicycle': 'Cyclist',  
            'pedestrian': 'Pedestrian',
            'traffic_cone': 'unknown', 
            'ignore': 'unknown'}

MAP_KITTI_TO_WAYMO = {
            'Car' : 'Vehicle',
            'Van' : 'Vehicle',
            'Truck' : 'Vehicle',
            'Tram' : 'Vehicle',
            'Pedestrian': 'Pedestrian',
            'Person_sitting': 'Pedestrian',
            'Cyclist': 'Cyclist',
            'DontCare': 'unknown',
            'Misc': 'unknown' }

MAP_LYFT_TO_WAYMO = {
            'car': 'Vehicle',
            'other_vehicle': 'unknown',
            'emergency_vehicle': 'unknown',
            'bus': 'Vehicle',            
            'truck': 'Vehicle',
            'bicycle': 'Cyclist',
            'motorcycle': 'Cyclist',
            'pedestrian': 'Pedestrian',
            'animal': 'unknown'
        }

MAP_ONCE_TO_WAYMO = {
            'Car': 'Vehicle',
            'Bus': 'Vehicle',
            'Truck': 'Vehicle',
            'Cyclist': 'Cyclist',
            'Pedestrian': 'Pedestrian'
        }

def transform_to_waymo_format(dataset_name, annos, is_gt):
    if dataset_name == 'NuScenesDataset':
        label_map = MAP_NUSCENES_TO_WAYMO
    elif dataset_name == 'KittiDataset':
        label_map = MAP_KITTI_TO_WAYMO
    elif dataset_name == 'LyftDataset':
        label_map = MAP_LYFT_TO_WAYMO
    elif dataset_name == 'ONCEDataset':
        label_map = MAP_ONCE_TO_WAYMO
    elif dataset_name == 'WaymoDataset':
        label_map = None
    else:
        raise NotImplementedError

    modified_annos = copy.deepcopy(annos)
    for anno in modified_annos:
        if is_gt:
            if dataset_name in ['NuScenesDataset', 'LyftDataset']:
                anno['name'] = anno['gt_names']
                anno['num_points_in_gt'] = anno['num_lidar_pts'] if 'num_lidar_pts' in anno else 100*np.ones(anno['name'].shape) # arbitrary num pts; L1 will be same as L2
                anno['difficulty'] = np.zeros(anno['name'].shape)
                anno['gt_boxes_lidar'] = anno['gt_boxes']
                anno.pop('gt_names')
                if 'num_lidar_pts' in anno:
                    anno.pop('num_lidar_pts')
            elif dataset_name in ['KittiDataset','CustomDataset']:
                anno['name'] = anno['annos']['name']
                anno['num_points_in_gt'] = anno['annos']['num_points_in_gt']
                anno['difficulty'] = np.zeros(anno['num_points_in_gt'].shape)
                anno['gt_boxes_lidar'] = anno['annos']['gt_boxes_lidar'] 
                anno['annos'].pop('name')
                anno['annos'].pop('num_points_in_gt')
                anno['annos'].pop('gt_boxes_lidar')
            elif dataset_name == 'ONCEDataset':
                anno['difficulty'] = np.zeros(anno['num_points_in_gt'].shape)
                anno['gt_boxes_lidar'] = anno['boxes_3d']
                anno.pop('boxes_3d')
                anno.pop('boxes_2d')    
            elif dataset_name == 'WaymoDataset':
                continue
            else:
                raise NotImplementedError
        
        anno['name'] = anno['name'].astype('<U17')  
        if label_map is None:
            continue 
        
        for k in range(anno['name'].shape[0]):                
            anno['name'][k] = label_map[anno['name'][k]]   
            
    return modified_annos

## ========= KITTI EVALUATION ==========

MAP_ONCE_TO_KITTI = {
            'Car': 'Car',
            'Bus': 'Truck',
            'Truck': 'Truck',
            'Cyclist': 'Cyclist',
            'Pedestrian': 'Pedestrian'
        }

MAP_NUSCENES_TO_KITTI = {
            'car' : 'Car',
            'truck' : 'Truck',
            'construction' : 'Truck',
            'bus' : 'Truck', 
            'trailer' : 'DontCare',
            'barrier': 'DontCare',
            'motorcycle': 'Cyclist', 
            'bicycle': 'Cyclist',  
            'pedestrian': 'Pedestrian',
            'traffic_cone': 'DontCare' }        

MAP_WAYMO_TO_KITTI = {
            'Vehicle': 'Car',
            'Cyclist': 'Cyclist',
            'Pedestrian': 'Pedestrian'
        }

def transform_to_kitti_format(dataset_name, annos, info_with_fakelidar=False, is_gt=False):
    if dataset_name == 'NuScenesDataset':
        map_name_to_kitti = MAP_NUSCENES_TO_KITTI
    elif dataset_name == 'ONCEDataset':
        map_name_to_kitti = MAP_ONCE_TO_KITTI
    elif dataset_name == 'WaymoDataset':
        map_name_to_kitti = MAP_WAYMO_TO_KITTI
    elif dataset_name == 'KittiDataset':
        map_name_to_kitti = None
    else:
        raise NotImplementedError

    modified_annos = copy.deepcopy(annos)
    for anno in modified_annos:
        if 'name' not in anno:
            anno['name'] = anno['gt_names']
            anno.pop('gt_names')

        if map_name_to_kitti is not None:            
            for k in range(anno['name'].shape[0]):
                if anno['name'][k] in map_name_to_kitti:
                    anno['name'][k] = map_name_to_kitti[anno['name'][k]]
                else:
                    anno['name'][k] = 'Person_sitting'

        if 'boxes_lidar' in anno:
            gt_boxes_lidar = anno['boxes_lidar'].copy()
        elif 'boxes_3d' in anno:                    
            gt_boxes_lidar = anno['boxes_3d'].copy()
        else:
            gt_boxes_lidar = anno['gt_boxes_lidar'].copy()

        anno['bbox'] = np.zeros((len(anno['name']), 4))
        anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = np.zeros(len(anno['name']))

        if len(gt_boxes_lidar) > 0:
            if info_with_fakelidar:
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)
            gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
            anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
            anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
            anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
            anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
            dxdydz = gt_boxes_lidar[:, 3:6]
            anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
            anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
            anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)
    return modified_annos
    
if __name__ == '__main__':
    pass