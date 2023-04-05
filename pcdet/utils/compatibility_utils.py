from pathlib import Path
import numpy as np
"""
Different datasets process infos differently
"""
def get_pose(dataset, frame_id):
    if dataset.dataset_cfg.DATASET == 'ONCEDataset':
        return dataset.get_pose(frame_id)
    elif dataset.dataset_cfg.DATASET == 'KittiDataset':
        infos_idx = dataset.frameid_to_idx[frame_id]
        pc_info = dataset.infos[infos_idx]['point_cloud']
        return dataset.get_pose(pc_info['lidar_sequence'], pc_info['sample_idx'])
    elif dataset.dataset_cfg.DATASET in ['NuScenesDataset','LyftDataset']:
        infos_idx = dataset.frameid_to_idx[frame_id]
        # transform from lidar frame -> vehicle ego frame then vehicle ego frame -> global frame
        pose = np.dot(np.linalg.inv(dataset.infos[infos_idx]['car_from_global']), 
               np.linalg.inv(dataset.infos[infos_idx]['ref_from_car']))
        return pose
    elif dataset.dataset_cfg.DATASET == 'WaymoDataset':
        infos_idx = dataset.frameid_to_idx[frame_id]
        return dataset.infos[infos_idx]['pose']
    else:
        raise NotImplementedError

def get_lidar(dataset, frame_id):
    """Returns 1-frame point cloud"""
    infos_idx = dataset.frameid_to_idx[frame_id]
    if dataset.dataset_cfg.DATASET == 'ONCEDataset':
        return dataset.get_lidar(dataset.infos[infos_idx]['sequence_id'], frame_id)    
    elif dataset.dataset_cfg.DATASET == 'KittiDataset':
        pc_info = dataset.infos[infos_idx]['point_cloud']
        return dataset.get_lidar_seq(pc_info['lidar_sequence'], pc_info['sample_idx'])
    elif dataset.dataset_cfg.DATASET in ['NuScenesDataset','LyftDataset']:
        return dataset.get_lidar_with_sweeps(infos_idx, max_sweeps=1)
    elif dataset.dataset_cfg.DATASET == 'WaymoDataset':
        sequence_name = get_sequence_name(dataset, frame_id)
        sample_idx = get_sample_idx(dataset, frame_id)
        return dataset.get_lidar(sequence_name, sample_idx)
    else:
        raise NotImplementedError

def get_frame_id(dataset, info):
    if dataset.dataset_cfg.DATASET == 'ONCEDataset':
        return info['frame_id']        
    elif dataset.dataset_cfg.DATASET == 'KittiDataset':
        return info['point_cloud']['frame_id']
    elif dataset.dataset_cfg.DATASET in ['NuScenesDataset','LyftDataset']:
        return Path(info['lidar_path']).stem
    elif dataset.dataset_cfg.DATASET == 'WaymoDataset':
        return info['frame_id']
    else:
        raise NotImplementedError

def get_sequence_name(dataset, frame_id):
    infos_idx = dataset.frameid_to_idx[frame_id]
    if dataset.dataset_cfg.DATASET == 'ONCEDataset':
        return dataset.infos[infos_idx]['sequence_id']
    elif dataset.dataset_cfg.DATASET == 'KittiDataset':        
        return dataset.infos[infos_idx]['point_cloud']['lidar_sequence']
    elif dataset.dataset_cfg.DATASET in ['NuScenesDataset','LyftDataset']:
        return dataset.infos[infos_idx]['scene_name']
    elif dataset.dataset_cfg.DATASET == 'WaymoDataset':
        return dataset.infos[infos_idx]['point_cloud']['lidar_sequence']
    else:
        raise NotImplementedError

def get_sample_idx(dataset, frame_id):
    infos_idx = dataset.frameid_to_idx[frame_id]
    if dataset.dataset_cfg.DATASET == 'ONCEDataset':        
        return dataset.infos[infos_idx]['sample_idx']
    elif dataset.dataset_cfg.DATASET == 'KittiDataset':
        return dataset.infos[infos_idx]['point_cloud']['sample_idx']
    elif dataset.dataset_cfg.DATASET in ['NuScenesDataset','LyftDataset']:
        return dataset.infos[infos_idx]['sample_idx']        
    elif dataset.dataset_cfg.DATASET == 'WaymoDataset':
        return dataset.infos[infos_idx]['point_cloud']['sample_idx']
    else:
        raise NotImplementedError    

def get_timestamp(dataset, frame_id):
    infos_idx = dataset.frameid_to_idx[frame_id]
    if dataset.dataset_cfg.DATASET == 'ONCEDataset':
        return dataset.infos[infos_idx]['timestamp']
    elif dataset.dataset_cfg.DATASET == 'KittiDataset':        
        return dataset.infos[infos_idx]['point_cloud']['timestamp']
    elif dataset.dataset_cfg.DATASET in ['NuScenesDataset', 'LyftDataset']:
        return dataset.infos[infos_idx]['timestamp']  
    elif dataset.dataset_cfg.DATASET == 'WaymoDataset':
        return dataset.infos[infos_idx]['metadata']['timestamp_micros']
    else:
        raise NotImplementedError                

def get_gt_boxes(dataset, frame_id):
    infos_idx = dataset.frameid_to_idx[frame_id]
    if dataset.dataset_cfg.DATASET == 'ONCEDataset':
        return dataset.infos[infos_idx]['annos']['boxes_3d']
    elif dataset.dataset_cfg.DATASET == 'KittiDataset':
        return dataset.infos[infos_idx]['annos']['gt_boxes_lidar']
    elif dataset.dataset_cfg.DATASET in ['NuScenesDataset','LyftDataset']:
        return dataset.infos[infos_idx]['gt_boxes']
    elif dataset.dataset_cfg.DATASET == 'WaymoDataset':
        return dataset.infos[infos_idx]['annos']['gt_boxes_lidar']
    else:
        raise NotImplementedError

def get_gt_names(dataset, frame_id):
    infos_idx = dataset.frameid_to_idx[frame_id]
    if dataset.dataset_cfg.DATASET == 'ONCEDataset':
        return dataset.infos[infos_idx]['annos']['name']
    elif dataset.dataset_cfg.DATASET == 'KittiDataset':
        return dataset.infos[infos_idx]['annos']['name']
    elif dataset.dataset_cfg.DATASET in ['NuScenesDataset','LyftDataset']:
        return dataset.infos[infos_idx]['gt_names']
    elif dataset.dataset_cfg.DATASET == 'WaymoDataset':
        return dataset.infos[infos_idx]['annos']['name']
    else:
        raise NotImplementedError        