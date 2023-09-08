from pathlib import Path
import numpy as np
"""
This file facilitates compatbility between different datasets, including loading in different target domain configs
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

def get_target_domain_cfg(cfg, dataset_name, sweeps, custom_target_scenes=False, use_tta=0):
    """
    Simplify the testing of a single pre-trained model on multiple target domains
    by adding in target dataset configs (i.e. DATA_CONFIG_TAR) rather than having
    to duplicate the yaml file and manually specify a DATA_CONFIG_TAR for each new 
    target domain.

    use_tta = 0: no_tta, 1: rwf, 2: rwr, 3: rwr+rwf
    """
    from easydict import EasyDict
    import yaml
    def load_yaml(fname):
        with open(fname, 'r') as f:
            try:
                ret = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                ret = yaml.safe_load(f)
        return ret
    
    # Modify cfg in-place to add DATA_CONFIG_TAR
    cfg.DATA_CONFIG_TAR = {}        
        
    if dataset_name == 'nuscenes':
        target_base_config = load_yaml('cfgs/dataset_configs/nuscenes_dataset_da.yaml')
        target_base_config['MAX_SWEEPS'] = sweeps
    elif dataset_name == 'waymo':
        target_base_config = load_yaml('cfgs/dataset_configs/waymo_dataset_multiframe_da.yaml')
        target_base_config['SEQUENCE_CONFIG']['SAMPLE_OFFSET'] = [-int(sweeps-1), 0]
    elif dataset_name == 'lyft':
        target_base_config = load_yaml('cfgs/dataset_configs/lyft_dataset_da.yaml')
        target_base_config['MAX_SWEEPS'] = sweeps
    elif dataset_name == 'custom':
        target_base_config = load_yaml('cfgs/dataset_configs/custom_dataset_da.yaml')
    else:
        raise NotImplementedError
    
    cfg.DATA_CONFIG_TAR.update(EasyDict(target_base_config))    
    cfg.DATA_CONFIG_TAR.DATA_PROCESSOR = cfg.DATA_CONFIG.DATA_PROCESSOR
    cfg.DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list = cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list
    cfg.DATA_CONFIG_TAR.SAVE_PKL_IN_GROUND_FRAME = True
    cfg.DATA_CONFIG_TAR.TARGET = True

    # if src was pre-trained with timestamp channel, data_config_tar also has to match
    if (cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.src_feature_list[-1] == 'timestamp') and \
        (cfg.DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list[-1] != 'timestamp'):
        cfg.DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list.append('timestamp')

    # if src was not pre-trained with timestamp channel, data_config_tar also has to match
    if (cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.src_feature_list[-1] != 'timestamp') and \
        (cfg.DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list[-1] == 'timestamp'):
        cfg.DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list.pop()

    # Remap class names
    cfg.DATA_CONFIG_TAR.CLASS_NAMES = []
    for class_name in cfg.CLASS_NAMES:
        cfg.DATA_CONFIG_TAR.CLASS_NAMES.append(cfg.DATA_CONFIG_TAR.CLASS_MAPPING[class_name])

    if use_tta != 0:
        cfg.DATA_CONFIG_TAR.USE_TTA = True
        cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR = {}    
        cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.AUG_CONFIG_LIST = []

        cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.AUG_CONFIG_LIST.append(EasyDict({'NAME':'random_world_flip',
                                                                   'ALONG_AXIS_LIST':['x','y']}))
        cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.AUG_CONFIG_LIST.append(EasyDict({'NAME':'random_world_rotation',
                                                                   'WORLD_ROT_ANGLE':[-3.1415926, 3.1415926]}))
        if use_tta == 1:
            cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['random_world_rotation']
        elif use_tta == 2:
            cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['random_world_flip']
        elif use_tta == 3:
            cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['placeholder']
        else:
            print('Choose 0, 1, 2 or 3 for use_tta')
            raise NotImplementedError

    if custom_target_scenes:
        cfg.DATA_CONFIG_TAR.USE_CUSTOM_TRAIN_SCENES = True    

