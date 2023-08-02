"""Export gt boxes in the format of our ps labels for training"""

import sys
sys.path.append('/MS3D')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
import pickle 
import numpy as np
from pathlib import Path
from pcdet.utils import compatibility_utils as compat

# Mapping of classes to super categories
SUPERCATEGORIES = ['Vehicle','Pedestrian','Cyclist']
SUPER_MAPPING = {'car': 'Vehicle',
                'truck': 'Vehicle',
                'bus': 'Vehicle',
                'Vehicle': 'Vehicle',
                'pedestrian': 'Pedestrian',
                'Pedestrian': 'Pedestrian',
                'motorcycle': 'Cyclist', 
                'bicycle': 'Cyclist',
                'Cyclist': 'Cyclist'}

# save_ps_path = '/MS3D/tools/cfgs/target_nuscenes/vox_c_st3d.pkl'
# cfg_file = '/MS3D/tools/cfgs/dataset_configs/nuscenes_dataset_da.yaml'
# det_pkl = '/MS3D/output/target_nuscenes/pretrained/waymo_voxel_rcnn_centerhead_4f_xyzt_allcls/default/eval/epoch_4/val/waymo4xyzt_nusc5xyzt_notta/result.pkl'
# veh_threshold = 0.5 # w-n
# ped_threshold = 0.4 # w-n

# save_ps_path = '/MS3D/tools/cfgs/target_waymo/vox_c_st3d.pkl'
# cfg_file = '/MS3D/tools/cfgs/dataset_configs/waymo_dataset_multiframe_da.yaml'
# det_pkl = '/MS3D/output/target_waymo/pretrained/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls/default/eval/epoch_3/val/lyft3xyzt_waymo5xyzt_custom190_notta/result.pkl'
# veh_threshold = 0.6 # l-w
# ped_threshold = 0.4 # l-w

save_ps_path = '/MS3D/tools/cfgs/target_lyft/vox_c_st3d.pkl'
cfg_file = '/MS3D/tools/cfgs/dataset_configs/lyft_dataset_da.yaml'
det_pkl = '/MS3D/output/target_lyft/pretrained/waymo_voxel_rcnn_centerhead_4f_xyzt_allcls/default/eval/epoch_4/val/waymo4xyzt_lyft3xyzt_notta/result.pkl'
veh_threshold = 0.6 # l-w
ped_threshold = 0.4 # l-w

# cfg_from_yaml_file(cfg_file, cfg)
# if cfg.get('USE_CUSTOM_TRAIN_SCENES', False):
#     cfg.USE_CUSTOM_TRAIN_SCENES = True
# dataset = load_dataset(split='train', sampled_interval=1)
# print('Dataset loaded')

# Final pseudo-labels are given in the format: (x,y,z,dx,dy,dz,heading,class_id,score). Only positive class_ids are used for training.
with open(det_pkl, 'rb') as f:
    det_annos = pickle.load(f)

# st3d settings for w-n
veh_threshold = 0.6
ped_threshold = 0.4

# Evaluate single pred annos
ps_dict = {}
for det_anno in det_annos:  
    pred_names = det_anno['name']
    superclass_ids = np.array([SUPERCATEGORIES.index(SUPER_MAPPING[name])+1 for name in pred_names])
    veh_neg_mask = np.logical_and(superclass_ids == 1, det_anno['score'] < veh_threshold)
    ped_neg_mask = np.logical_and(superclass_ids == 2, det_anno['score'] < ped_threshold)
    superclass_ids[veh_neg_mask] = -superclass_ids[veh_neg_mask]
    superclass_ids[ped_neg_mask] = -superclass_ids[ped_neg_mask]
    ps_boxes = np.hstack([det_anno['boxes_lidar'],superclass_ids[...,np.newaxis],det_anno['score'][...,np.newaxis]])

    ps_dict[det_anno['frame_id']] = {}
    ps_dict[det_anno['frame_id']]['gt_boxes'] = ps_boxes
    ps_dict[det_anno['frame_id']]['memory_counter'] = np.zeros(ps_boxes.shape[0], dtype=int)

with open(save_ps_path, 'wb') as f:
    pickle.dump(ps_dict, f)
    print(f"Saved: {save_ps_path}")