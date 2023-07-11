import sys
sys.path.append('/MS3D')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils, box_fusion_utils
from pcdet.datasets import build_dataloader
import pickle

def load_dataset(data_cfg, split):

    # Get target dataset    
    data_cfg.DATA_SPLIT.test = split
    data_cfg.SAMPLED_INTERVAL.test = 5
    logger = common_utils.create_logger('temp.txt', rank=cfg.LOCAL_RANK)
    target_set, _, _ = build_dataloader(
                dataset_cfg=data_cfg,
                class_names=data_cfg.CLASS_NAMES,
                batch_size=1, logger=logger, training=False, dist=False, workers=1
            )      
    return target_set

cfg_file = '/MS3D/tools/cfgs/target_waymo/ms3d_lyft_voxel_rcnn_centerhead.yaml'
pkl = '/MS3D/tools/cfgs/target_waymo/final_ps_labels/final_ps_dict.pkl'
with open(pkl, 'rb') as f:
    mydict = pickle.load(f)
# print(f'Loaded detections for {len(det_annos)} frames')

# cfg_from_yaml_file(cfg_file, cfg)
# dataset = load_dataset(cfg.DATA_CONFIG_TAR, 'val')
# result_str, result_dict = dataset.evaluation(
#         det_annos, dataset.class_names,
#         eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
#         output_path=''
#     )
# print('Evaluated: ', pkl)
# print(result_str)

import numpy as np
from pcdet.utils import generate_ps_utils
for frame_id in mydict.keys():

    # -ve class label if score < pos_th
    veh_mask = np.logical_and(abs(mydict[frame_id]['gt_boxes'][:,7]) == 1, 
                            mydict[frame_id]['gt_boxes'][:,8] < 0.6)
    mydict[frame_id]['gt_boxes'][:,7][veh_mask] = -abs(mydict[frame_id]['gt_boxes'][:,7][veh_mask])

    ped_mask = np.logical_and(abs(mydict[frame_id]['gt_boxes'][:,7]) == 2, 
                            mydict[frame_id]['gt_boxes'][:,8] < 0.5)
    mydict[frame_id]['gt_boxes'][:,7][ped_mask] = -abs(mydict[frame_id]['gt_boxes'][:,7][ped_mask])

    ped_mask = np.logical_and(abs(mydict[frame_id]['gt_boxes'][:,7]) == 3, 
                            mydict[frame_id]['gt_boxes'][:,8] < 0.5)
    mydict[frame_id]['gt_boxes'][:,7][ped_mask] = -abs(mydict[frame_id]['gt_boxes'][:,7][ped_mask])
    
generate_ps_utils.save_data(mydict, '/MS3D/tools/cfgs/target_waymo/final_ps_labels', name="final_ps_dict3.pkl")