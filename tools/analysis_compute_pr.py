"""
I think reweighing the conf by iou is a bit hard to justify...

It'd be more appropriate to report precision recall at just conf=pos_th (though this doesn't give a range of recall values...)

Recall = TP/(TP+FN) ----> TP+FN is simply the whole set of ground truths
"""

import pickle
import sys
import argparse
sys.path.append('/MS3D')
import numpy as np
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
import torch
from scipy.spatial import cKDTree
from pcdet.utils import compatibility_utils as compat
from tqdm import tqdm
from pathlib import Path

def load_pkl(file):
    with open(file,'rb') as f:
        data = pickle.load(f)
    return data

def load_dataset(split, sampled_interval):

    # Get target dataset    
    cfg.DATA_SPLIT.test = split
    cfg.SAMPLED_INTERVAL.test = sampled_interval
    logger = common_utils.create_logger('temp.txt', rank=cfg.LOCAL_RANK)
    target_set, _, _ = build_dataloader(
                dataset_cfg=cfg,
                class_names=cfg.CLASS_NAMES,
                batch_size=1, logger=logger, training=False, dist=False, workers=1
            )      
    return target_set

# Compute PR at one conf threshold
def compute_pr_onethreshold(dataset,ps_dict):
    return

# Compute PR for all conf thresholds
def compute_pr_all(dataset,ps_dict):
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

    for frame_id in tqdm(ps_dict.keys(), total=len(ps_dict.keys())):        
        frame_boxes = ps_dict[frame_id]['gt_boxes']
        
        gt_names = compat.get_gt_names(dataset, frame_id)
        class_mask = np.isin(gt_names, ['Vehicle'])
        gt_boxes_3d = compat.get_gt_boxes(dataset, frame_id)[class_mask]
        gt_boxes_3d[:,:3] += dataset.dataset_cfg.SHIFT_COOR


    return

if __name__ == '__main__':
    cfg_file = '/MS3D/tools/cfgs/dataset_configs/waymo_dataset_da.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    cfg.USE_CUSTOM_TRAIN_SCENES = True
    dataset = load_dataset(split='train', sampled_interval=2)
    print('Dataset loaded')

    pkl_file = '/MS3D/tools/cfgs/target_waymo/ps_labels_rnd2/final_ps_dict.pkl'
    ps_dict = load_pkl(pkl_file)

    compute_pr_all(dataset, ps_dict)