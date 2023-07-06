import sys
sys.path.append('/MS3D')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
import pickle 
import numpy as np
from pathlib import Path
from pcdet.utils import compatibility_utils as compat

def load_dataset(split):

    # Get target dataset    
    cfg.DATA_SPLIT.test = split
    cfg.SAMPLED_INTERVAL.test = 6
    logger = common_utils.create_logger('temp.txt', rank=cfg.LOCAL_RANK)
    target_set, _, _ = build_dataloader(
                dataset_cfg=cfg,
                class_names=cfg.CLASS_NAMES,
                batch_size=1, logger=logger, training=False, dist=False, workers=1
            )      
    return target_set

save_ps_path = '/MS3D/tools/cfgs/target_waymo/gt_as_ps_s190_6280.pkl'
cfg_file = '/MS3D/tools/cfgs/dataset_configs/waymo_dataset_da.yaml'
cfg_from_yaml_file(cfg_file, cfg)
cfg.USE_CUSTOM_TRAIN_SCENES = True
dataset = load_dataset(split='train')
print('Dataset loaded')

fake_ps = {}
for info in dataset.infos:
    # Only use for ground-truth boxes
    class_names = cfg.CLASS_NAMES
    frame_id = compat.get_frame_id(dataset, info)
    gt_names = compat.get_gt_names(dataset, frame_id)
    class_mask = np.isin(gt_names, class_names)
    boxes_3d = compat.get_gt_boxes(dataset, frame_id)[class_mask]
    boxes_3d[:,:3] += dataset.dataset_cfg.SHIFT_COOR
    boxes_3d = boxes_3d[:,:7]
    cls_ids = np.array([class_names.index(name)+1 for name in gt_names[class_mask]])
    boxes_3d = np.hstack([boxes_3d, cls_ids[...,np.newaxis]])
    boxes_3d = np.insert(boxes_3d, 8, 1,axis=1) # set conf score as 1

    gt_infos = {}
    gt_infos['gt_boxes'] = boxes_3d
    gt_infos['memory_counter'] = np.zeros(boxes_3d.shape[0], dtype=int)
    fake_ps[frame_id] = gt_infos

with open(save_ps_path, 'wb') as f:
    pickle.dump(fake_ps, f)
    print(f"Saved: {save_ps_path}")