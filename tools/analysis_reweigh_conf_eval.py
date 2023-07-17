"""
Main idea is that because confidence score plays a bit part in the final AP, I'll do the evaluation solely on
conf > pos_th, then for every remaining box, I'll set the conf=iou with GT if it matches. Else, zero (cause zero iou).
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

def find_nearest_gtbox(frame_gt_boxes, pred_box, return_iou=True):
    # Assess IOU of combined box with GT
    # Find closest GT to our chosen box    
    gt_tree = cKDTree(frame_gt_boxes[:,:3])
    nearest_gt = gt_tree.query_ball_point(pred_box[:3].reshape(1,-1), r=2.0)
    if len(nearest_gt[0]) == 0:        
        return None
    nearest_gt_box = frame_gt_boxes[nearest_gt[0][0]]
    if return_iou:
        gt_box = np.reshape(nearest_gt_box, (1, -1))
        gt_box_cuda = torch.from_numpy(gt_box.astype(np.float32)).cuda()
        pred_box_cuda = torch.from_numpy(pred_box.reshape(1,-1).astype(np.float32)).cuda()

        iou = iou3d_nms_utils.boxes_iou3d_gpu(gt_box_cuda, pred_box_cuda)
        return (nearest_gt_box, iou.item())
    else:
        return nearest_gt_box

def reweigh_box_by_class(dataset, frame_id, frame_boxes, cls_name, cls_id, pos_th):
    """frame_boxes: (N,9) -> (x,y,z,dx,dy,dz,heading,class,score)"""
    gt_names = compat.get_gt_names(dataset, frame_id)
    class_mask = np.isin(gt_names, [cls_name])
    gt_boxes_3d = compat.get_gt_boxes(dataset, frame_id)[class_mask]
    gt_boxes_3d[:,:3] += dataset.dataset_cfg.SHIFT_COOR

    cls_score_mask = np.logical_and(frame_boxes[:,8] > pos_th, abs(frame_boxes[:,7]) == cls_id)
    ps_boxes = frame_boxes[cls_score_mask]

    for idx, _ in enumerate(ps_boxes):        
        ret = find_nearest_gtbox(gt_boxes_3d[:,:7], ps_boxes[idx,:7], return_iou=True)
        if ret is None:
            ps_boxes[idx,8] = 0.0
        else:
            ps_boxes[idx,8] = ret[1]

    return ps_boxes

def reweigh_conf(dataset, ps_dict, veh_th, ped_th):

    new_ps_dict = {}
    for frame_id in tqdm(ps_dict.keys(), total=len(ps_dict.keys())):        
        frame_boxes = ps_dict[frame_id]['gt_boxes']
        veh_boxes = reweigh_box_by_class(dataset, frame_id, frame_boxes, 'Vehicle', cls_id=1, pos_th=veh_th)
        ped_boxes = reweigh_box_by_class(dataset, frame_id, frame_boxes, 'Pedestrian', cls_id=2, pos_th=ped_th)
        new_ps_dict[frame_id] = {}
        new_ps_dict[frame_id]['gt_boxes'] = np.vstack([veh_boxes, ped_boxes]) 

    return new_ps_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--ps_dict', type=str, help='Use kbf ps_dict')
    parser.add_argument('--veh_th', type=float)
    parser.add_argument('--ped_th', type=float)
    args = parser.parse_args()

    cfg_file = '/MS3D/tools/cfgs/dataset_configs/waymo_dataset_da.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    cfg.USE_CUSTOM_TRAIN_SCENES = True
    dataset = load_dataset(split='train', sampled_interval=1)
    print('Dataset loaded')

    ps_dict = load_pkl(args.ps_dict)
    new_ps_dict = reweigh_conf(dataset, ps_dict, veh_th=args.veh_th, ped_th=args.ped_th)
    with open(f'/MS3D/tools/cfgs/target_waymo/analysis/reweigh_conf_ps/{Path(args.ps_dict).parts[-2]}_{Path(args.ps_dict).stem}_confreweigh.pkl','wb') as f:
        pickle.dump(new_ps_dict, f)    