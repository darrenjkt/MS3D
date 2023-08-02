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

def load_pkl(file):
    with open(file,'rb') as f:
        data = pickle.load(f)
    return data

def load_dataset(split, sampled_interval):

    # Get target dataset    
    cfg.DATA_SPLIT.test = split
    if cfg.get('SAMPLED_INTERVAL',False):
        cfg.SAMPLED_INTERVAL.test = sampled_interval
    logger = common_utils.create_logger('temp.txt', rank=cfg.LOCAL_RANK)
    target_set, _, _ = build_dataloader(
                dataset_cfg=cfg,
                class_names=cfg.CLASS_NAMES,
                batch_size=1, logger=logger, training=False, dist=False, workers=1
            )      
    return target_set

# Compute PR for labels at various ranges
def compute_pr_all(dataset, ps_dict, cls_names, iou_th, range_eval=0, 
                   use_bev_iou=False, use_det_anno=False, score_th=0.0,
                   ped_cls_id=None):
    """
    Only tested for waymo.
    cls_name: choose 'Vehicle' or 'Pedestrian'
    range: 0 (all), 1 (0-30m), 2 (30-50m), 3 (50-80m)
    Lyft/Nusc ped class_id is 6
    """

    tps = 0
    fps = 0 
    tp_and_fn = 0
    if use_det_anno:
        iter_list = ps_dict
    else:
        iter_list = list(ps_dict.keys())
    
    for myiter in iter_list:
        if use_det_anno:
            frame_id = myiter['frame_id']
            frame_boxes = np.hstack([myiter['boxes_lidar'], myiter['pred_labels'][...,np.newaxis]])
            frame_boxes = frame_boxes[myiter['score'] > score_th]
        else:
            frame_id = myiter
            frame_boxes = ps_dict[frame_id]['gt_boxes']
            frame_boxes = frame_boxes[frame_boxes[:,8] > score_th]
        
        above_pos_mask = frame_boxes[:,7] > 0
        frame_boxes = frame_boxes[above_pos_mask]
        if np.count_nonzero(np.isin(["car","truck","bus","Vehicle"],cls_names)) > 0:
            cls_mask = frame_boxes[:,7] == 1
            frame_boxes = frame_boxes[cls_mask]
        elif np.count_nonzero(np.isin(["pedestrian","Pedestrian"],cls_names)) > 0:
            if ped_cls_id is None:
                cls_mask = frame_boxes[:,7] == 2
            else:
                cls_mask = frame_boxes[:,7] == ped_cls_id
            frame_boxes = frame_boxes[cls_mask]
        else:
            raise NotImplementedError

        gt_names = compat.get_gt_names(dataset, frame_id)
        class_mask = np.isin(gt_names, cls_names)
        gt_boxes_3d = compat.get_gt_boxes(dataset, frame_id)[class_mask]
        gt_boxes_3d[:,:3] += dataset.dataset_cfg.SHIFT_COOR

        if range_eval != 0:
            pred_dist = np.linalg.norm(frame_boxes[:,:2],axis=1)
            gt_dist = np.linalg.norm(gt_boxes_3d[:,:2],axis=1)
            if range_eval == 1:
                frame_boxes = frame_boxes[pred_dist < 30]
                gt_boxes_3d = gt_boxes_3d[gt_dist < 30]
            elif range_eval == 2:
                frame_boxes = frame_boxes[np.logical_and(pred_dist >= 30, pred_dist < 50)]
                gt_boxes_3d = gt_boxes_3d[np.logical_and(gt_dist >= 30, gt_dist < 50)]
            elif range_eval == 3:
                frame_boxes = frame_boxes[np.logical_and(pred_dist >= 50, pred_dist < 80)]
                gt_boxes_3d = gt_boxes_3d[np.logical_and(gt_dist >= 50, gt_dist < 80)]
            else:
                print('Please specify range_eval=0,1,2 or 3')
                raise NotImplementedError

        gt_box_a, _ = common_utils.check_numpy_to_torch(gt_boxes_3d)
        gt_box_b, _ = common_utils.check_numpy_to_torch(frame_boxes)
        gt_box_a, gt_box_b = gt_box_a.cuda(), gt_box_b.cuda()
        if gt_box_a.shape[0] == 0:
            if gt_box_b.shape[0] != 0:
                fps += frame_boxes.shape[0]
            continue
        if gt_box_b.shape[0] == 0:
            if gt_box_a.shape[0] != 0:
                tp_and_fn += gt_boxes_3d.shape[0]
            continue

        # get ious
        if use_bev_iou:
            iou_matrix = iou3d_nms_utils.boxes_iou_bev(gt_box_a[:, :7], gt_box_b[:, :7]).cpu()
        else:
            iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(gt_box_a[:, :7], gt_box_b[:, :7]).cpu()
        ious, match_idx = torch.max(iou_matrix, dim=1)
        ious, match_idx = ious.numpy(), match_idx.numpy()
        
        iou_mask = (ious >= iou_th)
        tps += np.count_nonzero(iou_mask)
        tp_and_fn += gt_boxes_3d.shape[0]
        fps += frame_boxes.shape[0] - np.count_nonzero(iou_mask)

    return tps, fps, tp_and_fn

def run(ps_dict,exp_name,use_bev_iou=True,use_det_anno=False, 
        veh_score_th=0.0, ped_score_th=0.0, ped_cls_id=None, waymo_cls_names=True):

    if waymo_cls_names:
        veh_cls_names = ['Vehicle']
        ped_cls_names = ['Pedestrian']
    else: # nusc/lyft
        veh_cls_names = ['car','truck','bus']
        ped_cls_names = ['pedestrian']

    pr = {}
    # VEHICLE
    tps, fps, tp_and_fn = compute_pr_all(dataset, ps_dict, cls_names=veh_cls_names, range_eval=1, 
                                         use_bev_iou=use_bev_iou, use_det_anno=use_det_anno, score_th=veh_score_th, iou_th=0.7)
    pr['VEH_0-30'] = {}
    pr['VEH_0-30']['precision'] = tps/(tps+fps)
    pr['VEH_0-30']['recall'] = tps/tp_and_fn

    tps, fps, tp_and_fn = compute_pr_all(dataset, ps_dict, cls_names=veh_cls_names, range_eval=2, 
                                         use_bev_iou=use_bev_iou, use_det_anno=use_det_anno, score_th=veh_score_th, iou_th=0.7)
    pr['VEH_30-50'] = {}
    pr['VEH_30-50']['precision'] = tps/(tps+fps)
    pr['VEH_30-50']['recall'] = tps/tp_and_fn

    tps, fps, tp_and_fn = compute_pr_all(dataset, ps_dict, cls_names=veh_cls_names, range_eval=3, 
                                         use_bev_iou=use_bev_iou, use_det_anno=use_det_anno, score_th=veh_score_th, iou_th=0.7)
    pr['VEH_50-80'] = {}
    pr['VEH_50-80']['precision'] = tps/(tps+fps)
    pr['VEH_50-80']['recall'] = tps/tp_and_fn

    tps, fps, tp_and_fn = compute_pr_all(dataset, ps_dict, cls_names=veh_cls_names, range_eval=0, 
                                         use_bev_iou=use_bev_iou, use_det_anno=use_det_anno, score_th=veh_score_th, iou_th=0.7)
    pr['VEH_0-80'] = {}
    pr['VEH_0-80']['precision'] = tps/(tps+fps)
    pr['VEH_0-80']['recall'] = tps/tp_and_fn            

    # PEDESTRIAN
    tps, fps, tp_and_fn = compute_pr_all(dataset, ps_dict, cls_names=ped_cls_names, range_eval=1, 
                                         use_bev_iou=use_bev_iou, use_det_anno=use_det_anno, score_th=ped_score_th, iou_th=0.5, ped_cls_id=ped_cls_id)
    pr['PED_0-30'] = {}
    pr['PED_0-30']['precision'] = tps/(tps+fps)
    pr['PED_0-30']['recall'] = tps/tp_and_fn

    tps, fps, tp_and_fn = compute_pr_all(dataset, ps_dict, cls_names=ped_cls_names, range_eval=2, 
                                         use_bev_iou=use_bev_iou, use_det_anno=use_det_anno, score_th=ped_score_th, iou_th=0.5, ped_cls_id=ped_cls_id)
    pr['PED_30-50'] = {}
    pr['PED_30-50']['precision'] = tps/(tps+fps)
    pr['PED_30-50']['recall'] = tps/tp_and_fn

    tps, fps, tp_and_fn = compute_pr_all(dataset, ps_dict, cls_names=ped_cls_names, range_eval=3, 
                                         use_bev_iou=use_bev_iou, use_det_anno=use_det_anno, score_th=ped_score_th, iou_th=0.5, ped_cls_id=ped_cls_id)
    pr['PED_50-80'] = {}
    pr['PED_50-80']['precision'] = tps/(tps+fps)
    pr['PED_50-80']['recall'] = tps/tp_and_fn

    tps, fps, tp_and_fn = compute_pr_all(dataset, ps_dict, cls_names=ped_cls_names, range_eval=0, 
                                         use_bev_iou=use_bev_iou, use_det_anno=use_det_anno, score_th=ped_score_th, iou_th=0.5, ped_cls_id=ped_cls_id)
    pr['PED_0-80'] = {}
    pr['PED_0-80']['precision'] = tps/(tps+fps)
    pr['PED_0-80']['recall'] = tps/tp_and_fn    

    out_k = ''
    out_pr = f'{exp_name}: '
    for k,v in pr.items():
        out_k += k + ','
        out_pr += f'{v["precision"]:0.6f}' + ',' + f'{v["recall"]:0.6f}' + ','

    print(out_pr)

if __name__ == '__main__':
    cfg_file = '/MS3D/tools/cfgs/dataset_configs/waymo_dataset_da.yaml'
    # cfg_file = '/MS3D/tools/cfgs/dataset_configs/nuscenes_dataset_da.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    cfg.USE_CUSTOM_TRAIN_SCENES = True
    dataset = load_dataset(split='train', sampled_interval=2)
    print('Dataset loaded')

    # pkl_file = '/MS3D/output/target_waymo/pretrained/ms3d_scratch_voxel_rcnn_anchorhead/4f_xyzt_vehped_rnd3/eval/epoch_30/val/waymo4xyzt_custom190_notta/result.pkl'
    # det_anno = load_pkl(pkl_file)
    # run(det_anno, use_bev_iou=True, exp_name='vanchorhead_bev',use_det_anno=True,veh_score_th=0.8,ped_score_th=0.6)

    # pkl_file = '/MS3D/output/target_waymo/pretrained/ms3d_scratch_voxel_rcnn_centerhead/4f_xyzt_vehped_rnd3/eval/epoch_30/val/waymo4xyzt_custom190_notta/result.pkl'
    # det_anno = load_pkl(pkl_file)
    # run(det_anno, use_bev_iou=True, exp_name='vcenterhead_bev',use_det_anno=True,veh_score_th=0.8,ped_score_th=0.6)
    

    # pkl_file = '/MS3D/output/target_nuscenes/pretrained/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls/default/eval/epoch_4/val/waymo4xyzt_nusc7xyzt_notta/result.pkl'
    # det_anno = load_pkl(pkl_file)
    # run(det_anno, use_bev_iou=True, exp_name='single_detector',use_det_anno=True,veh_score_th=0.6,ped_score_th=0.4, waymo_cls_names=False)

    # pkl_file = '/MS3D/tools/cfgs/target_nuscenes/ps_labels_exp/W_PC_VMFI.pkl'
    # ps_dict = load_pkl(pkl_file)
    # run(ps_dict, use_bev_iou=True, exp_name='W_PC_VMFI',veh_score_th=0.6,ped_score_th=0.4, waymo_cls_names=False)

    # pkl_file = '/MS3D/tools/cfgs/target_nuscenes/ps_labels_exp/W_PC_VMFI_TTA.pkl'
    # ps_dict = load_pkl(pkl_file)
    # run(ps_dict, use_bev_iou=True, exp_name='W_PC_VMFI_TTA',veh_score_th=0.6,ped_score_th=0.4, waymo_cls_names=False)

    # pkl_file = '/MS3D/tools/cfgs/target_waymo/exp_ps_dict/ps_labels/E39_L_TTA_VC_4.pkl'
    # ps_dict = load_pkl(pkl_file)
    # run(ps_dict, use_bev_iou=True, exp_name='E39_L_TTA_VC_4_bev',veh_score_th=0.6,ped_score_th=0.4)

    pkl_file = '/MS3D/tools/cfgs/target_waymo/ps_labels_rnd4/RND4_VOXA_VOXC.pkl'
    ps_dict = load_pkl(pkl_file)
    run(ps_dict, use_bev_iou=True, exp_name='rnd4_bev_ens',veh_score_th=0.8,ped_score_th=0.6)

    # pkl_file = '/MS3D/tools/cfgs/target_waymo/ps_labels_rnd4/final_ps_dict.pkl'
    # ps_dict = load_pkl(pkl_file)
    # run(ps_dict, use_bev_iou=True, exp_name='rnd4_bev_refined')
    # # run(ps_dict, use_bev_iou=False, exp_name='rnd4_3d_final')
