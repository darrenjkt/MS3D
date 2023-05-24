import sys
sys.path.append('/MS3D')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
import argparse
import numpy as np
from tqdm import tqdm
from pcdet.utils import box_fusion_utils
import copy
from pathlib import Path
import time

def conf_scaling_coeff(num_boxes, num_boxes_at_unity, eq_type='sqrt', max_scaling=2.0):
    """
    This function returns the scaling coefficient to scale the confidence score based on
    the number of boxes proposed for an object. Two scaling equations are provided.
    
    Note: This might be bad if many detectors detect but wrongly detect -> adjust max_scaling according
    
    num_boxes (list of ints): number of input boxes for each kbf box fusion
    num_boxes_at_unity (int): The point at which the scaling coeff is equal to 1. 
                        Less than this and coeff < 1; More than this and coeff > 1.
    """
    if eq_type == 'sqrt':
        # Non-linear scaling with sqrt(x)        
        scaling = (1/np.sqrt(num_boxes_at_unity)) * np.sqrt(num_boxes) 
        
    elif eq_type == 'linear':
        scaling = (1/num_boxes_at_unity) * num_boxes
        
    else:
        raise NotImplementedError
        
    return np.clip(scaling, a_max=max_scaling, a_min=0)

def prepare_fused_eval(ps_dict, dataset):
    pred_annos, gt_annos = [], []
    eval_gt_annos = copy.deepcopy(dataset.infos)
    for frame_id in tqdm(ps_dict.keys(), total=len(ps_dict.keys())):  
        boxes = ps_dict[frame_id]['gt_boxes'].copy()            
        boxes[:,:3] -= dataset.dataset_cfg.SHIFT_COOR # just for eval in this notebook
        p_anno = {"frame_id": frame_id,
                "name": np.array([dataset.dataset_cfg.CLASS_NAMES[int(abs(box[7]))-1] for box in boxes]),
                "pred_labels": np.array([abs(box[7]) for box in boxes]),
                "boxes_lidar": boxes[:,:7],
                "score": boxes[:,8]}
        pred_annos.append(p_anno)
        gt_annos.append(eval_gt_annos[dataset.frameid_to_idx[frame_id]]['annos'])
    return pred_annos, gt_annos

def prepare_single_eval(det_anno, dataset):
        # Evaluate single pred annos
    # baseline = det_annos['lyft_models.uda_pv_rcnn_plusplus_resnet_anchorhead.custom190_lyft10xyzt_waymo4xyzt_notta'][::3]
    pred_annos, gt_annos = [], []
    eval_gt_annos = copy.deepcopy(dataset.infos)
    for p_anno in tqdm(det_anno, total=len(det_anno)):  
        p_anno['boxes_lidar'][:,:3] -= dataset.dataset_cfg.SHIFT_COOR # just for eval in this notebook
        pred_annos.append(p_anno)
        gt_annos.append(eval_gt_annos[dataset.frameid_to_idx[p_anno['frame_id']]]['annos'])    
    return pred_annos, gt_annos

def pretty_print(ap_dict):
    item_key = 'SOURCE\tTARGET\tMODEL\t'
    item_res = f'{cfg.DATA_CONFIG.DATASET.replace("Dataset","")}\t{cfg.DATA_CONFIG_TAR.DATASET.replace("Dataset","")}\t{cfg.MODEL.NAME}\t'
    for k,v in ap_dict.items():
        if ('VEHICLE' in k) or ('PEDESTRIAN' in k) or ('CYCLIST' in k):
            key = k[11:].replace('LEVEL_','L').replace('PEDESTRIAN','PED').replace('VEHICLE','VEH').replace('CYCLIST','CYC').replace(' ','')
            item_key += f'{key}\t'
            item_res += f'{ap_dict[k][0]*100:0.2f}\t'
    item_key += '\n'
    item_res += '\n'
    return item_key, item_res

def waymo_eval(pred_annos, gt_annos, dataset):
    from pcdet.datasets.waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
    eval = OpenPCDetWaymoDetectionMetricsEstimator()

    ap_dict = eval.waymo_evaluation(
        pred_annos, gt_annos, class_name=dataset.dataset_cfg.CLASS_NAMES,
        distance_thresh=1000, fake_gt_infos=dataset.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
    )
    return ap_dict

#### --------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dets_txt', type=str, default=None,
                        help='specify the config for demo')
    parser.add_argument('--discard', type=int, required=True)
    parser.add_argument('--downsample', type=int, default=1)
    parser.add_argument('--pkl', type=str, default=None)
    args = parser.parse_args()

    cfg_file = '/MS3D/tools/cfgs/target_waymo/ms3d_nuscenes_voxel_rcnn_centerhead.yaml'

    # Get target dataset
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger('temp.txt', rank=cfg.LOCAL_RANK)
    if cfg.get('DATA_CONFIG_TAR', False):
        dataset_cfg = cfg.DATA_CONFIG_TAR
        classes = cfg.DATA_CONFIG_TAR.CLASS_NAMES
        cfg.DATA_CONFIG_TAR.USE_PSEUDO_LABEL=False
        if dataset_cfg.get('USE_TTA', False):
            dataset_cfg.USE_TTA=False
    else:
        dataset_cfg = cfg.DATA_CONFIG
        classes = cfg.CLASS_NAMES

    dataset_cfg.USE_CUSTOM_TRAIN_SCENES = True
    dataset_cfg.DATA_SPLIT.test = 'train'
    dataset_cfg.SAMPLED_INTERVAL.test = 2

    target_set, _, _ = build_dataloader(
                dataset_cfg=dataset_cfg,
                class_names=classes,
                batch_size=1, logger=logger, training=False, dist=False
            )

    if args.dets_txt is not None:
        ps_dict = {}
        det_annos = box_fusion_utils.load_src_paths_txt(args.dets_txt)
        print('Number of detectors: ', len(det_annos))
        combined_dets = box_fusion_utils.combine_box_pkls(det_annos, cfg.DATA_CONFIG_TAR.CLASS_NAMES)
        classes=target_set.dataset_cfg.CLASS_NAMES
        discard=[args.discard]*6
        radius=[1,1,1, 0.3,0.3, 0.2]
        kbf_nms=[0.1,0.1,0.1,0.5,0.5,0.5,0.5]
        min_score = 0.3

        # confidence scaling
        scale_conf_by_ndets = False
        num_boxes_at_unity = 4
        max_scale_score = cfg.SELF_TRAIN.SCORE_THRESH + 0.1

        # Downsample
        ds_combined_dets = combined_dets[::args.downsample]

        # Get class specific config
        cls_kbf_config = {}
        for enum, cls in enumerate(classes):
            if cls in cls_kbf_config.keys():
                continue
            cls_kbf_config[cls] = {}
            cls_kbf_config[cls]['cls_id'] = enum+1 # in OpenPCDet, cls_ids enumerate from 1
            cls_kbf_config[cls]['discard'] = discard[enum]
            cls_kbf_config[cls]['radius'] = radius[enum]
            cls_kbf_config[cls]['nms'] = kbf_nms[enum]

        ps_dict = {}    
        t0 = time.time()
        for frame_boxes in tqdm(ds_combined_dets, total=len(ds_combined_dets)):
                
            boxes_lidar = np.hstack([frame_boxes['boxes_lidar'],
                                    frame_boxes['class_ids'][...,np.newaxis],
                                    frame_boxes['score'][...,np.newaxis]])
            
            boxes_names = frame_boxes['names']
            unique_classes = np.unique(boxes_names)    
            ps_label_nms = []
            for cls in unique_classes:
                cls_mask = (boxes_names == cls)
                cls_boxes = boxes_lidar[cls_mask]
                score_mask = cls_boxes[:,8] > min_score
                cls_kbf_boxes, num_dets_per_box = box_fusion_utils.label_fusion(cls_boxes[score_mask], 'kde_fusion', 
                                            discard=cls_kbf_config[cls]['discard'], 
                                            radius=cls_kbf_config[cls]['radius'], 
                                            nms_thresh=cls_kbf_config[cls]['nms'])
                
                # E.g. car,truck,bus,motorcycle,bicycle,pedestrian (id: 1,2,3,4,5,6) -> Veh,Cyc,Ped (id: 1,4,6)
                cls_kbf_boxes[:,7] = cls_kbf_config[cls]['cls_id']
                
                # Scale confidence score by number of detections
                if scale_conf_by_ndets:
                    coeff = conf_scaling_coeff(num_dets_per_box, num_boxes_at_unity)
                    mask = cls_kbf_boxes[:,8] < cfg.SELF_TRAIN.SCORE_THRESH
                    cls_kbf_boxes[mask,8] *= coeff[mask]
                    cls_kbf_boxes[mask,8]= np.clip(cls_kbf_boxes[mask,8], 0, max_scale_score)
                    
                ps_label_nms.extend(cls_kbf_boxes)
                
            ps_label_nms = np.array(ps_label_nms)
            
            # neg_th < score < pos_th: ignore for training but keep for update step
            pred_boxes = ps_label_nms[:,:7]
            pred_labels = ps_label_nms[:,7]
            pred_scores = ps_label_nms[:,8]
            ignore_mask = pred_scores < cfg.SELF_TRAIN.SCORE_THRESH
            pred_labels[ignore_mask] = -pred_labels[ignore_mask]
            gt_box = np.concatenate((pred_boxes,
                                    pred_labels.reshape(-1, 1),
                                    pred_scores.reshape(-1, 1)), axis=1)    

            gt_infos = {
                'gt_boxes': gt_box,
            }
            ps_dict[frame_boxes['frame_id']] = gt_infos

        time_to_fuse = time.time() - t0
        from pcdet.utils import generate_ps_utils 
        generate_ps_utils.save_data(ps_dict, '/MS3D/tools/cfgs/target_waymo/kbf_combinations', name=f"{Path(args.dets_txt).stem}.pkl")

        # Evaluation
        pred_annos, gt_annos = prepare_fused_eval(ps_dict, target_set)
        exp_name = args.dets_txt

    elif args.pkl is not None:
        import pickle
        with open(args.pkl,'rb') as f:
            det_annos = pickle.load(f)
        
        pred_annos, gt_annos = prepare_single_eval(ps_dict, target_set)
        exp_name = args.pkl

    ap_dict = waymo_eval(pred_annos, gt_annos, target_set)    
    
    mystr = f'\nEVALUATED: [discard:{args.discard},data_len:{len(pred_annos)},time_taken={time_to_fuse:0.3f}] {exp_name}\n'
    item_key, item_res = pretty_print(ap_dict)

    out_txt = 'kbf_param_sel_output.txt'
    if Path(out_txt).exists():
        with open(out_txt, 'a') as f:
            f.writelines(mystr)
            f.writelines(item_key)
            f.writelines(item_res)
    else:
        with open(out_txt, 'w') as f:
            f.writelines(mystr)
            f.writelines(item_key)
            f.writelines(item_res)

if __name__ == '__main__':
    main()    