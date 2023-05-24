import sys
sys.path.append('/MS3D')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils, box_fusion_utils
from pcdet.datasets import build_dataloader

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
dataset_cfg.DATA_SPLIT.test = 'train'
dataset_cfg.USE_CUSTOM_TRAIN_SCENES = True
dataset_cfg.SAMPLED_INTERVAL.test = 2

target_set, _, _ = build_dataloader(
            dataset_cfg=dataset_cfg,
            class_names=classes,
            batch_size=1, logger=logger, training=False, dist=False
        )

from pcdet.utils import box_fusion_utils
from tqdm import tqdm 
import numpy as np

dets_txt = '/MS3D/tools/cfgs/target_waymo/kbf_combinations/pvrcnn_c.txt'
print(f'classes: {cfg.DATA_CONFIG_TAR.CLASS_NAMES}')

ps_dict = {}
det_annos = box_fusion_utils.load_src_paths_txt(dets_txt)
print('Number of detectors: ', len(det_annos))
combined_dets = box_fusion_utils.combine_box_pkls(det_annos, score_th=0.1)

def get_ps_dict(ds_combined_dets, classes, cls_kbf_config, scale_conf_by_ndets, detector_weights=None):
    ps_dict = {}    
    box_weight_dict = {}
    for frame_boxes in tqdm(ds_combined_dets, total=len(ds_combined_dets)):        
            
        boxes_lidar = np.hstack([frame_boxes['boxes_lidar'],
                                 frame_boxes['class_ids'][...,np.newaxis],
                                 frame_boxes['score'][...,np.newaxis]])
        if detector_weights is not None:
            box_weight_dict['heading'] = detector_weights['heading'][frame_boxes['source_id']]
            box_weight_dict['centroid'] = detector_weights['centroid'][frame_boxes['source_id']]
        
        boxes_names = frame_boxes['names']
        unique_classes = np.unique(boxes_names)    
        ps_label_nms = []
        for cls in unique_classes:
            if cls not in classes:
                continue
            cls_mask = (boxes_names == cls)
            cls_boxes = boxes_lidar[cls_mask]
            score_mask = cls_boxes[:,8] > min_score
            
            cls_kbf_boxes, num_dets_per_box = box_fusion_utils.label_fusion(cls_boxes[score_mask], 'kde_fusion', 
                                           discard=cls_kbf_config[cls]['discard'], 
                                           radius=cls_kbf_config[cls]['radius'], 
                                           nms_thresh=cls_kbf_config[cls]['nms'], 
                                           weights=box_weight_dict)

            # E.g. car,truck,bus,motorcycle,bicycle,pedestrian (id: 1,2,3,4,5,6) -> Veh,Cyc,Ped (id: 1,4,6)
            cls_kbf_boxes[:,7] = cls_kbf_config[cls]['cls_id']

            # Scale confidence score by number of detections
            if scale_conf_by_ndets:
                coeff = conf_scaling_coeff(num_dets_per_box, num_boxes_at_unity)
                temp_frame_id = frame_boxes['frame_id']
                mask = cls_kbf_boxes[:,8] < cfg.SELF_TRAIN.SCORE_THRESH
                cls_kbf_boxes[mask,8] *= coeff[mask]
                cls_kbf_boxes[mask,8] = np.clip(cls_kbf_boxes[mask,8], 0, max_scale_score)

            ps_label_nms.extend(cls_kbf_boxes)

        if ps_label_nms:
            ps_label_nms = np.array(ps_label_nms)
        else:
            ps_label_nms = np.empty((0,9))

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
    
    return ps_dict

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
        
    return np.clip(scaling, a_max=max_scaling, a_min=1)

# ['Vehicle', 'Vehicle', 'Vehicle', 'Cyclist', 'Cyclist', 'Pedestrian']
classes=target_set.dataset_cfg.CLASS_NAMES
discard=[4]*6
radius=[1,1,1, 0.3,0.3, 0.2]
kbf_nms=[0.1,0.1,0.1,0.5,0.5,0.5,0.5]
detector_weights = {}
detector_weights['heading'] = np.array([1,1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,3,3,3,3,3,3,3,3])
detector_weights['centroid'] = np.array([2,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,1,1,1,1,1,1,1,1])
min_score = 0.3

# confidence scaling
scale_conf_by_ndets = True
num_boxes_at_unity = int(len(det_annos) * 0.75)
max_scale_score = cfg.SELF_TRAIN.SCORE_THRESH + 0.1

# Downsample
# ds_combined_dets = combined_dets[::3]

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

ps_dict = get_ps_dict(combined_dets, ['Cyclist'], cls_kbf_config, 
                      scale_conf_by_ndets=scale_conf_by_ndets, detector_weights=detector_weights)