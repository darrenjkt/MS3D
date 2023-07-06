import sys
sys.path.append('../')
import numpy as np
from pcdet.utils import box_fusion_utils, generate_ps_utils
import argparse
from tqdm import tqdm
from pathlib import Path

# For MS3D labels, re-map every class into super categories with index:category of 1:VEH/CAR, 2:PED, 3:CYC
# When we load in the labels for fine-tuning the specific detector, we can re-index it based on the pretrained class index
SUPERCATEGORIES = ['Vehicle','Pedestrian','Cyclist']
SUPER_MAPPING = {'car': 'Vehicle',
                'truck': 'Vehicle',
                'bus': 'Vehicle',
                'Vehicle': 'Vehicle',
                'pedestrian': 'Pedestrian',
                'Pedestrian': 'Pedestrian',
                'motorcycle': 'Vehicle', # Waymo maps motorcycle to Vehicle
                'bicycle': 'Cyclist',
                'Cyclist': 'Cyclist'}

def get_multi_source_prelim_label(detection_sets, cls_kbf_config): 
    """
    This has been modified from the function in generate_ps_utils.py to now include
    various classes
    """
    ps_dict = {}

    for frame_boxes in tqdm(detection_sets, total=len(detection_sets), desc='get initial label set'):

        boxes_lidar = np.hstack([frame_boxes['boxes_lidar'],
                                 frame_boxes['class_ids'][...,np.newaxis],
                                 frame_boxes['score'][...,np.newaxis],
                                 frame_boxes['box_weights'][...,np.newaxis]])

        ps_label_nms = []
        for class_name in SUPERCATEGORIES:

            cls_mask = (frame_boxes['names'] == class_name)
            cls_boxes = boxes_lidar[cls_mask]
            cls_boxes = cls_boxes[cls_boxes[:,9] > 0] # Discard if box weight == 0
            if cls_boxes.shape[0] == 0:
                continue
            score_mask = cls_boxes[:,8] > cls_kbf_config[class_name]['neg_th']
            cls_kbf_boxes = box_fusion_utils.label_fusion(cls_boxes[score_mask],
                                           discard=cls_kbf_config[class_name]['discard'], 
                                           radius=cls_kbf_config[class_name]['radius'], 
                                           nms_thresh=cls_kbf_config[class_name]['nms'], 
                                           use_box_weights=True)            

            ignore_mask = cls_kbf_boxes[:,8] < cls_kbf_config[class_name]['pos_th']
            cls_kbf_boxes[:,7][ignore_mask] = -cls_kbf_boxes[:,7][ignore_mask]
            ps_label_nms.extend(cls_kbf_boxes)

        if ps_label_nms:
            ps_label_nms = np.array(ps_label_nms)
        else:
            ps_label_nms = np.empty((0,9))
            
        # neg_th < score < pos_th: ignore for training but keep for update step
        pred_boxes = ps_label_nms[:,:7]
        pred_labels = ps_label_nms[:,7]
        pred_scores = ps_label_nms[:,8]
        gt_box = np.concatenate((pred_boxes,
                                pred_labels.reshape(-1, 1),
                                pred_scores.reshape(-1, 1)), axis=1)    
        
        gt_infos = {
            'gt_boxes': gt_box,
        }
        ps_dict[frame_boxes['frame_id']] = gt_infos
    return ps_dict

def get_detection_sets(det_annos, score_th=0.1):
    """
    This function returns a list where each element contains all the detection sets for one frame.
    When we generate the detection sets with test.py, we map the class names to the target domain, but not the class_id.

    No matter the target domain, combine all detection sets as veh,ped,cyc with index 1,2,3 first. Later on, when we load in the labels for 
    training, we'll adapt it for the source pretrained model's classes with the DATA_CONFIG.CLASS_NAMES. This is because we need to maintain 
    the original pre-trained detector's class indexing. If motorcycle was idx=4, we should use cyclist as idx=4 at training.
    """
    detection_sets = []
    src_keys = list(det_annos.keys())
    src_keys.remove('det_cls_weights')
    len_data = len(det_annos[src_keys[0]])
    for idx in tqdm(range(len_data), total=len_data, desc='load detection sets'):
        frame_dets = {}
        frame_dets['boxes_lidar'], frame_dets['score'], frame_dets['source'], frame_dets['source_id'], frame_dets['frame_id'], frame_dets['class_ids'], frame_dets['names'], frame_dets['box_weights'] = [],[],[],[],[],[],[],[]
        for src_id, key in enumerate(src_keys):
            frame_dets['frame_id'] = det_annos[key][idx]['frame_id']
            score_mask = det_annos[key][idx]['score'] > score_th            
            frame_dets['boxes_lidar'].extend(det_annos[key][idx]['boxes_lidar'][score_mask])
            frame_dets['score'].extend(det_annos[key][idx]['score'][score_mask])
            frame_dets['source'].extend([key for i in range(len(det_annos[key][idx]['score'][score_mask]))])
            frame_dets['source_id'].extend([src_id for i in range(len(det_annos[key][idx]['score'][score_mask]))])
            
            # Remap every class_id to the supercategory class_id of 1:veh/car, 2:ped, 3:cyc
            # TODO: Confirm that this works for every target domain
            pred_names = det_annos[key][idx]['name'][score_mask]
            superclass_ids = np.array([SUPERCATEGORIES.index(SUPER_MAPPING[name])+1 for name in pred_names])
            frame_dets['class_ids'].extend(superclass_ids)
            frame_dets['names'].extend(pred_names)
            det_cls_weights = det_annos['det_cls_weights'][key]
            frame_dets['box_weights'].extend(np.array([det_cls_weights[cid-1] for cid in superclass_ids]))

        frame_dets['boxes_lidar'] = np.vstack(frame_dets['boxes_lidar']) if len(frame_dets['score']) != 0  else np.array([]).reshape(-1,7)
        frame_dets['score'] = np.hstack(frame_dets['score']) if len(frame_dets['score']) != 0 else np.array([])
        frame_dets['source'] = np.array(frame_dets['source']) if len(frame_dets['score']) != 0 else np.array([])
        frame_dets['source_id'] = np.array(frame_dets['source_id']) if len(frame_dets['score']) != 0 else np.array([])
        frame_dets['class_ids'] = np.array(frame_dets['class_ids'], dtype=np.int32) if len(frame_dets['score']) != 0 else  np.array([])
        frame_dets['names'] = np.array(frame_dets['names']) if len(frame_dets['score']) != 0 else  np.array([])
        frame_dets['box_weights'] = np.array(frame_dets['box_weights'], dtype=np.int32) if len(frame_dets['score']) != 0 else  np.array([])
        detection_sets.append(frame_dets)
        assert frame_dets['class_ids'].shape == frame_dets['score'].shape

    return detection_sets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')                   
    parser.add_argument('--save_dir', type=str, default='/MS3D/tools/cfgs/target_waymo/exp_ps_dict/ps_labels', help='where to save ps dict')    
    parser.add_argument('--dets_txt', type=str, help='Use kbf ps_dict')
    parser.add_argument('--interval', type=int, default=1, help='set interval')
    args = parser.parse_args()
    
    # Load detection sets
    det_annos = box_fusion_utils.load_src_paths_txt(args.dets_txt)
    num_det_sets = len(det_annos)-1 # minus det_cls_weights
    detection_sets = get_detection_sets(det_annos, score_th=0.1)
    print('Number of detection sets: ', num_det_sets)

    # TODO: Load in detection sets with weights. If no integers after the file name, assume weight=1.

    # Downsample for debugging
    detection_sets = detection_sets[::args.interval] # ::3 is 6280, ::2 is 9420. 9420 has closer results to the full 18840

    # Class-specific hyper parameters
    pos_th=[0.6,0.4,0.4]
    neg_th=[0.3,0.15,0.15]

    discard=[4,4,4] if num_det_sets >= 8 else [0,0,0] # 4 is good default
    radius=[1, 0.3, 0.2] # should not need to change
    kbf_nms=[0.1,0.5,0.5] # should not need to change

    # Get class specific config
    cls_kbf_config = {}
    for enum, cls in enumerate(SUPERCATEGORIES):
        if cls in cls_kbf_config.keys():
            continue
        cls_kbf_config[cls] = {}
        cls_kbf_config[cls]['cls_id'] = enum+1 # in OpenPCDet, cls_ids enumerate from 1
        cls_kbf_config[cls]['discard'] = discard[enum]
        cls_kbf_config[cls]['radius'] = radius[enum]
        cls_kbf_config[cls]['nms'] = kbf_nms[enum]
        cls_kbf_config[cls]['pos_th'] = pos_th[enum]
        cls_kbf_config[cls]['neg_th'] = neg_th[enum]

    ps_dict = get_multi_source_prelim_label(detection_sets, cls_kbf_config)
    generate_ps_utils.save_data(ps_dict, args.save_dir, name=f"{Path(args.dets_txt).stem}.pkl")
    print(f"saved: {Path(args.dets_txt).stem}.pkl\n")