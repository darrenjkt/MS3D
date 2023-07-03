import numpy as np
import copy
import sys
sys.path.append('../')
import argparse
from tqdm import tqdm
import pickle
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader

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

def load_dataset(split):

    # Get target dataset    
    cfg.DATA_SPLIT.test = split
    cfg.SAMPLED_INTERVAL.test = 1
    logger = common_utils.create_logger('temp.txt', rank=cfg.LOCAL_RANK)
    target_set, _, _ = build_dataloader(
                dataset_cfg=cfg,
                class_names=cfg.CLASS_NAMES,
                batch_size=1, logger=logger, training=False, dist=False, workers=1
            )      
    return target_set

def pretty_print(ap_dict):
    item_key = 'SOURCE\tTARGET\t'
    item_res = f'{cfg.DATASET.replace("Dataset","")}\t{cfg.DATASET.replace("Dataset","")}\t'
    for k,v in ap_dict.items():
        if ('VEHICLE' in k) or ('PEDESTRIAN' in k) or ('CYCLIST' in k):
            key = k[11:].replace('LEVEL_','L').replace('PEDESTRIAN','PED').replace('VEHICLE','VEH').replace('CYCLIST','CYC').replace(' ','')
            item_key += f'{key}\t'
            item_res += f'{ap_dict[k][0]*100:0.2f}\t'
    item_key += '\n'
    item_res += '\n'
    print(item_key)
    print(item_res)
    return item_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/MS3D/tools/cfgs/dataset_configs/waymo_dataset_da.yaml',
                        help='just use the target dataset cfg file')
    parser.add_argument('--ps_dict', type=str, help='Use kbf ps_dict')
    parser.add_argument('--det_pkl', type=str, help='result.pkl from test.py')
    parser.add_argument('--interval', type=int, default=1, help='set interval')
    args = parser.parse_args()
    
    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset = load_dataset(split='train')
    print('Evaluating for the classes: ', cfg.CLASS_NAMES)

    ps_dict = None
    if args.ps_dict:
        with open(args.ps_dict, 'rb') as f:
            ps_dict = pickle.load(f)

        # Evaluate
        pred_annos, gt_annos = [], []
        eval_gt_annos = copy.deepcopy(dataset.infos)
        for frame_id in tqdm(ps_dict.keys(), total=len(ps_dict.keys())):  
            boxes = ps_dict[frame_id]['gt_boxes'].copy()            
            boxes[:,:3] -= dataset.dataset_cfg.SHIFT_COOR # Translate ground frame bboxes to dataset label sensor frame
            p_anno = {"frame_id": frame_id,
                    "name": np.array([cfg.CLASS_NAMES[int(abs(box[7]))-1] for box in boxes]), # TODO: map supercategory to dataset specific class names
                    "pred_labels": np.array([abs(box[7]) for box in boxes]),
                    "boxes_lidar": boxes[:,:7],
                    "score": boxes[:,8]}
            
            # min threshold filter
            # veh_mask = np.zeros(len(p_anno['name']), dtype=bool)
            # cls_veh_mask = p_anno['name'] == 'Vehicle'
            # veh_mask.flat[np.flatnonzero(cls_veh_mask)[p_anno['score'][cls_veh_mask] > 0.6]] = True

            # ped_mask = np.zeros(len(p_anno['name']), dtype=bool)
            # cls_ped_mask = p_anno['name'] == 'Pedestrian'
            # ped_mask.flat[np.flatnonzero(cls_ped_mask)[p_anno['score'][cls_ped_mask] > 0.4]] = True

            # cyc_mask = np.zeros(len(p_anno['name']), dtype=bool)
            # cls_cyc_mask = p_anno['name'] == 'Cyclist'
            # cyc_mask.flat[np.flatnonzero(cls_cyc_mask)[p_anno['score'][cls_cyc_mask] > 0.4]] = True
                    
            # combined_mask = np.logical_or.reduce((veh_mask, ped_mask, cyc_mask))
            # for key in p_anno.keys():
            #     if key == 'frame_id':
            #         continue
            #     p_anno[key] = p_anno[key][combined_mask]

            pred_annos.append(p_anno)
            gt_annos.append(eval_gt_annos[dataset.frameid_to_idx[frame_id]]['annos'])
    
    if args.det_pkl:
        with open(args.det_pkl, 'rb') as f:
            det_pkl = pickle.load(f)

        det_pkl = det_pkl[::args.interval] # ::2 is 18840 -> 9420

        # Evaluate single pred annos
        pred_annos, gt_annos = [], []
        eval_gt_annos = copy.deepcopy(dataset.infos)
        for p_anno in tqdm(det_pkl, total=len(det_pkl)):  
            p_anno['boxes_lidar'][:,:3] -= dataset.dataset_cfg.SHIFT_COOR # Translate ground frame bboxes to dataset label sensor frame
            pred_annos.append(p_anno)
            gt_annos.append(eval_gt_annos[dataset.frameid_to_idx[p_anno['frame_id']]]['annos'])            

    from pcdet.datasets.waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
    eval = OpenPCDetWaymoDetectionMetricsEstimator()

    ap_dict = eval.waymo_evaluation(
        pred_annos, gt_annos, class_name=cfg.CLASS_NAMES,
        distance_thresh=1000, fake_gt_infos=dataset.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
    )
    item_result = pretty_print(ap_dict)
    if args.ps_dict:        
        print('Evaluated: ', args.ps_dict)
        item_result += f'Evaluated: {args.ps_dict}\n'
    if args.det_pkl:
        print('Evaluated: ', args.det_pkl)        
        item_result += f'Evaluated: {args.det_pkl}\n'

    with open('/MS3D/tools/cfgs/target_waymo/exp_ps_dict/r1_combinations/E1_17_results.txt', 'a') as f:
        f.write(item_result)