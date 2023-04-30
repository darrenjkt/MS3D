
import pickle
import sys
sys.path.append('/MS3D')
import copy
import argparse
from pathlib import Path
from eval_utils import cross_domain_eval_tools as cd_eval
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader

"""
python evaluation.py --cfg_file /MS3D/tools/cfgs/source-waymo/second_iou.yaml \
                     --pkl /MS3D/output/source-waymo/second_iou/1sweep/eval/epoch_30/val/waymo_1sweep_interval10/result.pkl \
                     --metric waymo
                     --veh_iou 0.7
python evaluation.py --cfg_file cfgs/source-waymo/secondiou.yaml \
                    --pkl ../output/source-waymo/secondiou/default/eval/epoch_8794/val/once/result.pkl \
                    --metric kitti                     
"""

def waymo_eval(eval_det_annos, eval_gt_annos, class_names, static_dyn=False, **kwargs):
    from pcdet.datasets.waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
    eval = OpenPCDetWaymoDetectionMetricsEstimator()
    veh_iou_threshold = kwargs['veh_iou_threshold'] if 'veh_iou_threshold' in kwargs else 0.7
    ped_iou_threshold = kwargs['ped_iou_threshold'] if 'ped_iou_threshold' in kwargs else 0.5
    eval_breakdown = kwargs['eval_breakdown'] if 'eval_breakdown' in kwargs else 'RANGE'

    ap_dict = eval.waymo_evaluation(
        eval_det_annos, eval_gt_annos, class_name=class_names,
        distance_thresh=1000, fake_gt_infos=False, 
        veh_iou_threshold=veh_iou_threshold, ped_iou_threshold=ped_iou_threshold,
        breakdown_generator_ids=eval_breakdown
    )
    ap_result_str = '\n'
    for key in ap_dict:
        ap_dict[key] = ap_dict[key][0]
        ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

    return ap_result_str, ap_dict

def kitti_eval(eval_det_annos, eval_gt_annos, class_names, **kwargs):
    from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

    ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
        gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=class_names
    )
    return ap_result_str, ap_dict   


    
def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='',
                        help='specify the config for demo')
    parser.add_argument('--pkl', type=str, default='',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--veh_iou', type=float, default=0.7,
                        help='iou threshold for vehicle evaluation')
    parser.add_argument('--ped_iou', type=float, default=0.5,
                        help='iou threshold for vehicle evaluation') 
    parser.add_argument('--eval_breakdown', type=str, default='RANGE',
                        help='RANGE or OBJECT_TYPE')                         
    parser.add_argument('--csv_pth', type=str, default='',
                        help='specify the csv file to save results')
    parser.add_argument('--shift_coor', action='store_true',
                        help='use shift coordinate frame')                          

    args = parser.parse_args()
    log_file = 'temp.txt'

    cfg_from_yaml_file(args.cfg_file, cfg)
    # cfg.DATA_CONFIG_TAR.DATA_SPLIT.test='train'
    if cfg.get('DATA_CONFIG_TAR', None):
        data_config = cfg.DATA_CONFIG_TAR
        cls_names = data_config.CLASS_NAMES
    else:
        data_config = cfg.DATA_CONFIG
        cls_names = cfg.CLASS_NAMES

    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    dataset, _, _ = build_dataloader(
            dataset_cfg=data_config,
            class_names=cls_names,
            batch_size=1, logger=logger, training=False, dist=False, workers=1
        )
    
    # Load detection pickle
    with open(args.pkl,'rb') as f:
        det_annos = pickle.load(f)

    # If I saved in ground frame, we need -SHIFT_COOR. If I didn't then comment this out
    if args.shift_coor:
        for anno in det_annos: # We only SHIFT_COOR for predictions
            anno['boxes_lidar'][:,:3] -= data_config.SHIFT_COOR 

    if data_config.DATASET in ['NuScenesDataset', 'LyftDataset']:
        eval_gt_annos = copy.deepcopy(dataset.infos) 
    else:
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in dataset.infos] # Waymo

    # We map all car/bus/truck to vehicle to have consistency across datasets
    eval_class_names = ['Vehicle', 'Pedestrian', 'Cyclist']

    mod_gt_annos = cd_eval.transform_to_waymo_format(data_config.DATASET, eval_gt_annos, is_gt=True)
    mod_det_annos = cd_eval.transform_to_waymo_format(data_config.DATASET, det_annos, is_gt=False)
    ap_result_str, ap_dict = waymo_eval(mod_det_annos, mod_gt_annos, eval_class_names, 
                                        veh_iou_threshold=args.veh_iou, ped_iou_threshold=args.ped_iou, eval_breakdown=args.eval_breakdown)    
    item_key = 'SOURCE\tTARGET\tMODEL\t'
    item_res = f'{data_config.DATASET.replace("Dataset","")}\t{data_config.DATASET.replace("Dataset","")}\t{cfg.MODEL.NAME}\t'
    for k,v in ap_dict.items():
        if ('VEHICLE' in k) or ('PEDESTRIAN' in k) or ('CYCLIST' in k):
            key = k[11:].replace('LEVEL_','L').replace('PEDESTRIAN','PED').replace('VEHICLE','VEH').replace('CYCLIST','CYC').replace(' ','')
            item_key += f'{key}\t'
            item_res += f'{ap_dict[k]*100:0.2f}\t'
    item_key += '\n'
    item_res += '\n'
    print(item_key)
    print(item_res)

    if args.csv_pth != '':
        csv_path = Path(args.csv_pth)
        if csv_path.exists():
            with open(csv_path, 'a') as f:
                f.writelines(item_res)
        else:
            with open(csv_path, 'w') as f:
                f.writelines(item_key)
                f.writelines(item_res)


if __name__ == '__main__':
    main()
