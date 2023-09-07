"""
MS3D Step 1

DESCRIPTION:
    Combine detection sets of the ensemble with KBF. Saves a pkl file containing a list of 
    dicts where each dict contains a single detection set (and optional metadata) for each frame

EXAMPLE:
    python ensemble_kbf.py --ps_cfg /MS3D/tools/cfgs/target_nuscenes/ms3d_ps_config_rnd1.yaml \
        --dets_txt /MS3D/tools/cfgs/target_nuscenes/ps_ensembling/W_L_VMFI_TTA_PA_PC_VA_VC_64.txt
"""

import sys
sys.path.append('../')
import argparse
import yaml
from pcdet.utils import box_fusion_utils, ms3d_utils

# Super categories for MS3D compatibility across datasets
SUPERCATEGORIES = ['Vehicle','Pedestrian','Cyclist']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')                   
    parser.add_argument('--ps_cfg', type=str, help='cfg file with MS3D parameters')
    parser.add_argument('--interval', type=int, default=1, help='set interval')
    parser.add_argument('--len', type=int, default=1, help='set interval')
    parser.add_argument('--save_dir', type=str, default=None, help='Overwrite save dir in the cfg file')
    parser.add_argument('--exp_name', type=str, default=None, help='Overwrite exp_name in the cfg file')
    args = parser.parse_args()
    
    ms3d_configs = yaml.load(open(args.ps_cfg,'r'), Loader=yaml.Loader)

    # Load detection sets
    det_annos = box_fusion_utils.load_src_paths_txt(ms3d_configs['DETS_TXT'])
    num_det_sets = len(det_annos)-1 # don't count det_annos[det_cls_weights] in the length
    detection_sets = box_fusion_utils.get_detection_sets(det_annos, score_th=0.1)
    print('Number of detection sets: ', num_det_sets)

    # Downsample for debugging
    if args.interval > 1:
        detection_sets = detection_sets[::args.interval]
    if args.len > 1:
        detection_sets = detection_sets[:args.len]

    # Get class specific config
    cls_kbf_config = {}
    for enum, cls in enumerate(SUPERCATEGORIES):
        if cls in cls_kbf_config.keys():
            continue
        cls_kbf_config[cls] = {}
        cls_kbf_config[cls]['cls_id'] = enum+1 # in OpenPCDet, cls_ids enumerate from 1
        cls_kbf_config[cls]['discard'] = ms3d_configs['ENSEMBLE_KBF']['DISCARD'][enum]
        cls_kbf_config[cls]['radius'] = ms3d_configs['ENSEMBLE_KBF']['RADIUS'][enum]
        cls_kbf_config[cls]['nms'] = ms3d_configs['ENSEMBLE_KBF']['NMS'][enum]
        cls_kbf_config[cls]['neg_th'] = ms3d_configs['PS_SCORE_TH']['NEG_TH'][enum]

    # Combine detection sets into a single set of initial pseudo-labels
    ps_dict = box_fusion_utils.get_initial_pseudo_labels(detection_sets, cls_kbf_config)
    save_dir = ms3d_configs['SAVE_DIR'] if args.save_dir is None else args.save_dir
    exp_name = ms3d_configs['EXP_NAME'] if args.exp_name is None else args.exp_name
    ms3d_utils.save_data(ps_dict, save_dir, name=f"{exp_name}.pkl")
    print(f"saved: {exp_name}.pkl\n")