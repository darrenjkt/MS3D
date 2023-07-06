import sys
sys.path.append('../')
import numpy as np
from pcdet.utils import box_fusion_utils, generate_ps_utils
import argparse
import pickle
from pcdet.utils import tracker_utils
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader

# For MS3D labels, re-map every class into super categories with index:category of 1:VEH/CAR, 2:PED, 3:CYC
# When we load in the labels for fine-tuning the specific detector, we can re-index it based on the pretrained class index
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')                   
    parser.add_argument('--cfg_file', type=str, default='/MS3D/tools/cfgs/dataset_configs/waymo_dataset_da.yaml',
                        help='just use the target dataset cfg file')
    parser.add_argument('--ps_dict', type=str, help='Use kbf ps_dict')
    parser.add_argument('--save_dir', type=str, default='/MS3D/tools/cfgs/target_waymo/ps_labels', help='where to save ps dict')    
    parser.add_argument('--cls_id', type=int, help='1: vehicle, 2: pedestrian, 3: cyclist')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset = load_dataset(split='train')
    ps_dict = None
    if args.ps_dict:
        with open(args.ps_dict, 'rb') as f:
            ps_dict = pickle.load(f)

    if args.cls_id == 1:
        trk_cfg = '/MS3D/tracker/configs/msda_configs/veh_kf_giou.yaml'
        save_fname = f"{Path(args.ps_dict).stem}_tracks_world_veh.pkl"
    elif args.cls_id == 2:
        trk_cfg = '/MS3D/tracker/configs/msda_configs/ped_kf_giou.yaml'
        save_fname = f"{Path(args.ps_dict).stem}_tracks_world_ped.pkl"
    elif args.cls_id == 3:
        trk_cfg = '/MS3D/tracker/configs/msda_configs/cyc_kf_giou.yaml'
        save_fname = f"{Path(args.ps_dict).stem}_tracks_world_cyc.pkl"
    else:
        print('Only support 3 classes at the moment (1: vehicle, 2: pedestrian, 3: cyclist)')
        raise NotImplementedError
    
    tracks_world = tracker_utils.get_tracklets(dataset, ps_dict, cfg_path=trk_cfg, cls_id=args.cls_id)

    generate_ps_utils.save_data(tracks_world, args.save_dir, name=save_fname)
    print(f"saved: {save_fname}\n")