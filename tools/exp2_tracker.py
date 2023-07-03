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
    parser.add_argument('--trk_cfg', type=str, default='/MS3D/tracker/configs/msda_configs/msda_1frame_giou.yaml',
                        help='Config for the tracker')
    parser.add_argument('--ps_dict', type=str, help='Use kbf ps_dict')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset = load_dataset(split='train')
    print('Evaluating for the classes: ', cfg.CLASS_NAMES)

    ps_dict = None
    if args.ps_dict:
        with open(args.ps_dict, 'rb') as f:
            ps_dict = pickle.load(f)
    
    tracks_world = tracker_utils.get_tracklets(dataset, ps_dict, cfg_path=args.trk_cfg, anno_frames_only=False)

    generate_ps_utils.save_data(tracks_world, args.save_dir, name=f"{Path(args.ps_dict).stem}_tracks_world.pkl")
    print(f"saved: {Path(args.ps_dict).stem}_tracks_world.pkl\n")