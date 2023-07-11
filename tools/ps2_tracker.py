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
import yaml 

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
    parser.add_argument('--ps_cfg', type=str, help='cfg file with MS3D parameters')
    parser.add_argument('--cls_id', type=int, help='1: vehicle, 2: pedestrian, 3: cyclist')
    parser.add_argument('--static_veh', action='store_true', default=False)
    parser.add_argument('--trk_cfg', type=str, default=None, help='overwrite default track configs')
    parser.add_argument('--save_name', type=str, default=None, help='overwrite default save name')

    args = parser.parse_args()

    ms3d_configs = yaml.load(open(args.ps_cfg,'r'), Loader=yaml.Loader)
    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset = load_dataset(split='train')

    ps_dict_pth = Path(ms3d_configs["save_dir"]) / f'{ms3d_configs["exp_name"]}.pkl'
    with open(ps_dict_pth, 'rb') as f:
        ps_dict = pickle.load(f)

    if args.cls_id == 1:        
        if args.static_veh:
            trk_cfg = ms3d_configs['tracking']['veh_static_cfg']
            save_fname = f"{ms3d_configs['exp_name']}_tracks_world_veh_static.pkl"            

            # # Downsample from 5Hz to 1.67Hz (i.e. skip 6)
            # selected_ids = list(range(0,len(ps_dict.keys()),3))
            # ds_ps_dict = {}
            # for i,k in enumerate(ps_dict.keys()):
            #     if i in selected_ids:
            #         ds_ps_dict[k] = ps_dict[k]
            # ps_dict = ds_ps_dict.copy()
            
        else:
            trk_cfg = ms3d_configs['tracking']['veh_all_cfg']
            save_fname = f"{ms3d_configs['exp_name']}_tracks_world_veh.pkl"
    elif args.cls_id == 2:
        trk_cfg = ms3d_configs['tracking']['ped_cfg']
        save_fname = f"{ms3d_configs['exp_name']}_tracks_world_ped.pkl"
    else:
        print('Only support 2 classes at the moment (1: vehicle, 2: pedestrian)')
        raise NotImplementedError    
    
    trk_cfg = args.trk_cfg if args.trk_cfg is not None else trk_cfg    
    save_fname = args.save_name if args.save_name is not None else save_fname
    
    tracks_world = tracker_utils.get_tracklets(dataset, ps_dict, cfg_path=trk_cfg, cls_id=args.cls_id)

    generate_ps_utils.save_data(tracks_world, ms3d_configs["save_dir"], name=save_fname)
    print(f"saved: {save_fname}\n")