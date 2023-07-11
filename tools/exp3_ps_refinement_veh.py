import sys
sys.path.append('../')
import numpy as np
from pcdet.utils import generate_ps_utils
import argparse
import pickle
from pcdet.utils import tracker_utils
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
import yaml

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

def load_pkl(file):
    with open(file, 'rb') as f:
        loaded_pkl = pickle.load(f)
    return loaded_pkl

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')                   
    parser.add_argument('--cfg_file', type=str, default='/MS3D/tools/cfgs/dataset_configs/waymo_dataset_da.yaml',
                        help='just use the target dataset cfg file')
    parser.add_argument('--pos_th_veh', type=float, default=0.6, help='Vehicle detections above this threshold is used as pseudo-label')
    parser.add_argument('--pos_th_ped', type=float, default=0.5, help='Pedestrian detections above this threshold is used as pseudo-label')    
    parser.add_argument('--min_dets_for_tracks_all', type=int, default=3, help='Every track should have a minimum of N detections')
    parser.add_argument('--min_dets_for_tracks_static', type=int, default=3, help='Every track (static) should have a minimum of N detections')
    parser.add_argument('--ps_label_dir', type=str, default='/MS3D/tools/cfgs/target_waymo/ps_labels',
                        help='Folder to save intermediate ps label pkl files')
    
    # Configs for refining boxes of static vehicles
    parser.add_argument('--min_static_score', type=float, default=0.7, help='Minimum score for static boxes after refinement')
    parser.add_argument('--rolling_kde_window', type=int, default=16, help='Minimum score for static boxes after refinement')

    # Configs for propogating boxes
    parser.add_argument('--propagate_boxes_min_dets', type=int, default=7, help='Minimum number of static boxes in order to decide if we want to propagate boxes')
    parser.add_argument('--n_extra_frames', type=int, default=100, help='Number of frames to propagate') # prev 40 for 1.67Hz
    parser.add_argument('--degrade_factor', type=float, default=0.99, help='For every propagated frame, the box score will be multiplied by degrade factor') # prev 0.95 for 1.67Hz
    parser.add_argument('--min_score_clip', type=float, default=0.5, help='Set minimum score that the box can be degraded to. This is not so necessary (more for experiments)')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset = load_dataset(split='train')
    
    # Paths
    # ps_pth = '/MS3D/tools/cfgs/target_waymo/ps_labels/N_L_VMFI_TTA_PA_PC_VA_VC_64_WEIGHTED.pkl'
    # tracks_veh_all_pth = '/MS3D/tools/cfgs/target_waymo/ps_labels/N_L_VMFI_TTA_PA_PC_VA_VC_64_WEIGHTED_tracks_world_veh.pkl'
    # tracks_veh_static_pth = '/MS3D/tools/cfgs/target_waymo/ps_labels/N_L_VMFI_TTA_PA_PC_VA_VC_64_WEIGHTED_tracks_world_veh_static_iou2d.pkl'
    # tracks_ped_pth = '/MS3D/tools/cfgs/target_waymo/ps_labels/N_L_VMFI_TTA_PA_PC_VA_VC_64_WEIGHTED_tracks_world_ped.pkl'

    # ps_dict = load_pkl(ps_pth)
    # tracks_veh_all = load_pkl(tracks_veh_all_pth)
    # tracks_veh_static = load_pkl(tracks_veh_static_pth)
    # tracks_ped = load_pkl(tracks_ped_pth)


    trk_cfg_veh_static = '/MS3D/tracker/configs/ms3d_configs/veh_static_kf_iou.yaml'
    configs = yaml.load(open(trk_cfg_veh_static, 'r'), Loader=yaml.Loader)
    trk_score_th_static = configs['running']['score_threshold']

    # # Start refinement
    # tracker_utils.delete_tracks(tracks_veh_all, min_score=args.pos_th_veh, num_min_dets=args.min_dets_for_tracks_all)                   
    # tracker_utils.delete_tracks(tracks_veh_static, min_score=trk_score_th_static, num_min_dets=args.min_dets_for_tracks_static)   
    
    #  # Get static boxes using tracking information
    # for trk_id in tracks_veh_all.keys():
    #     score_mask = tracks_veh_all[trk_id]['boxes'][:,7] > args.pos_th_veh
    #     tracks_veh_all[trk_id]['motion_state'] = tracker_utils.get_motion_state(tracks_veh_all[trk_id]['boxes'][score_mask])    
    # for trk_id in tracks_veh_static.keys():
    #     score_mask = tracks_veh_static[trk_id]['boxes'][:,7] > trk_score_th_static
    #     tracks_veh_static[trk_id]['motion_state'] = tracker_utils.get_motion_state(tracks_veh_static[trk_id]['boxes'][score_mask])        

    # # Updates motion-state of track dicts in-place
    # matched_trk_ids = generate_ps_utils.motion_state_refinement(tracks_veh_all, tracks_veh_static, list(ps_dict.keys())) # ~ 1:03HR for 18840                

    # # Merge disjointed tracks and assign one box per frame in the ego-vehicle frame
    # generate_ps_utils.merge_disjointed_tracks(tracks_veh_all, tracks_veh_static, matched_trk_ids)    
    # generate_ps_utils.save_data(tracks_veh_all, args.ps_label_dir, name="tracks_all_world_refined.pkl")
    # generate_ps_utils.save_data(tracks_veh_static, args.ps_label_dir, name="tracks_static_world_refined.pkl")


    tracks_veh_static_pth = '/MS3D/tools/cfgs/target_waymo/ps_labels/tracks_static_world_refined.pkl'
    tracks_veh_static = load_pkl(tracks_veh_static_pth)
    generate_ps_utils.get_track_rolling_kde_interpolation(dataset, tracks_veh_static, window=args.rolling_kde_window, 
                                                              static_score_th=trk_score_th_static, kdebox_min_score=args.min_static_score)  # 23:20MIN for 18840
    generate_ps_utils.save_data(tracks_veh_static, args.ps_label_dir, name="tracks_static_world_rkde2.pkl")

    # generate_ps_utils.propagate_static_boxes(dataset, tracks_veh_static, 
    #                                                  score_thresh=trk_score_th_static,
    #                                                  min_dets=args.propagate_boxes_min_dets,
    #                                                  n_extra_frames=args.n_extra_frames, 
    #                                                  degrade_factor=args.degrade_factor, 
    #                                                  min_score_clip=args.min_score_clip) # < 1 min for 18840
    # generate_ps_utils.save_data(tracks_veh_static, args.ps_label_dir, name="tracks_static_world_proprkde_prop100_deg99_2.pkl")

    # frame2box_key = 'frameid_to_extrollingkde'
    # ps_pth = '/MS3D/tools/cfgs/target_waymo/ps_labels/N_L_VMFI_TTA_PA_PC_VA_VC_64_WEIGHTED.pkl'
    # tracks_veh_all_pth = '/MS3D/tools/cfgs/target_waymo/ps_labels/tracks_all_world_refined.pkl'    
    # ps_dict = load_pkl(ps_pth)
    # tracks_veh_all = load_pkl(tracks_veh_all_pth)
    
    # final_ps_dict = generate_ps_utils.update_ps(dataset, ps_dict, tracks_veh_all, tracks_veh_static, score_th=args.pos_th_veh, 
    #                                             frame2box_key_16f=frame2box_key, frame2box_key_1f='frameid_to_box', frame_ids=list(ps_dict.keys()))
    # generate_ps_utils.save_data(final_ps_dict, args.ps_label_dir, name="final_ps_dict_prop100_deg99.pkl")