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
from tqdm import tqdm

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
    parser.add_argument('--min_dets_for_veh_tracks_all', type=int, default=3, help='Every track should have a minimum of N detections')
    parser.add_argument('--min_dets_for_veh_tracks_static', type=int, default=3, help='Every track (static) should have a minimum of N detections')
    parser.add_argument('--min_dets_for_ped_tracks', type=int, default=3, help='Every ped track should have a minimum of N detections')
    parser.add_argument('--ps_label_dir', type=str, default='/MS3D/tools/cfgs/target_waymo/ps_labels',
                        help='Folder to save intermediate ps label pkl files')
    
    # Configs for refining boxes of static vehicles
    parser.add_argument('--min_static_score', type=float, default=0.7, help='Minimum score for static boxes after refinement')
    parser.add_argument('--rolling_kde_window', type=int, default=16, help='Minimum score for static boxes after refinement')

    # Configs for propogating boxes
    parser.add_argument('--propagate_boxes_min_dets', type=int, default=7, help='Minimum number of static boxes in order to decide if we want to propagate boxes')
    parser.add_argument('--n_extra_frames', type=int, default=40, help='Number of frames to propagate')
    parser.add_argument('--degrade_factor', type=float, default=0.95, help='For every propagated frame, the box score will be multiplied by degrade factor')
    parser.add_argument('--min_score_clip', type=float, default=0.5, help='Set minimum score that the box can be degraded to. This is not so necessary (more for experiments)')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset = load_dataset(split='train')
    
    # Paths
    ps_pth = '/MS3D/tools/cfgs/target_waymo/ps_labels/N_L_VMFI_TTA_PA_PC_VA_VC_64_WEIGHTED.pkl'
    tracks_veh_all_pth = '/MS3D/tools/cfgs/target_waymo/ps_labels/N_L_VMFI_TTA_PA_PC_VA_VC_64_WEIGHTED_tracks_world_veh.pkl'
    tracks_veh_static_pth = '/MS3D/tools/cfgs/target_waymo/ps_labels/N_L_VMFI_TTA_PA_PC_VA_VC_64_WEIGHTED_tracks_world_veh_static_iou2d.pkl'
    tracks_ped_pth = '/MS3D/tools/cfgs/target_waymo/ps_labels/N_L_VMFI_TTA_PA_PC_VA_VC_64_WEIGHTED_tracks_world_ped.pkl'

    ps_dict = load_pkl(ps_pth)
    tracks_ped = load_pkl(tracks_ped_pth)

    # TODO: Add pedestrian refinement
    """
    2. Find all tracks where pedestrian is moving and tracked for at least 10 frames (~2s @ 5Hz)
    4. If valid track, assign box score of 0.5 for all track boxes
    5. Project dynamic tracks into all frames and do NMS
    """
    trk_cfg_ped = '/MS3D/tracker/configs/ms3d_configs/ped_kf_giou.yaml'
    configs = yaml.load(open(trk_cfg_ped, 'r'), Loader=yaml.Loader)
    trk_cfg_ped_th = configs['running']['score_threshold']
    tracker_utils.delete_tracks(tracks_ped, min_score=trk_cfg_ped_th, num_boxes_abv_score=args.min_dets_for_ped_tracks)                   
    for trk_id in tracks_ped.keys():
        score_mask = tracks_ped[trk_id]['boxes'][:,7] > trk_cfg_ped_th
        tracks_ped[trk_id]['motion_state'] = tracker_utils.get_motion_state(tracks_ped[trk_id]['boxes'][score_mask], s2e_th=1)  

    # Delete stationary pedestrian tracks, retain only dynamic tracks
    all_ids = list(tracks_ped.keys())
    for trk_id in all_ids:
        if tracks_ped[trk_id]['motion_state'] != 1:
            del tracks_ped[trk_id]
        else:
            mask = tracks_ped[trk_id]['boxes'][:,7] < args.pos_th_ped
            tracks_ped[trk_id]['boxes'][:,7][mask] = args.pos_th_ped

    from pcdet.utils.tracker_utils import get_frame_track_boxes
    from pcdet.utils.compatibility_utils import get_lidar, get_pose
    from pcdet.utils.transform_utils import world_to_ego
    from pcdet.utils.box_fusion_utils import nms
    from pcdet.datasets.augmentor.augmentor_utils import get_points_in_box

    final_ps_dict = {}
    final_ps_dict.update(ps_dict)
    for idx, (frame_id, ps) in enumerate(tqdm(final_ps_dict.items(), total=len(final_ps_dict.keys()), desc='update_ps')):        
        if idx > 1000:
            break
        cur_gt_boxes = final_ps_dict[frame_id]['gt_boxes']

        # Focus on pedestrian for now
        ped_mask = abs(ps_dict[frame_id]['gt_boxes'][:,7]) == 2
        cur_gt_boxes = cur_gt_boxes[ped_mask]

        track_boxes_ped = get_frame_track_boxes(tracks_ped, frame_id) # (N,9): [x,y,z,dx,dy,dz,heading,score,trk_id]
        pose = get_pose(dataset, frame_id)
        _, ego_track_boxes_ped = world_to_ego(pose, boxes=track_boxes_ped)
        ego_track_boxes_ped = np.insert(ego_track_boxes_ped[:,:8], 7, axis=1, values=2) # (N,9): [x,y,z,dx,dy,dz,heading,cls_id,score] ; ped class_id = 2        
        new_boxes = np.vstack([cur_gt_boxes, ego_track_boxes_ped])        
        if len(new_boxes) > 1:            
            nms_mask = nms(new_boxes[:,:7].astype(np.float32), 
                            new_boxes[:,8].astype(np.float32),
                            thresh=0.5) # ped_nms = 0.5, veh_nms = 0.05
            new_boxes = new_boxes[nms_mask]
        
        points_1frame = get_lidar(dataset, frame_id) # 1 sweep lidar
        points_1frame[:,:3] += dataset.dataset_cfg.SHIFT_COOR
        num_pts = []    
        for box in new_boxes:
            box_points, _ = get_points_in_box(points_1frame, box)
            num_pts.append(len(box_points))
        num_pts = np.array(num_pts)
        final_ps_dict[frame_id]['gt_boxes'] = new_boxes[num_pts > 1]
        final_ps_dict[frame_id]['num_pts'] = num_pts[num_pts > 1] 
        final_ps_dict[frame_id]['memory_counter'] = np.zeros(final_ps_dict[frame_id]['gt_boxes'].shape[0])

    generate_ps_utils.save_data(final_ps_dict, args.ps_label_dir, name="final_ps_dict_ped_min3.pkl")