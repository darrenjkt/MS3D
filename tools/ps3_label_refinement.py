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
from pathlib import Path
from tqdm import tqdm
from pcdet.utils.tracker_utils import get_frame_track_boxes
from pcdet.utils.compatibility_utils import get_lidar, get_pose
from pcdet.utils.transform_utils import world_to_ego
from pcdet.utils.box_fusion_utils import nms
from pcdet.datasets.augmentor.augmentor_utils import get_points_in_box

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

def update_ps(dataset, ps_dict, tracks_veh_all, tracks_veh_static, tracks_ped=None, 
              veh_pos_th=0.6, veh_nms_th=0.05, ped_nms_th=0.5, 
              frame2box_key_static='frameid_to_propboxes', frame2box_key='frameid_to_box', frame_ids=None):
    """
    Add everything to the frame and use NMS to filter out by score. 
    """
    
    final_ps_dict = {}
    if frame_ids is not None:
        for frame_id in ps_dict.keys():        
            # 16f is 2Hz whilst 1f is 5Hz. We could interpolate the box for 5Hz but for now we train at 2Hz for faster research iteration
            if frame_id not in frame_ids:
                continue
            final_ps_dict[frame_id] = ps_dict[frame_id]
    else:
        final_ps_dict.update(ps_dict)
    
    for idx, (frame_id, ps) in enumerate(tqdm(final_ps_dict.items(), total=len(final_ps_dict.keys()), desc='update_ps')):        

        cur_gt_boxes = final_ps_dict[frame_id]['gt_boxes']

        ## ----- Vehicle -----
        veh_mask = abs(cur_gt_boxes[:,7]) == 1
        cur_veh_boxes = cur_gt_boxes[veh_mask]
        
        # Add dynamic 1f interpolated/extrapolated tracks to replace lower scoring dets
        trackall_boxes = get_frame_track_boxes(tracks_veh_all, frame_id, frame2box_key=frame2box_key)
        pose = get_pose(dataset, frame_id)
        _, ego_trackall_boxes = world_to_ego(pose, boxes=trackall_boxes)
        ego_trackall_boxes = np.insert(ego_trackall_boxes[:,:8], 7,1,1)
        ego_trackall_boxes[:,8][np.where(ego_trackall_boxes[:,8] < veh_pos_th)[0]] = veh_pos_th
        
        # Add static objects
        trackstatic_boxes = get_frame_track_boxes(tracks_veh_static, frame_id, frame2box_key=frame2box_key_static)
        static_track_ids = np.array([trk_id for trk_id in trackstatic_boxes[:,-1] if tracks_veh_static[trk_id]['motion_state'] == 0])
        static_mask = np.isin(trackstatic_boxes[:,-1], static_track_ids)
        static_boxes = trackstatic_boxes[static_mask]
        _, ego_static_boxes = world_to_ego(pose, boxes=static_boxes)
        ego_static_boxes = np.insert(ego_static_boxes[:,:8], 7,1,1)
        
        new_veh_boxes = np.vstack([cur_veh_boxes, ego_trackall_boxes, ego_static_boxes])        
        if new_veh_boxes.shape[0] > 1:            
            nms_veh_mask = nms(new_veh_boxes[:,:7].astype(np.float32), 
                            new_veh_boxes[:,8].astype(np.float32),
                            thresh=veh_nms_th)
            new_veh_boxes = new_veh_boxes[nms_veh_mask]
        
        ## ----- Pedestrian -----
        if tracks_ped is not None:
            
            ped_mask = abs(cur_gt_boxes[:,7]) == 2
            cur_ped_boxes = cur_gt_boxes[ped_mask]            
            
            if cur_ped_boxes.shape[0] != 0:
                track_boxes_ped = get_frame_track_boxes(tracks_ped, frame_id) # (N,9): [x,y,z,dx,dy,dz,heading,score,trk_id]
                pose = get_pose(dataset, frame_id)
                _, ego_track_boxes_ped = world_to_ego(pose, boxes=track_boxes_ped)
                ego_track_boxes_ped = np.insert(ego_track_boxes_ped[:,:8], 7, axis=1, values=2) # (N,9): [x,y,z,dx,dy,dz,heading,cls_id,score] ; ped class_id = 2        
                new_ped_boxes = np.vstack([cur_ped_boxes, ego_track_boxes_ped])        
                if new_ped_boxes.shape[0] > 1:            
                    nms_mask = nms(new_ped_boxes[:,:7].astype(np.float32), 
                                    new_ped_boxes[:,8].astype(np.float32),
                                    thresh=ped_nms_th)
                    new_ped_boxes = new_ped_boxes[nms_mask]
            else:
                new_ped_boxes = np.empty((0,9))

            new_boxes = np.vstack([new_veh_boxes, new_ped_boxes])        
        else:
            new_boxes = new_veh_boxes

        # Only keep if more than 1 pt in the box - officially, the evaluation counts the num pts in the box for 1 lidar sweep
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

    # Remove boxes at the ego-vehicle position due to lidar hitting the roof/racks
    for frame_id in final_ps_dict.keys():
        boxes = final_ps_dict[frame_id]['gt_boxes']
        mask = ((np.abs(boxes[:, 0]) < 1.0) & (np.abs(boxes[:,1]) < 1.0))
        final_ps_dict[frame_id]['gt_boxes'] = boxes[~mask]
        final_ps_dict[frame_id]['num_pts'] = final_ps_dict[frame_id]['num_pts'][~mask]
        final_ps_dict[frame_id]['memory_counter'] = final_ps_dict[frame_id]['memory_counter'][~mask]        
        
    return final_ps_dict

def refine_veh_labels(tracks_veh_all, tracks_veh_static, 
                      trk_score_th_static, ms3d_configs):
    """
    Refine vehicle labels. Updates tracks_veh_all and tracks_veh_static in-place.
    Saving of pkl files makes it easier to analyze the results
    """

    # Use pos_th for static min_score so that we ensure to have some confident detections in the static track
    tracker_utils.delete_tracks(tracks_veh_all, min_score=ms3d_configs['ps_score_th']['pos_th'][0], num_boxes_abv_score=ms3d_configs['label_refinement']['track_filtering']['min_dets_above_pos_th_for_tracks_veh_all'])                   
    tracker_utils.delete_tracks(tracks_veh_static, min_score=ms3d_configs['ps_score_th']['pos_th'][0], num_boxes_abv_score=ms3d_configs['label_refinement']['track_filtering']['min_dets_above_pos_th_for_tracks_veh_static'])   
    
     # Get static boxes using tracking information
    for trk_id in tracks_veh_all.keys():
        score_mask = tracks_veh_all[trk_id]['boxes'][:,7] > ms3d_configs['ps_score_th']['pos_th'][0]
        tracks_veh_all[trk_id]['motion_state'] = tracker_utils.get_motion_state(tracks_veh_all[trk_id]['boxes'][score_mask])    
    for trk_id in tracks_veh_static.keys():
        score_mask = tracks_veh_static[trk_id]['boxes'][:,7] > trk_score_th_static
        tracks_veh_static[trk_id]['motion_state'] = tracker_utils.get_motion_state(tracks_veh_static[trk_id]['boxes'][score_mask])        

    # Updates motion-state of track dicts in-place
    matched_trk_ids = generate_ps_utils.motion_state_refinement(tracks_veh_all, tracks_veh_static, list(ps_dict.keys())) # ~ 1:03HR for 18840                

    # Merge disjointed tracks and assign one box per frame in the ego-vehicle frame
    generate_ps_utils.merge_disjointed_tracks(tracks_veh_all, tracks_veh_static, matched_trk_ids)    
    generate_ps_utils.save_data(tracks_veh_all, ms3d_configs["save_dir"], name="tracks_all_world_refined.pkl")
    generate_ps_utils.save_data(tracks_veh_static, ms3d_configs["save_dir"], name="tracks_static_world_refined.pkl")

    generate_ps_utils.get_track_rolling_kde_interpolation(dataset, tracks_veh_static, window=ms3d_configs['label_refinement']['rolling_kbf']['rolling_kde_window'], 
                                                              static_score_th=trk_score_th_static, kdebox_min_score=ms3d_configs['label_refinement']['rolling_kbf']['min_static_score'])  # 16MIN for 18840
    generate_ps_utils.save_data(tracks_veh_static, ms3d_configs["save_dir"], name="tracks_static_world_rkde.pkl")
    generate_ps_utils.propagate_static_boxes(dataset, tracks_veh_static, 
                                                     score_thresh=ms3d_configs['ps_score_th']['pos_th'][0],
                                                     min_static_tracks=ms3d_configs['label_refinement']['propagate_boxes']['min_static_tracks'],
                                                     n_extra_frames=ms3d_configs['label_refinement']['propagate_boxes']['n_extra_frames'], 
                                                     degrade_factor=ms3d_configs['label_refinement']['propagate_boxes']['degrade_factor'], 
                                                     min_score_clip=ms3d_configs['label_refinement']['propagate_boxes']['min_score_clip']) # < 1 min for 18840
    generate_ps_utils.save_data(tracks_veh_static, ms3d_configs["save_dir"], name="tracks_static_world_prop_boxes.pkl")
    return tracks_veh_all, tracks_veh_static

def refine_ped_labels(tracks_ped, ms3d_configs):
    """
    Refine pedestrian labels    
    """
    # Classify if track is static or dynamic
    pos_th_ped = ms3d_configs['ps_score_th']['pos_th'][1]
    tracker_utils.delete_tracks(tracks_ped, min_score=pos_th_ped, num_boxes_abv_score=ms3d_configs['label_refinement']['track_filtering']['min_dets_above_pos_th_for_tracks_ped'])                   
    for trk_id in tracks_ped.keys():
        tracks_ped[trk_id]['motion_state'] = tracker_utils.get_motion_state(tracks_ped[trk_id]['boxes'], s2e_th=1)  

    # Delete tracks if less than N tracks
    tracker_utils.delete_tracks(tracks_ped, min_score=0.0, num_boxes_abv_score=ms3d_configs['label_refinement']['track_filtering']['min_num_ped_tracks'])    

    # Delete stationary pedestrian tracks, retain only dynamic tracks and set score to pos_th_ped
    all_ids = list(tracks_ped.keys())
    for trk_id in all_ids:
        if tracks_ped[trk_id]['motion_state'] != 1:
            del tracks_ped[trk_id]
        else:
            mask = tracks_ped[trk_id]['boxes'][:,7] < pos_th_ped
            tracks_ped[trk_id]['boxes'][:,7][mask] = pos_th_ped
    return tracks_ped

def select_ps_by_th(ps_dict, pos_th):
    """
    Select which labels are not used as pseudo-labels by specifying -ve class label if score < pos_th

    pos_th: [veh_th, ped_th, cyc_th]

    Only supports two classes at the moment
    """
    for frame_id in ps_dict.keys():
        
        veh_mask = np.logical_and(abs(ps_dict[frame_id]['gt_boxes'][:,7]) == 1, 
                                ps_dict[frame_id]['gt_boxes'][:,8] < pos_th[0])
        ps_dict[frame_id]['gt_boxes'][:,7][veh_mask] = -abs(ps_dict[frame_id]['gt_boxes'][:,7][veh_mask])

        ped_mask = np.logical_and(abs(ps_dict[frame_id]['gt_boxes'][:,7]) == 2, 
                                ps_dict[frame_id]['gt_boxes'][:,8] < pos_th[1])
        ps_dict[frame_id]['gt_boxes'][:,7][ped_mask] = -abs(ps_dict[frame_id]['gt_boxes'][:,7][ped_mask])

        cyc_mask = np.logical_and(abs(ps_dict[frame_id]['gt_boxes'][:,7]) == 2, 
                                ps_dict[frame_id]['gt_boxes'][:,8] < pos_th[2])
        ps_dict[frame_id]['gt_boxes'][:,7][cyc_mask] = -abs(ps_dict[frame_id]['gt_boxes'][:,7][cyc_mask])

    return ps_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')                   
    parser.add_argument('--cfg_file', type=str, default='/MS3D/tools/cfgs/dataset_configs/waymo_dataset_da.yaml',
                        help='just use the target dataset cfg file')
    parser.add_argument('--ps_cfg', type=str, help='cfg file with MS3D parameters')
    
    # # Initial pseudo-label and track thresholds
    # parser.add_argument('--pos_th_veh', type=float, default=0.6, help='Vehicle detections above this threshold is used as pseudo-label')
    # parser.add_argument('--pos_th_ped', type=float, default=0.5, help='Pedestrian detections above this threshold is used as pseudo-label')    
    # parser.add_argument('--min_dets_for_tracks_all', type=int, default=4, help='Every track should have a minimum of N detections above the trk_cfg_veh_th')
    # parser.add_argument('--min_dets_for_tracks_static', type=int, default=6, help='Every track (static) should have a minimum of N detections above the trk_cfg_veh_th')
    # parser.add_argument('--min_dets_for_ped_tracks', type=int, default=6, help='Every ped track should have a minimum of N detections above the trk_cfg_veh_th')
    
    # # Configs for refining boxes of static vehicles
    # parser.add_argument('--min_static_score', type=float, default=0.7, help='Minimum score for static boxes after refinement')
    # parser.add_argument('--rolling_kde_window', type=int, default=16, help='Number of boxes to use for KBF of a single static object')

    # # Configs for propogating boxes
    # parser.add_argument('--propagate_boxes_min_dets', type=int, default=12, help='Minimum number of dets for static object to decide if we want to propagate boxes')
    # parser.add_argument('--n_extra_frames', type=int, default=100, help='Number of frames to propagate') # prev 40 for 1.67Hz
    # parser.add_argument('--degrade_factor', type=float, default=0.98, help='For every propagated frame, the box score will be multiplied by degrade factor') # prev 0.95 for 1.67Hz
    # parser.add_argument('--min_score_clip', type=float, default=0.5, help='Set minimum score that the box can be degraded to. This is not so necessary (more for experiments)')
    args = parser.parse_args()

    ms3d_configs = yaml.load(open(args.ps_cfg,'r'), Loader=yaml.Loader)
    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset = load_dataset(split='train')
    
    # Load pkls
    ps_pth = Path(ms3d_configs["save_dir"]) / f'{ms3d_configs["exp_name"]}.pkl'
    tracks_veh_all_pth = Path(ms3d_configs["save_dir"]) / f'{ms3d_configs["exp_name"]}_tracks_world_veh.pkl'
    tracks_veh_static_pth = Path(ms3d_configs["save_dir"]) / f'{ms3d_configs["exp_name"]}_tracks_world_veh_static.pkl'
    tracks_ped_pth = Path(ms3d_configs["save_dir"]) / f'{ms3d_configs["exp_name"]}_tracks_world_ped.pkl'
    ps_dict = load_pkl(ps_pth)
    tracks_veh_all = load_pkl(tracks_veh_all_pth)
    tracks_veh_static = load_pkl(tracks_veh_static_pth)
    tracks_ped = load_pkl(tracks_ped_pth)

    # Get vehicle labels
    print('Refining vehicle labels')
    trk_cfg_veh_static = yaml.load(open(ms3d_configs['tracking']['veh_static_cfg'], 'r'), Loader=yaml.Loader)
    tracks_veh_all, tracks_veh_static = refine_veh_labels(tracks_veh_all, 
                                                          tracks_veh_static, 
                                                          trk_cfg_veh_static['running']['score_threshold'], 
                                                          ms3d_configs)

    # Get pedestrian labels
    print('Refining pedestrian labels')
    trk_cfg_ped = yaml.load(open(ms3d_configs['tracking']['ped_cfg'], 'r'), Loader=yaml.Loader)
    tracks_ped = refine_ped_labels(tracks_ped, ms3d_configs)

    # Combine pseudo-labels for each class and filter with NMS
    print('Combining pseudo-labels for each class')
    final_ps_dict = update_ps(dataset, ps_dict, tracks_veh_all, tracks_veh_static, tracks_ped, 
              veh_pos_th=ms3d_configs['ps_score_th']['pos_th'][0], 
              veh_nms_th=0.05, ped_nms_th=0.5, 
              frame2box_key_static='frameid_to_propboxes', 
              frame2box_key='frameid_to_box', frame_ids=list(ps_dict.keys()))

    final_ps_dict = select_ps_by_th(final_ps_dict, ms3d_configs['ps_score_th']['pos_th'])

    generate_ps_utils.save_data(final_ps_dict, str(Path(ms3d_configs["save_dir"])), name="final_ps_dict.pkl")
    print('Finished generating pseudo-labels')