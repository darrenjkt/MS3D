import numpy as np
import yaml
from tqdm import tqdm
from mot_3d.frame_data import FrameData
from mot_3d.mot import MOTModel
import yaml
from mot_3d.data_protos import BBox
from pcdet.utils.transform_utils import ego_to_world
from pcdet.utils import compatibility_utils as compat_utils


def get_tracklets(dataset, ps_dict, configs, cls_id):
    """
    Uses SimpleTrack to generate tracklets for the dataset
    
    cls_id: the particular class of interest from the following 1: Vehicle, 2: Pedestrian, 3: Cyclist

    """

    tracks = {}
    tracker = None
    prev_count = 0
    prev_seq = ''
    for frame_id in tqdm(ps_dict.keys(), total=len(ps_dict.keys()), desc='generate_trks'):

        cur_seq = compat_utils.get_sequence_name(dataset, frame_id)
        
        if cur_seq != prev_seq:
            if tracker is not None:
                prev_count=tracker.count
                
            tracker = MOTModel(configs)
            tracker.count = prev_count
            prev_seq = cur_seq
            print(f'Initialising tracker for sequence: {cur_seq}, track_id: {tracker.count}')
            
        aux_info={'is_key_frame': True, 'velos': None}
        timestamp=compat_utils.get_timestamp(dataset, frame_id)
        pose = compat_utils.get_pose(dataset, frame_id)
        points = compat_utils.get_lidar(dataset, frame_id)

        boxes = ps_dict[frame_id]['gt_boxes'].copy()
        class_ids = np.array(abs(boxes[:,7]))
        points_global, boxes_global = ego_to_world(pose, points, boxes[class_ids == cls_id])
        points_global[:,:3] += dataset.dataset_cfg.SHIFT_COOR
        dets=list(np.hstack([boxes_global[:,:7], boxes_global[:,8].reshape(-1,1)]))
        
        frame_data = FrameData(dets=dets, ego=pose, pc=points_global, det_types=class_ids[class_ids == cls_id], 
                            aux_info=aux_info, time_stamp=timestamp, input_opd_format=True)

        results = tracker.frame_mot(frame_data)
        track_boxes = [BBox.bbox2array(trk[0], output_opd_format=True) for trk in results]
        track_ids = [trk[1] for trk in results]
        for list_idx, track_id in enumerate(track_ids):
            if track_id not in tracks.keys():
                tracks[track_id] = {}
                tracks[track_id]['boxes'] = []
                tracks[track_id]['frame_id'] = []
                
            tracks[track_id]['boxes'].append(track_boxes[list_idx])
            tracks[track_id]['frame_id'].append(f'{frame_id}')
        
    return tracks

def delete_tracks(tracks, min_score, num_boxes_abv_score=3):
    """
    Count the number of tracked boxes that are above 'min_score'. If total number of tracked boxes < num_min_dets, 
    delete the track ID
    """
    all_ids = list(tracks.keys())
    for trk_id in all_ids:    
        for k,v in tracks[trk_id].items():
            tracks[trk_id][k] = np.array(v)
        if len(np.argwhere(tracks[trk_id]['boxes'][:,7] > min_score)) < num_boxes_abv_score:
            del tracks[trk_id]      

def get_motion_state(box_seq, s2e_th=1, var_th=0.1):
    """
    Takes in numpy box seq and returns motion state
    Args:
        s2e_th (float): start to end distance threshold
        var_th (float): variance threshold of all centroids
    """
    xy_centroids = box_seq[:,:2]
    start2end_dist = np.linalg.norm(xy_centroids[0] - xy_centroids[-1])
    var = np.mean(np.var(xy_centroids, axis=0))
    if (start2end_dist < s2e_th) and (var < var_th):        
        return 0 # stationary
    else:
        return 1 # dynamic

def get_frame_track_boxes(tracks, frame_id, frame2box_key=None, nhistory=0):
    """
    Get boxes from tracklets for the given frame_id

    Args:
        tracks (dict): processed tracklets for each frame
        frame_id (str): frame id to retrieve tracks for
        nhistory (int): number of historical track boxes to include (so we get motion tail for dynamic objects)    
    
    Returns:
        boxes (N,9)
    """
    frame_boxes = []
    for k,v in tracks.items():
        if frame2box_key is not None:
            frame_ids = np.array(sorted(list(v[frame2box_key].keys())))
        else:
            frame_ids = np.array(v['frame_id'])
        if frame_id in frame_ids:
            frame_idx = np.where(frame_ids == frame_id)[0][0].item()                        
            nhist_start_idx = max(frame_idx-nhistory, 0)            
            track_hist_boxes = []
            # Get all historical tracked boxes and current frame box        
            for idx in range(nhist_start_idx, frame_idx):    
                if frame2box_key is not None: 
                    trk_box = np.hstack([v[frame2box_key][frame_ids[idx]], k])
                else:
                    trk_box = np.hstack([v['boxes'][idx], k])
                track_hist_boxes.append(trk_box)

            if frame2box_key is not None: 
                track_hist_boxes.append(np.hstack([v[frame2box_key][frame_ids[frame_idx]], k]))                
            else:
                track_hist_boxes.append(np.hstack([v['boxes'][frame_idx], k]))
            frame_boxes.extend(track_hist_boxes)     
        
    if len(frame_boxes) == 0:
        return np.empty((0,9))

    frame_boxes = np.array(frame_boxes)
    return frame_boxes


def prepare_track_cfg(ms3d_tracking_cfg):
    """
    This function fixes a few default configs for SimpleTrack to simplify the MS3D tracking config    
    ms3d_tracking_cfg: tracking config for veh_all, veh_static or ped e.g. ms3d_cfg['tracking']['veh_all']        
    """
    trk_cfg = {}    
    trk_cfg['running'] = {}        
    trk_cfg['running']['covariance'] = 'default'
    trk_cfg['running']['score_threshold'] = ms3d_tracking_cfg['RUNNING']['SCORE_TH']
    trk_cfg['running']['max_age_since_update'] = ms3d_tracking_cfg['RUNNING']['MAX_AGE_SINCE_UPDATE']
    trk_cfg['running']['min_hits_to_birth'] = ms3d_tracking_cfg['RUNNING']['MIN_HITS_TO_BIRTH']
    trk_cfg['running']['match_type'] = 'bipartite'
    trk_cfg['running']['has_velo'] = False
    trk_cfg['running']['motion_model'] = 'kf'
    trk_cfg['running']['asso'] = ms3d_tracking_cfg['RUNNING']['ASSO']
    trk_cfg['running']['asso_thres'] = {}
    trk_cfg['running']['asso_thres'][trk_cfg['running']['asso']] = ms3d_tracking_cfg['RUNNING']['ASSO_TH']
    
    trk_cfg['redundancy'] = {}    
    trk_cfg['redundancy']['mode'] = 'mm'
    trk_cfg['redundancy']['max_redundancy_age'] = ms3d_tracking_cfg['REDUNDANCY']['MAX_REDUNDANCY_AGE']
    trk_cfg['redundancy']['det_score_threshold'] = {}
    trk_cfg['redundancy']['det_score_threshold'][trk_cfg['running']['asso']] = ms3d_tracking_cfg['REDUNDANCY']['SCORE_TH']
    trk_cfg['redundancy']['det_dist_threshold'] = {}
    trk_cfg['redundancy']['det_dist_threshold'][trk_cfg['running']['asso']] = ms3d_tracking_cfg['REDUNDANCY']['ASSO_TH']
    return trk_cfg