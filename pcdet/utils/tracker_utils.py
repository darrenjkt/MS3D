import numpy as np
import yaml
from tqdm import tqdm
from mot_3d.frame_data import FrameData
from mot_3d.mot import MOTModel
import yaml
from mot_3d.data_protos import BBox
from pcdet.utils.transform_utils import ego_to_world
from pcdet.utils import compatibility_utils as compat_utils

def get_tracklets(dataset, ps_dict, cfg_path, cls_id):
    """
    Uses SimpleTrack to generate tracklets for the dataset
    
    cls_id: the particular class of interest from the following 1: Vehicle, 2: Pedestrian, 3: Cyclist

    """

    configs = yaml.load(open(cfg_path, 'r'), Loader=yaml.Loader)
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



### ------------------------- No longer used  -------------------------
# def associate_track_ids_to_dets(PSEUDO_LABELS, tracks, match_radius=3, potential_match_iou_th=0.1):
#     """
#     Iterate through all the boxes in each track_id and associate them with each frame's detection box.
#     This function modifies the PSEUDO_LABELS reference and assigns a track_id for each box
#     """
#     for track_id in tracks.keys():
#         boxes = tracks[track_id]['boxes']
#         frame_ids = tracks[track_id]['frame_id']
#         for enum, box in enumerate(boxes):
#             frame_id = frame_ids[enum] 
            
#             if 'track_ids' not in PSEUDO_LABELS[frame_id].keys():
#                 PSEUDO_LABELS[frame_id]['track_ids'] = {}
                
#             ps_boxes_global = PSEUDO_LABELS[frame_id]['gt_boxes_global']
#             nearby_mask = np.linalg.norm(ps_boxes_global[:,:2]-box[:2], axis=1) < match_radius
#             potential_det_matches = ps_boxes_global[nearby_mask]
#             # Compute IOU
#             iou_matrix = iou3d_nms_utils.boxes_bev_iou_cpu(box.reshape(1,-1)[:,:7], 
#                                                 potential_det_matches[:,:7])
#             ious_track2det = np.max(iou_matrix, axis=0)
#             overlap_mask = ious_track2det > potential_match_iou_th
            
#             # Use rounded xy (in global frame) as the det box key 
#             # (in case of multiple track id for one box)
#             for pdet in  potential_det_matches[overlap_mask]:
#                 det_box_key = tuple(np.round(pdet[:2],2))
#                 if det_box_key not in PSEUDO_LABELS[frame_id]['track_ids'].keys():
#                     PSEUDO_LABELS[frame_id]['track_ids'][det_box_key] = []
#                 PSEUDO_LABELS[frame_id]['track_ids'][det_box_key].append(track_id)       
#                 if 'det_boxes' not in tracks[track_id]:
#                     tracks[track_id]['det_boxes'] = []
#                 tracks[track_id]['det_boxes'].append(pdet)    
        
#     for frame_id, ps in PSEUDO_LABELS.items():
#         if 'track_ids' not in ps.keys():
#             continue
#         for k,v in ps['track_ids'].items():
#             if len(v) > 1:
#                 for trk_id in v:
#                     if 'overlap' not in tracks[trk_id].keys():
#                         tracks[trk_id]['overlap'] = {}
#                     tracks[trk_id]['overlap'][frame_id] = [ind for ind in v if ind != trk_id]           

# def clean_and_reassign_tracks(PSEUDO_LABELS,tracks):
#     """
#     Remove IDs from previous association which had overlapping tracks and
#     re-do association step
#     """
#     for key in tracks.keys():
#         tracks[key]['motion_state'] = get_motion_state(tracks[key]['boxes'])        
#         if 'overlap' in tracks[key].keys():
#             tracks[key].pop('overlap')
#         tracks[key].pop('det_boxes')
        
#     for frame_id in PSEUDO_LABELS.keys():
#         if 'track_ids' in PSEUDO_LABELS[frame_id].keys():
#             PSEUDO_LABELS[frame_id].pop('track_ids')                                    
    
#     associate_track_ids_to_dets(PSEUDO_LABELS, tracks)

# def make_track_heading_consistent(tracks):
#     for key in tracks.keys():
#         boxes = np.array(tracks[key]['boxes']).copy()
#         timestamps = tracks[key]['timestamp']
#         angle_bin_edges = np.array([get_abs_angle_diff(a1, a2) \
#                             for (a1,a2) in zip(boxes[1:,6], boxes[:-1,6])])    
#         angle_bin_edges = np.insert(angle_bin_edges, 0, 0)    
#         angle_th = np.pi/2
#         flip_mask  = angle_bin_edges > angle_th
#         if len(np.nonzero(flip_mask)[0]) > 0:
            
#             flipped_inds = sorted(np.nonzero(flip_mask)[0])
#             heading_bins = np.split(boxes[:,6], flipped_inds) # returns a reference to "boxes"
#             list_len = np.array([len(i) for i in heading_bins])

#             # Get majority heading in vector space
#             mean_hbin_vectors = np.array([np.mean(make_vector(hbin), axis=0) for hbin in heading_bins])
#             bin_lens = np.array([len(hbin) for hbin in heading_bins])
#             largest_bin_id = np.argmax(bin_lens)
#             majority_vector = mean_hbin_vectors[largest_bin_id]
#             for ind, hbin_vec in enumerate(mean_hbin_vectors):
#                 diff = np.arccos(np.clip(np.dot(hbin_vec, majority_vector), -1,1))
#                 if diff > angle_th:
#                     heading_bins[ind] += np.pi
                    
#         tracks[key]['boxes'] = boxes

# def get_overlapping_track_ids(overlap_boxes, all_ids):
#     """
#     Sometimes we get tracks that overlap a det box, but do not overlap
#     with each other. Here we return the list of tracks that actually
#     overlap
#     """
#     if len(overlap_boxes) == 0:
#         return [[]]

#     boxs_gpu = torch.from_numpy(overlap_boxes[:,:7]).cuda().float()
#     scores_gpu = torch.from_numpy(overlap_boxes[:,7]).cuda().float()
#     nms_inds = iou3d_nms_utils.nms_gpu(boxs_gpu, scores_gpu, thresh=0.1)[0].cpu().numpy()
#     nms_mask = np.zeros(boxs_gpu.shape[0], dtype=bool)
#     nms_mask[nms_inds] = 1
#     if len(nms_inds) == overlap_boxes.shape[0]:
#         return [[tid] for tid in all_ids]
    
#     elif len(nms_inds) > 1:
#         in_box_a = torch.from_numpy(overlap_boxes[nms_mask]).cuda().float()
#         in_box_b = torch.from_numpy(overlap_boxes[~nms_mask]).cuda().float()
#         iou_matrix = iou3d_nms_utils.boxes_iou_bev(in_box_a[:,:7], in_box_b[:,:7]).cpu().numpy()
#         matched_boxes = np.argwhere((iou_matrix > 0.1) == True)
        
#         matches = {}
#         for row in matched_boxes:
#             trk_ids_a = all_ids[nms_mask]
#             trk_ids_b = all_ids[~nms_mask]
#             if trk_ids_a[row[0]] not in matches.keys():
#                 matches[trk_ids_a[row[0]]] = [trk_ids_a[row[0]]]
#             matches[trk_ids_a[row[0]]].append(trk_ids_b[row[1]])

#         all_overlap_ids = list(matches.values())      
#         return all_overlap_ids
#     else:
#         return [list(all_ids)]

# def get_overlap_track_ids_to_discard(tracks, trk_id, 
#                                  diff_th=0.001, score_th=0.6,
#                                  verbose=False):
#     """
#     Returns a list of track ids to discard for the tracks that overlap.
#     Selection of track criterion is
#     0: continue track with max score
#     1: keep stationary track, discard dynamic one
#     2: multiple stationary overlapping, discard all
#     None: discard all
#     """
#     # 0: max score 1: stationary, 2: multiple stationary, None: discard all        
#     keep_ids = []
#     discard_ids_all = []
#     reasons = []
#     overlaps = []
#     for frame_id, overlap_ids in tracks[trk_id]['overlap'].items():
#         all_ids = np.array(overlap_ids + [trk_id])
#         overlap_boxes = []
#         for overlap_id in all_ids:
#             ind = np.argwhere(tracks[overlap_id]['frame_id'] == frame_id)
#             if len(ind) == 0:
#                 continue
#             track_box = tracks[overlap_id]['boxes'][ind.item()]
#             motion_state = get_motion_state(tracks[overlap_id]['boxes'])
#             track_box = np.hstack([track_box, overlap_id, motion_state])
#             overlap_boxes.append(track_box)
#         if len(overlap_boxes) > 0:
#             overlap_boxes = np.vstack(overlap_boxes)
#             overlaps.append(overlap_boxes)
#         if len(overlap_boxes) == 0:
#             continue

#         all_overlap_ids = get_overlapping_track_ids(overlap_boxes, all_ids) 

#         reason = None
#         keep_id = None
#         discard_ids_frame = []
#         for overlap_list in all_overlap_ids:
#             if len(overlap_list) == 1:
#                 keep_id = overlap_list[0]
#                 reason = 3

#             # Subtract max. 
#             # If all scores are similar, then diff will be small so we discard all
#             # If there is a large diff, then we just select highest score        
#             scores = overlap_boxes[:,7]
#             diff = np.abs(scores - np.max(scores))
#             if reason is None and ((np.max(scores) > score_th) or (len(np.nonzero(diff > diff_th)[0]) > 0)):            
#                 keep_id = np.array(overlap_list)[np.argmax(scores)]
#                 reason = 0

#             check_stationary = np.nonzero(overlap_boxes[:,-1] == 0)[0]
#             if (len(check_stationary) == 1) and (reason is None):
#                 keep_id = all_ids[check_stationary].item()
#                 reason = 1

#             if (len(check_stationary) > 1)  and (reason is None):
#                 # Multiple stationary overlapping means something is quite wrong
#                 keep_id = None
#                 reason = 2

#             keep_ids.append(keep_id)            
#             reasons.append(reason)
#             if keep_id is not None:
#                 overlap_list.pop(overlap_list.index(keep_id))                
#             discard_ids_frame.append(overlap_list)
#         discard_ids_all.append(discard_ids_frame)
        
#     if verbose:
#         return keep_ids, reasons, overlaps, discard_ids_all
#     else:
#         return discard_ids_all

# def filter_overlapping_tracks(tracks):
#     for trk_id in tracks.keys():        
#         if 'overlap' in tracks[trk_id].keys():
#             discard_ids = get_overlap_track_ids_to_discard(tracks, trk_id)        
#             for enum, d_id_list_frame in enumerate(discard_ids):
#                 for d_id_list_inst in d_id_list_frame:
#                     frame_id = list(tracks[trk_id]['overlap'].keys())[enum]
#                     for d_id in d_id_list_inst:
#                         end_ind = np.argwhere(tracks[d_id]['frame_id'] == frame_id)
#                         if len(end_ind) == 0:
#                             continue

#                         for k in tracks[d_id].keys():
#                             if k not in ['overlap', 'motion_state']:    
#                                 tracks[d_id][k] = tracks[d_id][k][:end_ind.item()]