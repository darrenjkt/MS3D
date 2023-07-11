import pcdet.utils.box_fusion_utils as box_fusion_utils
from pcdet.config import cfg
import numpy as np
from tqdm import tqdm

import copy
from pathlib import Path
import pickle as pkl
from pcdet.utils.transform_utils import ego_to_world, world_to_ego
from pcdet.utils.tracker_utils import get_frame_track_boxes
from pcdet.utils.box_fusion_utils import compute_iou, kbf, nms
from pcdet.datasets.augmentor.augmentor_utils import get_points_in_box
from pcdet.utils.compatibility_utils import get_lidar, get_pose, get_sequence_name, get_sample_idx, get_frame_id
from pcdet.utils import common_utils

def get_multi_source_prelim_label(ms_cfg, fusion_cfg, desc='gen_ps_labels', min_score=0.3): 
    ps_dict = {}
    det_annos = box_fusion_utils.load_src_paths_txt(fusion_cfg.PATH)
    combined_dets = box_fusion_utils.combine_box_pkls(det_annos, cfg.DATA_CONFIG_TAR.CLASS_NAMES)
    
    for frame_boxes in tqdm(combined_dets, total=len(combined_dets), desc=desc):

        boxes_lidar = np.hstack([frame_boxes['boxes_lidar'],
                                 frame_boxes['class_ids'][...,np.newaxis],
                                 frame_boxes['score'][...,np.newaxis]])

        # score < neg_th: discard                                 
        score_mask = boxes_lidar[:,8] > min_score
        ps_label_nms, _ = box_fusion_utils.label_fusion(boxes_lidar[score_mask], 
                                                                fusion_name=ms_cfg.FUSION, 
                                                                discard=fusion_cfg.DISCARD, radius=fusion_cfg.RADIUS)

        # neg_th < score < pos_th: ignore for training but keep for update step
        pred_boxes = ps_label_nms[:,:7]
        pred_labels = ps_label_nms[:,7]
        pred_scores = ps_label_nms[:,8]
        ignore_mask = pred_scores < cfg.SELF_TRAIN.SCORE_THRESH
        pred_labels[ignore_mask] = -pred_labels[ignore_mask]
        gt_box = np.concatenate((pred_boxes,
                                pred_labels.reshape(-1, 1),
                                pred_scores.reshape(-1, 1)), axis=1)    
        
        gt_infos = {
            'gt_boxes': gt_box,
        }
        ps_dict[frame_boxes['frame_id']] = gt_infos
    return ps_dict

def save_data(data, folder, name):
    ps_path = Path(folder) / name
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(str(ps_path), 'wb') as f:
        pkl.dump(data, f)

def load_if_exists(folder, name):
    ps_path = Path(folder) / name
    if ps_path.exists():
        with open(str(ps_path), 'rb') as f:
            data = pkl.load(f)
        return data
    return None

def merge_track_ids(trk_mapping, tracks):
    """
    trk_mapping (dict): matching track ids {track A id: [track B id 0,track B id 1,...]} 
                        when comparing one set of tracks to another. E.g track B might have 
                        3 track ids matched with 1 track id in track A. In this case, we 
                        merge all 3 track B ids into one.
    tracks (list): track to be modified
    """
    deleted_trk_ids = {}
    for _, overlapping_ids in trk_mapping.items():
        if len(overlapping_ids) == 1:
            continue
        # merge the two tracks
        merged_tracklet = {}
        sorted_ids = sorted(overlapping_ids)
        first_id = None
        for t_id in sorted_ids:        
            
            # If merging to an already-merged track, then find the first 
            # trk_id used for merging
            if t_id not in tracks.keys():
                while t_id in deleted_trk_ids.keys():
                    t_id = deleted_trk_ids[t_id]
                first_id = t_id

            for k,v in tracks[t_id].items():
                if k not in merged_tracklet.keys():
                    merged_tracklet[k] = []
                merged_tracklet[k].append(v)

        for k,v in merged_tracklet.items():
            if k == 'boxes':
                merged_tracklet[k] = np.vstack(v)
            else:
                merged_tracklet[k] = np.hstack(v)


        if first_id == sorted_ids[0]:                    
            continue # Probably a duplicate match
        
        merged_id = sorted_ids.pop(0) if first_id is None else first_id
        tracks[merged_id] = merged_tracklet
        # Remove track ids that were merged
        for del_ids in sorted_ids:
            deleted_trk_ids[del_ids] = merged_id
            if del_ids in tracks.keys():
                del tracks[del_ids]
                
def motion_state_refinement(tracks_1f, tracks_16f, frame_ids):
    """
    Compute iou of motion trails from the 1-frame tracklets against the 16-frame tracklets
    to refine motion states

    Returns:
        matched_trk_ids (list of tuples): list of (trk_id_16f, trk_id_1f) matched track ids
    """
    matched_trk_ids = []

    # Add 16f tracks to 1f tracks frame. Compare if they overlap
    for frame_id in tqdm(frame_ids, total=len(frame_ids), desc='motion_state_refinement'):
        boxes16f = get_frame_track_boxes(tracks_16f, frame_id, nhistory=0)
        if len(boxes16f) == 0:
            continue
            
        boxes1f = get_frame_track_boxes(tracks_1f, frame_id, nhistory=0)
        if len(boxes1f) == 0:
            continue

        # Get all motion trail if obj is dynamic
        for box in boxes1f:
            trk_id = box[-1]
            if tracks_1f[trk_id]['motion_state'] == 1:
                tboxes = np.hstack([tracks_1f[trk_id]['boxes'], np.ones((tracks_1f[trk_id]['boxes'].shape[0],1))*trk_id])
                boxes1f = np.vstack([boxes1f, tboxes])
                
        ious, matched_pairs = compute_iou(boxes16f, boxes1f)
        matched_pairs = matched_pairs[ious > 0.1]

        # Compare matched pairs. If one is labelled dynamic, then set both tracklet to dynamic 
        for pair in matched_pairs:
            trkid_16f = boxes16f[pair[0],-1]
            trkid_1f = boxes1f[pair[1],-1]
            motion_state = tracks_16f[trkid_16f]['motion_state'] | tracks_1f[trkid_1f]['motion_state']
            tracks_16f[trkid_16f]['motion_state'] = motion_state
            tracks_1f[trkid_1f]['motion_state'] = motion_state

            # Only keep track of static obj ID matches for merging disjointed tracks. It shouldn't matter that we 
            # got the whole motion trail
            if motion_state == 0:
                matched_trk_ids.append((int(trkid_16f), int(trkid_1f)))
    
    return list(set(matched_trk_ids))


def assign_box_to_frameid(tracks):
    # assign one box per frame_id; if duplicate, then keep higher scoring box
    for trk_id in tracks.keys():
        tracks[trk_id]['frameid_to_box'] = {}
        fid_list = tracks[trk_id]['frame_id']
        tracks[trk_id]['motion_state'] = int(np.median(tracks[trk_id]['motion_state']))
        for enum, f_id in enumerate(fid_list):
            if f_id in tracks[trk_id]['frameid_to_box'].keys():
                box_selection = np.stack([tracks[trk_id]['frameid_to_box'][f_id],
                                          tracks[trk_id]['boxes'][enum]])
                box = box_selection[np.argmax(box_selection[:,-1])]
            else:
                box = tracks[trk_id]['boxes'][enum]

            tracks[trk_id]['frameid_to_box'][f_id] = box

def merge_disjointed_tracks(tracks_all, tracks_static, matched_trk_ids):
    """Merge disjointed tracks and assign one box per frame """
    # First merge 1f tracks using 16f tracks as "glue" with f16-f1 map    
    matched_trk_ids = list(set(matched_trk_ids))
        
    # Get trk id mapping for 16f -> 1f trk ids  
    # Important to sort by 1f ids since we merge to the lowest trk_id -> this avoids merging to an already deleted track
    tstatic_tall_idmap = {}
    for tids in sorted(matched_trk_ids, key=lambda x:x[1]):
        tstatic_id = tids[0]
        tall_id = tids[1]
        if tstatic_id not in tstatic_tall_idmap.keys():
            tstatic_tall_idmap[tstatic_id] = []
        tstatic_tall_idmap[tstatic_id].append(tall_id)

    # Get trk id mapping for 1f -> 16f trk ids    
    # here we sort by 16f ids for same reason as above
    tall_tstatic_idmap = {}
    for tids in sorted(matched_trk_ids, key=lambda x:x[0]):
        tstatic_id = tids[0]
        tall_id = tids[1]
        if tall_id not in tall_tstatic_idmap.keys():
            tall_tstatic_idmap[tall_id] = []
        tall_tstatic_idmap[tall_id].append(tstatic_id)    
        
    merge_track_ids(tstatic_tall_idmap, tracks_all)

    # Remove track ids in the f1-f16 mapping dict that were merged 
    tall_keys = list(tall_tstatic_idmap.keys())
    for tall_id in tall_keys:
        if tall_id not in tracks_all.keys():
            del tall_tstatic_idmap[tall_id] 
            
    merge_track_ids(tall_tstatic_idmap, tracks_static)        

    # Remove track ids in the f1-f16 mapping dict that were merged 
    tstatic_keys = list(tstatic_tall_idmap.keys())
    for tstatic_id in tstatic_keys:
        if tstatic_id not in tracks_static.keys():
            del tstatic_tall_idmap[tstatic_id]

    assign_box_to_frameid(tracks_all)            
    assign_box_to_frameid(tracks_static)

def get_track_rolling_kde_interpolation(dataset, tracks_static, window, static_score_th, kdebox_min_score):
    """
    For static objects, combine historical N boxes as the current frame's box.
    If there are no N historical frames, we take future frames. 

    For missing detection frames, we use the previous historical kde box
    """
    for trk_id in tqdm(tracks_static.keys(), total=len(tracks_static), desc='rolling_kde_interp'):
        tracks_static[trk_id]['frameid_to_rollingkde'] = {}
        f2b = tracks_static[trk_id]['frameid_to_box']
        if tracks_static[trk_id]['motion_state'] != 0:
            tracks_static[trk_id]['frameid_to_rollingkde'].update(f2b)
            continue        
        
        trk_frame_inds_set = np.array(list(f2b.keys()))
            
        if len(trk_frame_inds_set) < window:
            boxes = np.array(list(f2b.values()))
            boxes = np.insert(boxes, 7,1,1) # Static refinement only done for vehicle class so we hardcode class ID
            kdebox = kbf(boxes, box_weights=boxes[:,-1], bw_score=1.0)
            if kdebox[8] > static_score_th:
                kdebox[8] = max(kdebox_min_score, kdebox[8])
            kdebox = np.delete(kdebox, 7)
            for frame_idx, f_id in enumerate(trk_frame_inds_set):
                tracks_static[trk_id]['frameid_to_rollingkde'][f_id] = kdebox
        
        else:
            # Get rolling KDE for each key frame
            for frame_idx, f_id in enumerate(trk_frame_inds_set):
                accum_inds = int(frame_idx) + np.arange(-window, 0)
                accum_inds[accum_inds < 0] = abs(accum_inds[accum_inds < 0]) + max(accum_inds)
                accum_inds = np.clip(accum_inds, 0, len(trk_frame_inds_set)-1)
                boxes = np.stack([f2b[key] for key in trk_frame_inds_set[np.unique(accum_inds)]])
                boxes = np.insert(boxes, 7,1,1) # Static refinement only done for vehicle class so we hardcode class ID
                kdebox = kbf(boxes, box_weights=boxes[:,-1], bw_score=1.0)
                if kdebox[8] > static_score_th:
                    kdebox[8] = max(kdebox_min_score, kdebox[8])
                kdebox = np.delete(kdebox, 7)
                tracks_static[trk_id]['frameid_to_rollingkde'][f_id] = kdebox
            
        # Interpolate between frames
        frame_ids = list(f2b.keys())
        first_frame_idx = dataset.frameid_to_idx[frame_ids[0]]
        last_frame_idx = dataset.frameid_to_idx[frame_ids[-1]]
        all_frame_inds = np.arange(first_frame_idx, last_frame_idx+1)
        prev_box = None
        for ind in all_frame_inds:
            frame_id = get_frame_id(dataset, dataset.infos[ind])
            if frame_id in tracks_static[trk_id]['frameid_to_rollingkde'].keys():
                prev_box = tracks_static[trk_id]['frameid_to_rollingkde'][frame_id]
            else:
                tracks_static[trk_id]['frameid_to_rollingkde'][frame_id] = prev_box            

def propagate_static_boxes(dataset, tracks_static, score_thresh, min_dets, n_extra_frames, degrade_factor, min_score_clip):
    for trk_id in tqdm(tracks_static.keys(), total=len(tracks_static), desc='propagate_static_boxes'):
        tracks_static[trk_id]['frameid_to_propboxes'] = {}
        roll_kde = tracks_static[trk_id]['frameid_to_rollingkde']
        tracks_static[trk_id]['frameid_to_propboxes'].update(roll_kde)
        if tracks_static[trk_id]['motion_state'] != 0:            
            continue       

        boxes = np.array(list(tracks_static[trk_id]['frameid_to_propboxes'].values()))
        if len(boxes[boxes[:,-1] > score_thresh]) < min_dets:
            continue
        
        frame_ids = list(roll_kde.keys())
        first_frame_idx = dataset.frameid_to_idx[frame_ids[0]]
        last_frame_idx = dataset.frameid_to_idx[frame_ids[-1]]
        first_box = roll_kde[frame_ids[0]]
        last_box = roll_kde[frame_ids[-1]]
        track_scene = get_sequence_name(dataset, frame_ids[0])
        seq_len = dataset.seq_name_to_len[track_scene]
        sample_idx = get_sample_idx(dataset, frame_ids[0])

        ext_first_idx = np.clip(first_frame_idx - n_extra_frames, 
                first_frame_idx - sample_idx, 
                first_frame_idx - sample_idx + seq_len - 1)
        ext_last_idx = np.clip(last_frame_idx + n_extra_frames, 
                first_frame_idx - sample_idx, 
                first_frame_idx - sample_idx + seq_len - 1)
        frame_inds = np.arange(ext_first_idx, ext_last_idx+1)

        prev_box = None
        for ind in frame_inds:
            frame_id = get_frame_id(dataset, dataset.infos[ind])
            if frame_id in tracks_static[trk_id]['frameid_to_propboxes'].keys():
                prev_box = tracks_static[trk_id]['frameid_to_propboxes'][frame_id]
            else:                            
                if ind < first_frame_idx:    
                    tracks_static[trk_id]['frameid_to_propboxes'][frame_id] = copy.deepcopy(first_box)
                    new_score = tracks_static[trk_id]['frameid_to_propboxes'][frame_id][7]
                    new_score = np.clip(new_score * degrade_factor**abs(first_frame_idx-ind), min_score_clip, 1.0)
                    tracks_static[trk_id]['frameid_to_propboxes'][frame_id][7] = new_score
                elif ind > last_frame_idx:
                    tracks_static[trk_id]['frameid_to_propboxes'][frame_id] = copy.deepcopy(last_box)
                    new_score = tracks_static[trk_id]['frameid_to_propboxes'][frame_id][7]
                    new_score = np.clip(new_score * degrade_factor**abs(ind-last_frame_idx), min_score_clip, 1.0)                
                    tracks_static[trk_id]['frameid_to_propboxes'][frame_id][7] = new_score                    
                else:
                    tracks_static[trk_id]['frameid_to_propboxes'][frame_id] = prev_box

def update_ps(dataset, ps_dict_1f, tracks_1f, tracks_16f, frame2box_key_16f, score_th, frame2box_key_1f=None, frame_ids=None):
    
    final_ps_dict = {}
    if frame_ids is not None:
        for frame_id in ps_dict_1f.keys():        
            # 16f is 2Hz whilst 1f is 5Hz. We could interpolate the box for 5Hz but for now we train at 2Hz for faster research iteration
            if frame_id not in frame_ids:
                continue
            final_ps_dict[frame_id] = ps_dict_1f[frame_id]
    else:
        final_ps_dict.update(ps_dict_1f)
    
    for idx, (frame_id, ps) in enumerate(tqdm(final_ps_dict.items(), total=len(final_ps_dict.keys()), desc='update_ps')):        
        if idx > 1000:
            return final_ps_dict # TODO: Delete
        cur_gt_boxes = final_ps_dict[frame_id]['gt_boxes']
        
        # Add dynamic 1f interpolated/extrapolated tracks to replace lower scoring dets
        track1f_boxes = get_frame_track_boxes(tracks_1f, frame_id, frame2box_key=frame2box_key_1f)
        pose = get_pose(dataset, frame_id)
        _, ego_track1f_boxes = world_to_ego(pose, boxes=track1f_boxes)
        ego_track1f_boxes = np.insert(ego_track1f_boxes[:,:8], 7,1,1)
        ego_track1f_boxes[:,8][np.where(ego_track1f_boxes[:,8] < score_th)[0]] = score_th
        
        # Add static 16f objects
        track16f_boxes = get_frame_track_boxes(tracks_16f, frame_id, frame2box_key=frame2box_key_16f)
        static_track_ids = np.array([trk_id for trk_id in track16f_boxes[:,-1] if tracks_16f[trk_id]['motion_state'] == 0])
        static_mask = np.isin(track16f_boxes[:,-1], static_track_ids)
        static16f_boxes = track16f_boxes[static_mask]
        _, ego_track16f_boxes = world_to_ego(pose, boxes=static16f_boxes)
        ego_track16f_boxes = np.insert(ego_track16f_boxes[:,:8], 7,1,1)
        
        new_boxes = np.vstack([cur_gt_boxes, ego_track1f_boxes, ego_track16f_boxes])        
        if len(new_boxes) > 1:            
            nms_mask = nms(new_boxes[:,:7].astype(np.float32), 
                            new_boxes[:,8].astype(np.float32),
                            thresh=0.05)
            new_boxes = new_boxes[nms_mask]
        
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

### ---- BELOW ARE CURRENTLY UNUSED ----
# def motion_state_refinement_old(tracks_1f, tracks_16f, frame_ids):
#     """
#     First msr function I used for target-nusc/target-waymo. 

#     Compute iou of motion trails from the 1-frame tracklets against the 16-frame tracklets
#     to refine motion states

#     Returns:
#         matched_trk_ids (list of tuples): list of (trk_id_16f, trk_id_1f) matched track ids
#     """
#     matched_trk_ids = []

#     # Add 16f tracks to 1f tracks frame. Compare if they overlap
#     for frame_id in tqdm(frame_ids, total=len(frame_ids), desc='motion_state_refinement'):
#         boxes16f = get_frame_track_boxes(tracks_16f, frame_id, nhistory=0)
#         if len(boxes16f) == 0:
#             continue
            
#         boxes1f = get_frame_track_boxes(tracks_1f, frame_id, nhistory=15)
#         if len(boxes1f) == 0:
#             continue
#         ious, matched_pairs = compute_iou(boxes16f, boxes1f)
#         matched_pairs = matched_pairs[ious > 0.1]
#         # Compare matched pairs. If one is labelled dynamic, then set tracklet to dynamic 
#         for pair in matched_pairs:
#             trkid_16f = boxes16f[pair[0],-1]
#             trkid_1f = boxes1f[pair[1],-1]
#             motion_state = tracks_16f[trkid_16f]['motion_state'] | tracks_1f[trkid_1f]['motion_state']
#             tracks_16f[trkid_16f]['motion_state'] = motion_state
#             tracks_1f[trkid_1f]['motion_state'] = motion_state
            
#             if motion_state == 0:
#                 matched_trk_ids.append((int(trkid_16f), int(trkid_1f)))
    
#     return list(set(matched_trk_ids))    


# def get_static_boxes_for_tracks(matched_trk_ids, tracks_1f, tracks_16f, min_static_score=0.8, min_score_th=0.6):
#     """
#     Updates the 1-frame tracks dictionary with a combined static box using all observations. 
#     KDE is used to merge all the boxes into one.
#     """
    
#     # First merge 1f tracks using 16f tracks as "glue" with f16-f1 map
#     f16_f1_idmap = {}
#     matched_trk_ids = list(set(matched_trk_ids))
#     for tids in matched_trk_ids:
#         f16_id = tids[0]
#         f1_id = tids[1]
#         if f16_id not in f16_f1_idmap.keys():
#             f16_f1_idmap[f16_id] = []
#         f16_f1_idmap[f16_id].append(f1_id)

#     deleted_trk_ids = {}
#     for f16_id, f1_ids in f16_f1_idmap.items():
#         if len(f1_ids) == 1:
#             continue
#         # merge the two tracks
#         merged_tracklet = {}
#         sorted_f1_ids = sorted(f1_ids)
#         first_id = None
#         for f1_id in sorted_f1_ids:        
            
#             # If merging to an already-merged track, then find the first trk_id used for merging
#             if f1_id not in tracks_1f.keys():                            
#                 first_id = deleted_trk_ids[f1_id]
#                 f1_id = first_id if first_id is not None else f1_id
                
#             for k,v in tracks_1f[f1_id].items():
#                 if k not in merged_tracklet.keys():
#                     merged_tracklet[k] = []
#                 merged_tracklet[k].append(v)
                    
#         for k,v in merged_tracklet.items():
#             if k == 'boxes':
#                 merged_tracklet[k] = np.vstack(v)
#             elif k == 'motion_state':
#                 merged_tracklet[k] = v[0]
#             else:
#                 merged_tracklet[k] = np.hstack(v)
        
#         merged_id = sorted_f1_ids.pop(0) if first_id is None else first_id
#         tracks_1f[merged_id] = merged_tracklet
#         # Remove track ids that were merged
#         for del_f1_ids in sorted_f1_ids:
#             deleted_trk_ids[del_f1_ids] = merged_id
#             if del_f1_ids in tracks_1f.keys():
#                 del tracks_1f[del_f1_ids]

#     # Remove track ids in the f1-f16 mapping dict that were merged 
#     f1_f16_idmap = {}
#     for tids in matched_trk_ids:
#         f16_id = tids[0]
#         f1_id = tids[1]
#         if f1_id not in f1_f16_idmap.keys():
#             f1_f16_idmap[f1_id] = []
#         f1_f16_idmap[f1_id].append(f16_id)
        
#     f1_keys = list(f1_f16_idmap.keys())
#     for f1_id in f1_keys:
#         if f1_id not in tracks_1f.keys():
#             del f1_f16_idmap[f1_id]    
    
#     # Combine 1-frame and 16-frame tracked boxes with KDE to form the new box
#     # Reason: 16-frame boxes are not necessarily better than 1-frame due to DA gap
#     for f1_id in tracks_1f.keys():
#         if tracks_1f[f1_id]['motion_state'] != 0:
#             continue
#         boxes = tracks_1f[f1_id]['boxes']
#         if f1_id in f1_f16_idmap.keys():
#             for f16_id in f1_f16_idmap[f1_id]:
#                 boxes = np.vstack([boxes, tracks_16f[f16_id]['boxes']])
                
#         boxes = boxes[boxes[:,-1] > min_score_th]        
#         boxes = np.insert(boxes, 7, 1, axis=1)
#         new_box = kde_fusion(boxes, src_weights=boxes[:,-1])
#         new_box[8] = max(new_box[8], min_static_score)    
#         tracks_1f[f1_id]['static_box'] = new_box

# def update_ps_with_static_boxes(dataset, tracks, ps_dict_1f, frameid_to_idx, n_extra=10, pc_range=80, anno_frames_only=False):
#     """
#     Updates the pseudo-label dict with the static boxes. Uses NMS to clean up overlapping labels.
#     Args:
#         frameid_to_idx (dict): mapping of frame id to list index of infos
#         n_extra (int): number of future/past frames to project the static box in outside the observed frames
#         pc_range (float): if static box is placed outside of point cloud range, it'll be removed
#     """        

#     if dataset.dataset_cfg.DATASET == 'ONCEDataset':
#         dataset.anno_frames_only = anno_frames_only
#         dataset.include_once_data('train', reload=True)
#     dataset_infos = dataset.infos

#     # Transform to world frame
#     for frame_id, val in ps_dict_1f.items():
#         pose = get_pose(dataset, frame_id)
#         _, boxes_global = ego_to_world(pose, boxes=val['gt_boxes'].copy())
#         ps_dict_1f[frame_id]['gt_boxes_global'] = boxes_global

#     for frame_id in ps_dict_1f.keys():
#         ps_dict_1f[frame_id]['refined'] = ps_dict_1f[frame_id]['gt_boxes_global'].copy()    
#         ps_dict_1f[frame_id]['static_boxes'] = {}
    
#     # Add static boxes to each frame's pseudo labels
#     for trk_id in tracks.keys():
#         if 'refined_static_box' in tracks[trk_id].keys():

#             s_indices = []
#             for enum, f_id in enumerate(tracks[trk_id]['frame_id']):
#                 infos = dataset_infos[frameid_to_idx[f_id]]
#                 if enum == 0:
#                     seq_len = dataset.seq_name_to_len[get_sequence_name(dataset, f_id)]
#                 s_indices.append(get_sample_idx(dataset, f_id))
#             start_sample_idx = min(s_indices)
#             end_sample_idx = max(s_indices)
            
#             # Get all frame indices +N before and after
#             all_inds = np.arange(np.clip(start_sample_idx-n_extra, 0, seq_len), 
#                                 np.clip(end_sample_idx+n_extra, 0, seq_len))
#             f_accum_box = tracks[trk_id]['refined_static_box']
#             for ind in all_inds:
#                 infos = seq_infos[ind]
#                 frame_id = get_frame_id(dataset, infos)
#                 pose = get_pose(dataset, frame_id)
#                 _, boxes_ego = world_to_ego(pose=pose, 
#                                             boxes=f_accum_box[:7].reshape(1,-1))
                
#                 # Do not add box if out of of pc range
#                 if len(np.argwhere(abs(boxes_ego[0,:2]) > pc_range)) > 0:
#                     continue
                
#                 new_boxes = np.vstack([ps_dict_1f[frame_id]['refined'],
#                                         f_accum_box[:9]])
#                 ps_dict_1f[frame_id]['refined'] = new_boxes
#                 ps_dict_1f[frame_id]['static_boxes'][tuple(np.round(f_accum_box[:2],2))] = trk_id
    
#     def update_static_boxes(ref_boxes, mask, ps_dict):
#         """
#         Args:
#             ref_boxes (N,9): boxes used to get the mask (before filtering)
#         """
#         rm_box_inds = np.nonzero(~mask)[0]
#         det_keys = np.round(ref_boxes[rm_box_inds,:2],2)
#         for key in det_keys:
#             if tuple(key) in list(ps_dict[frame_id]['static_boxes'].keys()):
#                 ps_dict[frame_id]['static_boxes'].pop(tuple(key))

#     # Cleanup overlapping labels with NMS
#     for idx, frame_id in tqdm(enumerate(ps_dict_1f.keys()), total=len(ps_dict_1f.keys()), desc='refine_ps_labels'): 
#         refined_boxes = ps_dict_1f[frame_id]['refined'].copy()
#         if len(refined_boxes) > 1:            
#             nms_mask = nms(refined_boxes[:,:7].astype(np.float32), 
#                             refined_boxes[:,8].astype(np.float32),
#                             thresh=0.1)  
#             update_static_boxes(refined_boxes, nms_mask, ps_dict_1f)
#             refined_boxes = refined_boxes[nms_mask]        
        
#         pose = get_pose(dataset, frame_id)
#         points_1frame = get_lidar(dataset, frame_id)
#         points_global, _ = ego_to_world(pose, points=points_1frame)
#         points_global[:,:3] += dataset.dataset_cfg.SHIFT_COOR

#         num_pts = []    
#         for box in refined_boxes:
#             box_points, _ = get_points_in_box(points_global, box)
#             num_pts.append(len(box_points))
#         num_pts = np.array(num_pts)
#         filter_pts_mask = num_pts > 1
#         update_static_boxes(refined_boxes, filter_pts_mask, ps_dict_1f)
#         ps_dict_1f[frame_id]['num_pts'] = num_pts[filter_pts_mask]
#         ps_dict_1f[frame_id]['refined'] = refined_boxes[filter_pts_mask]

# def get_static_obj_pts(dataset, tracks, enlarge_l=1, enlarge_w=0.3, enlarge_h=-0.5):
#     """Gets all the obj pts across each frame within an enlarged box. Frame index is appended to the last column"""
#     for data_dict in tqdm(dataset, total=len(dataset), desc='gather_obj_pts'):
#         points = data_dict['points'][:,:3]
#         frame_id = data_dict['frame_id']
#         pose = get_pose(dataset, frame_id)
#         trk_ids = get_frame_track_boxes(tracks, frame_id)[:,-1]
#         if len(trk_ids) == 0:
#             continue
            
#         points_global, _ = ego_to_world(pose, points=points.copy())
#         for trk_id in trk_ids:
#             if 'static_box' not in tracks[trk_id].keys():
#                 continue
#             if 'obj_pts' not in tracks[trk_id].keys():
#                 tracks[trk_id]['obj_pts'] = np.empty((0,4))            
#                 enlarged_box = tracks[trk_id]['static_box'].copy()
#                 enlarged_box[3:6] += np.array([enlarge_l, enlarge_w, enlarge_h])
#                 tracks[trk_id]['enlarged_box'] = enlarged_box
                
#             pts_in_box = get_points_in_box(points_global, tracks[trk_id]['enlarged_box'])[0]
#             pts_in_box = np.hstack([pts_in_box, np.ones((len(pts_in_box),1))*dataset.frameid_to_idx[frame_id]])
#             tracks[trk_id]['obj_pts'] = np.vstack([tracks[trk_id]['obj_pts'], pts_in_box])

# def refine_static_boxes_all_pts(tracks):    
#     """
#     This would work well if the vehicle ego-pose is 100% accurate. However when it isn't, the wrong accumulation of 
#     frames leads to elongated cars, and using 1 box for all frames leads to wrong positioning. If the ego-pose isn't
#     100%, it's better to use 16-frame box instead
#     """
#     for trk_id in tqdm(tracks.keys(), total=len(tracks.keys()), desc='resize_box'):
#         if 'obj_pts' not in tracks[trk_id].keys():
#             continue
#         new_box = resize_box(tracks[trk_id]['obj_pts'],
#                             tracks[trk_id]['static_box'],
#                             tracks[trk_id]['enlarged_box'])
#         tracks[trk_id]['refined_static_box'] = new_box

# def resize_box(pts_in_box, initial_box, enlarged_box, 
#                l_lim=3, w_lim=1, inc=0.05, dist_th=0.02,
#                l_skew_th=0.5, w_skew_th=0.1, 
#                side_mirror_spacing=0, loosen=0.00,
#                min_mean_intersect=20, min_intersect_pts=10, min_zero_intersect=300):
#     """
#     Resizes the box by computing the number of points that intersect the edge lines as we shift the line inwards from a distance away

#     inc (float): how much to increment/decrement the bounds for each loop
#     l_skew_th and w_skew_th (float): if the mean of the points is skewed too much to one side, it means 
#                     no observations on the opposite end, so we do not shrink on the 
#                     opposite end. l and w refer to length and width skew
#     w_lim (float): termination condition for shrinking box width
#     l_lim (float): termination condition for shrinking box length
#     side_mirror_spacing (float): Waymo/nuScenes label cars with side mirrors (set at ~0.05)
#     loosen (float): if you want to loosen the box a bit by adding a gap
#     """
#     pts_cn = common_utils.rotate_points_along_z((pts_in_box - initial_box[:3])[np.newaxis,...], 
#                                                 -np.array([initial_box[6]])).squeeze()
#     # Define box edges
#     l_min = (enlarged_box[3]-l_lim)/2
#     w_min = (enlarged_box[4]-w_lim)/2
#     pts_mean = np.mean(pts_cn,axis=0)[:2]
#     front_x, rear_x, right_y, left_y = None, None, None, None
#     if (abs(pts_mean[0]) < l_skew_th) or (pts_mean[0] > l_skew_th):
#         front_x = get_obj_bounds(pts_cn, line=np.array([enlarged_box[3]/2,0]), 
#                                inc=-np.array([inc,0]), lim=l_min, axis=0, loosen=loosen,
#                                dist_th=dist_th, min_pts=min_intersect_pts, min_mean_intersect=min_mean_intersect, 
#                                min_zero_intersect=min_zero_intersect)
        
#     if (abs(pts_mean[0]) < l_skew_th) or (pts_mean[0] < -l_skew_th):
#         rear_x = get_obj_bounds(pts_cn, line=np.array([-enlarged_box[3]/2,0]), 
#                               inc=np.array([inc,0]), lim=l_min, axis=0, loosen=-loosen,
#                                dist_th=dist_th, min_pts=min_intersect_pts, min_mean_intersect=min_mean_intersect, 
#                                min_zero_intersect=min_zero_intersect)
#     if (abs(pts_mean[1]) < w_skew_th) or (pts_mean[1] < -w_skew_th):
#         right_y = get_obj_bounds(pts_cn, line=np.array([0,-enlarged_box[4]/2]), 
#                                inc=np.array([0,inc]), lim=w_min, axis=1, loosen=-loosen,
#                                dist_th=dist_th, min_pts=min_intersect_pts, min_mean_intersect=min_mean_intersect, 
#                                min_zero_intersect=min_zero_intersect)        
#     if (abs(pts_mean[1]) < w_skew_th) or (pts_mean[1] > w_skew_th):
#         left_y = get_obj_bounds(pts_cn, line=np.array([0,enlarged_box[4]/2]), 
#                               inc=-np.array([0,inc]), lim=w_min, axis=1, loosen=loosen,
#                                dist_th=dist_th, min_pts=min_intersect_pts, min_mean_intersect=min_mean_intersect, 
#                                min_zero_intersect=min_zero_intersect)
        
#     front_x=initial_box[3]/2 if front_x is None else front_x
#     rear_x=-initial_box[3]/2 if rear_x is None else rear_x
#     right_y=-initial_box[4]/2 if right_y is None else right_y - side_mirror_spacing 
#     left_y=initial_box[4]/2 if left_y is None else left_y + side_mirror_spacing
    
#     new_l = front_x-rear_x
#     new_w = left_y-right_y
#     new_box = initial_box.copy()
#     adj_centre = np.array([[front_x - new_l/2, left_y - new_w/2, 0]])
#     rotated_adj_centre = common_utils.rotate_points_along_z(adj_centre[np.newaxis,...], 
#                                                             np.array([initial_box[6]])).squeeze()
#     new_box[:3] += rotated_adj_centre
#     new_box[3] = new_l
#     new_box[4] = new_w
#     return new_box

# def get_obj_bounds(pts_cn, line, inc, lim, loosen, axis=0, 
#                    dist_th=0.02, min_pts=10, min_mean_intersect=20, min_zero_intersect=300):
#     """
#     Moves the box edge line closer to the given limit and computes the number of intersecting points
#     min_zero_intersect (int): if the enlarged line (at index zero) already intersects min_zero_intersect points, we set box edge at that line
#     min_mean_intersect (int): number of points to consider as "spacing" between obstacles and the car itself
#     min_pts (int): set any num pts intersect below min_pts to min_pts cause e.g. 1->10 is a 10x increase but it's usually noise

#     """
#     intersects, lines = [], []
#     while(True):
#         dist = (pts_cn[:,:2] - line)[:,axis]
#         num_intersect = len(np.argwhere(abs(dist) < dist_th))
#         intersects.append(num_intersect)
#         lines.append(line[axis])
#         line += inc
#         if abs(line[axis]) < lim:
#             break
                
#     intersects = np.array(intersects)
#     intersects[intersects < min_pts] = min_pts
#     increase = (intersects[1:] - intersects[:-1])/intersects[:-1]    

#     # Filter out obstacles around car by looking for last min_pts before object body
#     start_idx = np.argwhere(intersects == min_pts)
#     if len(start_idx != 0):        
#         intersects = intersects[start_idx[-1].item():]
#         increase = increase[start_idx[-1].item():]
#         lines = lines[start_idx[-1].item():]
        
#     # Return
#     if intersects[0] > min_zero_intersect:
#         # sometimes box is not enlarged large enough and it already intersects object body
#         return lines[0] + loosen
#     elif np.mean(intersects) < min_mean_intersect:
#         return None
#     elif len(np.argwhere(intersects != min_pts)) > 0:
#         # return only if there are intersects above min_pts
#         return lines[np.argmax(increase)] + loosen
#     else:
#         return None
