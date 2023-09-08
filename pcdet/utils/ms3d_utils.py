from pcdet.config import cfg
import numpy as np
from tqdm import tqdm
import copy
from pathlib import Path
import pickle as pkl
from pcdet.utils.tracker_utils import get_frame_track_boxes, get_motion_state, delete_tracks
from pcdet.utils.box_fusion_utils import compute_iou, kbf
from pcdet.utils.compatibility_utils import get_sequence_name, get_sample_idx, get_frame_id

# For loader utils
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader

# ms3d key functions
from pcdet.utils.compatibility_utils import get_lidar, get_pose
from pcdet.utils.transform_utils import world_to_ego
from pcdet.utils.box_fusion_utils import nms
from pcdet.datasets.augmentor.augmentor_utils import get_points_in_box

# ======== Temporal and final label refinement key functions ========

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
        
        # Add dynamic interpolated/extrapolated tracks to replace lower scoring dets
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

def refine_veh_labels(dataset, frame_ids,
                      tracks_veh_all, tracks_veh_static, 
                      static_trk_score_th, veh_pos_th,
                      refine_cfg, save_dir=None):
    """
    Refine vehicle labels. Updates tracks_veh_all and tracks_veh_static in-place.
    
    If save_dir is specified, we save tracks after every few steps for inspection or 
    easy resuming of interrupted label refinement.
    """

    # Use pos_th for static min_score so that we ensure to have some confident detections in the static track
    delete_tracks(tracks_veh_all, min_score=veh_pos_th, 
                  num_boxes_abv_score=refine_cfg['TRACK_FILTERING']['MIN_DETS_ABOVE_POS_TH_FOR_TRACKS_VEH_ALL'])                   
    delete_tracks(tracks_veh_static, min_score=veh_pos_th, 
                  num_boxes_abv_score=refine_cfg['TRACK_FILTERING']['MIN_DETS_ABOVE_POS_TH_FOR_TRACKS_VEH_STATIC'])   
    delete_tracks(tracks_veh_static, min_score=0.0, 
                  num_boxes_abv_score=refine_cfg['TRACK_FILTERING']['MIN_NUM_STATIC_VEH_TRACKS'])   
    
     # Get static boxes using tracking information
    for trk_id in tracks_veh_all.keys():
        score_mask = tracks_veh_all[trk_id]['boxes'][:,7] > veh_pos_th
        tracks_veh_all[trk_id]['motion_state'] = get_motion_state(tracks_veh_all[trk_id]['boxes'][score_mask])    
    for trk_id in tracks_veh_static.keys():
        score_mask = tracks_veh_static[trk_id]['boxes'][:,7] > static_trk_score_th
        tracks_veh_static[trk_id]['motion_state'] = get_motion_state(tracks_veh_static[trk_id]['boxes'][score_mask])        

    # Updates motion-state of track dicts in-place
    matched_trk_ids = motion_state_refinement(tracks_veh_all, tracks_veh_static, frame_ids) # ~ 1:03HR for 18840                

    # Merge disjointed tracks and assign one box per frame in the ego-vehicle frame
    merge_disjointed_tracks(tracks_veh_all, tracks_veh_static, matched_trk_ids)    
    if save_dir is not None:
        save_data(tracks_veh_all, save_dir, name="tracks_world_veh_all_refined.pkl")
        save_data(tracks_veh_static, save_dir, name="tracks_world_veh_static_refined.pkl")

    get_track_rolling_kde_interpolation(dataset, tracks_veh_static, window=refine_cfg['ROLLING_KBF']['ROLLING_KDE_WINDOW'], 
                                                              static_score_th=static_trk_score_th, kdebox_min_score=refine_cfg['ROLLING_KBF']['MIN_STATIC_SCORE'])  # 16MIN for 18840
    propagate_static_boxes(dataset, tracks_veh_static, 
                                                     score_thresh=veh_pos_th,
                                                     min_static_tracks=refine_cfg['PROPAGATE_BOXES']['MIN_STATIC_TRACKS'],
                                                     n_extra_frames=refine_cfg['PROPAGATE_BOXES']['N_EXTRA_FRAMES'], 
                                                     degrade_factor=refine_cfg['PROPAGATE_BOXES']['DEGRADE_FACTOR'], 
                                                     min_score_clip=refine_cfg['PROPAGATE_BOXES']['MIN_SCORE_CLIP']) # < 1 min for 18840
    if save_dir is not None:
        save_data(tracks_veh_static, save_dir, name="tracks_world_veh_static_refined_prop_boxes.pkl")
    return tracks_veh_all, tracks_veh_static

def refine_ped_labels(tracks_ped, ped_pos_th, track_filtering_cfg):
    """
    Refine pedestrian labels    
    """
    # Classify if track is static or dynamic
    delete_tracks(tracks_ped, min_score=ped_pos_th, num_boxes_abv_score=track_filtering_cfg['MIN_DETS_ABOVE_POS_TH_FOR_TRACKS_PED'])                   
    for trk_id in tracks_ped.keys():
        tracks_ped[trk_id]['motion_state'] = get_motion_state(tracks_ped[trk_id]['boxes'], s2e_th=2) 

    # Delete tracks if less than N tracks
    delete_tracks(tracks_ped, min_score=0.0, num_boxes_abv_score=track_filtering_cfg['MIN_NUM_PED_TRACKS'])    

    # Set score to ped_pos_th and delete static tracks (unless specified with use_static_ped_tracks)
    all_ids = list(tracks_ped.keys())
    for trk_id in all_ids:
        mask = tracks_ped[trk_id]['boxes'][:,7] < ped_pos_th
        tracks_ped[trk_id]['boxes'][:,7][mask] = ped_pos_th
        if not track_filtering_cfg['USE_STATIC_PED_TRACKS']:
            if tracks_ped[trk_id]['motion_state'] != 1:            
                del tracks_ped[trk_id]
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

# ======== MS3D HELPER UTILS ========

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
                
def motion_state_refinement(tracks_all, tracks_static, frame_ids):
    """
    Compute iou of motion trails from the 1-frame tracklets against the 16-frame tracklets
    to refine motion states

    Returns:
        matched_trk_ids (list of tuples): list of (trk_id_16f, trk_id_1f) matched track ids
    """
    matched_trk_ids = []

    # Add 16f tracks to 1f tracks frame. Compare if they overlap
    for frame_id in tqdm(frame_ids, total=len(frame_ids), desc='motion_state_refinement'):
        boxes_static = get_frame_track_boxes(tracks_static, frame_id, nhistory=0)
        if len(boxes_static) == 0:
            continue
            
        boxes_all = get_frame_track_boxes(tracks_all, frame_id, nhistory=0)
        if len(boxes_all) == 0:
            continue

        # Get all motion trail if obj is dynamic
        for box in boxes_all:
            trk_id = box[-1]
            if tracks_all[trk_id]['motion_state'] == 1:
                tboxes = np.hstack([tracks_all[trk_id]['boxes'], np.ones((tracks_all[trk_id]['boxes'].shape[0],1))*trk_id])
                boxes_all = np.vstack([boxes_all, tboxes])
                
        ious, matched_pairs = compute_iou(boxes_static, boxes_all)
        matched_pairs = matched_pairs[ious > 0.1]

        # Compare matched pairs. If one is labelled dynamic, then set both tracklet to dynamic 
        for pair in matched_pairs:
            trkid_static = boxes_static[pair[0],-1]
            trkid_all = boxes_all[pair[1],-1]
            motion_state = tracks_static[trkid_static]['motion_state'] | tracks_all[trkid_all]['motion_state']
            tracks_static[trkid_static]['motion_state'] = motion_state
            tracks_all[trkid_all]['motion_state'] = motion_state

            # Only keep track of static obj ID matches for merging disjointed tracks. It shouldn't matter that we 
            # got the whole motion trail
            if motion_state == 0:
                matched_trk_ids.append((int(trkid_static), int(trkid_all)))
    
    return list(set(matched_trk_ids))


def assign_box_to_frameid(tracks):
    # assign one tracker box per frame_id; if duplicate, then keep higher scoring box
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
    """
    Merge disjointed tracks and assign one box per frame 
    TODO: used in MS3D++ but may not be that useful
    """
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
            boxes = np.insert(boxes, 7,1,1) # Static refinement only done for vehicle class so we hardcode class ID: 1
            kdebox = kbf(boxes, box_weights=boxes[:,-1], bw_score=1.0)
            if kdebox[8] > static_score_th:
                kdebox[8] = max(kdebox_min_score, kdebox[8])
            kdebox = np.delete(kdebox, 7)
            for frame_idx, f_id in enumerate(trk_frame_inds_set):
                tracks_static[trk_id]['frameid_to_rollingkde'][f_id] = kdebox
        
        else:
            # Get rolling KDE for each key frame
            for frame_idx, f_id in enumerate(trk_frame_inds_set):
                # TODO: It might actually work better to use [-window/2, window/2] here rather than historical only [-window, 0] because
                # seeing boxes from future frames may provide better context for refinement; though this may only be an incremental improvement
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

def propagate_static_boxes(dataset, tracks_static, score_thresh, min_static_tracks, n_extra_frames, degrade_factor, min_score_clip):
    for trk_id in tqdm(tracks_static.keys(), total=len(tracks_static), desc='propagate_static_boxes'):
        tracks_static[trk_id]['frameid_to_propboxes'] = {}
        roll_kde = tracks_static[trk_id]['frameid_to_rollingkde']
        tracks_static[trk_id]['frameid_to_propboxes'].update(roll_kde)
        if tracks_static[trk_id]['motion_state'] != 0:            
            continue       

        boxes = np.array(list(tracks_static[trk_id]['frameid_to_propboxes'].values()))
        if len(boxes[boxes[:,-1] > score_thresh]) < min_static_tracks:
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

# ======== MISC SAVE/LOADER UTILS ========

def save_data(data, folder, name):
    ps_path = Path(folder) / name
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(str(ps_path), 'wb') as f:
        pkl.dump(data, f)

def load_pkl(file):
    with open(file, 'rb') as f:
        loaded_pkl = pkl.load(f)
    return loaded_pkl

def load_if_exists(folder, name):
    ps_path = Path(folder) / name
    if ps_path.exists():
        return load_pkl(str(ps_path))        
    return None

def load_dataset(cfg, split):
    # Get target dataset    
    cfg.DATA_SPLIT.test = split
    if cfg.get('SAMPLED_INTERVAL', False):
        cfg.SAMPLED_INTERVAL.test = 1
    logger = common_utils.create_logger('unused_log.txt', rank=cfg.LOCAL_RANK) # TODO: remove the need to generate txt logger
    target_set, _, _ = build_dataloader(
                dataset_cfg=cfg,
                class_names=cfg.CLASS_NAMES,
                batch_size=1, logger=logger, training=False, dist=False, workers=1
            )      
    return target_set

