import torch
import os
import glob
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
from pcdet.config import cfg
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, commu_utils, memory_ensemble_utils
import pcdet.utils.box_fusion_utils as box_fusion_utils
from pcdet.utils.transform_utils import world_to_ego
import pickle as pkl
import re
from pathlib import Path
from pcdet.models.model_utils.dsnorm import set_ds_target
from pcdet.utils import generate_ps_utils 
from pcdet.utils import tracker_utils
from pcdet.utils import compatibility_utils as compat
import yaml

#     print("Using Manager.dict() for multi-gpu training")
#     from multiprocessing import Manager
#     PSEUDO_LABELS = Manager().dict()

PSEUDO_LABELS = {}
NEW_PSEUDO_LABELS = {}  

def check_already_exist_pseudo_label(ps_label_dir, start_epoch):
    """
    if we continue training, use this to directly
    load pseudo labels from exsiting result pkl
    if exsit, load latest result pkl to PSEUDO LABEL
    otherwise, return false and
    Args:
        ps_label_dir: dir to save pseudo label results pkls.
        start_epoch: start epoc
    Returns:
    """
    # support init ps_label given by cfg
    if start_epoch == 0 and cfg.SELF_TRAIN.get('INIT_PS', None):
        if os.path.exists(cfg.SELF_TRAIN.INIT_PS):
            init_ps_label = pkl.load(open(cfg.SELF_TRAIN.INIT_PS, 'rb'))
            filter_ps_by_neg_score(init_ps_label)
            PSEUDO_LABELS.update(init_ps_label)
            if cfg.LOCAL_RANK == 0:
                ps_path = os.path.join(ps_label_dir, "ps_label_e0.pkl")
                with open(ps_path, 'wb') as f:
                    pkl.dump(init_ps_label, f)

            return cfg.SELF_TRAIN.INIT_PS

    ps_label_list = glob.glob(os.path.join(ps_label_dir, 'ps_label_e*.pkl'))
    if len(ps_label_list) == 0:
        return

    ps_label_list.sort(key=os.path.getmtime, reverse=True)
    for cur_pkl in ps_label_list:
        num_epoch = re.findall('ps_label_e(.*).pkl', cur_pkl)
        assert len(num_epoch) == 1

        # load pseudo label and return
        if int(num_epoch[0]) <= start_epoch:
            latest_ps_label = pkl.load(open(cur_pkl, 'rb'))
            filter_ps_by_neg_score(latest_ps_label)
            PSEUDO_LABELS.update(latest_ps_label)            
            return cur_pkl

    return None


def save_pseudo_label_epoch(model, val_loader, rank, leave_pbar, ps_label_dir, cur_epoch):
    """
    Generate pseudo label with given model.
    Args:
        model: model to predict result for pseudo label
        val_loader: data_loader to predict pseudo label
        rank: process rank
        leave_pbar: tqdm bar controller
        ps_label_dir: dir to save pseudo label
        cur_epoch
    """
    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)

    if rank == 0:
        pbar = tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='generate_ps_e%d' % cur_epoch, dynamic_ncols=True)

    pos_ps_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))
    ign_ps_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))

    if cfg.SELF_TRAIN.get('DSNORM', None):
        model.apply(set_ds_target)

    model.eval()

    for cur_it in range(total_it_each_epoch):
        try:
            target_batch = next(val_dataloader_iter)
        except StopIteration:
            target_dataloader_iter = iter(val_loader)
            target_batch = next(target_dataloader_iter)

        # generate gt_boxes for target_batch and update model weights
        with torch.no_grad():
            load_data_to_gpu(target_batch)
            pred_dicts, ret_dict = model(target_batch)

        pos_ps_batch_nmeters, ign_ps_batch_nmeters = save_pseudo_label_batch(
            target_batch, pred_dicts=pred_dicts,
            need_update=(cfg.SELF_TRAIN.get('MEMORY_ENSEMBLE', None) and
                         cfg.SELF_TRAIN.MEMORY_ENSEMBLE.ENABLED and
                         cur_epoch > 0)
        )   

        # log to console and tensorboard
        pos_ps_nmeter.update(pos_ps_batch_nmeters)
        ign_ps_nmeter.update(ign_ps_batch_nmeters)
        pos_ps_result = pos_ps_nmeter.aggregate_result()
        ign_ps_result = ign_ps_nmeter.aggregate_result()

        disp_dict = {'pos_ps_box': pos_ps_result, 'ign_ps_box': ign_ps_result}

        if rank == 0:
            pbar.update()
            pbar.set_postfix(disp_dict)
            pbar.refresh()

    if rank == 0:
        pbar.close()

    gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch)

def gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch):
    commu_utils.synchronize()

    if dist.is_initialized():
        part_pseudo_labels_list = commu_utils.all_gather(NEW_PSEUDO_LABELS)

        new_pseudo_label_dict = {}
        for pseudo_labels in part_pseudo_labels_list:
            new_pseudo_label_dict.update(pseudo_labels)

        NEW_PSEUDO_LABELS.update(new_pseudo_label_dict)

    # dump new pseudo label to given dir
    if rank == 0:
        ps_path = os.path.join(ps_label_dir, "ps_label_e{}.pkl".format(cur_epoch))
        with open(ps_path, 'wb') as f:
            pkl.dump(NEW_PSEUDO_LABELS, f)

    commu_utils.synchronize()
    PSEUDO_LABELS.clear()
    PSEUDO_LABELS.update(NEW_PSEUDO_LABELS)
    NEW_PSEUDO_LABELS.clear()


def save_pseudo_label_batch(input_dict,
                            pred_dicts=None,
                            need_update=True):
    """
    Save pseudo label for give batch.
    If model is given, use model to inference pred_dicts,
    otherwise, directly use given pred_dicts.
    Args:
        input_dict: batch data read from dataloader
        pred_dicts: Dict if not given model.
            predict results to be generated pseudo label and saved
        need_update: Bool.
            If set to true, use consistency matching to update pseudo label
    """
    pos_ps_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))
    ign_ps_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))

    batch_size = len(pred_dicts)
    for b_idx in range(batch_size):
        if 'pred_boxes' in pred_dicts[b_idx]:
            # Exist predicted boxes passing self-training score threshold
            pred_boxes = pred_dicts[b_idx]['pred_boxes'].detach().cpu().numpy()
            pred_labels = pred_dicts[b_idx]['pred_labels'].detach().cpu().numpy()
            pred_scores = pred_dicts[b_idx]['pred_scores'].detach().cpu().numpy()

            # Transferring Lyft/nuScenes -> Waymo we want to combine car/truck/bus into Vehicle
            # Class merging - need to address other areas of code too
            # pred_labels = pred_labels[pred_labels <= len(cfg.CLASS_NAMES)]
            # pred_boxes = pred_boxes[pred_labels <= len(cfg.CLASS_NAMES)]
            # pred_scores = pred_scores[pred_labels <= len(cfg.CLASS_NAMES)]
            # pred_names = np.array(cfg.CLASS_NAMES)[pred_labels - 1]
            # mapped_names = np.array([cfg.DATA_CONFIG_TAR.CLASS_MAPPING[name] for name in pred_names])
            # pred_labels = np.array([cfg.DATA_CONFIG_TAR.CLASS_NAMES.index(n) + 1 for n in mapped_names], dtype=np.int32)
            
            # HARDCODED combination of all car/truck/bus/Vehicle to class_id=1
            # TODO: Adapt this for multi-class with above code (need to address other parts of code too)
            if np.unique(np.abs(pred_labels)).shape[0] > 1:
                pred_labels = np.ones(pred_labels.shape)

            # remove boxes under negative threshold
            if cfg.SELF_TRAIN.get('NEG_THRESH', None):                
                remain_mask = pred_scores >= cfg.SELF_TRAIN.NEG_THRESH
                pred_labels = pred_labels[remain_mask]
                pred_scores = pred_scores[remain_mask]
                pred_boxes = pred_boxes[remain_mask]

            ignore_mask = pred_scores < cfg.SELF_TRAIN.SCORE_THRESH
            pred_labels[ignore_mask] = -pred_labels[ignore_mask]

            gt_box = np.concatenate((pred_boxes,
                                     pred_labels.reshape(-1, 1),
                                     pred_scores.reshape(-1, 1)), axis=1)

        else:
            # no predicted boxes passes self-training score threshold
            gt_box = np.zeros((0, 9), dtype=np.float32)

        gt_infos = {
            'gt_boxes': gt_box,
            'memory_counter': np.zeros(gt_box.shape[0])
        }

        # record pseudo label to pseudo label dict
        if need_update:
            ensemble_func = getattr(memory_ensemble_utils, cfg.SELF_TRAIN.MEMORY_ENSEMBLE.NAME)
            gt_infos = memory_ensemble_utils.memory_ensemble(
                PSEUDO_LABELS[input_dict['frame_id'][b_idx]], gt_infos,
                cfg.SELF_TRAIN, ensemble_func
            )            

        # counter the number of ignore boxes for each class
        for i in range(ign_ps_nmeter.n):
            num_total_boxes = (np.abs(gt_infos['gt_boxes'][:, 7]) == (i+1)).sum()
            ign_ps_nmeter.update((gt_infos['gt_boxes'][:, 7] == -(i+1)).sum(), index=i)
            pos_ps_nmeter.update(num_total_boxes - ign_ps_nmeter.meters[i].val, index=i)

        NEW_PSEUDO_LABELS[input_dict['frame_id'][b_idx]] = gt_infos
    
    return pos_ps_nmeter, ign_ps_nmeter


def load_ps_label(frame_id):
    """
    :param frame_id: file name of pseudo label
    :return gt_box: loaded gt boxes (N, 9) [x, y, z, w, l, h, ry, label, scores]
    """
    if frame_id in PSEUDO_LABELS:
        gt_box = PSEUDO_LABELS[frame_id]['gt_boxes']
    else:
        raise ValueError('Cannot find pseudo label for frame: %s' % frame_id)

    return gt_box

def get_num_pts(frame_id):
    """
    :param frame_id: file name of pseudo label
    :return gt_box: loaded gt boxes (N, 9) [x, y, z, w, l, h, ry, label, scores]
    """
    if frame_id in PSEUDO_LABELS:
        num_pts = PSEUDO_LABELS[frame_id]['num_pts']
    else:
        raise ValueError('Cannot find num_pts for frame: %s' % frame_id)

    return num_pts

def init_multi_source_ps_label(dataset, ps_label_dir):
    # Set data to use 5Hz data (Waymo/Lyft)    
    if dataset.dataset_cfg.get('SAMPLED_INTERVAL', False):
        orig_interval = dataset.dataset_cfg.SAMPLED_INTERVAL.train
        if dataset.dataset_cfg.DATASET == 'WaymoDataset':
            dataset.dataset_cfg.SAMPLED_INTERVAL.train = 2
            dataset.reload_infos()
        if dataset.dataset_cfg.DATASET == 'LyftDataset':        
            dataset.dataset_cfg.SAMPLED_INTERVAL.train = 1
            dataset.reload_infos()

    ms_cfg = cfg.SELF_TRAIN.MS_DETECTOR_PS

    # Get preliminary pseudo labels for 1 and 16-frame point clouds
    ps_dict_1f = generate_ps_utils.load_if_exists(ps_label_dir, name="ps_dict_1f.pkl")
    if ps_dict_1f is None:
        ps_dict_1f = generate_ps_utils.get_multi_source_prelim_label(ms_cfg, ms_cfg.ACCUM1, desc='gen_ps_label_1f')    
        generate_ps_utils.save_data(ps_dict_1f, ps_label_dir, name="ps_dict_1f.pkl")

    ps_dict_16f = generate_ps_utils.load_if_exists(ps_label_dir, name="ps_dict_16f.pkl")
    if ps_dict_16f is None:
        ps_dict_16f = generate_ps_utils.get_multi_source_prelim_label(ms_cfg, ms_cfg.ACCUM16, desc='gen_ps_label_16f')    
        generate_ps_utils.save_data(ps_dict_16f, ps_label_dir, name="ps_dict_16f.pkl")

    # Get tracklets for refinement
    tracks_1f_world = generate_ps_utils.load_if_exists(ps_label_dir, name="tracks_1f_world.pkl")
    if tracks_1f_world is None:
        tracks_1f_world = tracker_utils.get_tracklets(dataset, ps_dict_1f, cfg_path=ms_cfg.TRACKING.ACCUM1_CFG, anno_frames_only=False)
        generate_ps_utils.save_data(tracks_1f_world, ps_label_dir, name="tracks_1f_world.pkl")

    tracks_16f_world = generate_ps_utils.load_if_exists(ps_label_dir, name="tracks_16f_world.pkl")
    if tracks_16f_world is None:
        tracks_16f_world = tracker_utils.get_tracklets(dataset, ps_dict_16f, cfg_path=ms_cfg.TRACKING.ACCUM16_CFG, anno_frames_only=True)
        generate_ps_utils.save_data(tracks_16f_world, ps_label_dir, name="tracks_16f_world.pkl")

    # Delete tracks if less than MIN_DETS_FOR_TRACK detections in the tracklet
    tracks_1f_world_refined = generate_ps_utils.load_if_exists(ps_label_dir, name="tracks_1f_world_refined.pkl")
    tracks_16f_world_refined = generate_ps_utils.load_if_exists(ps_label_dir, name="tracks_16f_world_refined.pkl")
    configs = yaml.load(open(ms_cfg.TRACKING.ACCUM16_CFG, 'r'), Loader=yaml.Loader)
    trk_score_th_16f = configs['running']['score_threshold']
    if (tracks_1f_world_refined is None) and (tracks_16f_world_refined is None):
        tracks_1f_world_refined = tracks_1f_world
        tracks_16f_world_refined = tracks_16f_world
        tracker_utils.delete_tracks(tracks_1f_world_refined, min_score=cfg.SELF_TRAIN.SCORE_THRESH, num_min_dets=ms_cfg.MIN_DETS_FOR_TRACK_1F)                   
        tracker_utils.delete_tracks(tracks_16f_world_refined, min_score=trk_score_th_16f, num_min_dets=ms_cfg.MIN_DETS_FOR_TRACK_16F)   
        
        # Get static boxes using tracking information
        for trk_id in tracks_1f_world_refined.keys():
            score_mask = tracks_1f_world_refined[trk_id]['boxes'][:,7] > cfg.SELF_TRAIN.SCORE_THRESH
            tracks_1f_world_refined[trk_id]['motion_state'] = tracker_utils.get_motion_state(tracks_1f_world_refined[trk_id]['boxes'][score_mask])    
        for trk_id in tracks_16f_world_refined.keys():
            score_mask = tracks_16f_world_refined[trk_id]['boxes'][:,7] > trk_score_th_16f
            tracks_16f_world_refined[trk_id]['motion_state'] = tracker_utils.get_motion_state(tracks_16f_world_refined[trk_id]['boxes'][score_mask])        

        # Updates motion-state of track dicts in-place
        matched_trk_ids = generate_ps_utils.motion_state_refinement(tracks_1f_world_refined, tracks_16f_world_refined, list(ps_dict_16f.keys()))                

        # Merge disjointed tracks and assign one box per frame in the ego-vehicle frame
        generate_ps_utils.merge_disjointed_tracks(tracks_1f_world_refined, tracks_16f_world_refined, matched_trk_ids)    
        generate_ps_utils.save_data(tracks_1f_world_refined, ps_label_dir, name="tracks_1f_world_refined.pkl")
        generate_ps_utils.save_data(tracks_16f_world_refined, ps_label_dir, name="tracks_16f_world_refined.pkl")
        
    tracks_16f_world_rke = generate_ps_utils.load_if_exists(ps_label_dir, name="tracks_16f_world_rkde.pkl")
    if tracks_16f_world_rke is None:
        tracks_16f_world_rke = tracks_16f_world_refined
        generate_ps_utils.get_track_rolling_kde_interpolation(dataset, tracks_16f_world_rke, window=ms_cfg.ROLLING_KDE_WINDOW, 
                                                              ps_score_th=cfg.SELF_TRAIN.SCORE_THRESH, kdebox_min_score=ms_cfg.MIN_STATIC_SCORE)
        generate_ps_utils.save_data(tracks_16f_world_rke, ps_label_dir, name="tracks_16f_world_rkde.pkl")

    if ms_cfg.PROPAGATE_STATIC_BOXES.ENABLED:
        tracks_16f_world_proprkde = generate_ps_utils.load_if_exists(ps_label_dir, name="tracks_16f_world_proprkde.pkl")
        if tracks_16f_world_proprkde is None:
            tracks_16f_world_proprkde = tracks_16f_world_rke
            generate_ps_utils.propagate_static_boxes(dataset, tracks_16f_world_proprkde, 
                                                     score_thresh=trk_score_th_16f,
                                                     min_dets=ms_cfg.PROPAGATE_STATIC_BOXES.MIN_DETS,
                                                     n_extra_frames=ms_cfg.PROPAGATE_STATIC_BOXES.N_EXTRA_FRAMES, 
                                                     degrade_factor=ms_cfg.PROPAGATE_STATIC_BOXES.DEGRADE_FACTOR, 
                                                     min_score_clip=ms_cfg.PROPAGATE_STATIC_BOXES.MIN_SCORE_CLIP)
            generate_ps_utils.save_data(tracks_16f_world_proprkde, ps_label_dir, name="tracks_16f_world_proprkde.pkl")
        
        frame2box_key = 'frameid_to_extrollingkde'
        tracks_16f_world_final = tracks_16f_world_proprkde
    else:
        frame2box_key = 'frameid_to_rollingkde'
        tracks_16f_world_final = tracks_16f_world_rke

    final_ps_dict = generate_ps_utils.update_ps(dataset, ps_dict_1f, tracks_1f_world_refined, tracks_16f_world_final, 
                                                frame2box_key_16f=frame2box_key, frame2box_key_1f='frameid_to_box', frame_ids=list(ps_dict_16f.keys()))
    NEW_PSEUDO_LABELS.update(final_ps_dict)
    gather_and_dump_pseudo_label_result(rank=0, ps_label_dir=ps_label_dir, cur_epoch=0)

    # Reset dataset infos to the cfg setting
    if dataset.dataset_cfg.get('SAMPLED_INTERVAL', False) and \
    dataset.dataset_cfg.DATASET in ['WaymoDataset', 'LyftDataset']:
        dataset.dataset_cfg.SAMPLED_INTERVAL.train = orig_interval
        dataset.reload_infos()    


def filter_ps_by_neg_score(mydict):
    for frame_id in mydict.keys():
        score_mask = mydict[frame_id]['gt_boxes'][:,8] > cfg.SELF_TRAIN.NEG_THRESH
        for key in mydict[frame_id].keys():
            mydict[frame_id][key] = mydict[frame_id][key][score_mask]        