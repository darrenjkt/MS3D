import torch
import yaml
import numpy as np
from easydict import EasyDict
from pcdet.utils.compatibility_utils import get_target_domain_cfg, get_lidar
from demo_dataset import DemoDataset
from pcdet.utils import common_utils
from torch.utils.data import DataLoader
from pcdet.models import build_network, load_data_to_gpu
from functools import partial
from tqdm import tqdm 
import matplotlib.pyplot as plt


def load_yaml(fname):
    with open(fname, 'r') as f:
        try:
            ret = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            ret = yaml.safe_load(f)
    return EasyDict(ret)

def load_dataset_and_model(data_dir, ckpt_path, model_yaml, ext='.pcd'):
    cfg = load_yaml(model_yaml)
    target_domain_cfg = get_target_domain_cfg(cfg,'custom',sweeps=4,use_tta=3) # we later modify demo_dataset.data_augmentor.augmentor_configs.DISABLE_AUG_LIST
    logger = common_utils.create_logger()
    demo_dataset = DemoDataset(
        dataset_cfg=target_domain_cfg, class_names=target_domain_cfg.CLASS_NAMES, training=False,
        root_path=data_dir, ext=ext, logger=logger, sweeps=4
    )
    cfg = load_yaml(model_yaml)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    return demo_dataset, model
    
def generate_predictions(model, dataset, batch_size, sweeps, tta_setting):
    dataset.max_sweeps = sweeps
    
    # Configure tta
    if tta_setting == 0:
        dataset.data_augmentor.augmentor_configs.DISABLE_AUG_LIST = ['random_world_flip','random_world_rotation']
    elif tta_setting == 1:
        dataset.data_augmentor.augmentor_configs.DISABLE_AUG_LIST = ['random_world_rotation']
    elif tta_setting == 2:
        dataset.data_augmentor.augmentor_configs.DISABLE_AUG_LIST = ['random_world_flip']
    elif tta_setting == 3:
        dataset.data_augmentor.augmentor_configs.DISABLE_AUG_LIST = ['placeholder']
    else:
        print('Choose 0, 1, 2 or 3 for tta_setting')
        raise NotImplementedError

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=1,
        shuffle=False, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=None, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=None)
    )
    det_annos = []
    for i, batch_dict in tqdm(enumerate(dataloader), total=len(dataloader)):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, _ = model(batch_dict)

        # Undo tta and format pred dict
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, dataset.class_names,
            output_path=None)
        det_annos += annos

    for anno in det_annos: # return boxes in ground plane
        anno['boxes_lidar'][:,:3] += dataset.dataset_cfg.SHIFT_COOR

    return det_annos

def format_ensemble_preds(det_annos):
    det_annos['det_cls_weights'] = {}
    det_cls_weights = np.ones(3, dtype=np.int32)
    for key in det_annos.keys():
        det_annos['det_cls_weights'][key] = det_cls_weights
    return det_annos


## ============ VISUALIZATION FUNCTIONS =================
def plot_boxes(ax, boxes_lidar, color=[0,0,1], 
               scores=None, label=None, cur_id=0, limit_range=None,
               source_id=None, source_labels=None, alpha=1.0, 
               linestyle='solid',linewidth=1.0, fontsize=12,
               show_score=True):
    if limit_range is not None:
        centroids = boxes_lidar[:,:3]
        mask = mask_points_by_range(centroids, limit_range) 
        boxes_lidar = boxes_lidar[mask]
        if source_labels is not None:
            source_labels = source_labels[mask] 
        if source_id is not None:
            source_id = source_id[mask] 
        if scores is not None:
            scores = scores[mask]
        
    box_pts = boxes_to_corners_3d(boxes_lidar)
    box_pts_bev = box_pts[:,:5,:2]        
    cmap = np.array(plt.get_cmap('tab20').colors)    
    prev_id = -1
    for idx, box in enumerate(box_pts_bev): 
        if source_id is not None:
            cur_id = source_id[idx]
            color = cmap[cur_id % len(cmap)]
            label = None
            if source_labels is not None:
                label = source_labels[idx]
        
        if cur_id != prev_id:
            ax.plot(box[:,0],box[:,1], color=color, label=label, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
            prev_id = cur_id
        else:
            ax.plot(box[:,0],box[:,1], color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        if (scores is not None) and show_score:
            ax.text(box[0,0], box[0,1], f'{scores[idx]:0.4f}', c=color, fontsize=fontsize)                             

def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def visualize_bev(target_set, idx, ps_dict=None, ps_dict2=None, detection_sets=None, tracks=None, 
                  ps_dict_legend=None, ps_dict2_legend=None, tracks_legend=None, 
                  point_cloud_range=80, show_legend=True,
                  show_score=True, conf_th=0.2, above_pos_th=False, figsize=(9,9),
                  show_trk_score=False, frame2box_key=None):
    """
    show_trk_score: if False, number=track_id, else number=conf_score
    frame2box_key: (only after refine_veh_labels)
    """
    frame_id = target_set.infos[idx]['frame_id']
    pts = get_lidar(target_set, frame_id)    
    limit_range = [-point_cloud_range, -point_cloud_range, -4.0, point_cloud_range, point_cloud_range, 2.0]
    mask = mask_points_by_range(pts, limit_range)
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    pts = pts[mask]
    ax.scatter(pts[:,0],pts[:,1],s=0.3, c='black', marker='o')

    if detection_sets:
        conf_mask = detection_sets[idx]['score'] >= conf_th
        plot_boxes(ax, detection_sets[idx]['boxes_lidar'][conf_mask], 
                scores=detection_sets[idx]['score'][conf_mask],
                source_id=detection_sets[idx]['source_id'][conf_mask] if 'source_id' in detection_sets[idx].keys() else None,
                source_labels=detection_sets[idx]['source'][conf_mask] if 'source' in detection_sets[idx].keys() else None,
                color=[0,0.8,0] if 'source_id' not in detection_sets[idx].keys() else [0,0,1],
                limit_range=limit_range, alpha=0.5 if 'source_id' in detection_sets[idx].keys() else 1.0,
                label='detection_sets' if 'source_id' not in detection_sets[idx].keys() else None, show_score=show_score)
        
    if ps_dict2:
        combined_mask = ps_dict2[frame_id]['gt_boxes'][:,8] >= conf_th
        if above_pos_th:
            above_pos_mask = ps_dict2[frame_id]['gt_boxes'][:,7] > 0
            combined_mask = np.logical_and(combined_mask, above_pos_mask)
        plot_boxes(ax, ps_dict2[frame_id]['gt_boxes'][combined_mask], 
                scores=ps_dict2[frame_id]['gt_boxes'][combined_mask][:,8],
                label='ps labels 2' if ps_dict2_legend is None else ps_dict2_legend, color=[0.8,0,0],
                limit_range=limit_range, alpha=1, show_score=show_score)

    if ps_dict:
        combined_mask = ps_dict[frame_id]['gt_boxes'][:,8] >= conf_th
        if above_pos_th:
            above_pos_mask = ps_dict[frame_id]['gt_boxes'][:,7] > 0
            combined_mask = np.logical_and(combined_mask, above_pos_mask)
        plot_boxes(ax, ps_dict[frame_id]['gt_boxes'][combined_mask], 
                scores=ps_dict[frame_id]['gt_boxes'][combined_mask][:,8],
                label='ps labels' if ps_dict_legend is None else ps_dict_legend, 
                color=[0,0.8,0], fontsize=14, linewidth=1.5,
                limit_range=limit_range, alpha=1, show_score=show_score)
        
    if tracks:
        from pcdet.utils import compatibility_utils as compat
        from pcdet.utils.transform_utils import world_to_ego
        from pcdet.utils.tracker_utils import get_frame_track_boxes
        track_boxes = get_frame_track_boxes(tracks, frame_id, frame2box_key=frame2box_key, nhistory=0)
        pose = compat.get_pose(target_set, frame_id)
        score_idx = 7 if show_trk_score else 8
        _, track_boxes_ego = world_to_ego(pose, boxes=track_boxes)
        if track_boxes_ego.shape[0] != 0:
            plot_boxes(ax, track_boxes_ego[:,:7], 
                    scores=track_boxes_ego[:,score_idx],
                    label='tracked boxes' if tracks_legend is None else tracks_legend, color=[1,0,0], linestyle='dotted',
                    limit_range=limit_range, alpha=1, show_score=show_score) 

    ax.set_title(f'Frame #{idx}, FID:{frame_id}')
    ax.set_aspect('equal')
    if show_legend:
        ax.legend(loc='upper right')
    plt.show()
    return fig, ax
                    