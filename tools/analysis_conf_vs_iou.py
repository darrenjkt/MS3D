import pickle
import sys
sys.path.append('/MS3D')
import numpy as np
import matplotlib.pyplot as plt
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
import torch
from scipy.spatial import cKDTree
from pcdet.utils import compatibility_utils as compat
from tqdm import tqdm

def load_pkl(file):
    with open(file,'rb') as f:
        data = pickle.load(f)
    return data

def load_dataset(split, sampled_interval):

    # Get target dataset    
    cfg.DATA_SPLIT.test = split
    cfg.SAMPLED_INTERVAL.test = sampled_interval
    logger = common_utils.create_logger('temp.txt', rank=cfg.LOCAL_RANK)
    target_set, _, _ = build_dataloader(
                dataset_cfg=cfg,
                class_names=cfg.CLASS_NAMES,
                batch_size=1, logger=logger, training=False, dist=False, workers=1
            )      
    return target_set

def find_nearest_gtbox(frame_gt_boxes, pred_box, return_iou=True):
    # Assess IOU of combined box with GT
    # Find closest GT to our chosen box    
    gt_tree = cKDTree(frame_gt_boxes[:,:3])
    nearest_gt = gt_tree.query_ball_point(pred_box[:3].reshape(1,-1), r=2.0)
    if len(nearest_gt[0]) == 0:        
        return None
    nearest_gt_box = frame_gt_boxes[nearest_gt[0][0]]
    if return_iou:
        gt_box = np.reshape(nearest_gt_box, (1, -1))
        gt_box_cuda = torch.from_numpy(gt_box.astype(np.float32)).cuda()
        pred_box_cuda = torch.from_numpy(pred_box.reshape(1,-1).astype(np.float32)).cuda()

        iou = iou3d_nms_utils.boxes_iou3d_gpu(gt_box_cuda, pred_box_cuda)
        return (nearest_gt_box, iou.item())
    else:
        return nearest_gt_box

def get_conf_iou(dataset, det_annos):
    confs = []
    ious = []
    for frame_preds in tqdm(det_annos, total=len(det_annos)):
        frame_id = frame_preds['frame_id']
        
        gt_names = compat.get_gt_names(dataset, frame_id)
        class_mask = np.isin(gt_names, ['Vehicle'])
        gt_boxes_3d = compat.get_gt_boxes(dataset, frame_id)[class_mask]
        gt_boxes_3d[:,:3] += dataset.dataset_cfg.SHIFT_COOR

        for idx, _ in enumerate(frame_preds['boxes_lidar']):        
            ret = find_nearest_gtbox(gt_boxes_3d[:,:7], frame_preds['boxes_lidar'][idx], return_iou=True)
            if ret is None:
                ious.append(0.0)
            else:
                ious.append(ret[1])
            confs.append(frame_preds['score'][idx])

    confs_np = np.array(confs)
    ious_np = np.array(ious)
    return confs_np, ious_np

cfg_file = '/MS3D/tools/cfgs/dataset_configs/waymo_dataset_da.yaml'
cfg_from_yaml_file(cfg_file, cfg)
cfg.USE_CUSTOM_TRAIN_SCENES = True
dataset = load_dataset(split='train', sampled_interval=2)
print('Dataset loaded')

pkl_file = '/MS3D/tools/cfgs/target_waymo/ps_labels_rnd2/final_ps_dict.pkl'

det1 = load_pkl(pkl_file)

# analysis=False
# confs_np, ious_np = get_conf_iou(dataset, det1)
# with open('/MS3D/tools/cfgs/target_waymo/analysis/voxa_l3_w5_ft_confs_np_18840.pkl','wb') as f:
#     pickle.dump(confs_np, f)
# with open('/MS3D/tools/cfgs/target_waymo/analysis/voxa_l3_w5_ft_ious_np_18840.pkl','wb') as f:
#     pickle.dump(ious_np, f)

# if not analysis:
#     exit()

with open('/MS3D/tools/cfgs/target_waymo/analysis/voxc_l3_w5_ft_confs_np_18840.pkl','rb') as f:
    # pickle.dump(confs_np, f)
    confs_np1 = pickle.load(f)

with open('/MS3D/tools/cfgs/target_waymo/analysis/voxc_l3_w5_ft_ious_np_18840.pkl','rb') as f:
    ious_np1 = pickle.load(f)

with open('/MS3D/tools/cfgs/target_waymo/analysis/voxa_l3_w5_ft_confs_np_18840.pkl','rb') as f:
    confs_np2 = pickle.load(f)    

with open('/MS3D/tools/cfgs/target_waymo/analysis/voxa_l3_w5_ft_ious_np_18840.pkl','rb') as f:
    ious_np2 = pickle.load(f)    

# plt.scatter(confs_np[ious_np > 0.1], ious_np[ious_np > 0.1], s=0.05)
# plt.xlabel('Confidence')
# plt.ylabel('IoU')
# plt.grid(True)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.show()

confs_np = confs_np2
ious_np = ious_np2
color='blue'

# conf_mask_init = confs_np > 0.3
# confs_np = confs_np[conf_mask_init]
# ious_np = ious_np[conf_mask_init]

tps = ious_np > 0.7
fps = ious_np < 0.7

bin_size = 0.1
conf_bins = []
num_tp_list = []
num_fp_list = []
precision_list = []
recall_list = []
counts = []
rounded_conf = np.floor(confs_np/bin_size)*bin_size
for val in np.arange(0.3,1.0,bin_size):
    conf_mask = rounded_conf >= val
    if np.count_nonzero(conf_mask) == 0:
        continue
    num_tp = np.count_nonzero(tps[conf_mask])
    num_fp = np.count_nonzero(fps[conf_mask])
    conf_bins.append(val)
    precision_list.append(num_tp/(num_tp+num_fp))
    num_tp_list.append(num_tp)
    num_fp_list.append(num_fp)
    counts.append(np.count_nonzero(conf_mask))

conf_bins = np.array(conf_bins)
num_tp_list = np.array(num_tp_list)
num_fp_list = np.array(num_fp_list)
precision_list = np.array(precision_list)
counts = np.array(counts)
pmf = counts/len(confs_np)

# hist, bin_edges = np.histogram(confs_np, bins=len(np.arange(0.0,1.0,bin_size)))
# pmf = hist/len(confs_np) # Sum of all bin heights equals to 1

plt.bar(conf_bins[1:], pmf[1:], alpha=0.3, width=bin_size/1.5, color=color, label='% of total predictions')
plt.plot(conf_bins[1:], precision_list[1:], color=color, label='precision')
plt.xticks(np.arange(0.2,1,bin_size))
plt.yticks(np.arange(0,1.05,0.1))
plt.grid(True)
plt.legend()
plt.xlabel('Confidence Threshold')
plt.show()

print('Done')