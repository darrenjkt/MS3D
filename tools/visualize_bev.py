import pickle
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append('/OpenPCDet')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from visual_utils import common_vis
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pcdet.utils import box_fusion_utils
from pcdet.utils import compatibility_utils as compat

"""
python visualize_bev.py --cfg_file cfgs/source-waymo/secondiou.yaml \
  --pkl ../output/source-kitti/centerpoint/default/eval/epoch_no_number/val/oncetrain/result.pkl \
        ../output/source-kitti/secondiou/default/eval/epoch_no_number/val/oncetrain/result.pkl \
        ../output/source-nuscenes/centerpoint/default/eval/epoch_no_number/val/oncetrain/result.pkl \
        ../output/source-nuscenes/secondiou/default/eval/epoch_no_number/val/oncetrain/result.pkl \
        ../output/source-waymo/centerpoint/default/eval/epoch_no_number/val/oncetrain/result.pkl \
        ../output/source-waymo/secondiou/default/eval/epoch_no_number/val/oncetrain/result.pkl

or

python visualize_bev.py --cfg_file cfgs/source-waymo/secondiou.yaml \
                        --dets_txt /OpenPCDet/tools/cfgs/source_detectors_train_tta.txt
"""

def plot_boxes(ax, boxes_lidar, color=[0,0,1], 
               scores=None, label=None, cur_id=0, limit_range=None,
               source_id=None, source_labels=None, alpha=1.0, linestyle='solid',linewidth=1.0):
    if limit_range is not None:
        centroids = boxes_lidar[:,:3]
        mask = common_vis.mask_points_by_range(centroids, limit_range) 
        boxes_lidar = boxes_lidar[mask]
        if source_labels is not None:
            source_labels = source_labels[mask] 
        if source_id is not None:
            source_id = source_id[mask] 
        if scores is not None:
            scores = scores[mask]
        
    box_pts = common_vis.boxes_to_corners_3d(boxes_lidar)
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
        if scores is not None:
            ax.text(box[0,0], box[0,1], f'{scores[idx]:0.4f}', c=color, size='medium')                             

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, required=True,
                        help='this config file just needs to have the correct target dataset')
    parser.add_argument('--dets_txt', type=str, default=None, required=False,
                        help='txt file containing detector pkl paths')                        
    parser.add_argument('--pkl', nargs='+', required=False,
                        help='Use saved detections from pkl path')
    parser.add_argument('--idx', type=int, default=0,
                        help='If you wish to only display a certain frame index')
    args = parser.parse_args()
    log_file = 'temp.txt'

    # Define which classes to display for gt_boxes
    # classes = ['Vehicle']
    classes = ['car','bus','truck']

    # Get target dataset
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    if cfg.get('DATA_CONFIG_TAR', False):
        tgt_dataset_cfg = cfg.DATA_CONFIG_TAR
        src_class_names = cfg.DATA_CONFIG_TAR.CLASS_NAMES
    else:
        tgt_dataset_cfg = cfg.DATA_CONFIG
        src_class_names = cfg.CLASS_NAMES
    tgt_dataset_cfg.DATA_SPLIT.test='train'
    if tgt_dataset_cfg.get('USE_TTA', False):
        tgt_dataset_cfg.USE_TTA=False

    # tgt_dataset_cfg.SEQUENCE_CONFIG.ENABLED = True
    # if tgt_dataset_cfg.SEQUENCE_CONFIG.ENABLED:
    #     tgt_dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET = [-15,0]
    #     # tgt_dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list=['x','y','z','intensity','timestamp']
    #     tgt_dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list=['x', 'y', 'z', 'intensity', 'elongation', 'timestamp']
    #     tgt_dataset_cfg.POINT_FEATURE_ENCODING.used_feature_list=['x','y','z']        
    cfg.DATA_CONFIG_TAR.USE_PSEUDO_LABEL=False
    target_set, _, _ = build_dataloader(
                dataset_cfg=tgt_dataset_cfg,
                class_names=src_class_names,
                batch_size=1, logger=logger, training=False, dist=False
            )

    if args.dets_txt is not None:
        det_annos = box_fusion_utils.load_src_paths_txt(args.dets_txt)
    else:
        det_annos = box_fusion_utils.load_src_paths_txt(args.pkl)    

    combined_dets = box_fusion_utils.combine_box_pkls(det_annos, classes, score_th=0.3)
    
    start_idx = args.idx
    start_frame_id = compat.get_frame_id(target_set, target_set.infos[start_idx])
    pts = target_set[start_idx]['points']

    pcr = 75
    limit_range = [-pcr, -pcr, -5.0, pcr, pcr, 3.0]
    mask = common_vis.mask_points_by_range(pts, limit_range)
    
    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot(111)
    scatter = ax.scatter(pts[mask][:,0],pts[mask][:,1],s=0.5, c='black', marker='o')
    # Plot GT boxes
    classes= ['car','truck','bus'] 
    class_mask = np.array([n in classes for n in compat.get_gt_names(target_set, start_frame_id)], dtype=np.bool_)
    plot_boxes(ax, compat.get_gt_boxes(target_set, start_frame_id)[class_mask], color=[0,0,1], 
                limit_range=limit_range, label='gt_boxes',
                scores=np.ones(compat.get_gt_boxes(target_set, start_frame_id)[class_mask].shape[0]))

    # Plot det boxes
    plot_boxes(ax, combined_dets[start_idx]['boxes_lidar'], 
            scores=combined_dets[start_idx]['score'],
            source_id=combined_dets[start_idx]['source_id'],
            source_labels=combined_dets[start_idx]['source'],
            limit_range=limit_range, alpha=0.3)
            
    ax.set_title(f'Frame: {start_idx}')
    ax.legend(loc='upper right')    

    def visualize(ind):
        frame_idx = ind % len(target_set)
        pts = target_set[frame_idx]['points']
        frame_id = compat.get_frame_id(target_set, target_set.infos[frame_idx])
        mask = common_vis.mask_points_by_range(pts, limit_range)              
        scatter.set_offsets(pts[mask][:,:2])
        ax.lines.clear()
        ax.texts.clear()
        # Plot GT boxes
        class_mask = np.array([n in classes for n in compat.get_gt_names(target_set, frame_id)], dtype=np.bool_)
        plot_boxes(ax, compat.get_gt_boxes(target_set, frame_id)[class_mask], color=[0,0,1], 
            limit_range=limit_range, label='gt_boxes',
            scores=np.ones(compat.get_gt_boxes(target_set, frame_id)[class_mask].shape[0]))

        # Plot det boxes
        plot_boxes(ax, combined_dets[frame_idx]['boxes_lidar'], 
                scores=combined_dets[frame_idx]['score'],
                source_id=combined_dets[frame_idx]['source_id'],
                source_labels=combined_dets[frame_idx]['source'],
                limit_range=limit_range, alpha=0.3)                        
        ax.set_aspect('equal')
        ax.set_title(f'Frame: {frame_idx}')
        plt.draw()

    class Index:
        ind = start_idx
        def next(self, event):
            self.ind += 1
            visualize(self.ind)

        def prev(self, event):
            self.ind -= 1
            visualize(self.ind)

        def fjump50(self, event):
            self.ind += 50
            visualize(self.ind)

        def bjump50(self, event):
            self.ind -= 50
            visualize(self.ind)

    callback = Index()
    axprev = fig.add_axes([0.75, 0.05, 0.05, 0.045])
    axnext = fig.add_axes([0.81, 0.05, 0.05, 0.045])
    axjumpf = fig.add_axes([0.87, 0.05, 0.05, 0.045])
    axjumpb = fig.add_axes([0.69, 0.05, 0.05, 0.045])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)
    fjump = Button(axjumpf, 'F_Jump')
    fjump.on_clicked(callback.fjump50)
    bjump = Button(axjumpb, 'B_Jump')
    bjump.on_clicked(callback.bjump50)

    ax.set_aspect('equal')
    plt.show()



if __name__ == '__main__':
    main()