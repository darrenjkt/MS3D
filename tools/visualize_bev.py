import pickle
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append('/MS3D')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from visual_utils import common_vis
import argparse
import matplotlib.pyplot as plt # apt-get update && apt-get install python3-tk (if given "Matplotlib is currently using agg" error)
from matplotlib.widgets import Button
from matplotlib.transforms import Bbox

from pcdet.utils import box_fusion_utils
from pcdet.utils import compatibility_utils as compat

"""
Updated 2023 Jun 27 at 12:11pm
- Now iterating with the det_annos or ps_dict instead of with target set
- This way we don't have to think about matching the target_set sample interval with the 
size of the detection sets/ps_label

python visualize_bev.py --cfg_file cfgs/dataset_configs/waymo_dataset_da.yaml \
                        --dets_txt /MS3D/tools/cfgs/target-nuscenes/raw_dets/det_1f_paths.txt

python visualize_bev.py --cfg_file cfgs/dataset_configs/waymo_dataset_da.yaml \
                        --ps_pkl cfgs/target_waymo/pretrained/ps_label_e0.pkl
"""

def plot_boxes(ax, boxes_lidar, color=[0,0,1], 
               scores=None, label=None, cur_id=0, limit_range=None,
               source_id=None, source_labels=None, alpha=1.0, 
               linestyle='solid',linewidth=1.0, fontsize=12,
               show_score=True):
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
        if (scores is not None) and show_score:
            ax.text(box[0,0], box[0,1], f'{scores[idx]:0.4f}', c=color, fontsize=fontsize)                             

def get_frame_id_from_dets(frame_id, detection_sets):
    for i, dset in enumerate(detection_sets):
        if dset['frame_id'] == frame_id:
            return i

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/MS3D/tools/cfgs/dataset_configs/waymo_dataset_da.yaml',
                        help='just use the target dataset cfg file')
    parser.add_argument('--dets_txt', type=str, default=None, required=False,
                        help='txt file containing detector pkl paths')       
    parser.add_argument('--det_pkl', type=str, default=None, required=False,
                        help='result.pkl file from test.py, assumd that boxes are in groud frame')
    parser.add_argument('--det_pkl2', type=str, default=None, required=False,
                        help='Another result.pkl file for comparison')
    parser.add_argument('--ps_pkl', type=str, required=False,
                        help='These are the ps_dict_*, ps_label_e*.pkl files generated from MS3D')
    parser.add_argument('--ps_pkl2', type=str, required=False,
                        help='Another ps_dict for comparison')
    parser.add_argument('--tracks_pkl', type=str, required=False,
                        help='Load in tracks, these are a dict with IDs as keys')
    parser.add_argument('--tracks_pkl2', type=str, required=False,
                        help='Load in tracks, these are a dict with IDs as keys')
    parser.add_argument('--idx', type=int, default=0,
                        help='If you wish to only display a certain frame index')
    parser.add_argument('--sweeps', type=int, default=None,
                        help='Num accum pc')
    parser.add_argument('--conf_th', type=float, default=0.0,
                        help='If you wish to only display a certain frame index')
    parser.add_argument('--split', type=str, default='train',
                        help='Specify train or test split')
    parser.add_argument('--show_trk_score', action='store_true', default=False)
    parser.add_argument('--hide_score', action='store_true', default=False)
    parser.add_argument('--custom_train_split', action='store_true', default=False)
    parser.add_argument('--above_pos_th', action='store_true', default=False)
    parser.add_argument('--color_height', action='store_true', default=False)
    parser.add_argument('--frame2box_key', type=str, required=False, default=None, 
                        help='options: frameid_to_box, frameid_to_rollingkde, frameid_to_propboxes. if None, then will use frameid_to_box')
    args = parser.parse_args()
    
    # Get target dataset
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.DATA_SPLIT.test = args.split
    if cfg.get('SAMPLED_INTERVAL', False):
        cfg.SAMPLED_INTERVAL.test = 1
    if args.custom_train_split:
        cfg.USE_CUSTOM_TRAIN_SCENES = True

    if args.sweeps is not None:
        # If dataset name in lyft,nusc
        cfg.MAX_SWEEPS = args.sweeps

        # else
        # dataset_cfg.SEQUENCE_CONFIG.ENABLED = True
    #     dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET = [-(args.sweeps-1),0]
    #     dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list=['x','y','z','intensity','timestamp']
    #     dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list=['x', 'y', 'z', 'intensity', 'elongation', 'timestamp']
    #     dataset_cfg.POINT_FEATURE_ENCODING.used_feature_list=['x','y','z']        
    
    logger = common_utils.create_logger('temp.txt', rank=cfg.LOCAL_RANK)
    target_set, _, _ = build_dataloader(
                dataset_cfg=cfg,
                class_names=cfg.CLASS_NAMES,
                batch_size=1, logger=logger, training=False, dist=False, workers=1
            )       
    idx_to_frameid = {v: k for k, v in target_set.frameid_to_idx.items()}     

    # Start with the first idx of the labels, not the target set
    start_idx = args.idx
    get_frame_id_from_ps = False
    start_frame_id = None
    if args.ps_pkl is not None:
        with open(args.ps_pkl,'rb') as f:
            ps_dict = pickle.load(f)

        ps_frame_ids = list(ps_dict.keys())
        start_frame_id = ps_frame_ids[start_idx]
    if args.ps_pkl2 is not None:
        with open(args.ps_pkl2,'rb') as f:
            ps_dict2 = pickle.load(f)
    if args.tracks_pkl is not None:
        with open(args.tracks_pkl,'rb') as f:
            tracks = pickle.load(f)
    if args.tracks_pkl2 is not None:
        with open(args.tracks_pkl2,'rb') as f:
            tracks2 = pickle.load(f)

    detection_sets = None
    if args.dets_txt is not None:
        det_annos = box_fusion_utils.load_src_paths_txt(args.dets_txt)
        detection_sets = box_fusion_utils.get_detection_sets(det_annos, score_th=0.1)
        if args.ps_pkl is None:
            start_frame_id = detection_sets[start_idx]['frame_id']   
        else:
            get_frame_id_from_ps = True

    if args.det_pkl is not None:
        with open(args.det_pkl,'rb') as f:
            detection_sets = pickle.load(f)
        if args.ps_pkl is None:
            start_frame_id = detection_sets[start_idx]['frame_id']   
        else:
            get_frame_id_from_ps = True

    if args.det_pkl2 is not None:
        with open(args.det_pkl2,'rb') as f:
            det2 = pickle.load(f)

    if start_frame_id is None:
        start_frame_id = idx_to_frameid[0]

    pts = target_set[target_set.frameid_to_idx[start_frame_id]]['points']
    pcr = 80
    limit_range = [-pcr, -pcr, -4.0, pcr, pcr, 2.0]
    mask = common_vis.mask_points_by_range(pts, limit_range)
    
    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot(111)
    fig.subplots_adjust(right=0.7)
    pts = pts[mask]
    if args.color_height: # is very slow - use only for paper vis. Buggy - doesn't do Next/Prev well
        cmap = plt.get_cmap('gray') # bone, ocean
        normed_z = (pts[:,2]-min(pts[:,2]))/(max(pts[:,2])-min(pts[:,2]))
        
        # Nusc color
        # mapped_z = 1/(1+np.exp(-15*(normed_z-0.5))) # sigmoid function mapped to [0,1] 
        # mapped_z = np.clip(mapped_z+0.2,0,1) # if you want ground to be darker

        # Waymo color
        mapped_z = 1/(1+np.exp(-5*(normed_z-0.5))) # sigmoid function mapped to [0,1] 
        mapped_z = np.clip(mapped_z+0.35,0,1) # if you want ground to be darker

        z_colors = np.array([cmap(1-n_z) for n_z in mapped_z])
        scatter = ax.scatter(pts[:,0],pts[:,1],s=0.1, c=z_colors, marker='o')
    else:
        scatter = ax.scatter(pts[:,0],pts[:,1],s=0.3, c='black', marker='o')

    # Plot GT boxes
    class_mask = np.isin(compat.get_gt_names(target_set, start_frame_id), ['Vehicle','car','truck','bus'])
    plot_boxes(ax, compat.get_gt_boxes(target_set, start_frame_id)[class_mask], color=[0,0,1],
        limit_range=limit_range, label='gt_vehicle', linewidth=2,
        scores=np.ones(compat.get_gt_boxes(target_set, start_frame_id)[class_mask].shape[0]), show_score=False if args.hide_score else True)
    
    class_mask = np.isin(compat.get_gt_names(target_set, start_frame_id), ['Pedestrian','pedestrian'])
    plot_boxes(ax, compat.get_gt_boxes(target_set, start_frame_id)[class_mask], color=[0.5,0,0.5],
        limit_range=limit_range, label='gt_pedestrian', linewidth=2,
        scores=np.ones(compat.get_gt_boxes(target_set, start_frame_id)[class_mask].shape[0]), show_score=False if args.hide_score else True)
    
    class_mask = np.isin(compat.get_gt_names(target_set, start_frame_id), ['Cyclist','bicycle','motorcycle'])
    plot_boxes(ax, compat.get_gt_boxes(target_set, start_frame_id)[class_mask], color=[0,0.5,0.5],
        limit_range=limit_range, label='gt_cyclist',
        scores=np.ones(compat.get_gt_boxes(target_set, start_frame_id)[class_mask].shape[0]), show_score=False if args.hide_score else True)

    # Plot det boxes
    if detection_sets is not None:
        det_start_idx = get_frame_id_from_dets(start_frame_id, detection_sets) if get_frame_id_from_ps else start_idx            
        conf_mask = detection_sets[det_start_idx]['score'] >= args.conf_th
        plot_boxes(ax, detection_sets[det_start_idx]['boxes_lidar'][conf_mask], 
                scores=detection_sets[det_start_idx]['score'][conf_mask],
                source_id=detection_sets[det_start_idx]['source_id'][conf_mask] if 'source_id' in detection_sets[det_start_idx].keys() else None,
                source_labels=detection_sets[det_start_idx]['source'][conf_mask] if 'source' in detection_sets[det_start_idx].keys() else None,
                color=[0,0.8,0] if 'source_id' not in detection_sets[det_start_idx].keys() else [0,0,1],
                limit_range=limit_range, alpha=0.5 if 'source_id' in detection_sets[det_start_idx].keys() else 1.0,
                label='det pkl' if 'source_id' not in detection_sets[det_start_idx].keys() else None, show_score=False if args.hide_score else True)
        
        if args.det_pkl2 is not None:
            conf_mask = det2[det_start_idx]['score'] >= args.conf_th
            plot_boxes(ax, det2[det_start_idx]['boxes_lidar'][conf_mask], 
                scores=det2[det_start_idx]['score'][conf_mask],
                label='det pkl 2', color=[0.6,0.4,0],
                limit_range=limit_range, alpha=1, show_score=False if args.hide_score else True)
        
    if args.ps_pkl is not None:
        combined_mask = ps_dict[ps_frame_ids[start_idx]]['gt_boxes'][:,8] >= args.conf_th
        if args.above_pos_th:
            above_pos_mask = ps_dict[ps_frame_ids[start_idx]]['gt_boxes'][:,7] > 0
            combined_mask = np.logical_and(combined_mask, above_pos_mask)
        plot_boxes(ax, ps_dict[ps_frame_ids[start_idx]]['gt_boxes'][combined_mask], 
                scores=ps_dict[ps_frame_ids[start_idx]]['gt_boxes'][combined_mask][:,8],
                label='ps labels', color=[0,0.8,0], fontsize=14, linewidth=1.5,
                limit_range=limit_range, alpha=1, show_score=False if args.hide_score else True)
        
    if args.ps_pkl2 is not None:
        combined_mask = ps_dict2[ps_frame_ids[start_idx]]['gt_boxes'][:,8] >= args.conf_th
        if args.above_pos_th:
            above_pos_mask = ps_dict2[ps_frame_ids[start_idx]]['gt_boxes'][:,7] > 0
            combined_mask = np.logical_and(combined_mask, above_pos_mask)
        plot_boxes(ax, ps_dict2[ps_frame_ids[start_idx]]['gt_boxes'][combined_mask], 
                scores=ps_dict2[ps_frame_ids[start_idx]]['gt_boxes'][combined_mask][:,8],
                label='ps labels 2', color=[1,0,0],
                limit_range=limit_range, alpha=1, show_score=False if args.hide_score else True) 
        
    if args.tracks_pkl is not None:     
        from pcdet.utils.transform_utils import world_to_ego   
        from pcdet.utils.tracker_utils import get_frame_track_boxes
        track_boxes = get_frame_track_boxes(tracks, start_frame_id, frame2box_key=args.frame2box_key, nhistory=0)
        pose = compat.get_pose(target_set, start_frame_id)
        score_idx = 7 if args.show_trk_score else 8
        _, track_boxes_ego = world_to_ego(pose, boxes=track_boxes)
        if track_boxes_ego.shape[0] != 0:
            plot_boxes(ax, track_boxes_ego[:,:7], 
                    scores=track_boxes_ego[:,score_idx],
                    label='tracked boxes', color=[1,0,0], linestyle='dotted',
                    limit_range=limit_range, alpha=1, show_score=False if args.hide_score else True) 
    if args.tracks_pkl2 is not None:     
        track_boxes2 = get_frame_track_boxes(tracks2, start_frame_id, nhistory=0)
        pose = compat.get_pose(target_set, start_frame_id)
        _, track_boxes_ego2 = world_to_ego(pose, boxes=track_boxes2)
        if track_boxes_ego2.shape[0] != 0:
            plot_boxes(ax, track_boxes_ego2[:,:7], 
                    scores=track_boxes_ego2[:,score_idx],
                    label='tracked boxes2', color=[1,0.7,0], linestyle='dotted',
                    limit_range=limit_range, alpha=1, show_score=False if args.hide_score else True) 
        
    # Plot ps labels
    # idx_to_frameid[start_idx]
            
    ax.set_title(f'Frame #{start_idx}, FID:{start_frame_id}')
    if 'scene_name' in target_set.infos[target_set.frameid_to_idx[start_frame_id]]:
        scene_name = target_set.infos[target_set.frameid_to_idx[start_frame_id]]['scene_name']
        ax.set_title(f'Frame #{start_idx}, SCENE: {scene_name}, FID:{start_frame_id}')
    else:
        ax.set_title(f'Frame #{start_idx}, FID:{start_frame_id}')    
        
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 0, 0.07, 1))    
    d = {"down" : 30, "up" : -30}
    def scroll_legend(evt):
        if legend.contains(evt):
            bbox = legend.get_bbox_to_anchor()
            bbox = Bbox.from_bounds(bbox.x0, bbox.y0+d[evt.button], bbox.width, bbox.height)
            tr = legend.axes.transAxes.inverted()
            legend.set_bbox_to_anchor(bbox.transformed(tr))
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", scroll_legend)

    def visualize(ind):
        if args.ps_pkl is not None:
            frame_idx = ind % len(ps_frame_ids)
            frame_id = ps_frame_ids[frame_idx]
        elif detection_sets is not None:
            frame_idx = ind % len(detection_sets)
            frame_id = detection_sets[frame_idx]['frame_id']        
        else:
            frame_idx = ind % len(idx_to_frameid.keys())
            frame_id = idx_to_frameid[frame_idx]

        pts = target_set[target_set.frameid_to_idx[frame_id]]['points']
        mask = common_vis.mask_points_by_range(pts, limit_range)              
        scatter.set_offsets(pts[mask][:,:2])
        ax.lines.clear()
        ax.texts.clear()

        # Plot GT boxes
        class_mask = np.isin(compat.get_gt_names(target_set, frame_id), ['Vehicle','car','truck','bus'])
        plot_boxes(ax, compat.get_gt_boxes(target_set, frame_id)[class_mask], color=[0,0,1],
            limit_range=limit_range, label='gt_vehicle', linewidth=2,
            scores=np.ones(compat.get_gt_boxes(target_set, frame_id)[class_mask].shape[0]), show_score=False if args.hide_score else True)
        
        class_mask = np.isin(compat.get_gt_names(target_set, frame_id), ['Pedestrian','pedestrian'])
        plot_boxes(ax, compat.get_gt_boxes(target_set, frame_id)[class_mask], color=[0.5,0,0.5],
            limit_range=limit_range, label='gt_pedestrian', linewidth=2,
            scores=np.ones(compat.get_gt_boxes(target_set, frame_id)[class_mask].shape[0]), show_score=False if args.hide_score else True)
        
        class_mask = np.isin(compat.get_gt_names(target_set, frame_id), ['Cyclist','bicycle','motorcycle'])
        plot_boxes(ax, compat.get_gt_boxes(target_set, frame_id)[class_mask], color=[0,0.5,0.5],
            limit_range=limit_range, label='gt_cyclist',
            scores=np.ones(compat.get_gt_boxes(target_set, frame_id)[class_mask].shape[0]), show_score=False if args.hide_score else True)

        # Plot det boxes
        if detection_sets is not None:
            det_frame_idx = get_frame_id_from_dets(frame_id, detection_sets) if get_frame_id_from_ps else frame_idx    
            conf_mask = detection_sets[det_frame_idx]['score'] >= args.conf_th
            plot_boxes(ax, detection_sets[det_frame_idx]['boxes_lidar'][conf_mask], 
                    scores=detection_sets[det_frame_idx]['score'][conf_mask],
                    source_id=detection_sets[det_frame_idx]['source_id'][conf_mask] if 'source_id' in detection_sets[det_frame_idx].keys() else None,
                    source_labels=detection_sets[det_frame_idx]['source'][conf_mask] if 'source' in detection_sets[det_frame_idx].keys() else None,
                    color=[0,0.8,0] if 'source_id' not in detection_sets[det_frame_idx].keys() else [0,0,1],
                    limit_range=limit_range, alpha=0.5 if 'source_id' in detection_sets[det_frame_idx].keys() else 1.0,
                    label='det pkl' if 'source_id' not in detection_sets[det_frame_idx].keys() else None, show_score=False if args.hide_score else True)

            if args.det_pkl2 is not None:
                conf_mask = det2[det_frame_idx]['score'] >= args.conf_th
                plot_boxes(ax, det2[det_frame_idx]['boxes_lidar'][conf_mask], 
                    scores=det2[det_frame_idx]['score'][conf_mask],
                    label='det pkl 2', color=[0.6,0.4,0],
                    limit_range=limit_range, alpha=1, show_score=False if args.hide_score else True)
             
        if args.ps_pkl is not None:
            combined_mask = ps_dict[ps_frame_ids[frame_idx]]['gt_boxes'][:,8] >= args.conf_th
            if args.above_pos_th:
                above_pos_mask = ps_dict[ps_frame_ids[frame_idx]]['gt_boxes'][:,7] > 0
                combined_mask = np.logical_and(combined_mask, above_pos_mask)
            plot_boxes(ax, ps_dict[ps_frame_ids[frame_idx]]['gt_boxes'][combined_mask], 
                scores=ps_dict[ps_frame_ids[frame_idx]]['gt_boxes'][combined_mask][:,8],
                label='ps labels', color=[0,0.8,0], fontsize=14, linewidth=1.5,
                limit_range=limit_range, alpha=1, show_score=False if args.hide_score else True) 
            
        if args.ps_pkl2 is not None:
            combined_mask = ps_dict2[ps_frame_ids[frame_idx]]['gt_boxes'][:,8] >= args.conf_th
            if args.above_pos_th:
                above_pos_mask = ps_dict2[ps_frame_ids[frame_idx]]['gt_boxes'][:,7] > 0
                combined_mask = np.logical_and(combined_mask, above_pos_mask)
            plot_boxes(ax, ps_dict2[ps_frame_ids[frame_idx]]['gt_boxes'][combined_mask], 
                scores=ps_dict2[ps_frame_ids[frame_idx]]['gt_boxes'][combined_mask][:,8],
                label='ps labels 2', color=[1,0,0],
                limit_range=limit_range, alpha=1, show_score=False if args.hide_score else True) 
            
        if args.tracks_pkl is not None:
            track_boxes = get_frame_track_boxes(tracks, frame_id, frame2box_key=args.frame2box_key, nhistory=0)
            pose = compat.get_pose(target_set, frame_id)
            _, track_boxes_ego = world_to_ego(pose, boxes=track_boxes)
            if track_boxes_ego.shape[0] != 0:
                plot_boxes(ax, track_boxes_ego[:,:7], 
                        scores=track_boxes_ego[:,score_idx],
                        label='tracked boxes', color=[1,0,0],linestyle='dotted',
                        limit_range=limit_range, alpha=1, show_score=False if args.hide_score else True) 
                
        if args.tracks_pkl2 is not None:     
            track_boxes2 = get_frame_track_boxes(tracks2, frame_id, nhistory=0)
            pose = compat.get_pose(target_set, frame_id)
            _, track_boxes_ego2 = world_to_ego(pose, boxes=track_boxes2)
            if track_boxes_ego2.shape[0] != 0:
                plot_boxes(ax, track_boxes_ego2[:,:7], 
                        scores=track_boxes_ego2[:,score_idx],
                        label='tracked boxes2', color=[1,0.7,0], linestyle='dotted',
                        limit_range=limit_range, alpha=1, show_score=False if args.hide_score else True) 
            
        if 'scene_name' in target_set.infos[target_set.frameid_to_idx[frame_id]]:
            scene_name = target_set.infos[target_set.frameid_to_idx[frame_id]]['scene_name']
            ax.set_title(f'Frame #{frame_idx}, SCENE: {scene_name}, FID:{frame_id}')
        else:
            ax.set_title(f'Frame #{frame_idx}, FID:{frame_id}')

        ax.set_aspect('equal')        
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
    fjump = Button(axjumpf, 'Jump +50')
    fjump.on_clicked(callback.fjump50)
    bjump = Button(axjumpb, 'Jump -50')
    bjump.on_clicked(callback.bjump50)


    ax.set_aspect('equal')
    plt.show()



if __name__ == '__main__':
    main()