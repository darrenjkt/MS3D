import torch
from pathlib import Path
import sys
sys.path.append('/MS3D')
from pcdet.models import build_network, load_data_to_gpu
import copy
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
import open3d as o3d
from visual_utils import open3d_vis_utils as V
import argparse
import pickle
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import box_fusion_utils
from pcdet.utils import compatibility_utils as compat
import numpy as np
"""
# Examples
python visualize_3d.py --cfg_file cfgs/target-nuscenes/ft_waymo_secondiou.yaml  \
    --idx 6 --dets_txt cfgs/target-nuscenes/raw_dets/det_1f_paths.txt

python visualize_3d.py --cfg_file cfgs/target-nuscenes/ft_waymo_secondiou.yaml \
    --ps_pkl ../output/target-nuscenes/ft_waymo_secondiou/default/ps_label/ps_label_e0.pkl \
    --split train --custom_train_split --idx 6

"""

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/MS3D/tools/cfgs/dataset_configs/waymo_dataset_da.yaml',
                        help='just use the target dataset cfg file')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='specify the model ckpt path')    
    parser.add_argument('--det_pkl', type=str, required=False,
                        help='These are the result.pkl files from test.py')
    parser.add_argument('--ps_pkl', type=str, required=False,
                        help='These are the ps_dict_*, ps_label_e*.pkl files generated from MS3D')
    parser.add_argument('--ps_pkl2', type=str, required=False, default=None,
                        help='These are the ps_dict_*, ps_label_e*.pkl files generated from MS3D')
    parser.add_argument('--dets_txt', type=str, default=None, required=False,
                        help='det_*f_paths.txt file containing detector pkl paths')                        
    parser.add_argument('--idx', type=int, default=0,
                        help='If you wish to only display a certain frame index')
    parser.add_argument('--split', type=str, default='train',
                        help='Specify train or test split')    
    parser.add_argument('--sampled_interval', type=int, default=None,
                        help='same as SAMPLED_INTERVAL config parameter')        
    parser.add_argument('--custom_train_split', action='store_true', default=False)
    parser.add_argument('--save_video', action='store_true', default=False)
    parser.add_argument('--show_gt', action='store_true', default=False)
    parser.add_argument('--sweeps', type=int, default=None)
    parser.add_argument('--bev_vis', action='store_true', default=False)
    parser.add_argument('--use_linemesh', action='store_true', default=False)
    parser.add_argument('--use_class_colors', action='store_true', default=False)    
    args = parser.parse_args()
    
    if args.bev_vis:
        from visualize_bev import plot_boxes
        import matplotlib.pyplot as plt

    # Load dataset or model+dataset
    cfg_from_yaml_file(args.cfg_file, cfg)
    if 'dataset_configs' in args.cfg_file:
        data_config = cfg      
    else:
        if cfg.get('DATA_CONFIG_TAR', None):
            data_config = cfg.DATA_CONFIG_TAR
        else:
            data_config = cfg.DATA_CONFIG

    cls_names = data_config.CLASS_NAMES
    data_config.DATA_SPLIT.test = args.split
    data_config.USE_TTA = False
    if args.sampled_interval is not None:          
        data_config.SAMPLED_INTERVAL.test = args.sampled_interval
    data_config.USE_CUSTOM_TRAIN_SCENES = args.custom_train_split
    logger = common_utils.create_logger('temp.txt', rank=cfg.LOCAL_RANK)

    if data_config.get('MAX_SWEEPS',False):
        if args.sweeps is not None:
            data_config.MAX_SWEEPS = args.sweeps

    if data_config.get('SEQUENCE_CONFIG',False):
        if data_config.SEQUENCE_CONFIG.ENABLED:
            if args.sweeps is not None:
                data_config.SEQUENCE_CONFIG.SAMPLE_OFFSET = [-(args.sweeps-1),0]

            # data_config.POINT_FEATURE_ENCODING.src_feature_list=['x','y','z','intensity','timestamp']
            data_config.POINT_FEATURE_ENCODING.src_feature_list=['x', 'y', 'z', 'intensity', 'elongation', 'timestamp']
            data_config.POINT_FEATURE_ENCODING.used_feature_list=['x','y','z','timestamp']        

    target_set, target_loader, _ = build_dataloader(
            dataset_cfg=data_config,
            class_names=cls_names,
            batch_size=1, logger=logger, training=False, dist=False, workers=1
        )          

    idx_to_frameid = {v: k for k, v in target_set.frameid_to_idx.items()}

    # If no pkl file, just show point cloud and gt boxes (optional)
    if (args.det_pkl is None) and (args.ps_pkl is None) and (args.dets_txt is None) and (args.ckpt is None):    
        for idx, data_dict in enumerate(target_loader):
            if idx < args.idx:
                print(f'Skipping {idx}/{args.idx}')
                continue
            V.draw_scenes(points=data_dict['points'][:, 1:], 
                          gt_boxes=data_dict['gt_boxes'][0] if args.show_gt else None,                           
                          draw_origin=False, use_linemesh=args.use_linemesh, ref_labels=list(data_dict['gt_boxes'][0][:,7].astype(int)))

    # Visualize pkls
    if (args.det_pkl is not None) or (args.ps_pkl is not None) or (args.dets_txt is not None):    

        # Load detection pickle
        if args.det_pkl is not None:
            with open(args.det_pkl,'rb') as f:
                det_annos = pickle.load(f)

            # eval_det_annos = copy.deepcopy(det_annos)   
            for idx, det_anno in enumerate(det_annos):
                if idx < args.idx:
                    print(f'Skipping {idx}/{args.idx} out of {len(det_annos)} total samples')
                    continue
                frame_id = det_anno['frame_id']
                if frame_id not in target_set.frameid_to_idx.keys():
                    print(f"{frame_id} not found in frameid_to_idx, skipping frame")
                    continue
                print(f'Visualizing frame idx: {idx}, frame_id: {frame_id}')
                pts = target_set[target_set.frameid_to_idx[frame_id]]['points']
                gt_boxes = target_set[target_set.frameid_to_idx[frame_id]]['gt_boxes']
                # class_mask = np.isin(compat.get_gt_names(target_set, frame_id), ['Vehicle','car','truck','bus'])

            # for idx, data_dict in enumerate(target_loader):
            #     if idx < args.idx:
            #         print(f'Skipping {idx}/{args.idx}')
            #         continue
                V.draw_scenes(points=pts, 
                                ref_boxes=det_anno['boxes_lidar'][det_anno['score'] > 0.2],                         
                                ref_scores=det_anno['score'][det_anno['score'] > 0.2], 
                                ref_labels=[1 for i in range(len(det_anno['boxes_lidar'][det_anno['score'] > 0.2]))],
                                gt_boxes=gt_boxes if args.show_gt else None, use_class_colors=args.use_class_colors,
                                draw_origin=False, use_linemesh=args.use_linemesh)
        if args.ps_pkl is not None:
            with open(args.ps_pkl,'rb') as f:
                ps_dict = pickle.load(f)

            if args.ps_pkl2 is not None:
                with open(args.ps_pkl2,'rb') as f:
                    ps_dict2 = pickle.load(f)

            for idx, data_dict in enumerate(target_loader):
                if idx < args.idx:
                    print(f'Skipping {idx}/{args.idx}')
                    continue

                frame_id = idx_to_frameid[idx]
                if frame_id not in ps_dict.keys():
                    print(f"{frame_id} not in ps_dict, skipping frame")
                    continue
                # mask = ps_dict[frame_id]['gt_boxes'][:,8] > 0.4 #0.6 for ps_label, 0.4 for ps_dict_1f
                print(f'Visualizing frame idx: {idx}, frame_id: {frame_id}')
                V.draw_scenes(points=data_dict['points'][:, 1:], 
                                ref_boxes=ps_dict[frame_id]['gt_boxes'][:,:7][ps_dict[frame_id]['gt_boxes'][:,8] > 0.4],
                                ref_boxes2=ps_dict2[frame_id]['gt_boxes'][:,:7] if args.ps_pkl2 is not None else None,                         
                                ref_scores=ps_dict[frame_id]['gt_boxes'][:,8][ps_dict[frame_id]['gt_boxes'][:,8] > 0.4], 
                                ref_labels=list(abs(ps_dict[frame_id]['gt_boxes'][:,7][ps_dict[frame_id]['gt_boxes'][:,8] > 0.4].astype(int))),
                                gt_boxes=data_dict['gt_boxes'][0] if args.show_gt else None, 
                                draw_origin=False, use_linemesh=args.use_linemesh, use_class_colors=args.use_class_colors,)
        else:                        
            det_annos = box_fusion_utils.load_src_paths_txt(args.dets_txt)
            src_keys = list(det_annos.keys())
            src_keys.remove('det_cls_weights')
            len_data = len(det_annos[src_keys[0]])

            for idx in range(len_data):
                if idx < args.idx:
                    print(f'Skipping {idx}/{args.idx}')
                    continue
                frame_id = det_annos[src_keys[0]][idx]['frame_id']
                if frame_id not in target_set.frameid_to_idx.keys():
                    print(f"{frame_id} not found in frameid_to_idx, skipping frame")
                    continue
                print(f'Visualizing frame idx: {idx}, frame_id: {frame_id}')
                pts = target_set[target_set.frameid_to_idx[frame_id]]['points']
                gt_boxes = target_set[target_set.frameid_to_idx[frame_id]]['gt_boxes']        
                
                geom = V.draw_scenes_msda(points=pts, 
                                          idx=idx,
                                          det_annos=det_annos,                                        
                                          gt_boxes=gt_boxes if args.show_gt else None,
                                          use_linemesh=args.use_linemesh)
            
    else:
        # Load trained model for inference
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=target_set)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
        if args.save_video:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
        
        with torch.no_grad():
            for idx, data_dict in enumerate(target_loader):
                if idx < args.idx:
                    print(f'Skipping {idx}/{args.idx}')
                    continue
                
                load_data_to_gpu(data_dict)
                print(f'\nVisualizing frame: {idx}')  
                print('Points: ', data_dict['points'].shape[0])

                pred_dicts, _ = model.forward(data_dict)   
                if 'gt_boxes' in data_dict.keys():
                    gt_boxes = data_dict['gt_boxes'][0]

                    # For filtering out gt boxes with 0 pts in waymo scenes
                    # class_mask = np.in1d(target_set.infos[idx]['annos']['name'], target_set.class_names)
                    # class_num_pts = target_set.infos[idx]['annos']['num_points_in_gt'][class_mask]
                    # gt_boxes = gt_boxes[class_num_pts > 0]
                else:
                    gt_boxes = None
   
                if args.save_video:
                    geom = V.get_geometries(
                                points=data_dict['points'][:, 1:], gt_boxes=gt_boxes if args.show_gt else None, ref_boxes=pred_dicts[0]['pred_boxes'], 
                                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'], use_linemesh=args.use_linemesh
                            )
                    vis.clear_geometries()
                    for g in geom:                
                        vis.add_geometry(g)
                        
                    ctr = vis.get_view_control()    
                    vis.get_render_option().point_size = 2.0

                    ctr.set_front([ -0.63703010546300987, 0.031621802576535914, 0.77019004559627835 ])
                    ctr.set_lookat([ 24.513087638782164, 2.8039754478129324, -1.0959739482593087 ])
                    ctr.set_up([ 0.77082292993229895, 0.019695916704257327, 0.63674491089899199 ])
                    ctr.set_zoom(0.16)
                    
                    vis.update_renderer()         
                    vis.poll_events()
                    
                    Path('demo_data/save_frames').mkdir(parents=True, exist_ok=True)
                    vis.capture_screen_image(f'demo_data/save_frames/frame-{idx}.jpg')

                else:

                    ref_boxes = pred_dicts[0]['pred_boxes']
                    ref_labels = pred_dicts[0]['pred_labels']
                    ref_scores = pred_dicts[0]['pred_scores']

                    print('Frame ID: ', data_dict['frame_id'])
                    print('Predicted: ', int(ref_boxes.shape[0]))
                    print('Ground truth: ', int(gt_boxes.shape[0]))
                    if args.bev_vis:

                        pts = data_dict['points'][:, 1:].cpu().numpy()
                        fig = plt.figure(figsize=(20,20))
                        ax = plt.subplot(111)
                        fig.subplots_adjust(right=0.7)
                        if (args.sweeps is not None) and (args.sweeps > 1):
                            ax.scatter(pts[:,0],pts[:,1],s=0.1, c='black', marker='o')
                        else:
                            ax.scatter(pts[:,0],pts[:,1],s=0.5, c='black', marker='o')

                        plot_boxes(ax, ref_boxes.cpu().numpy(), 
                            scores=ref_scores.cpu().numpy(),
                            source_labels=ref_labels.cpu().numpy(),
                            limit_range=[-80, -80, -5.0, 80, 80, 3.0], color=[0,1,0])
                        
                        if args.show_gt:
                            plot_boxes(ax, gt_boxes.cpu().numpy(), 
                                        scores=None,
                                        source_labels=None,
                                        limit_range=[-80, -80, -5.0, 80, 80, 3.0], color=[0,0,1])

                        ax.set_aspect('equal')
                        plt.show(block=True)
                    else:
                        V.draw_scenes(
                            points=data_dict['points'][:, 1:], gt_boxes=gt_boxes if args.show_gt else None, ref_boxes=ref_boxes, 
                            ref_scores=ref_scores, ref_labels=ref_labels, use_linemesh=args.use_linemesh, use_class_colors=args.use_class_colors
                        )


if __name__ == '__main__':
    main()