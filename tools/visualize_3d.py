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
    parser.add_argument('--cfg_file', type=str, default=None, required=True,
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='specify the model ckpt path')    
    parser.add_argument('--det_pkl', type=str, required=False,
                        help='These are the result.pkl files from test.py')
    parser.add_argument('--ps_pkl', type=str, required=False,
                        help='These are the ps_dict_*, ps_label_e*.pkl files generated from MS3D')
    parser.add_argument('--dets_txt', type=str, default=None, required=False,
                        help='det_*f_paths.txt file containing detector pkl paths')                        
    parser.add_argument('--idx', type=int, default=0,
                        help='If you wish to only display a certain frame index')
    parser.add_argument('--split', type=str, default='train',
                        help='Specify train or test split')    
    parser.add_argument('--sampled_interval', type=int, default=1,
                        help='same as SAMPLED_INTERVAL config parameter')        
    parser.add_argument('--custom_train_split', action='store_true', default=False)
    parser.add_argument('--save_video', action='store_true', default=False)
    parser.add_argument('--show_gt', action='store_true', default=False)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    if cfg.get('DATA_CONFIG_TAR', None):
        data_config = cfg.DATA_CONFIG_TAR
        cls_names = cfg.DATA_CONFIG_TAR.CLASS_NAMES
    else:
        data_config = cfg.DATA_CONFIG
        cls_names = cfg.CLASS_NAMES

    data_config.DATA_SPLIT.test = args.split
    if data_config.get('SAMPLED_INTERVAL', None):        
        data_config.SAMPLED_INTERVAL.test = args.sampled_interval
    data_config.USE_CUSTOM_TRAIN_SCENES = args.custom_train_split
    logger = common_utils.create_logger('temp.txt', rank=cfg.LOCAL_RANK)
    target_set, target_loader, _ = build_dataloader(
                dataset_cfg=data_config,
                class_names=cls_names,
                batch_size=1, logger=logger, training=False, dist=False, workers=1
            )
    frameid_to_idx = target_set.frameid_to_idx    
    idx_to_frameid = {v: k for k, v in frameid_to_idx.items()}
    if (args.det_pkl is not None) or (args.ps_pkl is not None) or (args.dets_txt is not None):    

        # Load detection pickle
        if args.det_pkl is not None:
            with open(args.det_pkl,'rb') as f:
                det_annos = pickle.load(f)

            eval_det_annos = copy.deepcopy(det_annos)   
            for idx, data_dict in enumerate(target_loader):
                if idx < args.idx:
                    print(f'Skipping {idx}/{args.idx}')
                    continue
                V.draw_scenes(points=data_dict['points'][:, 1:], 
                                        ref_boxes=eval_det_annos[idx]['boxes_lidar'][eval_det_annos[idx]['score'] > 0.6],                         
                                        ref_scores=eval_det_annos[idx]['score'][eval_det_annos[idx]['score'] > 0.6], 
                                        ref_labels=[1 for i in range(len(eval_det_annos[idx]['boxes_lidar'][eval_det_annos[idx]['score'] > 0.6]))],
                                        gt_boxes=data_dict['gt_boxes'][0] if args.show_gt else None, 
                                        draw_origin=False)
        if args.ps_pkl is not None:
            with open(args.ps_pkl,'rb') as f:
                ps_dict = pickle.load(f)

            for idx, data_dict in enumerate(target_loader):
                if idx < args.idx:
                    print(f'Skipping {idx}/{args.idx}')
                    continue

                frame_id = idx_to_frameid[idx]
                mask = ps_dict[frame_id]['gt_boxes'][:,8] > 0.4 #0.6 for ps_label, 0.4 for ps_dict_1f
                V.draw_scenes(points=data_dict['points'][:, 1:], 
                                        ref_boxes=ps_dict[frame_id]['gt_boxes'][mask],                         
                                        ref_scores=ps_dict[frame_id]['gt_boxes'][mask][:,8], 
                                        ref_labels=[1 for i in range(len(ps_dict[frame_id]['gt_boxes'][mask]))],
                                        gt_boxes=data_dict['gt_boxes'][0] if args.show_gt else None, 
                                        draw_origin=False)
        else:                        
            det_annos = box_fusion_utils.load_src_paths_txt(args.dets_txt)
            
            for idx, data_dict in enumerate(target_loader):
                if idx < args.idx:
                    print(f'Skipping {idx}/{args.idx}')
                    continue                
                
                geom = V.draw_scenes_msda(points=data_dict['points'][:, 1:], 
                                          idx=idx,
                                          det_annos=det_annos,                                        
                                          gt_boxes=data_dict['gt_boxes'][0] if args.show_gt else None)
            
    else:

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
                else:
                    gt_boxes = None
   
                if args.save_video:
                    geom = V.get_geometries(
                                points=data_dict['points'][:, 1:], gt_boxes=gt_boxes if args.show_gt else None, ref_boxes=pred_dicts[0]['pred_boxes'], 
                                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
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

                    print('Predicted: ', int(ref_boxes.shape[0]))
                    print('Ground truth: ', int(gt_boxes.shape[0]))
                    V.draw_scenes(
                        points=data_dict['points'][:, 1:], gt_boxes=gt_boxes if args.show_gt else None, ref_boxes=ref_boxes, 
                        ref_scores=ref_scores, ref_labels=ref_labels
                    )


if __name__ == '__main__':
    main()