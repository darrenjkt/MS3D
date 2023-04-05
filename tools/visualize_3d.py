import torch
from pathlib import Path
import sys
sys.path.append('/OpenPCDet')
from pcdet.models import build_network, load_data_to_gpu
import copy
import open3d as o3d
from visual_utils import open3d_vis_utils as V
import argparse
import pickle
from common_jupy_utils import *
from pcdet.utils import box_fusion_utils
"""
# Examples

python visualize_3d.py --cfg_file cfgs/target-nuscenes/ft_waymo_secondiou.yaml  \
    --idx 6 --dets_txt cfgs/target-nuscenes/det_1f_paths_temp.txt

python visualize_3d.py --cfg_file cfgs/target-nuscenes/ft_waymo_secondiou.yaml  \
    --idx 6 --ps_pkl ../output/target-nuscenes/ft_waymo_secondiou/final_cfgs_noupdate_minstaticscore0.7/ps_label/ps_dict_1f.pkl

"""

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, required=True,
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='specify the model ckpt path')    
    parser.add_argument('--pkl', type=str, required=False,
                        help='Use saved detections from pkl path')
    parser.add_argument('--ps_pkl', type=str, required=False)
    parser.add_argument('--dets_txt', type=str, default=None, required=False,
                        help='txt file containing detector pkl paths')                        
    parser.add_argument('--idx', type=int, default=0,
                        help='If you wish to only display a certain frame index')
    parser.add_argument('--save_video', action='store_true', default=False)
    parser.add_argument('--only_car', action='store_true', default=False)
    args = parser.parse_args()

    # # Get target dataset
    # cfg_from_yaml_file(args.cfg_file, cfg)
    # logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    # if cfg.get('DATA_CONFIG_TAR', False):
    #     tgt_dataset_cfg = cfg.DATA_CONFIG_TAR
    #     src_class_names = ['Car','Bus','Truck'] if cfg.DATA_CONFIG_TAR.CLASS_NAMES == ['Car'] else cfg.DATA_CONFIG_TAR.CLASS_NAMES
    # else:
    #     tgt_dataset_cfg = cfg.DATA_CONFIG
    #     src_class_names = ['Car','Bus','Truck'] if cfg.CLASS_NAMES == ['Car'] else cfg.CLASS_NAMES
    # tgt_dataset_cfg.DATA_SPLIT.test='train'
    # if tgt_dataset_cfg.get('USE_TTA', False):
    #     tgt_dataset_cfg.USE_TTA=False

    # # Extra cfg changes here
    # # tgt_dataset_cfg.ANNO_FRAMES_ONLY = True
    # # tgt_dataset_cfg.SEQUENCE_CONFIG.ENABLED = False
    # # if tgt_dataset_cfg.SEQUENCE_CONFIG.ENABLED:
    # #     tgt_dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET = [-15,0]
    # #     tgt_dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list=['x','y','z','intensity','timestamp']
    # #     tgt_dataset_cfg.POINT_FEATURE_ENCODING.used_feature_list=['x','y','z','timestamp']

    # target_set, target_loader, sampler = build_dataloader(
    #             dataset_cfg=tgt_dataset_cfg,
    #             class_names=src_class_names,
    #             batch_size=1, logger=logger, training=False, dist=False, workers=1
    #         )
    
    
    
    # if cfg.get('DATA_CONFIG_TAR', None):
    #     cfg = get_cfg(args.cfg_file, split='train')
    #     data_config = cfg.DATA_CONFIG_TAR
    #     cfg.DATA_CONFIG_TAR.MAX_SWEEPS=1
    #     cfg.DATA_CONFIG_TAR.USE_PSEUDO_LABEL=False
    # # cfg.DATA_CONFIG_TAR.SAMPLED_INTERVAL.test=6
    # else:
    cfg_from_yaml_file(args.cfg_file, cfg)
    data_config = cfg.DATA_CONFIG

    target_set, target_loader, logger, frameid_to_idx = get_dataset(data_config, cls_names=cfg.CLASS_NAMES, training=False)
    idx_to_frameid = {v: k for k, v in frameid_to_idx.items()}
    if (args.pkl is not None) or (args.ps_pkl is not None) or (args.dets_txt is not None):    

        # Load detection pickle
        if args.pkl is not None:
            with open(args.pkl,'rb') as f:
                det_annos = pickle.load(f)

            eval_det_annos = copy.deepcopy(det_annos)   
            for idx, data_dict in enumerate(target_loader):
                if idx < args.idx:
                    print(f'Skipping {idx}/{args.idx}')
                    continue
                # eval_det_annos[idx]['boxes_lidar'][:,:3] += [0,0,1.6]
                V.draw_scenes(points=data_dict['points'][:, 1:], 
                                        ref_boxes=eval_det_annos[idx]['boxes_lidar'][eval_det_annos[idx]['score'] > 0.6],                         
                                        ref_scores=eval_det_annos[idx]['score'][eval_det_annos[idx]['score'] > 0.6], 
                                        ref_labels=[1 for i in range(len(eval_det_annos[idx]['boxes_lidar'][eval_det_annos[idx]['score'] > 0.6]))],
                                        gt_boxes=data_dict['gt_boxes'][0], 
                                        draw_origin=True)
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
                                        gt_boxes=data_dict['gt_boxes'][0], 
                                        draw_origin=True)
        else:                        
            det_annos = box_fusion_utils.load_src_paths_txt(args.dets_txt)
            
            for idx, data_dict in enumerate(target_loader):
                if idx < args.idx:
                    print(f'Skipping {idx}/{args.idx}')
                    continue                
                
                geom = V.draw_scenes_msda(points=data_dict['points'][:, 1:], 
                                          idx=idx,
                                          det_annos=det_annos,                                        
                                          gt_boxes=data_dict['gt_boxes'][0])
            
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
                                points=data_dict['points'][:, 1:], gt_boxes=gt_boxes, ref_boxes=pred_dicts[0]['pred_boxes'], 
                                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                            )
                    vis.clear_geometries()
                    for g in geom:                
                        vis.add_geometry(g)
                        
                    ctr = vis.get_view_control()    
                    vis.get_render_option().point_size = 2.0

                    # KITTI raw
                    # ctr.set_front([ -0.82471032579302606, 0.00018059423526749826, 0.56555534292948129 ])
                    # ctr.set_lookat([ 19.844308125762407, -0.13879944104704953, -6.4286119410902804 ])
                    # ctr.set_up([ 0.56547045688150521, 0.017591491015727621, 0.82458092497829805 ])
                    # ctr.set_zoom(0.259)

                    # BARAJA 64 PT
                    # ctr.set_front([ -0.82511184454554687, -0.060818814664189341, 0.56168631439041539 ])
                    # ctr.set_lookat([ 23.014725722772752, -0.047830774361222357, -1.4670256867434726 ])
                    # ctr.set_up([ 0.56201377939248653, 0.013202930753420827, 0.82702248723507754 ])
                    # ctr.set_zoom(0.16)

                    # BARAJA 128 U
                    ctr.set_front([ -0.63703010546300987, 0.031621802576535914, 0.77019004559627835 ])
                    ctr.set_lookat([ 24.513087638782164, 2.8039754478129324, -1.0959739482593087 ])
                    ctr.set_up([ 0.77082292993229895, 0.019695916704257327, 0.63674491089899199 ])
                    ctr.set_zoom(0.16)
                    
                    vis.update_renderer()         
                    vis.poll_events()
                    
                    Path('demo_data/save_frames').mkdir(parents=True, exist_ok=True)
                    vis.capture_screen_image(f'demo_data/save_frames/frame-{idx}.jpg')

                else:
                                                      
                    if args.only_car:
                        class_idx_of_interest = 1 # car
                        mask = pred_dicts[0]['pred_labels'] == class_idx_of_interest
                        ref_boxes = pred_dicts[0]['pred_boxes'][mask]
                        ref_labels = pred_dicts[0]['pred_labels'][mask]
                        ref_scores = pred_dicts[0]['pred_scores'][mask]
                    else:
                        ref_boxes = pred_dicts[0]['pred_boxes']
                        ref_labels = pred_dicts[0]['pred_labels']
                        ref_scores = pred_dicts[0]['pred_scores']

                    print('Predicted: ', int(ref_boxes.shape[0]))
                    print('Ground truth: ', int(gt_boxes.shape[0]))
                    V.draw_scenes(
                        points=data_dict['points'][:, 1:], gt_boxes=gt_boxes, ref_boxes=ref_boxes, 
                        ref_scores=ref_scores, ref_labels=ref_labels
                    )


if __name__ == '__main__':
    main()