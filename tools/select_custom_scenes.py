import sys
sys.path.append('/MS3D')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
import numpy as np
from tqdm import tqdm 

def load_dataset(data_cfg, split):

    # Get target dataset    
    data_cfg.DATA_SPLIT.test = split
    logger = common_utils.create_logger('temp.txt', rank=cfg.LOCAL_RANK)
    target_set, _, _ = build_dataloader(
                dataset_cfg=data_cfg,
                class_names=data_cfg.CLASS_NAMES,
                batch_size=1, logger=logger, training=False, dist=False, workers=1
            )      
    return target_set

if __name__ == '__main__':
    cfg_file_path = '/MS3D/tools/cfgs/dataset_configs/nuscenes_dataset_da.yaml'
    cfg_from_yaml_file(cfg_file_path, cfg)

    dataset = load_dataset(cfg,'train')

    print('done')

    scene_cls_counts = {}
    for info in tqdm(dataset.infos, total=len(dataset.infos)):
        if info['scene_name'] not in scene_cls_counts:
            scene_cls_counts[info['scene_name']] = {}
            scene_cls_counts[info['scene_name']]['ped'] = 0
            scene_cls_counts[info['scene_name']]['veh'] = 0
            scene_cls_counts[info['scene_name']]['num_frames'] = 0
        scene_cls_counts[info['scene_name']]['veh'] += np.count_nonzero(np.isin(info['gt_names'], ['car','truck','bus']))
        scene_cls_counts[info['scene_name']]['ped'] += np.count_nonzero(np.isin(info['gt_names'], ['pedestrian']))
        scene_cls_counts[info['scene_name']]['num_frames'] += 1

    scene_cls_list = []
    for k,v in scene_cls_counts.items():
        scene_cls_list.append((k,v['veh'],v['ped']))

    veh_desc = sorted(scene_cls_list, key=lambda x:x[1], reverse=True)
    ped_desc = sorted(scene_cls_list, key=lambda x:x[2], reverse=True)

    # Pick top 100 ped and veh scenes
    selected_scenes = []
    for v_scene, p_scene in zip(veh_desc[:100], ped_desc[:100]):
        selected_scenes.append(v_scene[0])
        selected_scenes.append(p_scene[0])
    selected_scenes = list(set(selected_scenes))
    print(selected_scenes)

    # with open('/MS3D/tools/custom_train_scenes_193.txt', 'w') as f:
    #     for scene in selected_scenes:
    #         f.write(scene+'\n')
    print('processed')        
        
