import sys
sys.path.append('/MS3D')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils, box_fusion_utils
from pcdet.datasets import build_dataloader
import pickle

def load_dataset(data_cfg, split):

    # Get target dataset    
    data_cfg.DATA_SPLIT.test = split
    data_cfg.SAMPLED_INTERVAL.test = 5
    logger = common_utils.create_logger('temp.txt', rank=cfg.LOCAL_RANK)
    target_set, _, _ = build_dataloader(
                dataset_cfg=data_cfg,
                class_names=data_cfg.CLASS_NAMES,
                batch_size=1, logger=logger, training=False, dist=False, workers=1
            )      
    return target_set

cfg_file = '/MS3D/tools/cfgs/target_waymo/ms3d_lyft_voxel_rcnn_centerhead.yaml'
pkl = '/MS3D/output/waymo_models/uda_voxel_rcnn_centerhead/4f_xyzt_allcls/eval/eval_with_train/epoch_30/val/result.pkl'
with open(pkl, 'rb') as f:
    det_annos = pickle.load(f)
print(f'Loaded detections for {len(det_annos)} frames')

cfg_from_yaml_file(cfg_file, cfg)
dataset = load_dataset(cfg.DATA_CONFIG_TAR, 'val')
result_str, result_dict = dataset.evaluation(
        det_annos, dataset.class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=''
    )
print('Evaluated: ', pkl)
print(result_str)
