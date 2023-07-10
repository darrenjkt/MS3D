import argparse
import glob
from pathlib import Path
import sys
sys.path.insert(0, '../')
try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# TODO: Figure out how to properly incorporate this into the repo for a quick demo

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin', sample_offset=[-3,0]):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        self.sample_offset = sample_offset 
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_fnames = set([Path(fname).stem for fname in data_file_list])
        saved_list = glob.glob(str(root_path.parent / 'labels' / '*.pkl'))
        saved_fnames = set([Path(fname).stem for fname in saved_list])
        unlabeled_pcd_files = list(data_fnames - saved_fnames)
        unlabeled_file_list = [str(root_path / f'{pcd_fname}.pcd') for pcd_fname in unlabeled_pcd_files]

        unlabeled_file_list.sort()

        
        self.sample_file_list = unlabeled_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.pcd':
            pcd = open3d.io.read_point_cloud(self.sample_file_list[index])
            points = np.asarray(pcd.points)
            # print('loading current frame: ', self.sample_file_list[index])              
            if self.sample_offset[0] != 0:
                sample_idx_pre_list = np.clip(index + np.arange(self.sample_offset[0], self.sample_offset[1]), 0, 0x7FFFFFFF)
                sample_idx_pre_list = sample_idx_pre_list[::-1]

                all_points = []
                points = np.hstack([points, np.zeros((points.shape[0], 1)).astype(points.dtype)]) # current frame
                all_points.append(points)
                for sample_idx_pre in sample_idx_pre_list:
                    pcd = open3d.io.read_point_cloud(self.sample_file_list[sample_idx_pre])
                    # print('loading sweeps: ', self.sample_file_list[sample_idx_pre])
                    points_pre = np.asarray(pcd.points)
                    points_pre = np.hstack([points_pre, 0.1 * (index - sample_idx_pre) * np.ones((points_pre.shape[0], 1)).astype(points_pre.dtype)])  # one frame 0.1s
                    all_points.append(points_pre)
                points = np.vstack(all_points)
        else:
            raise NotImplementedError

        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            'points': points,
            'frame_id': Path(self.sample_file_list[index]).stem,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    from tqdm import tqdm
    import pickle
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in tqdm(enumerate(demo_dataset), total=len(demo_dataset)):
            # logger.info(f'Visualized sample index: \t{idx + 1}')
            
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            label_dir = Path(args.data_path).parent / 'labels'
            fname = label_dir / f'{data_dict["frame_id"][0]}.pkl'
            with open(str(fname),'wb') as f:
                pred_dicts[0]['pred_boxes'] = pred_dicts[0]['pred_boxes'].cpu().numpy()
                pred_dicts[0]['pred_labels'] = pred_dicts[0]['pred_labels'].cpu().numpy()
                pred_dicts[0]['pred_scores'] = pred_dicts[0]['pred_scores'].cpu().numpy()
                pred_dicts[0]['pred_boxes'][:,:3] -= demo_dataset.dataset_cfg.SHIFT_COOR
                pickle.dump(pred_dicts, f)
                print('Saved: ', str(fname))

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'], draw_origin=True
            # )

            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()