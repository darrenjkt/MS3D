import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, box_utils
from ..dataset import DatasetTemplate


class LyftDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        self.root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=self.root_path, logger=logger
        )
        self.infos = []
        self.split = dataset_cfg.DATA_SPLIT['train'] if training else dataset_cfg.DATA_SPLIT['test']
        self.frameid_to_idx = {}
        self.seq_name_to_infos = self.include_lyft_data()

    def reload_infos(self):
        self.infos = []
        self.frameid_to_idx = {}
        self.seq_name_to_infos = self.include_lyft_data()

    def include_lyft_data(self):
        if self.logger is not None:
            self.logger.info('Loading lyft dataset')
        lyft_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[self.split]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                lyft_infos.extend(infos)

        self.infos.extend(lyft_infos)
        if self.logger is not None:
            self.logger.info('Total samples for lyft dataset: %d' % (len(lyft_infos)))
        
        seq_name_to_infos = {}
        seq_name_to_len = {}
        for i in range(len(self.infos)):
            seq_id = self.infos[i]['scene_name']
            if seq_id not in seq_name_to_infos.keys():
                seq_name_to_infos[seq_id] = []            
            seq_name_to_infos[seq_id].append(self.infos[i])
            seq_name_to_len[seq_id] = len(self.infos[i])

        # Downsample data from 5Hz
        if self.dataset_cfg.SAMPLED_INTERVAL[self.mode] > 1:
            sampled_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[self.mode]):
                sampled_infos.append(self.infos[k])
            self.infos = sampled_infos

            seq_name_to_len = {}
            for info in self.infos:
                if info['scene_name'] not in seq_name_to_len.keys():
                    seq_name_to_len[info['scene_name']] = 0
                seq_name_to_len[info['scene_name']] += 1
        
        for idx, info in enumerate(self.infos):
            self.frameid_to_idx[Path(info['lidar_path']).stem] = idx        

        if self.logger is not None:
                self.logger.info('Total sampled samples for lyft dataset: %d' % len(self.infos))
        self.seq_name_to_len = seq_name_to_len
        return seq_name_to_infos

    @staticmethod
    def remove_ego_points(points, center_radius=1.0):
        mask = ~((np.abs(points[:, 0]) < center_radius*1.5) & (np.abs(points[:, 1]) < center_radius))
        return points[mask]

    def get_sweep(self, sweep_info):
        lidar_path = self.root_path / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1)
        if points_sweep.shape[0] % 5 != 0:
            points_sweep = points_sweep[: points_sweep.shape[0] - (points_sweep.shape[0] % 5)]
        points_sweep = points_sweep.reshape([-1, 5])[:, :4]

        points_sweep = self.remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]
        lidar_path = self.root_path / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1)        
        if points.shape[0] % 5 != 0:
            points = points[: points.shape[0] - (points.shape[0] % 5)]
        points = points.reshape([-1, 5])[:, :4]
        points = self.remove_ego_points(points, center_radius=1.5)
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        # Don't do random selection from infos cause my re-generated infos are for 16 frames
        # for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
        for k in np.random.choice(max_sweeps, max_sweeps - 1, replace=False): # not sure why they don't do it sequentially? maybe data aug effect?
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }

        if 'gt_boxes' in info:
            input_dict.update({
                'gt_boxes': info['gt_boxes'],
                'gt_names': info['gt_names']
            })
            if self.dataset_cfg.get('SHIFT_COOR', None):
                input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR
            
            if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
                input_dict['gt_boxes'] = None
            
        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            self.fill_pseudo_labels(input_dict)
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    # def kitti_eval(self, eval_det_annos, eval_gt_annos, class_names):
    #     from ..kitti.kitti_object_eval_python import eval as kitti_eval
    #     from ..kitti import kitti_utils

    #     map_name_to_kitti = {
    #         'car': 'Car',
    #         'pedestrian': 'Pedestrian',
    #         'truck': 'Car',
    #         'emergency_vehicle': 'DontCare',
    #         'other_vehicle': 'DontCare',
    #         'bus': 'Car',
    #         'bicycle': 'Cyclist',
    #         'motorcycle': 'Cyclist',
    #         'animal': 'DontCare'
    #     }

    #     kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
    #     kitti_utils.transform_annotations_to_kitti_format(
    #         eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
    #         info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
    #     )

    #     kitti_class_names = [map_name_to_kitti[x] for x in class_names]

    #     ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
    #         gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
    #     )
    #     return ap_result_str, ap_dict

    @staticmethod
    def extract_fov_gt(gt_boxes, fov_degree, heading_angle):
        """
        Args:
            anno_dict:
            fov_degree: [0~180]
            heading_angle: [0~360] in lidar coords, 0 is the x-axis, increase clockwise
        Returns:
        """
        half_fov_degree = fov_degree / 180 * np.pi / 2
        heading_angle = -heading_angle / 180 * np.pi
        gt_boxes_lidar = copy.deepcopy(gt_boxes)
        gt_boxes_lidar = common_utils.rotate_points_along_z(
            gt_boxes_lidar[np.newaxis, :, :], np.array([heading_angle])
        )[0]
        gt_boxes_lidar[:, 6] += heading_angle
        gt_angle = np.arctan2(gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0])
        fov_gt_mask = ((np.abs(gt_angle) < half_fov_degree) & (gt_boxes_lidar[:, 0] > 0))
        return fov_gt_mask

    def kitti_eval(self, eval_det_annos, eval_gt_annos, class_names):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        map_name_to_kitti = {
            'car': 'Car',
            'pedestrian': 'Pedestrian',
            'truck': 'Car',
            'emergency_vehicle': 'DontCare',
            'other_vehicle': 'DontCare',
            'bus': 'Car',
            'bicycle': 'Cyclist',
            'motorcycle': 'Cyclist',
            'animal': 'DontCare'
        }

        def transform_to_kitti_format(annos, info_with_fakelidar=False, is_gt=False):
            for anno in annos:
                if 'name' not in anno:
                    anno['name'] = anno['gt_names']
                    anno.pop('gt_names')

                for k in range(anno['name'].shape[0]):
                    if anno['name'][k] in map_name_to_kitti:
                        anno['name'][k] = map_name_to_kitti[anno['name'][k]]
                    else:
                        anno['name'][k] = 'Person_sitting'

                if 'boxes_lidar' in anno:
                    gt_boxes_lidar = anno['boxes_lidar'].copy()
                else:
                    gt_boxes_lidar = anno['gt_boxes'].copy()

                # filter boxes outside of range                
                point_cloud_range = self.point_cloud_range
                mask = box_utils.mask_boxes_outside_range_numpy(gt_boxes_lidar,
                                                                point_cloud_range,
                                                                min_num_corners=1)
                gt_boxes_lidar = gt_boxes_lidar[mask]
                anno['name'] = anno['name'][mask]
                if not is_gt:
                    anno['score'] = anno['score'][mask]
                    anno['pred_labels'] = anno['pred_labels'][mask]

                anno['bbox'] = np.zeros((len(anno['name']), 4))
                anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
                anno['truncated'] = np.zeros(len(anno['name']))
                anno['occluded'] = np.zeros(len(anno['name']))

                if len(gt_boxes_lidar) > 0:
                    if info_with_fakelidar:
                        gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

                    gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
                    anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
                    anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
                    anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
                    anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
                    dxdydz = gt_boxes_lidar[:, 3:6]
                    anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
                    anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
                    anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
                else:
                    anno['location'] = anno['dimensions'] = np.zeros((0, 3))
                    anno['rotation_y'] = anno['alpha'] = np.zeros(0)

        transform_to_kitti_format(eval_det_annos)
        transform_to_kitti_format(eval_gt_annos, info_with_fakelidar=False, is_gt=True)

        kitti_class_names = []
        for x in class_names:
            if x in map_name_to_kitti:
                kitti_class_names.append(map_name_to_kitti[x])
            else:
                kitti_class_names.append('Person_sitting')
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
        )
        return ap_result_str, ap_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if kwargs['eval_metric'] == 'kitti':
            eval_det_annos = copy.deepcopy(det_annos)
            eval_gt_annos = copy.deepcopy(self.infos)
            return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
        elif kwargs['eval_metric'] == 'lyft':
            return self.lyft_eval(det_annos, class_names, 
                                  iou_thresholds=self.dataset_cfg.EVAL_LYFT_IOU_LIST)
        else:
            raise NotImplementedError
    
    def lyft_eval(self, det_annos, class_names, iou_thresholds=[0.5]):
        from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
        from . import lyft_utils
        # from lyft_dataset_sdk.eval.detection.mAP_evaluation import get_average_precisions
        from .lyft_mAP_eval.lyft_eval import get_average_precisions

        lyft = Lyft(json_path=self.root_path / 'data', data_path=self.root_path, verbose=True)

        det_lyft_boxes, sample_tokens = lyft_utils.convert_det_to_lyft_format(lyft, det_annos)
        gt_lyft_boxes = lyft_utils.load_lyft_gt_by_tokens(lyft, sample_tokens)

        average_precisions = get_average_precisions(gt_lyft_boxes, det_lyft_boxes, class_names, iou_thresholds)

        ap_result_str, ap_dict = lyft_utils.format_lyft_results(average_precisions, class_names, iou_thresholds, version=self.dataset_cfg.VERSION)

        return ap_result_str, ap_dict

    def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
        import torch

        database_save_path = self.root_path / f'gt_database'
        db_info_save_path = self.root_path / f'lyft_dbinfos_{max_sweeps}sweeps.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_lyft_info(version, data_path, save_path, split, max_sweeps=10):
    from lyft_dataset_sdk.lyftdataset import LyftDataset
    from . import lyft_utils
    data_path = data_path / version
    save_path = save_path / version
    split_path = data_path.parent / 'ImageSets'

    if split is not None:
        save_path = save_path / split
        split_path = split_path / split

    save_path.mkdir(exist_ok=True)

    assert version in ['trainval', 'one_scene', 'test']

    if version == 'trainval':
        train_split_path = split_path / 'train.txt'
        val_split_path = split_path / 'val.txt'
    elif version == 'test':
        train_split_path = split_path / 'test.txt'
        val_split_path = None
    elif version == 'one_scene':
        train_split_path = split_path / 'one_scene.txt'
        val_split_path = split_path / 'one_scene.txt'
    else:
        raise NotImplementedError

    train_scenes = [x.strip() for x in open(train_split_path).readlines()] if train_split_path.exists() else []
    val_scenes = [x.strip() for x in open(val_split_path).readlines()] if val_split_path is not None and val_split_path.exists() else []

    lyft = LyftDataset(json_path=data_path / 'data', data_path=data_path, verbose=True)

    available_scenes = lyft_utils.get_available_scenes(lyft)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_lyft_infos, val_lyft_infos = lyft_utils.fill_trainval_infos(
        data_path=data_path, lyft=lyft, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps
    )

    if version == 'test':
        print('test sample: %d' % len(train_lyft_infos))
        with open(save_path / f'lyft_infos_test_16sweeps.pkl', 'wb') as f:
            pickle.dump(train_lyft_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_lyft_infos), len(val_lyft_infos)))
        with open(save_path / f'lyft_infos_train_16sweeps.pkl', 'wb') as f:
            pickle.dump(train_lyft_infos, f)
        with open(save_path / f'lyft_infos_val_16sweeps.pkl', 'wb') as f:
            pickle.dump(val_lyft_infos, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_lyft_infos', help='')
    parser.add_argument('--version', type=str, default='trainval', help='')
    parser.add_argument('--split', type=str, default=None, help='')
    parser.add_argument('--max_sweeps', type=int, default=16, help='')
    args = parser.parse_args()

    if args.func == 'create_lyft_infos':
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        dataset_cfg.MAX_SWEEPS = args.max_sweeps 
        create_lyft_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'lyft',
            save_path=ROOT_DIR / 'data' / 'lyft',
            split=args.split,
            max_sweeps=dataset_cfg.MAX_SWEEPS
        )

        # lyft_dataset = LyftDataset(
        #     dataset_cfg=dataset_cfg, class_names=None,
        #     root_path=ROOT_DIR / 'data' / 'lyft',
        #     logger=common_utils.create_logger(), training=True
        # )

        # if args.version != 'test':
        #     lyft_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
