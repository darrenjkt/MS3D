import copy
import pickle

import numpy as np
from skimage import io
import open3d as o3d
import json

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate

class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG

        split_dir = self.root_path / 'ImageSets' / 'test.txt'
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []                
        self.seq_name_to_infos = self.include_data(self.mode)

    def include_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading CustomDataset dataset')
        custom_infos = []
        seq_name_to_infos = {}

        for k in range(len(self.sample_sequence_list)):
            sequence_name = self.sample_sequence_list[k]
            info_path = self.data_path / sequence_name / f'{sequence_name}.pkl'            
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                custom_infos.extend(infos)

            seq_name_to_infos[infos[0]['point_cloud']['lidar_sequence']] = infos

        self.infos.extend(custom_infos)

        if self.logger is not None:
            self.logger.info('Total samples for CustomDataset dataset: %d' % (len(self.infos)))

        use_sequence_data = self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED
        if not use_sequence_data:
            seq_name_to_infos = None 
        return seq_name_to_infos

    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.data_path / sequence_name / 'pcd' / ('%06d.pcd' % sample_idx)
        assert lidar_file.exists(), f'No file found at {str(lidar_file)}'
        pcd = o3d.io.read_point_cloud(str(lidar_file))
        return np.asarray(pcd.points, dtype=np.float32)

    def get_sequence_data(self, info, points, sequence_name, sample_idx, sequence_cfg, load_pred_boxes=False):
        """
        Args:
            info:
            points:
            sequence_name:
            sample_idx:
            sequence_cfg:
        Returns:
        """

        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        def load_pred_boxes_from_dict(sequence_name, sample_idx):
            """
            boxes: (N, 11)  [x, y, z, dx, dy, dn, raw, vx, vy, score, label]
            """
            sequence_name = sequence_name.replace('training_', '').replace('validation_', '')
            load_boxes = self.pred_boxes_dict[sequence_name][sample_idx]
            assert load_boxes.shape[-1] == 11
            load_boxes[:, 7:9] = -0.1 * load_boxes[:, 7:9]  # transfer speed to negtive motion from t to t-1
            return load_boxes


        pose_cur = info['pose'].reshape((4, 4)) 
        num_pts_cur = points.shape[0]
        sample_idx_pre_list = np.clip(sample_idx + np.arange(sequence_cfg.SAMPLE_OFFSET[0], sequence_cfg.SAMPLE_OFFSET[1]), 0, 0x7FFFFFFF)
        sample_idx_pre_list = sample_idx_pre_list[::-1]
        
        if sequence_cfg.get('ONEHOT_TIMESTAMP', False):
            onehot_cur = np.zeros((points.shape[0], len(sample_idx_pre_list) + 1)).astype(points.dtype)
            onehot_cur[:, 0] = 1
            points = np.hstack([points, onehot_cur])
        else:
            points = np.hstack([points, np.zeros((points.shape[0], 1)).astype(points.dtype)])
        
        points_pre_all = []
        num_points_pre = []

        pose_all = [pose_cur]
        pred_boxes_all = []
        if load_pred_boxes:
            pred_boxes = load_pred_boxes_from_dict(sequence_name, sample_idx)
            pred_boxes_all.append(pred_boxes)

        sequence_info = self.seq_name_to_infos[sequence_name]

        for idx, sample_idx_pre in enumerate(sample_idx_pre_list):

            points_pre = self.get_lidar(sequence_name, sample_idx_pre)
            pose_pre = sequence_info[sample_idx_pre]['pose'].reshape((4, 4))
            expand_points_pre = np.concatenate([points_pre[:, :3], np.ones((points_pre.shape[0], 1))], axis=-1)
            points_pre_global = np.dot(expand_points_pre, pose_pre.T)[:, :3]
            expand_points_pre_global = np.concatenate([points_pre_global, np.ones((points_pre_global.shape[0], 1))], axis=-1)
            points_pre2cur = np.dot(expand_points_pre_global, np.linalg.inv(pose_cur.T))[:, :3]
            points_pre = np.concatenate([points_pre2cur, points_pre[:, 3:]], axis=-1)
            if sequence_cfg.get('ONEHOT_TIMESTAMP', False):
                onehot_vector = np.zeros((points_pre.shape[0], len(sample_idx_pre_list) + 1))
                onehot_vector[:, idx + 1] = 1
                points_pre = np.hstack([points_pre, onehot_vector])
            else:
                # Append relative timestamp (each frame is 0.1s apart - 10Hz lidar)
                points_pre = np.hstack([points_pre, 0.1 * (sample_idx - sample_idx_pre) * np.ones((points_pre.shape[0], 1)).astype(points_pre.dtype)])  # one frame 0.1s
            
            points_pre = remove_ego_points(points_pre, 1.0)
            points_pre_all.append(points_pre)
            num_points_pre.append(points_pre.shape[0])
            pose_all.append(pose_pre)

            if load_pred_boxes:
                pose_pre = sequence_info[sample_idx_pre]['pose'].reshape((4, 4))
                pred_boxes = load_pred_boxes_from_dict(sequence_name, sample_idx_pre)
                pred_boxes = self.transform_prebox_to_current(pred_boxes, pose_pre, pose_cur)
                pred_boxes_all.append(pred_boxes)

        points = np.concatenate([points] + points_pre_all, axis=0).astype(np.float32)
        num_points_all = np.array([num_pts_cur] + num_points_pre).astype(np.int32)
        poses = np.concatenate(pose_all, axis=0).astype(np.float32)

        if load_pred_boxes:
            temp_pred_boxes = self.reorder_rois_for_refining(pred_boxes_all)
            pred_boxes = temp_pred_boxes[:, :, 0:9]
            pred_scores = temp_pred_boxes[:, :, 9]
            pred_labels = temp_pred_boxes[:, :, 10]
        else:
            pred_boxes = pred_scores = pred_labels = None

        return points, num_points_all, sample_idx_pre_list, poses, pred_boxes, pred_scores, pred_labels

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI        
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)


    def __getitem__(self, index):

        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])

        sample_idx = int(info['point_cloud']['sample_idx'])
        sequence_name = info['point_cloud']['lidar_sequence']
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
        }        

        points = self.get_lidar(sequence_name, sample_idx)


        if self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED:
            points, num_points_all, sample_idx_pre_list, poses, pred_boxes, pred_scores, pred_labels = self.get_sequence_data(
                info, points, sequence_name, sample_idx, self.dataset_cfg.SEQUENCE_CONFIG,
                load_pred_boxes=self.dataset_cfg.get('USE_PREDBOX', False)
            )

        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
        
        if 'annos' in info:
            annos = info['annos']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            num_points_in_gt = annos['num_points_in_gt']
            if self.dataset_cfg.get('SHIFT_COOR', None):
                gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR
            
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar[:,:7],
                'num_points_in_gt': annos.get('num_points_in_gt', None)

            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

        input_dict.update({'points': points,})
        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict.pop('num_points_in_gt', None)
        return data_dict
