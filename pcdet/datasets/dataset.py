import torch
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder

from ..utils import common_utils, self_training_utils
from ..ops.roiaware_pool3d import roiaware_pool3d_utils

class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training or self.dataset_cfg.get('USE_TTA', False) else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None
            
    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        
        def get_template_prediction(num_samples):
            box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict
            
            if self.dataset_cfg.get('SHIFT_COOR', None):
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR
                
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        if not self.training and self.dataset_cfg.get('USE_TTA', False):
            from .augmentor import augmentor_utils
            # Need to undo the data augmentations in reverse order
            for aug in reversed(self.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST):
                if aug['NAME'] == 'random_world_flip':
                    if 'flip_x' in batch_dict.keys():
                        for idx, enable in enumerate(batch_dict['flip_x']):
                            pred_dicts[idx]['pred_boxes'], _ = augmentor_utils.random_flip_along_x(pred_dicts[idx]['pred_boxes'], np.zeros((1,3)), enable=bool(enable))
                    if 'flip_y' in batch_dict.keys():
                        for idx, enable in enumerate(batch_dict['flip_y']):
                            pred_dicts[idx]['pred_boxes'], _ = augmentor_utils.random_flip_along_y(pred_dicts[idx]['pred_boxes'], np.zeros((1,3)), enable=bool(enable))
                if aug['NAME'] == 'random_world_rotation':
                    if 'noise_rot' in batch_dict.keys():
                        for idx, noise_rot in enumerate(batch_dict['noise_rot']):
                            unrotated_boxes, _ = augmentor_utils.global_rotation(pred_dicts[idx]['pred_boxes'].cpu(), np.zeros((1,3)), [], return_rot=False, noise_rotation=-noise_rot.item())
                            pred_dicts[idx]['pred_boxes'] = unrotated_boxes.cuda()
                            
        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def fill_pseudo_labels(self, input_dict, psid2clsid):
        """
        All labels are loaded with the index: class as 1:Vehicle, 2:Pedestrian, 3:Cyclist so we need to re-map them
        to the same class indices that the pre-trained detector has learnt
        """
        
        gt_boxes = self_training_utils.load_ps_label(input_dict['frame_id'])                

        gt_scores = gt_boxes[:, 8]
        gt_classes = gt_boxes[:, 7]
        remapped_classes = np.array([psid2clsid[cls_id] for cls_id in gt_classes])
        gt_boxes = gt_boxes[:, :7]
        gt_names = np.array(self.class_names)[np.abs(remapped_classes.astype(np.int32)) - 1]

        input_dict['gt_boxes'] = gt_boxes
        input_dict['gt_names'] = gt_names
        input_dict['gt_classes'] = remapped_classes
        input_dict['gt_scores'] = gt_scores
        input_dict['pos_ps_bbox'] = np.zeros((len(self.class_names)), dtype=np.float32)
        input_dict['ign_ps_bbox'] = np.zeros((len(self.class_names)), dtype=np.float32)
        for i in range(len(self.class_names)):
            num_total_boxes = (np.abs(remapped_classes) == (i+1)).sum()
            num_ps_bbox = (remapped_classes == (i+1)).sum()
            input_dict['pos_ps_bbox'][i] = num_ps_bbox
            input_dict['ign_ps_bbox'][i] = num_total_boxes - num_ps_bbox
        
        input_dict.pop('num_points_in_gt', None)

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            # filter gt_boxes without points
            num_points_in_gt = data_dict.get('num_points_in_gt', None)
            if num_points_in_gt is None:
                num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(data_dict['points'][:, :3]),
                    torch.from_numpy(data_dict['gt_boxes'][:, :7])).numpy().sum(axis=1)

            mask = (num_points_in_gt >= self.dataset_cfg.get('MIN_POINTS_OF_GT', 1))
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            data_dict['gt_names'] = data_dict['gt_names'][mask]
            if 'gt_classes' in data_dict:
                data_dict['gt_classes'] = data_dict['gt_classes'][mask]
                data_dict['gt_scores'] = data_dict['gt_scores'][mask]

            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
            
            if 'calib' in data_dict:
                calib = data_dict['calib']
            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if 'calib' in data_dict:
                data_dict['calib'] = calib

        if not self.training and self.dataset_cfg.get('USE_TTA', False):            
            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                }
            )
        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
             # for pseudo label has ignore labels.
            if 'gt_classes' not in data_dict:
                gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            else:
                gt_classes = data_dict['gt_classes'][selected]
                data_dict['gt_scores'] = data_dict['gt_scores'][selected]
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)
        data_dict.pop('gt_classes', None)

        return data_dict

    def eval(self):
        self.training = False
        self.data_processor.eval()

    def train(self):
        self.training = True
        self.data_processor.train()        

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['roi_boxes']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k,:, :val[k].shape[1], :] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['roi_scores', 'roi_labels']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k,:, :val[k].shape[1]] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        pad_value = 0

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                elif key in ['calib']:
                    ret[key] = val
                elif key in ["points_2d"]:
                    max_len = max([len(_val) for _val in val])
                    pad_value = 0
                    points = []
                    for _points in val:
                        pad_width = ((0, max_len-len(_points)), (0,0))
                        points_pad = np.pad(_points,
                                pad_width=pad_width,
                                mode='constant',
                                constant_values=pad_value)
                        points.append(points_pad)
                    ret[key] = np.stack(points, axis=0)
                elif key in ['gt_scores']:
                    max_gt = max([len(x) for x in val])
                    batch_scores = np.zeros((batch_size, max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_scores[k, :val[k].__len__()] = val[k]
                    ret[key] = batch_scores
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret