import copy
import pickle
import numpy as np

from PIL import Image
import torch
import torch.nn.functional as F
from pathlib import Path

from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils
from .once_toolkits import Octopus

# For the sweeps
from scipy.spatial.transform import Rotation

class ONCEDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
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
        self.split = dataset_cfg.DATA_SPLIT['train'] if training else dataset_cfg.DATA_SPLIT['test']
        assert self.split in ['train', 'val', 'test', 'raw_small', 'raw_medium', 'raw_large']

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_seq_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        self.cam_names = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
        self.cam_tags = ['top', 'top2', 'left_back', 'left_front', 'right_front', 'right_back', 'back']
        self.toolkits = Octopus(self.root_path)
        self.anno_frames_only = dataset_cfg.get('ANNO_FRAMES_ONLY', True)
        self.frameid_to_idx = {}        
        self.infos = [] # Full length training dataset
        self.st_infos = None # Constantly re-adjusting length of self-training dataset each update
        self.seq_name_to_infos = self.include_once_data(self.split)

        if dataset_cfg.get('DATA_LEN', False) and self.training:
            from random import sample
            self.datainds = sample(list(range(len(self.infos))), dataset_cfg.DATA_LEN)
            self.infos = [self.infos[ind] for ind in self.datainds]
            self.logger.info(f'Sampled {len(self.infos)} for fine-tuning')

    def include_once_data(self, split, reload=False):
        if reload:
            self.infos = []

        if self.logger is not None:
            self.logger.info('Loading ONCE dataset')
        once_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[split]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                once_infos.extend(infos)

        def check_annos(info):
            return 'annos' in info
        
        # This section is for get_sequence_data since getitem idx is diff to sequence-sample_idx
        prev_seq = ''
        for info in once_infos:
            seq_id = info['sequence_id']    
            if seq_id != prev_seq:
                prev_seq = seq_id
                sample_idx = 0
            info['sample_idx'] = sample_idx
            sample_idx += 1            
        seq_name_to_infos = {}
        for i in range(len(once_infos)):
            seq_id = once_infos[i]['sequence_id']
            if seq_id not in seq_name_to_infos.keys():
                seq_name_to_infos[seq_id] = []            
            seq_name_to_infos[seq_id].append(once_infos[i])

        if self.split != 'raw' and self.anno_frames_only:
            once_infos = list(filter(check_annos,once_infos))

        self.infos.extend(once_infos)

        if self.logger is not None:
            self.logger.info('Total samples for ONCE dataset: %d' % (len(once_infos)))
        
        for idx, data in enumerate(self.infos):
            self.frameid_to_idx[data['frame_id']] = idx
        
        # use_sequence_data = self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED
        # if not use_sequence_data:
            # seq_name_to_infos = None 
        return seq_name_to_infos

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_seq_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, sequence_id, frame_id):        
        return self.toolkits.load_point_cloud(sequence_id, frame_id)

    def get_once_pose_as_hom(self, transform_data):
        """
        Return ONCE pose as homogenous transformation matrix (4x4)
        ONCE pose: (list) [quat_x, quat_y, quat_z, quat_w, trans_x, trans_y, trans_z]
        """            
        rotation = Rotation.from_quat(transform_data[:4]).as_matrix()
        translation = np.array(transform_data[4:]).transpose()
        return np.vstack([np.hstack([rotation, translation.reshape(-1,1)]), [0,0,0,1]])

    def get_pose(self, frame_id):
        pose = self.infos[self.frameid_to_idx[frame_id]]['pose']
        return self.get_once_pose_as_hom(pose)

    def get_sequence_data(self, points, sequence_name, sample_idx, sequence_cfg):
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
        
        sample_idx_pre_list = np.clip(sample_idx + np.arange(sequence_cfg.SAMPLE_OFFSET[0], sequence_cfg.SAMPLE_OFFSET[1]), 0, 0x7FFFFFFF)
        sample_idx_pre_list = sample_idx_pre_list[::-1]
        sequence_info = self.seq_name_to_infos[sequence_name]
        points = np.hstack([points, np.zeros((points.shape[0], 1)).astype(points.dtype)])
        num_pts_cur = points.shape[0]
        pose_cur = self.get_once_pose_as_hom(sequence_info[sample_idx]['pose'])

        points_pre_all = []
        num_points_pre = []
        pose_all = [pose_cur]
        for sample_idx_pre in sample_idx_pre_list:
            
            points_pre = self.get_lidar(sequence_name, sequence_info[sample_idx_pre]['frame_id'])
            pose_pre = self.get_once_pose_as_hom(sequence_info[sample_idx_pre]['pose'])
            expand_points_pre = np.concatenate([points_pre[:, :3], np.ones((points_pre.shape[0], 1))], axis=-1)
            points_pre_global = np.dot(expand_points_pre, pose_pre.T)[:,:3]        
            expand_points_pre_global = np.concatenate([points_pre_global, np.ones((points_pre_global.shape[0], 1))], axis=-1)
            points_pre2cur = np.dot(expand_points_pre_global, np.linalg.inv(pose_cur.T))[:,:3]
            points_pre = np.concatenate([points_pre2cur, points_pre[:, 3:]], axis=-1)
            
            # Add timestamp 
            points_pre = np.hstack([points_pre, 0.1 * (sample_idx - sample_idx_pre) * np.ones((points_pre.shape[0], 1)).astype(points_pre.dtype)])  # one frame 0.1s
            points_pre = remove_ego_points(points_pre, 2.0)
            points_pre_all.append(points_pre)
            num_points_pre.append(points_pre.shape[0])
            pose_all.append(pose_pre)

        points = np.concatenate([points] + points_pre_all, axis=0).astype(np.float32)
        num_points_all = np.array([num_pts_cur] + num_points_pre).astype(np.int32)
        poses = np.concatenate(pose_all, axis=0).astype(np.float32)
        return points, poses

    def get_image(self, sequence_id, frame_id, cam_name):
        return self.toolkits.load_image(sequence_id, frame_id, cam_name)

    def project_lidar_to_image(self, sequence_id, frame_id):
        return self.toolkits.project_lidar_to_image(sequence_id, frame_id)

    def remove_ego_points(self, points, center_radius=1.0):
        mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
        return points[mask]

    def __len__(self):
        if (self.st_infos is not None) and (self.training):
            data_infos = self.st_infos
        else:
            data_infos = self.infos  
        if self._merge_all_iters_to_one_epoch:
            return len(data_infos) * self.total_epochs

        return len(data_infos)

    def __getitem__(self, index):        
        
        if (self.st_infos is not None) and (self.training):
            data_infos = self.st_infos
        else:
            data_infos = self.infos  

        if self._merge_all_iters_to_one_epoch:
            index = index % len(data_infos)

        info = copy.deepcopy(data_infos[index])        
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        sample_idx = info['sample_idx']
        input_dict = {
            'sample_idx': sample_idx
        }
        points = self.get_lidar(seq_id, frame_id)
        points = self.remove_ego_points(points, 2.0)
        if self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED:            
            points, poses = self.get_sequence_data(points, seq_id, sample_idx, self.dataset_cfg.SEQUENCE_CONFIG)
            input_dict['poses'] = poses
            
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict.update({
            'points': points,
            'frame_id': frame_id,
        })

        if 'annos' in info:
            annos = info['annos']
            gt_boxes_lidar = annos['boxes_3d']
            if self.dataset_cfg.get('SHIFT_COOR', None):
                gt_boxes_lidar[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

            if self.dataset_cfg.get('REMOVE_ORIGIN_GTS', None) and self.training:
                input_dict['points'] = box_utils.remove_points_in_boxes3d(input_dict['points'], input_dict['gt_boxes'])
                mask = np.zeros(gt_boxes_lidar.shape[0], dtype=np.bool_)
                input_dict['gt_boxes'] = input_dict['gt_boxes'][mask]
                input_dict['gt_names'] = input_dict['gt_names'][mask]

            if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
                input_dict['gt_boxes'] = None

        elif self.anno_frames_only:
            input_dict.update({
                'gt_names': np.empty(0),
                'gt_boxes': np.empty((0,7)),
                'num_points_in_gt': np.empty(0)
            })

        # load saved pseudo label for unlabel data
        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            self.fill_pseudo_labels(input_dict)
           
        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict.pop('num_points_in_gt', None)
        return data_dict

    def get_infos(self, num_workers=4, sample_seq_list=None):
        import concurrent.futures as futures
        import json
        root_path = self.root_path
        cam_names = self.cam_names

        """
        # dataset json format
        {
            'meta_info': 
            'calib': {
                'cam01': {
                    'cam_to_velo': list
                    'cam_intrinsic': list
                    'distortion': list
                }
                ...
            }
            'frames': [
                {
                    'frame_id': timestamp,
                    'annos': {
                        'names': list
                        'boxes_3d': list of list
                        'boxes_2d': {
                            'cam01': list of list
                            ...
                        }
                    }
                    'pose': list
                },
                ...
            ]
        }
        # open pcdet format
        {
            'meta_info':
            'sequence_id': seq_idx
            'frame_id': timestamp
            'timestamp': timestamp
            'lidar': path
            'cam01': path
            ...
            'calib': {
                'cam01': {
                    'cam_to_velo': np.array
                    'cam_intrinsic': np.array
                    'distortion': np.array
                }
                ...
            }
            'pose': np.array
            'annos': {
                'name': np.array
                'boxes_3d': np.array
                'boxes_2d': {
                    'cam01': np.array
                    ....
                }
            }          
        }
        """
        def process_single_sequence(seq_idx):
            print('%s seq_idx: %s' % (self.split, seq_idx))
            seq_infos = []
            seq_path = Path(root_path) / 'data' / seq_idx
            json_path = seq_path / ('%s.json' % seq_idx)
            with open(json_path, 'r') as f:
                info_this_seq = json.load(f)
            meta_info = info_this_seq['meta_info']
            calib = info_this_seq['calib']
            for f_idx, frame in enumerate(info_this_seq['frames']):
                frame_id = frame['frame_id']
                if f_idx == 0:
                    prev_id = None
                else:
                    prev_id = info_this_seq['frames'][f_idx-1]['frame_id']
                if f_idx == len(info_this_seq['frames'])-1:
                    next_id = None
                else:
                    next_id = info_this_seq['frames'][f_idx+1]['frame_id']
                pc_path = str(seq_path / 'lidar_roof' / ('%s.bin' % frame_id))
                pose = np.array(frame['pose'])
                frame_dict = {
                    'sequence_id': seq_idx,
                    'frame_id': frame_id,
                    'timestamp': int(frame_id),
                    'prev_id': prev_id,
                    'next_id': next_id,
                    'meta_info': meta_info,
                    'lidar': pc_path,
                    'pose': pose
                }
                calib_dict = {}
                for cam_name in cam_names:
                    cam_path = str(seq_path / cam_name / ('%s.jpg' % frame_id))
                    frame_dict.update({cam_name: cam_path})
                    calib_dict[cam_name] = {}
                    calib_dict[cam_name]['cam_to_velo'] = np.array(calib[cam_name]['cam_to_velo'])
                    calib_dict[cam_name]['cam_intrinsic'] = np.array(calib[cam_name]['cam_intrinsic'])
                    calib_dict[cam_name]['distortion'] = np.array(calib[cam_name]['distortion'])
                frame_dict.update({'calib': calib_dict})

                if 'annos' in frame:
                    annos = frame['annos']
                    boxes_3d = np.array(annos['boxes_3d'])
                    if boxes_3d.shape[0] == 0:
                        print(frame_id)
                        continue
                    boxes_2d_dict = {}
                    for cam_name in cam_names:
                        boxes_2d_dict[cam_name] = np.array(annos['boxes_2d'][cam_name])
                    annos_dict = {
                        'name': np.array(annos['names']),
                        'boxes_3d': boxes_3d,
                        'boxes_2d': boxes_2d_dict
                    }

                    points = self.get_lidar(seq_idx, frame_id)
                    corners_lidar = box_utils.boxes_to_corners_3d(np.array(annos['boxes_3d']))
                    num_gt = boxes_3d.shape[0]
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    for k in range(num_gt):
                        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annos_dict['num_points_in_gt'] = num_points_in_gt

                    frame_dict.update({'annos': annos_dict})
                seq_infos.append(frame_dict)
            return seq_infos

        sample_seq_list = sample_seq_list if sample_seq_list is not None else self.sample_seq_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_sequence, sample_seq_list)
        all_infos = []
        for info in infos:
            all_infos.extend(info)
        return all_infos

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('once_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            if 'annos' not in infos[k]:
                continue
            print('gt_database sample: %d' % (k + 1))
            info = infos[k]
            frame_id = info['frame_id']
            seq_id = info['sequence_id']
            points = self.get_lidar(seq_id, frame_id)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['boxes_3d']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (frame_id, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_3d': np.zeros((num_samples, 7))
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
            pred_dict['boxes_3d'] = pred_boxes
            return pred_dict

        if self.dataset_cfg.get('USE_TTA', False):
            from ..augmentor import augmentor_utils
            # Need to undo the data augmentations in reverse order because...maths
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
            frame_id = batch_dict['frame_id'][index]
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                raise NotImplementedError
        return annos

    def kitti_eval(self, eval_det_annos, eval_gt_annos, class_names):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval
        # Treat "Car" as "Vehicle"
        map_name_to_kitti = {
            'Car': 'Car',
            'Truck': 'Car',
            'Bus': 'Car',
            'Pedestrian': 'Pedestrian',      
            'Cyclist': 'Cyclist',            
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

                """
                if 'boxes_lidar' in anno:
                    gt_boxes_lidar = anno['boxes_lidar'].copy()
                else:
                    gt_boxes_lidar = anno['gt_boxes'].copy()
                """
                gt_boxes_lidar = anno['boxes_3d'].copy()

                # filter by fov
                if is_gt and self.dataset_cfg.get('GT_FILTER', None):
                    if self.dataset_cfg.GT_FILTER.get('FOV_FILTER', None):
                        fov_gt_flag = self.extract_fov_gt(
                            gt_boxes_lidar, self.dataset_cfg['FOV_DEGREE'], self.dataset_cfg['FOV_ANGLE']
                        )
                        gt_boxes_lidar = gt_boxes_lidar[fov_gt_flag]
                        anno['name'] = anno['name'][fov_gt_flag]

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

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
        elif kwargs['eval_metric'] == 'once':
            from .once_eval.evaluation import get_evaluation_results
            ap_result_str = '\nEvaluation at vehicle iou=0.5'
            ap_result_str_05, ap_dict = get_evaluation_results(eval_gt_annos, eval_det_annos, class_names, 
                                                            iou_thresholds={'Vehicle': 0.5})
            ap_result_str += ap_result_str_05
            suffix = '/0.5'
            for key in list(ap_dict.keys()):
                ap_dict[key + suffix] = ap_dict.pop(key)
            ap_result_str_07, ap_dict_07 = get_evaluation_results(eval_gt_annos, eval_det_annos, class_names, 
                                                            iou_thresholds=kwargs['iou_thresholds'] if 'iou_thresholds' in kwargs else None)
            ap_dict.update(ap_dict_07)  
            ap_result_str += '\nEvaluation at vehicle iou=0.7'
            ap_result_str += ap_result_str_07                                                          
            return ap_result_str, ap_dict
        else:
            raise NotImplementedError

def create_once_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = ONCEDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)

    splits = ['train', 'val', 'test', 'raw_small', 'raw_medium', 'raw_large']
    ignore = ['test']

    print('---------------Start to generate data infos---------------')
    for split in splits:
        if split in ignore:
            continue

        filename = 'once_infos_%s.pkl' % split
        filename = save_path / Path(filename)
        dataset.set_split(split)
        once_infos = dataset.get_infos(num_workers=workers)
        with open(filename, 'wb') as f:
            pickle.dump(once_infos, f)
        print('ONCE info %s file is saved to %s' % (split, filename))

    train_filename = save_path / 'once_infos_train.pkl'
    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split('train')
    dataset.create_groundtruth_database(train_filename, split='train')

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    parser.add_argument('--runs_on', type=str, default='server', help='')
    args = parser.parse_args()

    if args.func == 'create_once_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))


        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        once_data_path = ROOT_DIR / 'data' / 'once'
        once_save_path = ROOT_DIR / 'data' / 'once'

        if args.runs_on == 'cloud':
            once_data_path = Path('/cache/once/')
            once_save_path = Path('/cache/once/')
            dataset_cfg.DATA_PATH = dataset_cfg.CLOUD_DATA_PATH

        create_once_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Bus', 'Truck', 'Pedestrian', 'Bicycle'],
            data_path=once_data_path,
            save_path=once_save_path
        )