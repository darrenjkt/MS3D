import copy
import pickle

import numpy as np
from skimage import io

from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate


class KittiDataset(DatasetTemplate):
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
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.use_raw_data = self.dataset_cfg.get('USE_RAW_DATA', False)
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.infos = []
        self.frameid_to_idx = {} # Mapping of frame_id to infos index
        self.seq_name_to_infos = self.include_kitti_data(self.mode)        

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        kitti_infos = []

        if self.use_raw_data:
            seq_name_to_infos = {}

            for k in range(len(self.sample_id_list)): # sample_id_list is actually sample_seq_list here
                sequence_name = self.sample_id_list[k]
                info_path = self.root_path / 'sequences' / sequence_name / f'{sequence_name}.pkl'            
                if not info_path.exists():
                    continue
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    kitti_infos.extend(infos)

                seq_name_to_infos[infos[0]['point_cloud']['lidar_sequence']] = infos

            self.infos.extend(kitti_infos)
        else:
            for info_path in self.dataset_cfg.INFO_PATH[mode]:
                info_path = self.root_path / info_path
                if not info_path.exists():
                    print('no file found at : ', info_path)
                    continue
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    kitti_infos.extend(infos)

            self.infos.extend(kitti_infos)
            seq_name_to_infos = None
            
        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

        for idx, data in enumerate(self.infos):
            if self.use_raw_data:
                self.frameid_to_idx[data['point_cloud']['frame_id']] = idx
            else:
                self.frameid_to_idx[data['point_cloud']['lidar_idx']] = idx
        
        return seq_name_to_infos

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_pose(self, sequence_name, sample_idx):
        sequence_info = self.seq_name_to_infos[sequence_name]
        return sequence_info[sample_idx]['pose']        

    def get_lidar_seq(self, sequence_name, sample_idx):
        sequence_info = self.seq_name_to_infos[sequence_name]
        fname = sequence_info[sample_idx]['point_cloud']['frame_id'].split('_')[-1]
        path = self.root_path / 'sequences' / sequence_name / 'velodyne_points' / 'data' / f'{fname}.bin'
        if path.exists():
            return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
        else:
            return None

    def get_lidar_accum(self, sequence_name, sample_idx):
        sequence_info = self.seq_name_to_infos[sequence_name]
        fname = sequence_info[sample_idx]['point_cloud']['frame_id'].split('_')[-1]
        path = self.root_path / 'sequences' / sequence_name / 'velodyne_points' / 'accum' / f'{fname}'
        if path.with_suffix('.npz').exists():
            return np.load(str(path.with_suffix('.npz')))['data']
        elif path.with_suffix('.npy').exists():
            return np.load(str(path.with_suffix('.npy')))
        else:
            return None
    
    def save_lidar_accum(self, sequence_name, sample_idx, pts, compressed=False):
        sequence_info = self.seq_name_to_infos[sequence_name]
        fname = sequence_info[sample_idx]['point_cloud']['frame_id'].split('_')[-1]
        path = self.root_path / 'sequences' / sequence_name / 'velodyne_points' / 'accum' / f'{fname}'
        path.parent.mkdir(parents=True, exist_ok=True)
        if compressed:
            np.savez_compressed(str(path), data=pts) # auto adds the .npz extension
        else:
            np.save(str(path), pts) # auto adds the .npy extension

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
            
        sequence_info = self.seq_name_to_infos[sequence_name]
        
        # # KITTI raw_data is missing a few frames so the list indexing id is diff to the file name sample id
        # sample_to_list_idx = dict([(sinfo['sample_idx'], enum) for enum, sinfo in enumerate(sequence_info)])        
        # list_to_sample_idx = {v: k for k, v in sample_to_list_idx.items()}
        
        sample_idx_pre_list = np.clip(sample_idx + np.arange(sequence_cfg.SAMPLE_OFFSET[0], sequence_cfg.SAMPLE_OFFSET[1]), 0, 0x7FFFFFFF)
        sample_idx_pre_list = sample_idx_pre_list[::-1]
        
        points = np.hstack([points, np.zeros((points.shape[0], 1)).astype(points.dtype)])        
        points_pre_all = []
        pose_cur = sequence_info[sample_idx]['pose']
        pose_all = [pose_cur]
        
        for idx, sample_idx_pre in enumerate(sample_idx_pre_list):
                        
            points_pre = self.get_lidar_seq(sequence_name, sample_idx_pre)
            pose_pre = sequence_info[sample_idx_pre]['pose'].reshape((4, 4))
            expand_points_pre = np.concatenate([points_pre[:, :3], np.ones((points_pre.shape[0], 1))], axis=-1)
            points_pre_global = np.dot(expand_points_pre, pose_pre.T)[:, :3]
            expand_points_pre_global = np.concatenate([points_pre_global, np.ones((points_pre_global.shape[0], 1))], axis=-1)
            points_pre2cur = np.dot(expand_points_pre_global, np.linalg.inv(pose_cur.T))[:, :3]
            points_pre = np.concatenate([points_pre2cur, points_pre[:, 3:]], axis=-1)
            
            # Append relative timestamp (each frame is 0.1s apart - 10Hz lidar)
            points_pre = np.hstack([points_pre, 0.1 * (sample_idx - sample_idx_pre) * np.ones((points_pre.shape[0], 1)).astype(points_pre.dtype)])  # one frame 0.1s
            
            points_pre = remove_ego_points(points_pre, 1.0)
            points_pre_all.append(points_pre)
            pose_all.append(pose_pre)

        points = np.concatenate([points] + points_pre_all, axis=0).astype(np.float32)
        poses = np.concatenate(pose_all, axis=0).astype(np.float32)

        return points, poses

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_seq(self, sequence_name, sample_idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_path / 'sequences' / sequence_name / 'image_02' / 'data' / f'{sample_idx:010}.png'
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)
    
    def get_image_shape_seq(self, sequence_name, sample_idx):
        img_file = self.root_path / 'sequences' / sequence_name / 'image_02' / 'data' / f'{sample_idx:010}.png'
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_calib_seq(self, sequence_name):
        calib_folder = self.root_path / 'sequences' / sequence_name / 'calib' 
        assert calib_folder.exists()
        return calibration_kitti.Calibration(calib_folder, use_raw_data=True)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib, margin=0):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0 - margin, pts_img[:, 0] < img_shape[1] + margin)
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0 - margin, pts_img[:, 1] < img_shape[0] + margin)
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_raw_data_infos(self):
        import os
        from . import parseTrackletXML
        try:
            import kiss_icp
        except ImportError as e:
            os.system("pip install kiss-icp")

        def get_tracklets_for_sample_idx(sample_idx, tracklets):        
            
            frame_boxes, names = [], []
            for track in tracklets:
                box_idx = np.argwhere(track.sample_indices == sample_idx)
                if len(box_idx) == 1:
                    frame_boxes.append(track.boxes[box_idx.item()])
                    names.append(track.objectType)

            if len(frame_boxes) == 0:
                return np.empty((0,7)), np.empty((0))
            else:
                return np.array(frame_boxes), np.array(names)

        # get lidar-odom for each sequence
        print('--------------- Getting lidar odometry ---------------')
        sequences = list(self.root_path.glob('sequences/*'))
        for seq_idx in range(len(sequences)):
            if (sequences[seq_idx] / 'odom.npy').exists():
                continue

            _ = os.system(f"cd {str(sequences[seq_idx])} && kiss_icp_pipeline velodyne_points/data/")
            odom_pth = list(sequences[seq_idx].glob('results/*/data_poses.npy'))[0]
            os.system(f"mv {str(odom_pth)} {str(sequences[seq_idx] / 'odom.npy')}")
            os.system(f"rm -rf {str(sequences[seq_idx] / 'results')}")

        def load_timestamps(pth):  
            with open(str(pth), 'r') as f:
                timestamps = [line.split('\n')[0] for line in f.readlines()]
            for idx in range(len(timestamps)):
                timestamps[idx] = ''.join(c for c in timestamps[idx] if c.isdigit())
            return timestamps
        print('--------------- Processing sequence infos ---------------')
        
        for sequence in sequences:
            seq_info = []
            lidar_paths = sorted(sequence.glob('velodyne_points/data/*.bin'))
            timestamps = load_timestamps(sequence / 'velodyne_points' / 'timestamps.txt')
            tracklets = parseTrackletXML.parseXML(str(sequence / 'tracklet_labels.xml'))
            odom = np.load(str(sequence / 'odom.npy'))
            for idx, lpath in enumerate(lidar_paths):
                frame_dict = {}
                frame_dict['point_cloud'] = {}
                frame_dict['point_cloud']['lidar_sequence'] = sequence.stem
                frame_dict['point_cloud']['sample_idx'] = idx
                frame_dict['point_cloud']['lidar_idx'] = idx # just to fit with the convention for kitti infos
                frame_dict['point_cloud']['timestamp'] = int(timestamps[int(lpath.stem)])
                frame_dict['point_cloud']['frame_id'] = f'{sequence.stem}_{lpath.stem}'
                frame_dict['pose'] = odom[idx]                                
                
                frame_dict['image'] = {}
                frame_dict['image']['image_idx'] = idx
                frame_dict['image']['image_shape'] = self.get_image_shape_seq(frame_dict['point_cloud']['lidar_sequence'], frame_dict['point_cloud']['sample_idx'])
                calib = self.get_calib_seq(frame_dict['point_cloud']['lidar_sequence'])
                frame_dict['calib'] = {'P2': calib.P2, 'R0_rect': calib.R0, 'Tr_velo_to_cam': calib.V2C}

                gt_boxes_lidar, names = get_tracklets_for_sample_idx(idx, tracklets)                                
                frame_dict['annos'] = {}
                frame_dict['annos']['gt_boxes_lidar'] = gt_boxes_lidar
                frame_dict['annos']['name'] = names

                # For raw_data we'll evaluate on vehicle category, but for 3d_object_detection, we'll stick with car category for comparison sake
                CLASS_MAPPING = {
                    'Car': 'Car',        
                    'Pedestrian': 'Pedestrian',
                    'Cyclist': 'Cyclist',
                    'Person (sitting)': 'Pedestrian',
                    'Tram': 'Tram',
                    'Van': 'Car',
                    'Truck': 'Car',
                    'Misc': 'Misc',
                    'DontCare': 'DontCare'}
                kitti_utils.transform_annotations_to_kitti_format([frame_dict['annos']], map_name_to_kitti=CLASS_MAPPING)
                seq_info.append(frame_dict)
                
            fname = sequence / f'{sequence.stem}.pkl'
            with open(fname, 'wb') as f:
                pickle.dump(seq_info, f)


    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict
            
            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()

            if self.dataset_cfg.get('SHIFT_COOR', None):
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR
                
            # BOX FILTER
            if self.dataset_cfg.get('TEST', None) and self.dataset_cfg.TEST.BOX_FILTER['FOV_FILTER']:
                box_preds_lidar_center = pred_boxes[:, 0:3]
                pts_rect = calib.lidar_to_rect(box_preds_lidar_center)
                fov_flag = self.get_fov_flag(pts_rect, image_shape, calib, margin=5)
                pred_boxes = pred_boxes[fov_flag]
                pred_labels = pred_labels[fov_flag]
                pred_scores = pred_scores[fov_flag]
            
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

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

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):

        if 'annos' not in self.infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])

        if self.use_raw_data:
            sample_idx = info['point_cloud']['sample_idx']
            frame_id = info['point_cloud']['frame_id']
            sequence_name = info['point_cloud']['lidar_sequence']
            calib = self.get_calib_seq(sequence_name)
        else:
            frame_id = info['point_cloud']['lidar_idx']
            calib = self.get_calib(frame_id)

        img_shape = info['image']['image_shape']
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': frame_id,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            
            gt_names = annos['name']
            if self.use_raw_data:
                gt_boxes_lidar = annos['gt_boxes_lidar']                
            else:
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
                
            if self.dataset_cfg.get('SHIFT_COOR', None):
                gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]
            
            if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
                input_dict['gt_boxes'] = None

            road_plane = self.get_road_plane(frame_id)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            if self.use_raw_data:
                
                points = self.get_lidar_seq(sequence_name, sample_idx)                
                if self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED:            
                    accum_pts = self.get_lidar_accum(sequence_name, sample_idx)
                    if accum_pts is None:
                        points, _ = self.get_sequence_data(points, sequence_name, sample_idx, self.dataset_cfg.SEQUENCE_CONFIG)  
                        if self.dataset_cfg.SEQUENCE_CONFIG.get('SAVE_ACCUM_FRAMES', False):
                            self.save_lidar_accum(sequence_name, sample_idx, points, compressed=True)
                    else:
                        points = copy.deepcopy(accum_pts)
            else:
                points = self.get_lidar(frame_id)

            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]

            if self.dataset_cfg.get('SHIFT_COOR', None):
                points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            input_dict['points'] = points

        if "images" in get_item_list:
            if self.use_raw_data:
                input_dict['images'] = self.get_image_seq(sequence_name, sample_idx)
            else:
                input_dict['images'] = self.get_image(frame_id)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(frame_id)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        # load saved pseudo label for unlabel data
        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            self.fill_pseudo_labels(input_dict)

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict


def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, use_raw_data, workers=4):
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    if use_raw_data:
        print('---------------Start to generate kitti raw_data infos---------------')
        dataset.get_raw_data_infos()
    else:
        train_split, val_split = 'train', 'val'

        train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
        val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
        trainval_filename = save_path / 'kitti_infos_trainval.pkl'
        test_filename = save_path / 'kitti_infos_test.pkl'

        print('---------------Start to generate data infos---------------')

        dataset.set_split(train_split)
        kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
        with open(train_filename, 'wb') as f:
            pickle.dump(kitti_infos_train, f)
        print('Kitti info train file is saved to %s' % train_filename)

        dataset.set_split(val_split)
        kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
        with open(val_filename, 'wb') as f:
            pickle.dump(kitti_infos_val, f)
        print('Kitti info val file is saved to %s' % val_filename)

        with open(trainval_filename, 'wb') as f:
            pickle.dump(kitti_infos_train + kitti_infos_val, f)
        print('Kitti info trainval file is saved to %s' % trainval_filename)

        dataset.set_split('test')
        kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
        with open(test_filename, 'wb') as f:
            pickle.dump(kitti_infos_test, f)
        print('Kitti info test file is saved to %s' % test_filename)

        print('---------------Start create groundtruth database for data augmentation---------------')
        dataset.set_split(train_split)
        dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys

    # python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_raw_dataset.yaml
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        parts = list(Path(dataset_cfg.DATA_PATH).parts)
        parts[0] = str(ROOT_DIR)
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=Path(*parts),
            save_path=Path(*parts),
            use_raw_data=dataset_cfg.USE_RAW_DATA
        )
