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
        self.data_path = self.root_path / 'sequences'
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []                
        self.frameid_to_idx = {}
        self.seq_name_to_len = {}
        self.seq_name_to_infos = self.include_data()

    def include_data(self):
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
            self.seq_name_to_len[infos[0]['point_cloud']['lidar_sequence']] = len(infos)

        self.infos.extend(custom_infos)
        for idx, data in enumerate(self.infos):
            self.frameid_to_idx[data['frame_id']] = idx

        if self.logger is not None:
            self.logger.info('Total samples for CustomDataset dataset: %d' % (len(self.infos)))

        use_sequence_data = self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED
        if not use_sequence_data:
            seq_name_to_infos = None 
        return seq_name_to_infos

    def get_lidar(self, index):
        lidar_path = self.infos[index]['lidar_path']
        points = np.asarray(o3d.io.read_point_cloud(lidar_path).points) # only loads (N,3)          
        return points
    
    def get_sequence_data(self, points, sample_idx, max_sweeps):
        """
        Transform historical frames to current frame with odometry and concatenate them
        """
        sample_idx_pre_list = np.clip(sample_idx + np.arange(-int(max_sweeps-1), 0), 0, 0x7FFFFFFF)
        sample_idx_pre_list = sample_idx_pre_list[::-1]
        
        points = np.hstack([points, np.zeros((points.shape[0], 1)).astype(points.dtype)])        
        points_pre_all = []
        pose_cur = self.infos[sample_idx]['pose']
        pose_all = [pose_cur]

        for sample_idx_pre in sample_idx_pre_list:
            pcd = o3d.io.read_point_cloud(self.infos[sample_idx_pre]['lidar_path'])
            # print('loading sweeps: ', self.sample_file_list[sample_idx_pre])
            points_pre = np.asarray(pcd.points)
            pose_pre = self.infos[sample_idx_pre]['pose'].reshape((4, 4))
            expand_points_pre = np.concatenate([points_pre[:, :3], np.ones((points_pre.shape[0], 1))], axis=-1)
            points_pre_global = np.dot(expand_points_pre, pose_pre.T)[:, :3]
            expand_points_pre_global = np.concatenate([points_pre_global, np.ones((points_pre_global.shape[0], 1))], axis=-1)
            points_pre2cur = np.dot(expand_points_pre_global, np.linalg.inv(pose_cur.T))[:, :3]
            points_pre = np.concatenate([points_pre2cur, points_pre[:, 3:]], axis=-1)
            
            # Append relative timestamp (each frame is 0.1s apart for 10Hz lidar)
            points_pre = np.hstack([points_pre, 0.1 * (sample_idx - sample_idx_pre) * np.ones((points_pre.shape[0], 1)).astype(points_pre.dtype)])  # one frame 0.1s
            
            points_pre = self.remove_ego_points(points_pre, 1.5)
            points_pre_all.append(points_pre)
            pose_all.append(pose_pre)
            
        points = np.concatenate([points] + points_pre_all, axis=0).astype(np.float32)
        poses = np.concatenate(pose_all, axis=0).astype(np.float32)

        return points, poses

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)


    def __getitem__(self, index):

        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])

        # sample_idx is the point cloud index within the sequence [0,199]
        sample_idx = int(info['point_cloud']['sample_idx'])
        points = self.get_lidar(sample_idx)
        points, _ = self.get_sequence_data(points, index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)

        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)        

        input_dict = {
            'frame_id': sample_idx,
            'points': points,
            'frame_id': info['frame_id'],
        }
        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            # Remap indices from pseudo-label 1-3 to order of det head classes; pseudo-labels ids are always 1:Vehicle, 2:Pedestrian, 3:Cyclist
            # Make sure DATA_CONFIG_TAR.CLASS_NAMES is same order/length as DATA_CONFIG.CLASS_NAMES (i.e. the pretrained class indices)
            
            psid2clsid = {}
            if 'Vehicle' in self.class_names:
                psid2clsid[1] = self.class_names.index('Vehicle') + 1
            if 'Pedestrian' in self.class_names:
                psid2clsid[2] = self.class_names.index('Pedestrian') + 1
            if 'Cyclist' in self.class_names:
                psid2clsid[3] = self.class_names.index('Cyclist') + 1
            self.fill_pseudo_labels(input_dict, psid2clsid)
                        
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


def create_infos(data_path):
    """
    Assumes the folder structure and file naming convention

    data_path
    |   |--- sequences
    |   |   |--- sequence_0
    |   |   |   |--- lidar
    |   |   |   |   |--- 123456_12324.pcd # extension does not matter for this function
    |   |   |   |--- lidar_odom.npy # generate with kiss_icp for each sequence
    |   |   |--- sequence_999
    """
    seq_dir = data_path / 'sequences'
    sample_sequences = list(seq_dir.glob('*'))
    seq_names = [seq.stem for seq in sample_sequences]
    
    for seq_idx, seq_name in enumerate(seq_names):
        seq_info = []
        sample_file_list = sorted(list((seq_dir / seq_name / 'lidar').glob('*')))
        seq_lidar_odom = np.load(str(seq_dir / seq_name / 'lidar_odom.npy'))
        for frame_idx, fname in enumerate(sample_file_list):
            lidar_timestamp = np.int64(''.join(fname.stem.split('_')))                        
            frame_info = {}
            frame_info['frame_id'] = str(lidar_timestamp)
            frame_info['timestamp'] = lidar_timestamp
            frame_info['point_cloud'] = {}
            frame_info['point_cloud']['lidar_sequence'] = seq_name
            frame_info['point_cloud']['sample_idx'] = frame_idx
            frame_info['point_cloud']['num_features'] = 3 # just xyz for each point cloud
            frame_info['lidar_path'] = str(fname)
            frame_info['pose'] = seq_lidar_odom[frame_idx]
            seq_info.append(frame_info)

        save_fname = str(seq_dir / seq_name / f'{seq_name}.pkl')
        with open(save_fname, 'wb') as f:
            pickle.dump(seq_info, f)
            print(f'{seq_idx+1}/{len(seq_names)} Saved: {save_fname}')

if __name__ == '__main__':
    import sys

    # python -m pcdet.datasets.custom.custom_dataset create_infos /MS3D/data/sydney_ouster
    from pathlib import Path
    absolute_data_path = sys.argv[2]
    create_infos(data_path=Path(absolute_data_path))