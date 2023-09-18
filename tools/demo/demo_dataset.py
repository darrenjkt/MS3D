from pathlib import Path
import glob
from pcdet.datasets import DatasetTemplate
import numpy as np
import open3d as o3d
            
class DemoDataset(DatasetTemplate):
    
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.pcd', sweeps=4):
        """
        Demo data should be a parent dir e.g. ouster_sydney_sequence_60 with a lidar folder and odometry.csv

        If using own data, use timestamps to name point cloud files so it can be easily sorted. We assume all 
        point clouds in this folder are temporally continuous.

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
        self.root_path = Path(root_path) 
        self.ext = ext       
        self.max_sweeps = sweeps 
        self.sample_file_list = sorted(glob.glob(str(self.root_path / 'lidar' / f'*{self.ext}')))    
        self.lidar_timestamps = [np.int64(''.join(Path(fname).stem.split('_'))) for fname in self.sample_file_list]    
        self.lidar_odom = self.load_lidar_odom()
        self.frameid_to_idx = {}
        self.seq_name_to_len = {}
        self.infos = self.get_infos()     

    def __len__(self):
        return len(self.sample_file_list)
    
    def get_infos(self):
        """
        Prepare a dict for each frame with metadata
        """
        infos = []
        for idx, fname in enumerate(self.sample_file_list):
            lidar_timestamp = np.int64(''.join(Path(fname).stem.split('_')))                        
            frame_info = {}
            frame_info['frame_id'] = str(lidar_timestamp)
            frame_info['timestamp'] = lidar_timestamp
            frame_info['point_cloud'] = {}
            frame_info['point_cloud']['lidar_sequence'] = 'demo_sequence'
            frame_info['point_cloud']['sample_idx'] = idx
            frame_info['point_cloud']['num_features'] = 3 # just xyz for each point cloud
            frame_info['lidar_path'] = fname
            frame_info['pose'] = self.lidar_odom[lidar_timestamp]
            
            # Placeholder gt annotations. If you have gt labels, include them here
            frame_info['annos'] = {}
            frame_info['annos']['name'] = []
            frame_info['annos']['gt_boxes_lidar'] = np.empty((0,7))
            
            infos.append(frame_info)

            self.frameid_to_idx[frame_info['frame_id']] = idx

        self.seq_name_to_len['demo_sequence'] = len(self.sample_file_list)
        return infos
    
    def load_lidar_odom(self):
        kicp_odom = np.load(str(self.root_path / 'lidar_odom.npy')) 
        poses = {}
        for idx, timestamp in enumerate(self.lidar_timestamps):
            poses[timestamp] = kicp_odom[idx]
        return poses
    
    def remove_ego_points(self, points, center_radius=1.0):
        mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
        return points[mask]

    def get_sequence_data(self, points, sample_idx):
        """
        Transform historical frames to current frame with odometry and concatenate them
        """
        sample_idx_pre_list = np.clip(sample_idx + np.arange(-int(self.max_sweeps-1), 0), 0, 0x7FFFFFFF)
        sample_idx_pre_list = sample_idx_pre_list[::-1]
        
        points = np.hstack([points, np.zeros((points.shape[0], 1)).astype(points.dtype)])        
        points_pre_all = []
        pose_cur = self.infos[sample_idx]['pose']
        pose_all = [pose_cur]

        for sample_idx_pre in sample_idx_pre_list:
            pcd = o3d.io.read_point_cloud(self.infos[sample_idx_pre]['lidar_path'])
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

    def get_lidar(self, lidar_path):
        """We only use x,y,z lidar channels cause intensity/elongation/etc adds another layer to the domain gap"""        
        if self.ext == '.bin':
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:,:3]
        elif self.ext == '.npy':
            points = np.load(lidar_path)[:,:3]
        elif self.ext == '.pcd':
            points = np.asarray(o3d.io.read_point_cloud(lidar_path).points)
        else:
            raise NotImplementedError
        return points # (N,3)

    def __getitem__(self, index):
        lidar_path = self.infos[index]['lidar_path']
        points = self.get_lidar(lidar_path)
        points = self.remove_ego_points(points, 1.5)
        accum_points, _ = self.get_sequence_data(points, index)
        if self.dataset_cfg.get('SHIFT_COOR', None):
            accum_points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            'frame_id': self.infos[index]['frame_id'], 
            'points': accum_points,
            'gt_boxes': np.empty((0,7)), # placeholders for data_augmentor functions
            'gt_names': np.empty((0))
            
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
    