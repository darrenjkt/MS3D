from functools import partial

import numpy as np

from ...utils import common_utils
from . import augmentor_utils, database_sampler
from pcdet.config import cfg

class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger
        self.augmentor_configs = augmentor_configs
        
        self.prog_range = False
        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            if cur_cfg.NAME == 'progressive_range':
                self.prog_range = True
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_object_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_rotation, config=config)

        gt_boxes, points = augmentor_utils.rotate_objects(
            data_dict['gt_boxes'],
            data_dict['points'],
            data_dict['gt_boxes_mask'],
            rotation_perturb=config['ROT_UNIFORM_NOISE'],
            prob=config['ROT_PROB'],
            num_try=50
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_object_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_scaling, config=config)
        points, gt_boxes = augmentor_utils.scale_pre_object(
            data_dict['gt_boxes'], data_dict['points'],
            gt_boxes_mask=data_dict['gt_boxes_mask'],
            scale_perturb=config['SCALE_UNIFORM_NOISE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points, enable = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points, return_flip=True
            )
            data_dict['flip_%s'%cur_axis] = enable
            if 'roi_boxes' in data_dict.keys():
                num_frame, num_rois,dim = data_dict['roi_boxes'].shape
                roi_boxes, _, _ = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                data_dict['roi_boxes'].reshape(-1,dim), np.zeros([1,3]), return_flip=True, enable=enable
                )
                data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois,dim)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, noise_rot = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range, return_rot=True
        )
        if 'roi_boxes' in data_dict.keys():
            num_frame, num_rois,dim = data_dict['roi_boxes'].shape
            roi_boxes, _, _ = augmentor_utils.global_rotation(
            data_dict['roi_boxes'].reshape(-1, dim), np.zeros([1, 3]), rot_range=rot_range, return_rot=True, noise_rotation=noise_rot)
            data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois,dim)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_rot'] = noise_rot
        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        trans_range = config['WORLD_TRANS_RANGE']
        if not isinstance(trans_range, list):
            trans_range = [-trans_range, trans_range]
        gt_boxes, points, trans = augmentor_utils.global_translation(
            data_dict['gt_boxes'], data_dict['points'], trans_range=trans_range, return_trans=True
        )
        if 'roi_boxes' in data_dict.keys():
            num_frame, num_rois,dim = data_dict['roi_boxes'].shape
            roi_boxes, _, _ = augmentor_utils.global_translation(
            data_dict['roi_boxes'].reshape(-1, dim), np.zeros([1, 3]), trans_range=trans_range, return_trans=True, translation=trans)
            data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois,dim)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['trans'] = trans
        return data_dict        

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        
        if 'roi_boxes' in data_dict.keys():
            gt_boxes, roi_boxes, points, noise_scale = augmentor_utils.global_scaling_with_roi_boxes(
                data_dict['gt_boxes'], data_dict['roi_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], return_scale=True
            )
            data_dict['roi_boxes'] = roi_boxes
        else:
            gt_boxes, points, noise_scale = augmentor_utils.global_scaling(
                data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], return_scale=True
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_scale'] = noise_scale
        return data_dict
        
    def normalize_object_size(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.normalize_object_size, config=config)
        points, gt_boxes = augmentor_utils.normalize_object_size(
            data_dict['gt_boxes'], data_dict['points'], data_dict['gt_boxes_mask'], config['SIZE_RES']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib,
            )

        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config['LOCAL_TRANSLATION_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_local_translation_along_%s' % cur_axis)(
                gt_boxes, points, offset_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_ring_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_ring_dropout, config=config)

        copy_gtbox = data_dict['gt_boxes'].copy()
        copy_points = data_dict['points'].copy()
        copy_points[:,:3] -= cfg.DATA_CONFIG_TAR.SHIFT_COOR
        copy_gtbox[:,:3] -= cfg.DATA_CONFIG_TAR.SHIFT_COOR
        points = augmentor_utils.local_ring_dropout(copy_gtbox, copy_points, config['MIN_INOUT_PTS'])
        points[:,:3] += cfg.DATA_CONFIG_TAR.SHIFT_COOR
        data_dict['points'] = points
        return data_dict

    def random_local_rotation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_rotation, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.local_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_scaling(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_scaling, config=config)
        gt_boxes, points = augmentor_utils.local_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['LOCAL_SCALE_RANGE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_world_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'global_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'local_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def adjust_range(self, cur_epoch):
        
        for aug_id, aug_func in enumerate(self.data_augmentor_queue):
            func_name = aug_func.keywords['config']['NAME']
            if func_name == 'progressive_range':
                cur_cfg = aug_func.keywords['config']
                range_aug_id = aug_id
        
                # Set new range if we're at the update epoch
                aug_idx = cur_cfg.UPDATE_AUG.index(cur_epoch) if cur_epoch in cur_cfg.UPDATE_AUG else None
                if aug_idx is not None:
                    for data_proc in cfg.DATA_CONFIG_TAR.DATA_PROCESSOR:
                        if data_proc.NAME == 'transform_points_to_voxels':
                            vox_size = data_proc.VOXEL_SIZE
                    approx_range = cur_cfg.RANGE[aug_idx]
                    xy_range = (np.round(approx_range/(vox_size[0]*16))*(vox_size[0]*16))
                    xy_range = np.round(xy_range,1)
                    full_range = np.array(cfg.DATA_CONFIG_TAR.POINT_CLOUD_RANGE)
                    full_range[[0,1]] = -xy_range
                    full_range[[3,4]] = xy_range
                    cur_cfg['CUR_RANGE'] = full_range

                    self.data_augmentor_queue[range_aug_id] = getattr(self, 'progressive_range')(config=cur_cfg)
        return

    def progressive_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.progressive_range, config=config)

        range_mask = common_utils.mask_points_by_range(data_dict['points'][:,:3], limit_range=list(config['CUR_RANGE']))
        box_mask = common_utils.mask_points_by_range(data_dict['gt_boxes'][:,:3], limit_range=list(config['CUR_RANGE']))

        data_dict['points'] = data_dict['points'][range_mask]
        data_dict['gt_boxes'] = data_dict['gt_boxes'][box_mask]
        data_dict['gt_names'] = data_dict['gt_names'][box_mask]
        data_dict['gt_boxes_mask'] = data_dict['gt_boxes_mask'][box_mask]

        return data_dict

    def random_local_pyramid_aug(self, data_dict=None, config=None):
        """
        Refer to the paper:
            SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud
        """
        if data_dict is None:
            return partial(self.random_local_pyramid_aug, config=config)

        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']

        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_dropout(gt_boxes, points, config['DROP_PROB'])
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_sparsify(gt_boxes, points,
                                                                            config['SPARSIFY_PROB'],
                                                                            config['SPARSIFY_MAX_NUM'],
                                                                            pyramids)
        gt_boxes, points = augmentor_utils.local_pyramid_swap(gt_boxes, points,
                                                                 config['SWAP_PROB'],
                                                                 config['SWAP_MAX_NUM'],
                                                                 pyramids)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        # if 'calib' in data_dict:
        #     data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]

            data_dict.pop('gt_boxes_mask')
        return data_dict


    def re_prepare(self, augmentor_configs=None, intensity=None):
        self.data_augmentor_queue = []

        if augmentor_configs is None:
            augmentor_configs = self.augmentor_configs

        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            # scale data augmentation intensity
            if intensity is not None:
                cur_cfg = self.adjust_augment_intensity(cur_cfg, intensity)
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)


    def adjust_augment_intensity(self, config, intensity):
        adjust_map = {
            'random_object_scaling': 'SCALE_UNIFORM_NOISE',
            'random_object_rotation': 'ROT_UNIFORM_NOISE',
            'random_world_rotation': 'WORLD_ROT_ANGLE',
            'random_world_scaling': 'WORLD_SCALE_RANGE',
        }

        def cal_new_intensity(config, flag):
            """
            Not sure why, but flag is a hardcoded value for the initial value. 
            For random_object_scaling, flag=1 means no scaling. For W-K, you want [0.7,0.95]
            This will give you 1-0.7=0.3, and 0.95-1=-0.05 which is not close at all hence assert fails.
            Works well if [0.95,1.05] which is most of the ST3D tuning; maybe we'll fix the intensity of data aug?
            """
            origin_intensity_list = config.get(adjust_map[config.NAME])
            assert len(origin_intensity_list) == 2
            assert np.isclose(flag - origin_intensity_list[0], origin_intensity_list[1] - flag)
            
            noise = origin_intensity_list[1] - flag
            new_noise = noise * intensity
            new_intensity_list = [flag - new_noise, new_noise + flag]
            return new_intensity_list

        if config.NAME not in adjust_map:
            return config
        
        # for data augmentations that init with 1
        if config.NAME in ["random_object_scaling", "random_world_scaling"]:
            new_intensity_list = cal_new_intensity(config, flag=1)
            setattr(config, adjust_map[config.NAME], new_intensity_list)
            return config
        elif config.NAME in ['random_object_rotation', 'random_world_rotation']:
            new_intensity_list = cal_new_intensity(config, flag=0)
            setattr(config, adjust_map[config.NAME], new_intensity_list)
            return config
        else:
            raise NotImplementedError
