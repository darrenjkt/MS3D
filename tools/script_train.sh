#!/bin/bash

python train.py --cfg_file cfgs/nuscenes_models/uda_pv_rcnn_plusplus_resnet_anchorhead.yaml \
                --extra_tag 10sweep_xyzt_allcls

python train.py --cfg_file cfgs/nuscenes_models/uda_pv_rcnn_plusplus_resnet_anchorhead.yaml \
                --extra_tag 1sweep_xyz_allcls --set DATA_CONFIG.MAX_SWEEPS 1 DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z'