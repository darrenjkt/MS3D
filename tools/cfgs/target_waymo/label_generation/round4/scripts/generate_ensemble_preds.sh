#!/bin/bash

python test.py --cfg_file cfgs/target_waymo/ms3d_scratch_voxel_rcnn_anchorhead.yaml \
                --ckpt ../output/target_waymo/ms3d_scratch_voxel_rcnn_anchorhead/4f_xyzt_vehped_rnd3/ckpt/checkpoint_epoch_30.pth \
                --eval_tag waymo4xyzt_custom190_notta \
                --extra_tag 4f_xyzt_vehped_rnd3 \
                --target_dataset waymo --sweeps 4 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file cfgs/target_waymo/ms3d_scratch_voxel_rcnn_anchorhead.yaml \
                --ckpt ../output/target_waymo/ms3d_scratch_voxel_rcnn_anchorhead/4f_xyzt_vehped_rnd3/ckpt/checkpoint_epoch_30.pth \
                --eval_tag waymo4xyzt_custom190_rwr \
                --extra_tag 4f_xyzt_vehped_rnd3 \
                --target_dataset waymo --sweeps 4 --use_tta 2 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file cfgs/target_waymo/ms3d_scratch_voxel_rcnn_anchorhead.yaml \
                --ckpt ../output/target_waymo/ms3d_scratch_voxel_rcnn_anchorhead/4f_xyzt_vehped_rnd3/ckpt/checkpoint_epoch_30.pth \
                --eval_tag waymo4xyzt_custom190_rwf \
                --extra_tag 4f_xyzt_vehped_rnd3 \
                --target_dataset waymo --sweeps 4 --use_tta 1 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file cfgs/target_waymo/ms3d_scratch_voxel_rcnn_anchorhead.yaml \
                --ckpt ../output/target_waymo/ms3d_scratch_voxel_rcnn_anchorhead/4f_xyzt_vehped_rnd3/ckpt/checkpoint_epoch_30.pth \
                --eval_tag waymo4xyzt_custom190_rwf_rwr \
                --extra_tag 4f_xyzt_vehped_rnd3 \
                --target_dataset waymo --sweeps 4 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none  

python test.py --cfg_file cfgs/target_waymo/ms3d_scratch_voxel_rcnn_centerhead.yaml \
                --ckpt ../output/target_waymo/ms3d_scratch_voxel_rcnn_centerhead/4f_xyzt_vehped_rnd3/ckpt/checkpoint_epoch_30.pth \
                --eval_tag waymo4xyzt_custom190_notta \
                --extra_tag 4f_xyzt_vehped_rnd3 \
                --target_dataset waymo --sweeps 4 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file cfgs/target_waymo/ms3d_scratch_voxel_rcnn_centerhead.yaml \
                --ckpt ../output/target_waymo/ms3d_scratch_voxel_rcnn_centerhead/4f_xyzt_vehped_rnd3/ckpt/checkpoint_epoch_30.pth \
                --eval_tag waymo4xyzt_custom190_rwr \
                --extra_tag 4f_xyzt_vehped_rnd3 \
                --target_dataset waymo --sweeps 4 --use_tta 2 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file cfgs/target_waymo/ms3d_scratch_voxel_rcnn_centerhead.yaml \
                --ckpt ../output/target_waymo/ms3d_scratch_voxel_rcnn_centerhead/4f_xyzt_vehped_rnd3/ckpt/checkpoint_epoch_30.pth \
                --eval_tag waymo4xyzt_custom190_rwf \
                --extra_tag 4f_xyzt_vehped_rnd3 \
                --target_dataset waymo --sweeps 4 --use_tta 1 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file cfgs/target_waymo/ms3d_scratch_voxel_rcnn_centerhead.yaml \
                --ckpt ../output/target_waymo/ms3d_scratch_voxel_rcnn_centerhead/4f_xyzt_vehped_rnd3/ckpt/checkpoint_epoch_30.pth \
                --eval_tag waymo4xyzt_custom190_rwf_rwr \
                --extra_tag 4f_xyzt_vehped_rnd3 \
                --target_dataset waymo --sweeps 4 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none                                                                

