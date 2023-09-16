#!/bin/bash

python test.py --cfg_file cfgs/target_nuscenes/ms3d_waymo_voxel_rcnn_centerhead.yaml \
                --ckpt ../output/target_nuscenes/ms3d_waymo_voxel_rcnn_centerhead/10f_xyzt_vehped_rnd1/ckpt/checkpoint_epoch_30.pth \
                --eval_tag nusc10xyzt_notta \
                --extra_tag 10f_xyzt_vehped_rnd1 \
                --target_dataset nuscenes --sweeps 10 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file cfgs/target_nuscenes/ms3d_waymo_voxel_rcnn_centerhead.yaml \
                --ckpt ../output/target_nuscenes/ms3d_waymo_voxel_rcnn_centerhead/10f_xyzt_vehped_rnd1/ckpt/checkpoint_epoch_30.pth \
                --eval_tag nusc10xyzt_rwr \
                --extra_tag 10f_xyzt_vehped_rnd1 \
                --target_dataset nuscenes --sweeps 10 --use_tta 2 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                

python test.py --cfg_file cfgs/target_nuscenes/ms3d_waymo_voxel_rcnn_centerhead.yaml \
                --ckpt ../output/target_nuscenes/ms3d_waymo_voxel_rcnn_centerhead/10f_xyzt_vehped_rnd1/ckpt/checkpoint_epoch_30.pth \
                --eval_tag nusc10xyzt_rwr \
                --extra_tag 10f_xyzt_vehped_rnd1 \
                --target_dataset nuscenes --sweeps 10 --use_tta 1 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                

python test.py --cfg_file cfgs/target_nuscenes/ms3d_waymo_voxel_rcnn_centerhead.yaml \
                --ckpt ../output/target_nuscenes/ms3d_waymo_voxel_rcnn_centerhead/10f_xyzt_vehped_rnd1/ckpt/checkpoint_epoch_30.pth \
                --eval_tag nusc10xyzt_rwr \
                --extra_tag 10f_xyzt_vehped_rnd1 \
                --target_dataset nuscenes --sweeps 10 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                                

python test.py --cfg_file cfgs/target_nuscenes/ms3d_waymo_voxel_rcnn_anchorhead.yaml \
                --ckpt ../output/target_nuscenes/ms3d_waymo_voxel_rcnn_anchorhead/10f_xyzt_vehped_rnd1/ckpt/checkpoint_epoch_30.pth \
                --eval_tag nusc10xyzt_notta \
                --extra_tag 10f_xyzt_vehped_rnd1 \
                --target_dataset nuscenes --sweeps 10 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                                

python test.py --cfg_file cfgs/target_nuscenes/ms3d_waymo_voxel_rcnn_anchorhead.yaml \
                --ckpt ../output/target_nuscenes/ms3d_waymo_voxel_rcnn_anchorhead/10f_xyzt_vehped_rnd1/ckpt/checkpoint_epoch_30.pth \
                --eval_tag nusc10xyzt_rwr \
                --extra_tag 10f_xyzt_vehped_rnd1 \
                --target_dataset nuscenes --sweeps 10 --use_tta 2 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                                

python test.py --cfg_file cfgs/target_nuscenes/ms3d_waymo_voxel_rcnn_anchorhead.yaml \
                --ckpt ../output/target_nuscenes/ms3d_waymo_voxel_rcnn_anchorhead/10f_xyzt_vehped_rnd1/ckpt/checkpoint_epoch_30.pth \
                --eval_tag nusc10xyzt_rwf \
                --extra_tag 10f_xyzt_vehped_rnd1 \
                --target_dataset nuscenes --sweeps 10 --use_tta 1 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                                

python test.py --cfg_file cfgs/target_nuscenes/ms3d_waymo_voxel_rcnn_anchorhead.yaml \
                --ckpt ../output/target_nuscenes/ms3d_waymo_voxel_rcnn_anchorhead/10f_xyzt_vehped_rnd1/ckpt/checkpoint_epoch_30.pth \
                --eval_tag nusc10xyzt_rwf_rwr \
                --extra_tag 10f_xyzt_vehped_rnd1 \
                --target_dataset nuscenes --sweeps 10 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                                