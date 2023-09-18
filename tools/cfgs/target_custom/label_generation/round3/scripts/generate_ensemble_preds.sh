#!/bin/bash

python test.py --cfg_file cfgs/target_custom/ms3d_waymo_voxel_rcnn_centerhead.yaml \
                --ckpt ../output/target_custom/ms3d_waymo_voxel_rcnn_centerhead/4f_xyzt_vehped_rnd2/ckpt/checkpoint_epoch_30.pth \
                --eval_tag custom4xyzt_notta \
                --extra_tag 4f_xyzt_vehped_rnd2 \
                --target_dataset custom --sweeps 4 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file cfgs/target_custom/ms3d_waymo_voxel_rcnn_centerhead.yaml \
                --ckpt ../output/target_custom/ms3d_waymo_voxel_rcnn_centerhead/4f_xyzt_vehped_rnd2/ckpt/checkpoint_epoch_30.pth \
                --eval_tag custom4xyzt_rwr \
                --extra_tag 4f_xyzt_vehped_rnd2 \
                --target_dataset custom --sweeps 4 --use_tta 2 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                

python test.py --cfg_file cfgs/target_custom/ms3d_waymo_voxel_rcnn_centerhead.yaml \
                --ckpt ../output/target_custom/ms3d_waymo_voxel_rcnn_centerhead/4f_xyzt_vehped_rnd2/ckpt/checkpoint_epoch_30.pth \
                --eval_tag custom4xyzt_rwr \
                --extra_tag 4f_xyzt_vehped_rnd2 \
                --target_dataset custom --sweeps 4 --use_tta 1 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                

python test.py --cfg_file cfgs/target_custom/ms3d_waymo_voxel_rcnn_centerhead.yaml \
                --ckpt ../output/target_custom/ms3d_waymo_voxel_rcnn_centerhead/4f_xyzt_vehped_rnd2/ckpt/checkpoint_epoch_30.pth \
                --eval_tag custom4xyzt_rwr \
                --extra_tag 4f_xyzt_vehped_rnd2 \
                --target_dataset custom --sweeps 4 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                                

python test.py --cfg_file cfgs/target_custom/ms3d_waymo_voxel_rcnn_anchorhead.yaml \
                --ckpt ../output/target_custom/ms3d_waymo_voxel_rcnn_anchorhead/4f_xyzt_vehped_rnd2/ckpt/checkpoint_epoch_30.pth \
                --eval_tag custom4xyzt_notta \
                --extra_tag 4f_xyzt_vehped_rnd2 \
                --target_dataset custom --sweeps 4 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                                

python test.py --cfg_file cfgs/target_custom/ms3d_waymo_voxel_rcnn_anchorhead.yaml \
                --ckpt ../output/target_custom/ms3d_waymo_voxel_rcnn_anchorhead/4f_xyzt_vehped_rnd2/ckpt/checkpoint_epoch_30.pth \
                --eval_tag custom4xyzt_rwr \
                --extra_tag 4f_xyzt_vehped_rnd2 \
                --target_dataset custom --sweeps 4 --use_tta 2 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                                

python test.py --cfg_file cfgs/target_custom/ms3d_waymo_voxel_rcnn_anchorhead.yaml \
                --ckpt ../output/target_custom/ms3d_waymo_voxel_rcnn_anchorhead/4f_xyzt_vehped_rnd2/ckpt/checkpoint_epoch_30.pth \
                --eval_tag custom4xyzt_rwf \
                --extra_tag 4f_xyzt_vehped_rnd2 \
                --target_dataset custom --sweeps 4 --use_tta 1 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                                

python test.py --cfg_file cfgs/target_custom/ms3d_waymo_voxel_rcnn_anchorhead.yaml \
                --ckpt ../output/target_custom/ms3d_waymo_voxel_rcnn_anchorhead/4f_xyzt_vehped_rnd2/ckpt/checkpoint_epoch_30.pth \
                --eval_tag custom4xyzt_rwf_rwr \
                --extra_tag 4f_xyzt_vehped_rnd2 \
                --target_dataset custom --sweeps 4 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                                