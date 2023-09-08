#!/bin/bash

# ---- single frame ----
python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_anchorhead_1f_xyz_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_anchorhead_1f_xyz_allcls.pth \
                --eval_tag lyft1xyz_waymo1xyz_custom190_notta \
                --target_dataset waymo --sweeps 1 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_anchorhead_1f_xyz_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_anchorhead_1f_xyz_allcls.pth \
                --eval_tag lyft1xyz_waymo1xyz_custom190_rwr \
                --target_dataset waymo --sweeps 1 --use_tta 2 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_anchorhead_1f_xyz_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_anchorhead_1f_xyz_allcls.pth \
                --eval_tag lyft1xyz_waymo1xyz_custom190_rwf \
                --target_dataset waymo --sweeps 1 --use_tta 1 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_anchorhead_1f_xyz_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_anchorhead_1f_xyz_allcls.pth \
                --eval_tag lyft1xyz_waymo1xyz_custom190_rwf_rwr \
                --target_dataset waymo --sweeps 1 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none                                                              

# ---- multi frame ----

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_anchorhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_anchorhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_waymo3xyzt_custom190_notta \
                --target_dataset waymo --sweeps 3 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_anchorhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_anchorhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_waymo3xyzt_custom190_rwf_rwr \
                --target_dataset waymo --sweeps 3 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none  

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_anchorhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_anchorhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_waymo5xyzt_custom190_notta \
                --target_dataset waymo --sweeps 5 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_anchorhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_anchorhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_waymo5xyzt_custom190_rwf_rwr \
                --target_dataset waymo --sweeps 5 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 2 MODEL.POST_PROCESSING.EVAL_METRIC none                                

# ---------------------