#!/bin/bash

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_nusc1xyzt_notta \
                --target_dataset nuscenes --sweeps 1 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_nusc1xyzt_rwf_rwr \
                --target_dataset nuscenes --sweeps 1 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_nusc5xyzt_notta \
                --target_dataset nuscenes --sweeps 5 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_nusc5xyzt_rwf_rwr \
                --target_dataset nuscenes --sweeps 5 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none     

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_nusc9xyzt_notta \
                --target_dataset nuscenes --sweeps 9 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_nusc9xyzt_rwf_rwr \
                --target_dataset nuscenes --sweeps 9 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none             

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_nusc9xyzt_rwf \
                --target_dataset nuscenes --sweeps 9 --use_tta 1 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_nusc9xyzt_rwr \
                --target_dataset nuscenes --sweeps 9 --use_tta 2 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                                                

# ---------------------