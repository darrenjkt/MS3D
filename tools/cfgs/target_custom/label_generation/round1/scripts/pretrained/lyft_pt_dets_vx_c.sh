#!/bin/bash

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_custom1xyzt_notta \
                --target_dataset custom --sweeps 1 --batch_size 8 --use_tta 0 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_custom1xyzt_rwf_rwr \
                --target_dataset custom --sweeps 1 --batch_size 8 --use_tta 3 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_custom2xyzt_notta \
                --target_dataset custom --sweeps 2 --batch_size 8 --use_tta 0 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_custom2xyzt_rwf_rwr \
                --target_dataset custom --sweeps 2 --batch_size 8 --use_tta 3 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none     

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_custom4xyzt_notta \
                --target_dataset custom --sweeps 4 --batch_size 8 --use_tta 0 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/lyft_pretrained/cfgs/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/lyft_pretrained/lyft_voxel_rcnn_centerhead_3f_xyzt_allcls.pth \
                --eval_tag lyft3xyzt_custom4xyzt_rwf_rwr \
                --target_dataset custom --sweeps 4 --batch_size 8 --use_tta 3 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none             
# ---------------------