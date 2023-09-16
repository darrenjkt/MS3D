#!/bin/bash

python test.py --cfg_file ../model_zoo/waymo_pretrained/cfgs/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/waymo_pretrained/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.pth \
                --eval_tag waymo4xyzt_nusc1xyzt_notta \
                --target_dataset nuscenes --sweeps 1 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/waymo_pretrained/cfgs/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/waymo_pretrained/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.pth \
                --eval_tag waymo4xyzt_nusc1xyzt_rwf_rwr \
                --target_dataset nuscenes --sweeps 1 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none 

python test.py --cfg_file ../model_zoo/waymo_pretrained/cfgs/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/waymo_pretrained/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.pth \
                --eval_tag waymo4xyzt_nusc3xyzt_notta \
                --target_dataset nuscenes --sweeps 3 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/waymo_pretrained/cfgs/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/waymo_pretrained/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.pth \
                --eval_tag waymo4xyzt_nusc3xyzt_rwf_rwr \
                --target_dataset nuscenes --sweeps 3 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                

python test.py --cfg_file ../model_zoo/waymo_pretrained/cfgs/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/waymo_pretrained/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.pth \
                --eval_tag waymo4xyzt_nusc5xyzt_notta \
                --target_dataset nuscenes --sweeps 5 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/waymo_pretrained/cfgs/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/waymo_pretrained/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.pth \
                --eval_tag waymo4xyzt_nusc5xyzt_rwf_rwr \
                --target_dataset nuscenes --sweeps 5 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                

python test.py --cfg_file ../model_zoo/waymo_pretrained/cfgs/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/waymo_pretrained/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.pth \
                --eval_tag waymo4xyzt_nusc7xyzt_notta \
                --target_dataset nuscenes --sweeps 7 --use_tta 0 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/waymo_pretrained/cfgs/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/waymo_pretrained/waymo_pv_rcnn_plusplus_resnet_centerhead_4f_xyzt_allcls.pth \
                --eval_tag waymo4xyzt_nusc7xyzt_rwf_rwr \
                --target_dataset nuscenes --sweeps 7 --use_tta 3 --custom_target_scenes \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                                

# ---------------------