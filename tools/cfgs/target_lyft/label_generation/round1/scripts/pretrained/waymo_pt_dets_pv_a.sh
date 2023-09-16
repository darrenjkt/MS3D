#!/bin/bash

python test.py --cfg_file ../model_zoo/waymo_pretrained/cfgs/waymo_pv_rcnn_plusplus_resnet_anchorhead_4f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/waymo_pretrained/waymo_pv_rcnn_plusplus_resnet_anchorhead_4f_xyzt_allcls.pth \
                --eval_tag waymo4xyzt_lyft1xyzt_notta \
                --target_dataset lyft --sweeps 1 --use_tta 0 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/waymo_pretrained/cfgs/waymo_pv_rcnn_plusplus_resnet_anchorhead_4f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/waymo_pretrained/waymo_pv_rcnn_plusplus_resnet_anchorhead_4f_xyzt_allcls.pth \
                --eval_tag waymo4xyzt_lyft1xyzt_rwf_rwr \
                --target_dataset lyft --sweeps 1 --use_tta 3 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none 

python test.py --cfg_file ../model_zoo/waymo_pretrained/cfgs/waymo_pv_rcnn_plusplus_resnet_anchorhead_4f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/waymo_pretrained/waymo_pv_rcnn_plusplus_resnet_anchorhead_4f_xyzt_allcls.pth \
                --eval_tag waymo4xyzt_lyft2xyzt_notta \
                --target_dataset lyft --sweeps 2 --use_tta 0 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/waymo_pretrained/cfgs/waymo_pv_rcnn_plusplus_resnet_anchorhead_4f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/waymo_pretrained/waymo_pv_rcnn_plusplus_resnet_anchorhead_4f_xyzt_allcls.pth \
                --eval_tag waymo4xyzt_lyft2xyzt_rwf_rwr \
                --target_dataset lyft --sweeps 2 --use_tta 3 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/waymo_pretrained/cfgs/waymo_pv_rcnn_plusplus_resnet_anchorhead_4f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/waymo_pretrained/waymo_pv_rcnn_plusplus_resnet_anchorhead_4f_xyzt_allcls.pth \
                --eval_tag waymo4xyzt_lyft3xyzt_notta \
                --target_dataset lyft --sweeps 3 --use_tta 0 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file ../model_zoo/waymo_pretrained/cfgs/waymo_pv_rcnn_plusplus_resnet_anchorhead_4f_xyzt_allcls.yaml \
                --ckpt ../model_zoo/waymo_pretrained/waymo_pv_rcnn_plusplus_resnet_anchorhead_4f_xyzt_allcls.pth \
                --eval_tag waymo4xyzt_lyft3xyzt_rwf_rwr \
                --target_dataset lyft --sweeps 3 --use_tta 3 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                                

# ---------------------