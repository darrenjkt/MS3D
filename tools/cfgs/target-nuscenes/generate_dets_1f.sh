#!/bin/bash

# 150 scenes @ 2 Hz
python test.py --cfg_file cfgs/target-nuscenes/waymo_centerpoint.yaml \
                --ckpt ../model_zoo/waymo_centerpoint.pth \
                --eval_tag customtrain_1f_no_tta --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.USE_TTA False

python test.py --cfg_file cfgs/target-nuscenes/waymo_centerpoint.yaml \
                --ckpt ../model_zoo/waymo_centerpoint.pth \
                --eval_tag customtrain_1f_tta-rwf-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST placeholder

python test.py --cfg_file cfgs/target-nuscenes/waymo_centerpoint.yaml \
                --ckpt ../model_zoo/waymo_centerpoint.pth \
                --eval_tag customtrain_1f_tta-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_flip

python test.py --cfg_file cfgs/target-nuscenes/waymo_centerpoint.yaml \
                --ckpt ../model_zoo/waymo_centerpoint.pth \
                --eval_tag customtrain_1f_tta-rwf --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_rotation                                        

python test.py --cfg_file cfgs/target-nuscenes/waymo_secondiou.yaml \
                --ckpt ../model_zoo/waymo_secondiou.pth \
                --eval_tag customtrain_1f_no_tta --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.USE_TTA False

python test.py --cfg_file cfgs/target-nuscenes/waymo_secondiou.yaml \
                --ckpt ../model_zoo/waymo_secondiou.pth \
                --eval_tag customtrain_1f_tta-rwf-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST placeholder                            

python test.py --cfg_file cfgs/target-nuscenes/waymo_secondiou.yaml \
                --ckpt ../model_zoo/waymo_secondiou.pth \
                --eval_tag customtrain_1f_tta-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_flip                                                 

python test.py --cfg_file cfgs/target-nuscenes/waymo_secondiou.yaml \
                --ckpt ../model_zoo/waymo_secondiou.pth \
                --eval_tag customtrain_1f_tta-rwf --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_rotation

# LYFT
python test.py --cfg_file cfgs/target-nuscenes/lyft_centerpoint.yaml \
                --ckpt ../model_zoo/lyft_centerpoint_fivesweeps_vehicle.pth \
                --eval_tag customtrain_1f_no_tta --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.USE_TTA False

python test.py --cfg_file cfgs/target-nuscenes/lyft_centerpoint.yaml \
                --ckpt ../model_zoo/lyft_centerpoint_fivesweeps_vehicle.pth \
                --eval_tag customtrain_1f_tta-rwf-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST placeholder  

python test.py --cfg_file cfgs/target-nuscenes/lyft_centerpoint.yaml \
                --ckpt ../model_zoo/lyft_centerpoint_fivesweeps_vehicle.pth \
                --eval_tag customtrain_1f_tta-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_flip

python test.py --cfg_file cfgs/target-nuscenes/lyft_centerpoint.yaml \
                --ckpt ../model_zoo/lyft_centerpoint_fivesweeps_vehicle.pth \
                --eval_tag customtrain_1f_tta-rwf --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_rotation                          

python test.py --cfg_file cfgs/target-nuscenes/lyft_secondiou.yaml \
                --ckpt ../model_zoo/lyft_secondiou_fivesweeps_vehicle.pth \
                --eval_tag customtrain_1f_no_tta --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.USE_TTA False

python test.py --cfg_file cfgs/target-nuscenes/lyft_secondiou.yaml \
                --ckpt ../model_zoo/lyft_secondiou_fivesweeps_vehicle.pth \
                --eval_tag customtrain_1f_tta-rwf-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST placeholder                

python test.py --cfg_file cfgs/target-nuscenes/lyft_secondiou.yaml \
                --ckpt ../model_zoo/lyft_secondiou_fivesweeps_vehicle.pth \
                --eval_tag customtrain_1f_tta-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_flip                                                 

python test.py --cfg_file cfgs/target-nuscenes/lyft_secondiou.yaml \
                --ckpt ../model_zoo/lyft_secondiou_fivesweeps_vehicle.pth \
                --eval_tag customtrain_1f_tta-rwf --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_rotation                                      

   
