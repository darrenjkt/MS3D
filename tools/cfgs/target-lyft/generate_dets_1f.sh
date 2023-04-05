#!/bin/bash

python test.py --cfg_file cfgs/target-lyft/nuscenes_centerpoint.yaml \
                --ckpt ../model_zoo/nuscenes_centerpoint_tensweeps_vehicle.pth \
                --eval_tag train_1f_5hz_no_tta --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.USE_TTA False                 

python test.py --cfg_file cfgs/target-lyft/nuscenes_centerpoint.yaml \
                --ckpt ../model_zoo/nuscenes_centerpoint_tensweeps_vehicle.pth \
                --eval_tag train_1f_5hz_tta-rwf-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST placeholder

python test.py --cfg_file cfgs/target-lyft/nuscenes_centerpoint.yaml \
                --ckpt ../model_zoo/nuscenes_centerpoint_tensweeps_vehicle.pth \
                --eval_tag train_1f_5hz_tta-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_flip

python test.py --cfg_file cfgs/target-lyft/nuscenes_centerpoint.yaml \
                --ckpt ../model_zoo/nuscenes_centerpoint_tensweeps_vehicle.pth \
                --eval_tag train_1f_5hz_tta-rwf --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_rotation                                        

python test.py --cfg_file cfgs/target-lyft/nuscenes_secondiou.yaml \
                --ckpt ../model_zoo/nuscenes_secondiou_tensweeps_vehicle.pth \
                --eval_tag train_1f_5hz_no_tta --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.USE_TTA False

python test.py --cfg_file cfgs/target-lyft/nuscenes_secondiou.yaml \
                --ckpt ../model_zoo/nuscenes_secondiou_tensweeps_vehicle.pth \
                --eval_tag train_1f_5hz_tta-rwf-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST placeholder                            

python test.py --cfg_file cfgs/target-lyft/nuscenes_secondiou.yaml \
                --ckpt ../model_zoo/nuscenes_secondiou_tensweeps_vehicle.pth \
                --eval_tag train_1f_5hz_tta-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_flip                                                 

python test.py --cfg_file cfgs/target-lyft/nuscenes_secondiou.yaml \
                --ckpt ../model_zoo/nuscenes_secondiou_tensweeps_vehicle.pth \
                --eval_tag train_1f_5hz_tta-rwf --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_rotation

# LYFT
python test.py --cfg_file cfgs/target-lyft/waymo_centerpoint.yaml \
                --ckpt ../model_zoo/waymo_centerpoint.pth \
                --eval_tag train_1f_5hz_no_tta --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.USE_TTA False

python test.py --cfg_file cfgs/target-lyft/waymo_centerpoint.yaml \
                --ckpt ../model_zoo/waymo_centerpoint.pth \
                --eval_tag train_1f_5hz_tta-rwf-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST placeholder  

python test.py --cfg_file cfgs/target-lyft/waymo_centerpoint.yaml \
                --ckpt ../model_zoo/waymo_centerpoint.pth \
                --eval_tag train_1f_5hz_tta-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_flip

python test.py --cfg_file cfgs/target-lyft/waymo_centerpoint.yaml \
                --ckpt ../model_zoo/waymo_centerpoint.pth \
                --eval_tag train_1f_5hz_tta-rwf --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_rotation                          

python test.py --cfg_file cfgs/target-lyft/waymo_secondiou.yaml \
                --ckpt ../model_zoo/waymo_secondiou.pth \
                --eval_tag train_1f_5hz_no_tta --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.USE_TTA False

python test.py --cfg_file cfgs/target-lyft/waymo_secondiou.yaml \
                --ckpt ../model_zoo/waymo_secondiou.pth \
                --eval_tag train_1f_5hz_tta-rwf-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST placeholder                

python test.py --cfg_file cfgs/target-lyft/waymo_secondiou.yaml \
                --ckpt ../model_zoo/waymo_secondiou.pth \
                --eval_tag train_1f_5hz_tta-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_flip                                                 

python test.py --cfg_file cfgs/target-lyft/waymo_secondiou.yaml \
                --ckpt ../model_zoo/waymo_secondiou.pth \
                --eval_tag train_1f_5hz_tta-rwf --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.MAX_SWEEPS 1 DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 1 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_rotation                                      

   
