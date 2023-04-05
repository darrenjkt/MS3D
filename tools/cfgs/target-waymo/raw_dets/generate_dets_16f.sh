#!/bin/bash

# 190 scenes @ 2 Hz
python test.py --cfg_file cfgs/target-waymo/nuscenes_centerpoint.yaml \
                --ckpt ../model_zoo/nuscenes_centerpoint_tensweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_no_tta --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.USE_TTA False

python test.py --cfg_file cfgs/target-waymo/nuscenes_centerpoint.yaml \
                --ckpt ../model_zoo/nuscenes_centerpoint_tensweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_tta-rwf-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST placeholder

python test.py --cfg_file cfgs/target-waymo/nuscenes_centerpoint.yaml \
                --ckpt ../model_zoo/nuscenes_centerpoint_tensweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_tta-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_flip

python test.py --cfg_file cfgs/target-waymo/nuscenes_centerpoint.yaml \
                --ckpt ../model_zoo/nuscenes_centerpoint_tensweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_tta-rwf --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_rotation                                        

python test.py --cfg_file cfgs/target-waymo/nuscenes_secondiou.yaml \
                --ckpt ../model_zoo/nuscenes_secondiou_tensweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_no_tta --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.USE_TTA False

python test.py --cfg_file cfgs/target-waymo/nuscenes_secondiou.yaml \
                --ckpt ../model_zoo/nuscenes_secondiou_tensweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_tta-rwf-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST placeholder                            

python test.py --cfg_file cfgs/target-waymo/nuscenes_secondiou.yaml \
                --ckpt ../model_zoo/nuscenes_secondiou_tensweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_tta-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_flip                                                 

python test.py --cfg_file cfgs/target-waymo/nuscenes_secondiou.yaml \
                --ckpt ../model_zoo/nuscenes_secondiou_tensweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_tta-rwf --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_rotation

# LYFT
python test.py --cfg_file cfgs/target-waymo/lyft_centerpoint.yaml \
                --ckpt ../model_zoo/lyft_centerpoint_fivesweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_no_tta --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.USE_TTA False

python test.py --cfg_file cfgs/target-waymo/lyft_centerpoint.yaml \
                --ckpt ../model_zoo/lyft_centerpoint_fivesweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_tta-rwf-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST placeholder  

python test.py --cfg_file cfgs/target-waymo/lyft_centerpoint.yaml \
                --ckpt ../model_zoo/lyft_centerpoint_fivesweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_tta-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_flip

python test.py --cfg_file cfgs/target-waymo/lyft_centerpoint.yaml \
                --ckpt ../model_zoo/lyft_centerpoint_fivesweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_tta-rwf --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_rotation                          

python test.py --cfg_file cfgs/target-waymo/lyft_secondiou.yaml \
                --ckpt ../model_zoo/lyft_secondiou_fivesweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_no_tta --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.USE_TTA False                

python test.py --cfg_file cfgs/target-waymo/lyft_secondiou.yaml \
                --ckpt ../model_zoo/lyft_secondiou_fivesweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_tta-rwf-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST placeholder                

python test.py --cfg_file cfgs/target-waymo/lyft_secondiou.yaml \
                --ckpt ../model_zoo/lyft_secondiou_fivesweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_tta-rwr --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_flip                                                 

python test.py --cfg_file cfgs/target-waymo/lyft_secondiou.yaml \
                --ckpt ../model_zoo/lyft_secondiou_fivesweeps_vehicle.pth \
                --eval_tag customtrain_16f_s190_2hz_tta-rwf --batch_size 4 --extra_tag ms3d \
                --set DATA_CONFIG_TAR.SEQUENCE_CONFIG.ENABLED True DATA_CONFIG_TAR.SEQUENCE_CONFIG.SAMPLE_OFFSET [-15,0] \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.used_feature_list 'x','y','z' \
                DATA_CONFIG_TAR.POINT_FEATURE_ENCODING.src_feature_list 'x','y','z','intensity','elongation','timestamp' \
                DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.SAMPLED_INTERVAL.test 6 DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST random_world_rotation                                      

   
