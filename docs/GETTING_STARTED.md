

# Usage

### Preliminary
Even if you already have generated infos from OpenPCDet, you need to re-generate the infos for nuScenes and Lyft (gt database not required) because we updated infos to include sequence metadata and we use 16 sweeps.

If you'd like to train a separate detector to the ones we've provided, you can follow [nuScenes SECOND-IoU](../tools/cfgs/nuscenes_models/uda_secondiou_vehicle.yaml) for reference. 

### 1. Generate predictions on unlabelled target domain

First we generate predictions with TTA on the unlabelled data with multiple detectors with 1-frame and 16-frame detection. Here is an example of how we can generate detections with one detector. We provide the TTA predictions for this section on [target-nuscenes](https://drive.google.com/drive/folders/1KAFrrE9oNG6rRrbII_Myzdz5bcCXA69t?usp=share_link) if you'd like to save time and try it out.

```shell

# 1-frame
python test.py --cfg_file cfgs/target-nuscenes/waymo_centerpoint.yaml \
                --ckpt ../model_zoo/waymo_centerpoint.pth \
                --extra_tag ms3d --eval_tag train_1f_tta-rwf-rwr \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 1 \
                DATA_CONFIG_TAR.DATA_SPLIT.test train \
                DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST placeholder  

# 16-frame
python test.py --cfg_file cfgs/target-nuscenes/waymo_centerpoint.yaml \
                --ckpt ../model_zoo/waymo_centerpoint.pth \
                --extra_tag msda --eval_tag train_16f_tta-rwf-rwr \
                --set DATA_CONFIG_TAR.MAX_SWEEPS 16 DATA_CONFIG_TAR.DATA_SPLIT.test train DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST placeholder                
```
We train MS3D with lidar data at 5Hz (except nuScenes). For 1-frame detections, we get predictions at 5Hz but for 16-frame detections we get predictions at 1.67Hz. For pseudo-label generation, we interpolate 16-frame detections to 5Hz.

Every dataset has different key frame rates. To downsample the frame rate for the above, we set `DATA_CONFIG_TAR.SAMPLED_INTERVAL.test` at different interval.
1. Waymo (10Hz): 1-frame (skip 2), 16-frame (skip 6)
2. nuScenes (2Hz): no downsampling, both at 2Hz, and we train at 2 Hz
3. Lyft (5Hz): 1-frame (skip 0), 16-frame (skip 3)

We provide a script for each target domain which you can reference for how we generated detections for our experiments for 1-frame (e.g. [waymo script](../tools/cfgs/target-waymo/generate_dets_1f.sh)) and 16-frame (e.g. [waymo script](../tools/cfgs/target-waymo/generate_dets_16f.sh)). Our reference script assumes the file structure:
```bash
MS3D
├── tools
├── ├── cfgs
│   │   ├── target-waymo
│   │   │   │── nuscenes_centerpoint.yaml # source-trained detector
│   │   │   │── nuscenes_secondiou.yaml 
│   │   │   │── lyft_centerpoint.yaml
│   │   │   │── lyft_secondiou.yaml
│   │   │   │── ft_nuscenes_secondiou.yaml # pre-trained detector you want to fine-tune
    - ...
```

After generating all detections, place their result.pkl paths in a file. Specify the file path in the config ACCUM1.PATH and ACCUM16.PATH. You can reference this file: [waymo paths](../tools/cfgs/target-waymo/det_16f_paths_s190_2hz.txt).
```
MS_DETECTOR_PS:
    FUSION: kde_fusion
    ACCUM1:
        PATH: '/MS3D/tools/cfgs/target-nuscenes/det_1f_paths.txt'
        DISCARD: 4
        RADIUS: 2.0
    ACCUM16:
        PATH: '/MS3D/tools/cfgs/target-nuscenes/det_16f_paths.txt'
        DISCARD: 3
        RADIUS: 1.0
```


### 2. Self-training: Generate pseudo-labels and fine-tune detector
Run the following command to:
1. Fuse detections, generate tracks, refine static objects, then generate pseudo-labels
2. Fine-tune the given pre-trained detector

```shell
python train.py \
        --cfg_file ${CONFIG_FILE} \
        --pretrained_model ${MODEL_PTH} \
        --extra_tag ${EXPERIMENT_NAME}

# here is an example of fine-tuning the waymo secondiou detector to the nuscenes domain with our pseudo-labels
python train.py \
        --cfg_file cfgs/target-nuscenes/ft_waymo_secondiou.yaml \
        --pretrained_model ../model_zoo/waymo_secondiou.pth  \
        --extra_tag exp_name
```
That's it!

**Note**
- Make sure "used_feature_list" is the same between source and target domain configs
- If pre-trained model was trained with "timestamp" channel, then test with "timestamp" channel for optimal performance even if we only use 1-frame for detection on target domain.

## General OpenPCDet Train/Test usage

### Test and evaluate the pretrained models
* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* To test with multiple GPUs:
```shell script
python -m torch.distributed.launch --nproc_per_node=4 --master_port 47771 test.py --launcher pytorch \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}

# or    

sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}

# or

sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}`, `--epochs ${EPOCHS}`, `--extra_tag ${EXPERIMENT_NAME}` to specify your preferred parameters.

* Train with multiple GPUs or multiple machines
```shell script
python -m torch.distributed.launch --nproc_per_node=4 --master_port 47771 train.py --launcher pytorch \
    --cfg_file ${CONFIG_FILE} --extra_tag ${EXPERIMENT_NAME}

# or    

sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# or 

sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE}
```

### Self-training with MS3D
This is basically the same as training a model, but we use our pseudo-labels to fine-tune an existing pre-trained detector. Testing is the same as above.

```shell
python train.py \
        --cfg_file ${CONFIG_FILE} \
        --pretrained_model ${CKPT_FILE}  \
        --extra_tag ${EXPERIMENT_NAME}
```

You can change parameters of the config file with `--set`. For example:
```shell
python train.py \
        --cfg_file cfgs/target-nuscenes/ft_waymo_secondiou.yaml \
        --pretrained_model ../model_zoo/nuscenes_secondiou_vehicle.pth  \
        --extra_tag 10sweep --set DATA_CONFIG_TAR.MAX_SWEEPS 10
```
