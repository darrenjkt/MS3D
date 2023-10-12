
# General OpenPCDet Train/Test usage

Here are just some basic commands for using OpenPCDet.

### Training a pretrained model for MS3D
If you'd like to train a separate detector to the ones we've provided, you can use the same config files/commands as OpenPCDet, just make sure to include the `DATA_CONFIG._BASE_CONFIG_` as our provided `nuscenes_dataset_da.yaml` or `waymo_dataset_da.yaml`. Refer to [nuScenes VoxelRCNN (Anchor)](../tools/cfgs/nuscenes_models/uda_voxel_rcnn_anchorhead.yaml) as an example. 

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