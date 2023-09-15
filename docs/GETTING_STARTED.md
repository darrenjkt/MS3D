

# MS3D Usage

## Preliminary
Even if you already have generated infos from OpenPCDet, you need to re-generate the infos for nuScenes and Lyft (gt database not required) because we modified the infos to include sequence metadata.

If you'd like to train a separate detector to the ones we've provided, you can use the same config files as OpenPCDet, just make sure to include the `DATA_CONFIG._BASE_CONFIG_` as our provided `nuscenes_dataset_da.yaml` or `waymo_dataset_da.yaml`. You can see [nuScenes VoxelRCNN (Anchor)](../tools/cfgs/nuscenes_models/uda_voxel_rcnn_anchorhead.yaml) for reference. 

Now let's go through how to use MS3D! We also provide a simple [tutorial](../tools/demo/ms3d_demo_tutorial.ipynb) which steps through the functions used in MS3D.

## Overview
For MS3D you only need to modify 3 files: `generate_ensemble_preds.sh`, `ensemble_detections.txt` and `ps_config.yaml`. 

The first two are for generating the ensemble and `ps_config.yaml` sets the configs for MS3D, which we run with `bash run_ms3d.sh`


We've organised each target domain to follow the structure as shown below.

```
MS3D
├── data
├── pcdet
├── tools
|   ├── cfgs
|   |   ├── target_nuscenes
|   |   |   ├── label_generation
|   |   |   |   ├── round1
|   |   |   |   |   ├── cfgs
|   |   |   |   |   |   ├── ps_config.yaml
|   |   |   |   |   |   ├── ensemble_detections.txt
|   |   |   |   |   ├── scripts
|   |   |   |   |   |   ├── generate_ensemble_preds.sh
|   |   |   |   |   |   ├── run_ms3d.sh
|   |   |   |   ├── round2
|   |   |   |   ├── ...
|   |   ├── target_waymo
|   |   ├── ...
```
We now explain each of the files.

## 1. Generate predictions on unlabelled target domain

In this step we simply need to download the pre-trained models provided and generate predictions for our unlabeled point clouds. We use test-time augmentation and VMFI to generate multiple detection sets for each pre-trained model.

**Generate Detections**: We provide a script to run this in each target domain's script folder. Simply run it with:
```bash
bash generate_ensemble_preds.sh
```
You can reference how we generate detections for nuScenes as the target domain [here](../tools/cfgs/target_nuscenes/label_generation/round1/scripts/pretrained/). Modify the pre-trained model's ckpt and cfg file to the location of your downloaded files.

**Specify Detection Paths**: After generating all detections, put the absolute file paths of the result.pkl in a text file e.g. refer to [nuScenes' ensemble_detections.txt](../tools/cfgs/target_nuscenes/label_generation/round1/cfgs/ensemble_detections.txt)

**Additional comments**:
For our paper's results, we train detectors with lidar data at 5Hz (except nuScenes, which is at 2Hz). 

Every dataset has different key frame rates. To downsample the frame rate for the above, we set `DATA_CONFIG_TAR.SAMPLED_INTERVAL.test` at different intervals.
1. Waymo (10Hz): 1-frame (skip 2), 16-frame (skip 6)
2. nuScenes (2Hz): no downsampling, both at 2Hz, and we train at 2 Hz
3. Lyft (5Hz): 1-frame (skip 0), 16-frame (skip 3)

If you have the compute, we recommend higher sampling rates to reduce tracking errors for better pseudo-label quality. 

We follow OpenPCDet's format for detector predictions which should make it easy if you wish to load in detection sets from another repo rather than use our scripts above.

Now that we have a lot of detection sets, we can feed them into our MS3D++ framework! 

## 2. MS3D++ Framework
Our framework is contained in 3 files in `/MS3D/tools`.
1. `ensemble_kbf.py`  fuses the detection_sets specified in `ensemble_detections.txt` to give us our initial pseudo-label set.
2. `generate_tracks.py`  uses SimpleTrack to generate tracks for the initial pseudo-labels. We use this to generate 3 sets of tracks (veh_all, veh_static, ped). 
3. `temporal_refinement.py` refines the pseudo-labels with temporal information and object characteristics. This file gives us our final pseudo-labels. 

These can all be run with `run_ms3d.sh` in their respective target_domain folders.
```bash
bash run_ms3d.sh
```

The `ps_config.yaml` is already pre-set for each target domain to achieve the results reported in our MS3D++ paper. If you wish to tweak it, we explain how to do so in [MS3D_PARAMETERS.md](../docs/MS3D_PARAMETERS.md).

If you wish to use MS3D++ for your own point cloud data, refer to [CUSTOM_DATASET_TUTORIAL.md](../docs/CUSTOM_DATASET_TUTORIAL.md).

That's it for the auto-labeling!

## 3. Training a model with MS3D labels
To train a model, simply specify the pseudo-label path in the detector config file with the following format:
```yaml
SELF_TRAIN:
    INIT_PS: /MS3D/tools/cfgs/target_nuscenes/label_generation/round1/ps_labels/final_ps_dict.pkl
```
Take a look at [nuScenes' VoxelRCNN (Anchor)](tools/cfgs/target_nuscenes/ms3d_waymo_voxel_rcnn_anchorhead.yaml) config file for reference.

Following this, we just train the model like in OpenPCDet. Sometimes it helps to initialize from an existing pre-trained model. 
```bash
python train.py --cfg_file ${CONFIG_FILE} --extra_tag ${EXTRA_TAG} --pretrained_model ${EXTRA_TAG}
```
You can change parameters of the config file with `--set` for both `train.py` and `test.py`.

# General OpenPCDet Train/Test usage

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