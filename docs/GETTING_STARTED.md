

# MS3D Usage

## Problem Definition
Our MS3D framework falls in the category of Unsupervised Domain Adaptation (UDA) works, where the task is to adapt the off-the-shelf model(s) to a new domain that it has not seen during its training. In the UDA setting, we assume that no labels are present in the target domain (i.e. unsupervised) for the adaptation process. 

## Preliminary
Even if you already have generated infos from OpenPCDet, you need to re-generate the infos for nuScenes and Lyft (gt database not required) because we modified the infos to include sequence metadata.

If you'd like to train a separate detector to the ones we've provided, you can use the same config files as OpenPCDet, just make sure to include the `DATA_CONFIG._BASE_CONFIG_` as our provided `nuscenes_dataset_da.yaml` or `waymo_dataset_da.yaml`. You can see [nuScenes VoxelRCNN (Anchor)](../tools/cfgs/nuscenes_models/uda_voxel_rcnn_anchorhead.yaml) for reference. 

Now let's go through how to use MS3D! We also provide a simple [tutorial](../tools/demo/ms3d_demo_tutorial.ipynb) which steps through the functions used in MS3D.

## Overview
For MS3D you only need to modify 3 files: `generate_ensemble_preds.sh`, `ensemble_detections.txt` and `ps_config.yaml`. 

The first two are for generating the ensemble and `ps_config.yaml` sets the configs for MS3D, which we run with `bash run_ms3d.sh`. All bash scripts should be run from the `/MS3D/tools` folder.

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
cd /MS3D/tools && bash cfgs/target_dataset/label_generation/roundN/scripts/generate_ensemble_preds.sh
```
You can reference how we generate detections for nuScenes as the target domain [here](../tools/cfgs/target_nuscenes/label_generation/round1/scripts/pretrained/). Modify the pre-trained model's ckpt and cfg file to the location of your downloaded files.

**Specify Detection Paths**: After generating all detections, put the absolute file paths of the result.pkl in a text file e.g. refer to [nuScenes' ensemble_detections.txt](../tools/cfgs/target_nuscenes/label_generation/round1/cfgs/ensemble_detections.txt)

Now that we have a lot of detection sets, we can feed them into our MS3D++ framework! 

### Additional Comments
For our paper's results, we train detectors with lidar data at 5Hz (except nuScenes, which is at 2Hz). 

Every dataset has different key frame rates. To downsample the frame rate for the above, we set `DATA_CONFIG_TAR.SAMPLED_INTERVAL.test` at different intervals.
1. Waymo (10Hz): skip 2
2. nuScenes (2Hz): no downsampling, both at 2Hz, and we train at 2 Hz
3. Lyft (5Hz): skip 0

If you have the compute, we recommend higher sampling rates to reduce tracking errors for better pseudo-label quality. 

We follow OpenPCDet's format for detector predictions which should make it easy if you wish to load in detection sets from another repo rather than use our scripts above. 

**Important Note:** Pre-trained detectors for the UDA setting should be trained with point clouds and labels in the ground plane for better cross-domain performance. I.e. the ground plane of the point cloud should be roughly at `z=0`. This means measuring (approximately) the lidar mounting height and adding `SHIFT_COOR=[0,0,lidar_height]` to the point cloud points. For context, this is also the same coordinate frame used in the Waymo dataset's point clouds. 

**Tip**: Experiment with including different pre-trained detectors in the ensemble. Here are some ideas.
- Multi-modal: images can help distinguish many smaller object classes like pedestrians, or far-range objects.
- Multi-frame: any state-of-the-art multi-frame models (e.g. MPPNet) can be used to generate detections.
- Transformer-based: better performance than many voxel/point-based methods, maybe it has better domain generalisation too.
- Class-specific: 3D detectors trained solely for a specific class may have better performance
- Varying voxel sizes: ensembling voxel-based detectors with different voxel sizes may be able to have better generalization to different lidar scan patterns.

## 2. Auto-labeling with MS3D++
Our framework is contained in 3 files in `/MS3D/tools` that load in our auto-labeling configurations from `label_generation/round1/cfgs/ps_config.yaml`.
1. `ensemble_kbf.py`  fuses the detection_sets specified in `ensemble_detections.txt` to give us our initial pseudo-label set.
2. `generate_tracks.py`  uses SimpleTrack to generate tracks for the initial pseudo-labels. We use this to generate 3 sets of tracks (veh_all, veh_static, ped). 
3. `temporal_refinement.py` refines the pseudo-labels with temporal information and object characteristics. This file gives us our final pseudo-labels. 

These can all be run with `run_ms3d.sh` in their respective target_domain folders. This will give you a set of labels for each point cloud frame stored in `label_generation/round1/ps_labels/final_ps_dict.pkl`.
```bash
cd /MS3D/tools && bash cfgs/target_dataset/label_generation/roundN/scripts/run_ms3d.sh
```

`ps_config.yaml` is already pre-set for each target domain to achieve the results reported in our MS3D++ paper. If you wish to tweak it, we explain how to do so in [MS3D_PARAMETERS.md](../docs/MS3D_PARAMETERS.md).

If you wish to use MS3D++ for your own point cloud data, refer to [CUSTOM_DATASET_TUTORIAL.md](../docs/CUSTOM_DATASET_TUTORIAL.md).

That's it for the auto-labeling! 

## 3. Training a model with MS3D labels
To train a model, simply specify the pseudo-label path in the detector config file with the following format e.g.
```yaml
SELF_TRAIN:
    INIT_PS: /MS3D/tools/cfgs/target_nuscenes/label_generation/round1/ps_labels/final_ps_dict.pkl
```
Take a look at [nuScenes' VoxelRCNN (Anchor)](tools/cfgs/target_nuscenes/ms3d_waymo_voxel_rcnn_anchorhead.yaml) config file for reference.

Following this, we just train the model like in OpenPCDet. Sometimes it helps to initialize from an existing pre-trained model. 
```bash
python train.py --cfg_file ${CONFIG_FILE} --extra_tag ${EXTRA_TAG} --pretrained_model ${EXTRA_TAG}

# Example
python train.py --cfg_file cfgs/target_nuscenes/ms3d_waymo_voxel_rcnn_anchorhead.yaml --extra_tag 10xyzt_vehped --pretrained_model ../model_zoo/waymo_voxel_rcnn_anchorhead.pth
```
You can change parameters of the config file with `--set` for both `train.py` and `test.py`.

## 4. Multi-stage self-training
To do multiple rounds of self-training for iterative label refinement, you can follow our scripts and configs provided in each target domain which runs the same functions as steps 1-2 above for label generation. 

The main idea is to use our trained models from the 1st round to re-generate the label set for the same data. The trained models should have improved label quality and be able to detect objects that were not initially labeled. 

In our paper, we only used MS3D++ trained models for the current round to form the ensemble for the next round. Feel free to experiment with other ensemble combinations such as:
- including the trained models in the initial pre-trained ensemble with detector weighting.
- training different detector types for the ensemble (e.g. multi-modal, transformer-based, etc) 

**Tip**: Try to use strict ps_config settings for earlier rounds to reduce false positives. It's very hard to get rid of false positives in later rounds. 

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