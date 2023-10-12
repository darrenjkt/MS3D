

# MS3D Usage for Reproducing Results

Here we give a guide for reproducing the results in our papers. 

If you simply wish to use MS3D++ to auto-label your own point cloud data, refer to our [**auto-label your data guide**](../docs/AUTO_LABEL_YOUR_DATA.md) instead.

We assume you've successfully followed our [**installation guide**](../docs/INSTALL.md).

## Problem Definition
Our MS3D framework falls in the category of Unsupervised Domain Adaptation (UDA) works, where the task is to adapt the off-the-shelf model(s) to a new domain that it has not seen during its training. In the UDA setting, we assume that no labels are present in the target domain (i.e. unsupervised) for the adaptation process. 

## Data Preparation
Download the datasets and set them up by following our [**dataset preparation guide**](../docs/DATASET_PREPARATION.md)

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

## 2. Auto-labeling with MS3D++
Our framework is contained in 3 files in `/MS3D/tools` that load in our auto-labeling configurations from `label_generation/round1/cfgs/ps_config.yaml`.
1. `ensemble_kbf.py`  fuses the detection_sets specified in `ensemble_detections.txt` to give us our initial pseudo-label set.
2. `generate_tracks.py`  uses SimpleTrack to generate tracks for the initial pseudo-labels. We use this to generate 3 sets of tracks (veh_all, veh_static, ped). 
3. `temporal_refinement.py` refines the pseudo-labels with temporal information and object characteristics. This file gives us our final pseudo-labels. 

We can run the auto-labeling framework as follows. Replace the `target_dataset` and `roundN` in the file path below as necessary.
```bash
cd /MS3D/tools && bash cfgs/target_dataset/label_generation/roundN/scripts/run_ms3d.sh
```
 This will give you a set of labels for each point cloud frame stored in `label_generation/roundN/ps_labels/final_ps_dict.pkl`.

Your main workspace will be within the `target_dataset` folder:
```bash
MS3D
├── tools
│   ├── cfgs
│   │   ├── target_dataset # e.g. target_waymo, nusc, lyft
│   │   |   ├── label_generation
│   │   |   │   ├── round1
│   │   |   │   │   ├── cfgs
│   │   |   │   │   │   ├── ensemble_detections.txt
│   │   |   │   │   │   ├── ps_config.yaml
│   │   |   │   │   ├── scripts
│   │   |   │   │   │   ├── generate_ensemble_preds.sh
│   │   |   │   │   │   ├── run_ms3d.sh
│   │   |   │   │   ├── ps_labels # pseudo-labels/tracks are saved here
│   │   |   │   ├── round2
│   │   |   │   ├── round3
│   │   |   │   ├── ...
```

`ps_config.yaml` is already pre-set for each target domain to achieve the results reported in our MS3D++ paper. If you wish to tweak it, we explain how to do so in [MS3D_PARAMETERS.md](../docs/MS3D_PARAMETERS.md).

That's it for the auto-labeling! 

## 3. Training a model with MS3D labels
To train a model with the pseudo-labels, specify the pseudo-label absolute path in the detector config file as follows:
```yaml
SELF_TRAIN:
    INIT_PS: /MS3D/tools/cfgs/target_nuscenes/label_generation/round1/ps_labels/final_ps_dict.pkl
```
Take a look at [nuScenes' VoxelRCNN (Anchor)](tools/cfgs/target_nuscenes/ms3d_waymo_voxel_rcnn_anchorhead.yaml) config file for reference.

Following this, we just train the model like in OpenPCDet. 
```bash
python train.py --cfg_file ${CONFIG_FILE} --extra_tag ${EXTRA_TAG}
```
Sometimes it helps to initialize from an existing pre-trained model. 
```bash
python train.py --cfg_file ${CONFIG_FILE} --extra_tag ${EXTRA_TAG} --pretrained_model ${EXTRA_TAG}

# Example
python train.py --cfg_file cfgs/target_nuscenes/ms3d_waymo_voxel_rcnn_anchorhead.yaml --extra_tag 10xyzt_vehped --pretrained_model ../model_zoo/waymo_voxel_rcnn_anchorhead.pth
```
You can change parameters of the config file with `--set` for both `train.py` and `test.py`.

## 4. Iterative Label Refinement
With each round, you'll get a trained detector for the target dataset. We trained multiple detectors (e.g. VoxelRCNN-anchorhead, VoxelRCNN-centerhead) and used them for auto-labelling the same dataset. 

We provided the configs we used to achieve the reported results for each round. 
```
target_dataset
├── label_generation
│   ├── round1
│   │   ├── cfgs
│   │   │   ├── ps_config.yaml
│   │   ├── scripts
│   ├── round2
│   ├── round3
```

We believe using `gt_database` augmentation to build a database of high confidence pseudo-labels for each round to paste rare/hard samples into the scenes for training can further boost performance.

# OpenPCDet Usage
For a guide on general OpenPCDet commands, refer to our [OpenPCDet guide](../docs/OPENPCDET_USAGE.md).