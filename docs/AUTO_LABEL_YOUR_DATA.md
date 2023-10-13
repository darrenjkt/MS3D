# Auto-label your data

In this guide, we walk through how you can auto-label your point cloud sequences with MS3D.

We assume you've successfully followed our [**installation guide**](../docs/INSTALL.md).

#### Contents
1. [Data Collection for MS3D](#data-collection-for-ms3d)
2. [Data Preparation](#data-preparation)
3. [Auto-labeling Your Data](#auto-labeling-your-data)
3. [Training](#training)
3. [Iterative Label Refinement](#iterative-label-refinement)

## Data Collection for MS3D
We recommend collecting the data in sequences that are selected to have the most density of objects in the scene. For our own data, we tend to follow nuScenes/Waymo to use 20s scenes but longer scenes would be fine as well.

If saving `.pcd` files from rosbags, `pcl::io::savePCDFileASCII` worked well for us as the pcd file is easily readable by Open3D.

A few important things to note for your data:
- Each sequence must be temporally continuous and sortable. We use `sorted()` to sort the file names of the point clouds. Using timestamps as point cloud file names work best to ensure the sequence is in order.
- Our code assumes that you have no ground-truth boxes for your dataset. If you want to load in your own labels for a validation dataset, modify `create_infos` in `custom_dataset.py` for kitti AP evaluation.

**Tip**: the highest quality pseudo-labels come from scenes with lots of **dynamic pedestrians** and **stationary vehicles** (such as carpark scenes).

## Data Preparation
### 1. Folder organisation
Please organise your files into the following structure.
```
MS3D
├── data
│   ├── custom_dataset_name
│   │   │── sequences
│   │   │   │── sequence_0
│   │   │   │   │── lidar
│   │   │   │   │── lidar_odom.npy # we generate this (see below) 
│   │   │   │   │── sequence_0.pkl # we also generate this (see below) 
│   │   │   │── sequence_1
│   │   │   │── ...
│   │   │── ImageSets
│   │   │   │── train.txt
│   │   │   │── val.txt
├── pcdet
├── tools
```
Generate `ImageSets/train.txt` by running `ls sequences > train.txt` from the `data/custom_dataset_name` folder. Instead of using all sequences for training, you can have a small portion in a `val.txt` file. If not, just have an empty `val.txt` for OpenPCDet compatibility.

### 2. Modify dataset config
In [`tools/cfgs/custom_dataset_da.yaml`](../tools/cfgs/dataset_configs/custom_dataset_da.yaml), specify the following:
- `DATA_PATH=/MS3D/data/custom_dataset_name`: Set your own dataset path
- `SHIFT_COOR`: Change this according to your lidar's height above the ground
- `MAX_SWEEPS`: If your lidar is 10Hz, we recommend `MAX_SWEEPS=4 or 5`. This should not be more than 0.45s of frame accumulation.

### 3. Localization/Odometry
MS3D relies on having the pose of the ego-vehicle for multi-frame accumulation (i.e. sweeps) and refining static vehicle boxes. We assume the ego-vehicle's pose is a `.npy` file with an array of `(N,4,4)` containing the `4x4` transform matrix for each sequence of `N` point clouds.

To get pose using lidar odometry, we provide a script that uses KISS-ICP to generate the pose for each of your sequences: [`data/custom/generate_lidar_odom.sh`](../data/custom/generate_lidar_odom.sh)
```bash
# Run the script from the custom dataset folder
cd /MS3D/data/custom_dataset_name && bash generate_lidar_odom.sh
```

**Note:** If you have your own ego-pose with odometry or localization, you can skip this part but make sure that the poses are timesynced to your point clouds and in the format described above.

### 4. Generate Infos
Create infos for your dataset as an easy way for us to access frame metadata during training. Run the following from the `/MS3D` directory.
```
python -m pcdet.datasets.custom.custom_dataset create_infos /MS3D/data/custom_dataset_name
```
## Auto-labeling Your Data
### Overview of MS3D++ file organization
Our framework is contained in 3 files in `/MS3D/tools` that load in our auto-labeling configurations from `label_generation/round1/cfgs/ps_config.yaml`.
1. `ensemble_kbf.py`  fuses the detection_sets specified in `ensemble_detections.txt` to give us our initial pseudo-label set.
2. `generate_tracks.py`  uses SimpleTrack to generate tracks for the initial pseudo-labels. We use this to generate 3 sets of tracks (veh_all, veh_static, ped). 
3. `temporal_refinement.py` refines the pseudo-labels with temporal information and object characteristics. This file gives us our final pseudo-labels. 

You will primarily be working within the folder `target_custom`.
```bash
MS3D
├── tools
│   ├── cfgs
│   │   ├── target_custom # custom is your own dataset
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

Now we'll walk through how to generate pseudo-labels for your dataset.

### 1. Generating the ensemble predictions

Download the pre-trained models from our [pre-trained model zoo](../README.md) or train your own. 

We provide some example [scripts](../tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/) to generate ensemble predictions with a few models; modify this as you wish. The scripts can be run with the following command from `/MS3D/tools` directory.
```
bash cfgs/target_custom/label_generation/round1/scripts/generate_ensemble_preds.sh
```

Once you finish generating predictions, specify their absolute file paths in [`ensemble_detections.txt`](../tools/cfgs/target_custom/label_generation/round1/cfgs/ensemble_detections.txt)

#### Tips on tailoring the ensemble
1. Use a pre-trained detector that has been trained on point cloud data most similar to yours.

If your lidar has high point cloud density, then including pre-trained models that are trained on high density point clouds like Waymo would be better for the ensemble. If your lidar's point cloud is sparse, then nuScenes' detectors would be good. 

2. Improve generalization to your lidar

For a more generalizable ensemble, use pre-trained detectors from different training datasets, rather than using just one. e.g. [Lyft, nuScenes, Waymo] or [nuScenes, Waymo]. In our paper we showed that the detector's training set (rather than architecture) was more influential on their cross-domain performance.

3. Experiment with including different pre-trained detectors in the ensemble. 

- Multi-frame: any SOTA multi-frame models (e.g. MPPNet) can be used to generate detections.
- Transformer-based: better performance than many voxel/point-based methods, maybe it has better domain generalisation too.
- Multi-modal: images can help distinguish many smaller object classes like pedestrians, or far-range objects.
- Class-specific: 3D detectors trained solely for a specific class may have better performance
- Varying voxel sizes: ensembling voxel-based detectors with different voxel sizes may be able to have better generalization to different lidar scan patterns.

### 2. MS3D++ framework

#### Running MS3D++
Once you've set up the pseudo-label configs above, you can run our pipeline with the following command.
```bash
cd /MS3D/tools && bash cfgs/target_custom/label_generation/round1/scripts/run_ms3d.sh
```
After this finishes, you should have pseudo-labels for your dataset saved at `label_generation/round1/ps_labels/final_ps_dict.pkl`

It is helpful, especially after `ensemble_kbf.py`, to visualize the saved labels in `label_generation/round1/ps_labels` at every step of the script above in order to set good thresholds in `ps_config.yaml`. Our visualization tool can be run with the following command.
```bash
# show the initial_ps_labels.pkl or final_ps_labels.pkl
python visualize_bev.py --cfg_file cfgs/dataset_configs/custom_dataset_da.yaml \
                        --ps_pkl /MS3D/tools/cfgs/target_custom/label_generation/round1/ps_labels/initial_pseudo_labels.pkl                                             
```

We have provided default parameters in our `ps_config.yaml` file but you may need to modify them for optimal auto-labeling of your dataset which we elabote on below. 

#### Configuring [**`ps_config.yaml`**](../tools/cfgs/target_custom/label_generation/round1/cfgs/ps_config.yaml)
We recommend setting `ps_configs.yaml` after visualizing the `initial_pseudo_labels.pkl` from `ensemble_kbf.py`. With visual assessment, try to set thresholds that minimise false positives as much as possible. Refer to [`MS3D_PARAMETERS.md`](../docs/MS3D_PARAMETERS.md) for an explanation of our config file parameters.
 
- If you plan to run multiple rounds (recommended), set strict thresholds to minimise false positives (e.g. `POS_TH` and Tracking's `SCORE_TH`). False positive labels are very hard to get rid of with multiple self-training rounds. 
- If you only wish to run a single round, you can loosen the thresholds a bit to increase overall model recall.

Localization/odometry can drift over time. This affects the following params: `ROLLING_KBF.ROLLING_KDE_WINDOW`, `PROPAGATE_BOXES.N_EXTRA_FRAMES` and `PROPAGATE_BOXES.DEGRADE_FACTOR`. 

- Due to ego-pose drift, we don't want to use too many past/future boxes to refine the current frame's box nor propagate the boxes too far in the past or future. Generally for a 10s window, localization remains pretty good. 
- Propagating a static box label too far in the past/future may be detrimental for the following (or similar) scenario: stationary car at a traffic light but moves at the end of the sequence.


**Tip:** If you have an accompanying camera feed, it is also helpful to check the corresponding image, or even better, project the 3D label into the image to cross-check.


## Training
To train a model with the pseudo-labels, specify the pseudo-label absolute path in the detector config file in `/MS3D/tools/cfgs/target_custom` as follows:
```yaml
SELF_TRAIN:
    INIT_PS: /MS3D/tools/cfgs/target_nuscenes/label_generation/round1/ps_labels/final_ps_dict.pkl
```

Following this, we just train the model like in OpenPCDet. 
```bash
python train.py --cfg_file ${CONFIG_FILE} --extra_tag ${EXTRA_TAG}
```
Sometimes it helps to initialize from an existing pre-trained model.
```bash
python train.py --cfg_file ${CONFIG_FILE} --extra_tag ${EXTRA_TAG} --pretrained_model ${EXTRA_TAG}

# Example
python train.py --cfg_file cfgs/target_custom/ms3d_waymo_voxel_rcnn_centerhead.yaml --extra_tag round1 --pretrained_model ../model_zoo/waymo_uda_voxel_rcnn_centerhead_4xyzt_allcls.pth
```

## Iterative Label Refinement
With each round, you'll get a trained detector for your own data. You can train multiple detectors (e.g. VoxelRCNN-anchorhead, VoxelRCNN-centerhead) and use them for auto-labelling the same dataset. This would lead to better pseudo-labels. 

We provide a few default configs for each round with the same file structure as above. You can do as many rounds as you wish.

The process is the same as above. I.e. 
1. Create script to generate ensemble predictions with newly trained models
2. Generate initial pseudo-labels
3. Modify ps_configs thresholds to ensure few false positives (IMPORTANT)
4. Resume the other auto-labelling steps

You can also use `gt_database` augmentation to build a database of high confidence pseudo-labels for each round to paste more difficult samples into the scenes for training.

# OpenPCDet Usage
For a guide on general OpenPCDet commands, refer to our [OpenPCDet guide](../docs/OPENPCDET_USAGE.md).
