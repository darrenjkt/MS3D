# Custom Dataset Tutorial

In this section, we walk through how you can use MS3D to auto-label your point cloud sequences.

#### Contents
1. [Data Collection for MS3D](#data-collection-for-ms3d)
2. [Data Preparation](#data-preparation)
3. [Auto-labeling Your Data](#auto-labeling-your-data)

## Data Collection for MS3D
MS3D works best with scenes containing lots of **dynamic pedestrians** and **static cars** (such as carpark scenes). 

If such scenes are not accessible, it's still okay. We can tweak the config file to be more conservative. You can also use `gt_database` augmentation to build a database of high confidence pseudo-labels for each round to paste into the scenes for training.

We recommend collecting the data in sequences that are selected to have the most density of objects in the scene. For our own data, we tend to follow nuScenes/Waymo to use 20s scenes but longer scenes would be fine as well.

A few important things to note for your data:
- Each sequence must be temporally continuous and sortable. We use `sorted()` to sort the file names of the point clouds. Using timestamps as point cloud file names work best to ensure the sequence is in order.
- Our code at the moment assumes that you have no ground-truth boxes for your dataset. If you want to load in your own labels for a validation dataset, modify `create_infos` in `custom_dataset.py` for kitti AP evaluation.

## Data Preparation
Please organise your files into the following structure:
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
Please generate `ImageSets/train.txt` by running `ls sequences > train.txt` from the `data/custom_dataset_name` folder. Instead of using all sequences for training, you can have a small portion in a `val.txt` file. If not, just have an empty `val.txt` for OpenPCDet compatibility.

In [`tools/cfgs/custom_dataset_da.yaml`](../tools/cfgs/dataset_configs/custom_dataset_da.yaml), specify the following:
- `DATA_PATH=/MS3D/data/custom_dataset_name`: Set your own dataset path
- `SHIFT_COOR`: Change this according to your lidar's height above the ground
- `MAX_SWEEPS`: If your lidar is 10Hz, we recommend `MAX_SWEEPS=4 or 5`. This should not be more than 0.45s of frame accumulation.

### Lidar Odometry
MS3D relies on having the ego-pose of the vehicle in order for multi-frame accumulation (i.e. sweeps) and for refining static vehicle boxes. We assume the odometry is a `.npy` file with an array of `(N,4,4)` containing the `4x4` transform matrix for each sequence of `N` point clouds.

If you have your own odometry, you can skip this part but make sure that the odometry poses are timesynced to your point clouds and in the format described above.

If you don't have odometry, we provide a script that uses KISS-ICP to generate lidar odometry for each of your sequences: [`data/custom/generate_lidar_odom.sh`](../data/custom/generate_lidar_odom.sh)
```bash
# Run the script from the custom dataset folder
cd /MS3D/data/custom_dataset_name && bash generate_lidar_odom.sh
```
### Generate Infos
Create infos for your dataset as an easy way for us to access frame metadata during training.
```
python -m pcdet.datasets.custom.custom_dataset create_infos /MS3D/data/custom_dataset_name
```
## Auto-labeling Your Data

### Tailoring the ensemble
Depending on what lidar type you have, different pre-trained detectors may perform differently. If your lidar has high point cloud density, then pre-trained models that are trained on high density point clouds like Waymo would be good for the ensemble. If your lidar's point cloud is sparse, then nuScenes' detectors would be good. 

For a more generalizable ensemble, we suggest to use pre-trained detectors from different source domains, rather than using just one. e.g. [Lyft, nuScenes, Waymo] or [nuScenes, Waymo]. In our paper we showed that the detector's training set (rather than architecture) was more influential on their cross-domain performance.

We provide [scripts](../tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/) that use VoxelRCNN (Anchor) and VoxelRCNN (Center), trained on [Lyft, nuScenes and Waymo] to generate predictions for the ensemble. You can use other 3D detectors if you wish (we just find that VoxelRCNN is fast to train/test with high performance).

Once you finish generating predictions, specify their absolute file paths in [`ensemble_detections.txt`](../tools/cfgs/target_custom/label_generation/round1/cfgs/ensemble_detections.txt)

### MS3D++ framework
#### Setting [**`ps_config.yaml`**](../tools/cfgs/target_custom/label_generation/round1/cfgs/ps_config.yaml)
We provided default settings in `ps_config.yaml` but you may need to further tweak the configs for your dataset. Refer to [`MS3D_PARAMETERS.md`](../docs/MS3D_PARAMETERS.md) for an explanation of our config file parameters. 
- If you plan to run multiple rounds, set strict thresholds to minimise false positives (e.g. `POS_TH` and Tracking's `SCORE_TH`). False positive labels are very hard to get rid of with multiple self-training rounds. 
- If you only wish to run a single round, you can loosen the thresholds a bit to increase overall model recall.

It is helpful to visualize the labels at every step of the process in order to set good thresholds. You can refer to our [tutorial](../tools/demo/ms3d_demo_tutorial.ipynb) to see how we assess the pseudo-labels. For example:
```bash
# show the initial_ps_labels.pkl or final_ps_labels.pkl
python visualize_bev.py --cfg_file cfgs/dataset_configs/custom_dataset_da.yaml \
                        --ps_pkl /MS3D/tools/cfgs/target_custom/label_generation/round1/ps_labels/final_ps_dict.pkl                                             
```

If using lidar odometry like KISS-ICP, the localization can start to drift at longer distances. This affects the following params: `ROLLING_KBF.ROLLING_KDE_WINDOW`, `PROPAGATE_BOXES.N_EXTRA_FRAMES` and `PROPAGATE_BOXES.DEGRADE_FACTOR`. With poor long sequence localization, we don't want to use too many past/future boxes to refine the current frame's box nor propagate the boxes too far in the past or future. Generally for a short 15-20 frames window, the localization remains pretty good. 

**Tip:** If you have an accompanying camera feed, it is also helpful to check the corresponding image, or even better, project the 3D label into the image to cross-check.

#### Running MS3D++
Once you've set up the pseudo-label configs above, you can run our pipeline as such.
```
bash run_ms3d.sh
```
After this finishes, you should have labels for each point cloud frame saved at `label_generation/round1/ps_labels/final_ps_dict.pkl`

### Training/Multi-stage Self-training
Now that you have the labels, you can train your own 3D detector. This section is the same as section 3 and 4 in [`GETTING_STARTED.md`](../docs/GETTING_STARTED.md). 