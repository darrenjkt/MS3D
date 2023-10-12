# Dataset Preparation
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs](../tools/cfgs) for different datasets. 

**Note:** Even if you already have generated infos from OpenPCDet, you need to re-generate the infos for nuScenes and Lyft (gt database not required) because we updated infos to include sequence metadata.

### NuScenes Dataset
* Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and 
organize the downloaded files as follows: 
```
MS3D
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
├── pcdet
├── tools
```

* Install the `nuscenes-devkit` with version `1.0.5` by running the following command: 
```shell script
pip install nuscenes-devkit==1.0.5
```

* Generate the data infos by running the following command. `gt_database` augmentation should be able to improve performance but for MS3D experiments we did not use it so feel free to comment that section out in `nuscenes_dataset.py`.

```python 
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval
```

### Waymo Open Dataset
* Please download the official [Waymo Open Dataset](https://waymo.com/open/download/), 
including the training data `training_0000.tar~training_0031.tar` and the validation 
data `validation_0000.tar~validation_0007.tar`.
* Unzip all the above `xxxx.tar` files to the directory of `data/waymo/raw_data` as follows (You should get 798 *train* tfrecord and 202 *val* tfrecord ):  
```
MS3D
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_v0_5_0
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/  (old, for single-frame)
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl  (old, for single-frame)
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy (optional, old, for single-frame)
│   │   │── waymo_processed_data_v0_5_0_infos_train.pkl (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_val.pkl (optional)
|   |   |── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0 (new, for single/multi-frame)
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1_multiframe_-4_to_0.pkl (new, for single/multi-frame)
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0_global.np  (new, for single/multi-frame)
 
├── pcdet
├── tools
```
* Install the official `waymo-open-dataset` by running the following command: 
```shell script
pip3 install --upgrade pip
# tf 2.0.0
pip3 install waymo-open-dataset-tf-2-5-0 --user
```

* Extract point cloud data from tfrecord and generate data infos by running the following command. MS3D does not use gt database for our experiments so you can skip the gt database generation step: 
```python 
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset_multiframe.yaml
# Ignore 'CUDA_ERROR_NO_DEVICE' error as this process does not require GPU.
```

Note that you do not need to install `waymo-open-dataset` if you have already processed the data before and do not need to evaluate with official Waymo Metrics. 


### Lyft Dataset
**Note**: Last I checked (15 September 2023) it seems like the Lyft Level5 perception dataset has been discontinued so the link below is not active. I'll still keep this section here for reference.

* Please download the official [Lyft Level5 perception dataset](https://level-5.global/data/perception) and 
organize the downloaded files as follows: 
```
MS3D
├── data
│   ├── lyft
│   │   │── ImageSets
│   │   │── trainval
│   │   │   │── data & maps(train_maps) & images(train_images) & lidar(train_lidar) & train_lidar
│   │   │── test
│   │   │   │── data & maps(test_maps) & test_images & test_lidar
├── pcdet
├── tools
```

* Install the `lyft-dataset-sdk` with version `0.0.8` by running the following command: 
```shell script
pip install -U lyft_dataset_sdk==0.0.8
```

* Generate the training & validation data infos by running the following command (it may take several hours): 
```python 
python -m pcdet.datasets.lyft.lyft_dataset --func create_lyft_infos \
    --cfg_file tools/cfgs/dataset_configs/lyft_dataset.yaml
```