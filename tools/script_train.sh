#!/bin/bash

pid=2106
tail --pid=$pid -f /dev/null

python train.py --cfg_file cfgs/nuscenes_models/uda_voxel_rcnn_centerhead.yaml \
                --extra_tag 1sweep_xyz