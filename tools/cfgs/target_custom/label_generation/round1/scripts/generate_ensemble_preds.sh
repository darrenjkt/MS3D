#!/bin/bash

# We set sweeps based on the assumption that your lidar data is at 10Hz
bash /MS3D/tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/lyft_pt_dets_vx_a.sh
bash /MS3D/tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/lyft_pt_dets_vx_c.sh
bash /MS3D/tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/nusc_pt_dets_vx_a.sh
bash /MS3D/tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/nusc_pt_dets_vx_c.sh
bash /MS3D/tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/waymo_pt_dets_vx_a.sh
bash /MS3D/tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/waymo_pt_dets_vx_c.sh