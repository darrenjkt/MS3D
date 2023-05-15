#!/bin/bash

# Modify these paths and GPU ids
DATA_PATH="/media/darren/ITS_2TB/data"
CODE_PATH="/media/darren/ITS_2TB/code/MS3D"
GPU_ID="0"

ENVS="  --env=NVIDIA_VISIBLE_DEVICES=$GPU_ID
        --env=CUDA_VISIBLE_DEVICES=$GPU_ID
        --env=NVIDIA_DRIVER_CAPABILITIES=all"

# Modify these paths to mount the data
VOLUMES="       --volume=$DATA_PATH:/MS3D/data"

# Setup environmetns for pop-up visualization of point cloud 
VISUAL="        --env=DISPLAY
                --env=QT_X11_NO_MITSHM=1
                --volume=/tmp/.X11-unix:/tmp/.X11-unix"
xhost +local:docker

echo "Running the docker image [GPUS: ${GPU_ID}]"
docker_image="darrenjkt/openpcdet:v0.6.0"

# Start docker image
docker  run -d -it --rm \
$VOLUMES \
$ENVS \
$VISUAL \
--mount type=bind,source=$CODE_PATH,target=/MS3D \
--runtime=nvidia \
--gpus $GPU_ID \
--privileged \
--net=host \
--shm-size=30G \
--workdir=/MS3D \
$docker_image   
