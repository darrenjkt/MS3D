# Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04)
* Python 3.6+
* PyTorch 1.1 or higher (tested on PyTorch 1.1, 1,3, 1,5~1.10)
* CUDA 9.0 or higher (PyTorch 1.3+ needs CUDA 9.2+)
* [`spconv v1.0 (commit 8da6f96)`](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634) or [`spconv v1.2`](https://github.com/traveller59/spconv) or [`spconv v2.x`](https://github.com/traveller59/spconv)


### Install MS3D
MS3D runs on OpenPCDet `pcdet v0.6.0`. 

a. Clone this repository.
```shell
git clone https://github.com/darrenjkt/MS3D.git
```
b. Using Docker

We highly recommend running this repository on docker for out-of-box usage. You can build our [Dockerfile](../docker/Dockerfile) or pull our provided docker image.
```
docker pull darrenjkt/openpcdet:v0.6.0
```
For easy running of the image we provide a script. Change the file paths, number of GPUs and then run it. Use `docker ps` to find the container name. 

```
bash docker/run.sh
docker exec -it ${CONTAINER_NAME} /bin/bash
```

c. Within the container, install `pcdet` and the tracker with the following commands
```shell
python setup.py develop
cd tracker && pip install -e . --user
```
Note that if you want to use dynamic voxelization (e.g. Voxel-RCNN), you need torch-scatter. This should already be pre-installed in our docker image but you can install with the following commands if need be.
```shell
pip install torch==1.8.1 torchvision==0.9.1
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.8.1+cu111.html
python setup.py develop # rebuild repository
```
