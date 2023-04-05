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
b. Docker

We highly recommend running this repository on docker for an out-of-box functionality. We provide a docker image and [Dockerfile](../docker/Dockerfile). You can pull the docker image with:
```
docker pull darrenjkt/openpcdet:v0.6.0
```
For easy running of the image we provide a script. Change the file paths, number of GPUs and then run it 

```
bash docker/run.sh
```

c. Install this `pcdet` library and its dependent libraries by running the following command:
```shell
python setup.py develop
```
