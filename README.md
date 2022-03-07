## Introduction
## Prepare environment
1. Create a conda virtual environment and activate it
```
  conda create -n openmmlab python=3.8 -y
  conda activate openmmlab
```
2. Install PyTorch and torchvision
```
  conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```
3. Install mmdetection
```
  pip install mmcv-full==1.3.2 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html 
  pip install -r requirements/build.txt 
  pip install -v -e.
```
## Prepare the dataset
Download CLCXray from [here](https://pan.baidu.com/s/1fYwxiyGG8cJndebMO4Bn9A) (password: clcx) and move it to the "data" folder. The folder structure is as follow:
```
MMDETECTION
|-data
|  |-coco
```
## Acknowledgement
We thanks MMDetection for their code.


## Citation

