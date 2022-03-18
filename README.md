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
## Training
### Command
```
CUDA_VISIBLE_DEVICES={GPU id} python tools/train.py {config}  --work-dir {output folder}
```
### Sample
```
CUDA_VISIBLE_DEVICES=0 python tools/train.py ./configs/atss_la/LAcls_r50_fpn_1x_coco.py  --work-dir atss_LAcls_new
``` 
(we set samples_per_gpu=8 and workers_per_gpu=8 in ./configs/_base_/datasets/coco_detection.py when using a single GPU)
or 
```
CUDA_VISIBLE_DEVICES=0，1 tools/dist_train.sh ./configs/atss_la/LAcls_r50_fpn_1x_coco.py  2 --work-dir atss_LAcls_new
```
## Test trained models
Download the trained model from [here](https://pan.baidu.com/s/1HcB_RcIQRtExzPyoTm5xQg?pwd=CLCX)
```
CUDA_VISIBLE_DEVICES=0 python tools/test.py ./configs/atss_la/LAcls_r50_fpn_1x_coco.py ./trained/atss_cls/epoch_12_311.pth --eval bbox
```
## Acknowledgement
We thanks MMDetection for their code.

## Citation
```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
