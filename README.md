# DeepLiDAR
This repository contains the code (in PyTorch) for "[DeepLiDAR: Deep Surface Normal Guided Depth Prediction for Outdoor Scene
from Sparse LiDAR Data and Single Color Image](https://arxiv.org/pdf/1812.00488.pdf)" paper (CVPR 2019) by [Jiaxiong Qiu](https://jiaxiongq.github.io/), [Zhaopeng Cui](https://zhpcui.github.io/), [Yinda Zhang](https://www.zhangyinda.com/), [Xingdi Zhang](https://github.com/crazyzxd), [Shuaicheng Liu](http://www.liushuaicheng.org/), Bing Zeng and [Marc Pollefeys](https://www.inf.ethz.ch/personal/marc.pollefeys/index.html).
## Introduction
In this work, we propose an end-to-end deep learning system to produce dense depth from sparse LiDAR data and a color image taken from outdoor on-road scenes leveraging surface normal as the intermediate representation.
![image](https://github.com/JiaxiongQ/Need2Adjust/blob/master/pipline.PNG)
## Requirements
- [Python2.7](https://www.python.org/downloads/)
- [PyTorch(0.4.0+)](http://pytorch.org)
- torchvision 0.2.0 (higher version may cause issues)
- [KITTI Depth Completion](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)
### Pretrained Model
※NOTE: The pretrained model were saved in .tar; however, you don't need to untar it. Use torch.load() to load it.
[Google Drive](https://drive.google.com/file/d/1eaOCtl_CGzqqqJDbVawsdniND255ZaP8/view?usp=sharing)
## Train
1. Get the surface normal of Lidar dataset by the code in the folder named [surface_normal](https://github.com/crazyzxd).
2. Use the following command to train the part I of our net on [1].
```
python trainN.py --datapath (your surface normal folder)\
               --epochs 20\
               --loadmodel (optional)\
               --savemodel (path for saving model)
```
3. Use the following command to train the part II of our net on KITTI(after load the parameters of [2]).
```
python trainD.py --datapath (your KITTI dataset folder)\
               --epochs 20\
               --batch_size 6\
               --gpu_nums 3\
               --loadmodel (optional)\
               --savemodel (path for saving model)
```
3. Use the following command to train the part III of our net on KITTI(after load the parameters of [2,3]).
```
python train.py --datapath (your KITTI dataset folder)\
               --epochs 20\
               --batch_size 6\
               --gpu_nums 3\
               --loadmodel (optional)\
               --savemodel (path for saving model)
```
## Evaluation
Use the following command to evaluate the trained on your own data(after fill folder names).
```
python test.py --loadmodel (your trained model)
```
## Citation 
If you use our code or method in your work, please cite the following:
```
@inproceedings{qiu2018deeplidar,
  title={DeepLiDAR: Deep Surface Normal Guided Depth Prediction for Outdoor Scene from Sparse LiDAR Data and Single Color Image},
  author={Qiu, Jiaxiong and Cui, Zhaopeng and Zhang, Yinda and Zhang, Xingdi and Liu, Shuaicheng and Zeng, Bing and Pollefeys, Marc},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
Please direct any questions to [Jiaxiong Qiu](https://jiaxiongq.github.io/) at qiujiaxiong727@gmail.com

