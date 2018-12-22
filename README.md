# Optical-RemoteSensing-Image-Resolution
This repository contains source code necessary to reproduce some of the main results in the paper:
[Deep Memory Connected Neural Network for Optical Remote Sensing Image Restoration](https://www.mdpi.com/2072-4292/10/12/1893)

**If you use this software in an academic article, please consider citing:**

@article{xu2018deep,
  title={Deep Memory Connected Neural Network for Optical Remote Sensing Image Restoration},
  author={Xu, Wenjia and Xu, Guangluan and Wang, Yang and Sun, Xian and Lin, Daoyu and Wu, Yirong},
  journal={Remote Sensing},
  volume={10},
  number={12},
  pages={1893},
  year={2018},
  publisher={Multidisciplinary Digital Publishing Institute}
}

## Method overview
We propose a novel method named deep memory connected network (DMCN) based on the convolutional neural network to achieve image restoration. We build local and global memory connections to combine image detail with global information. To further reduce parameters and ease time consumption, we propose Downsampling Units, shrinking the spatial size of feature maps. The network can achieve Gaussian image denoising and single image super-resolution (SR).
![](https://github.com/wenjiaXu/Optical-RemoteSensing-Image-Resolution/blob/master/intro.png)

## Requirements
* Python 3
* PyTorch

### Python Packages:
* matplotlib
* cv2
* h5py
* numpy
* skimage
* tensorboardX
* torchvision


