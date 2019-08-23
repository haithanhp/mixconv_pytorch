# Mixconv_pytorch

This repo is the pytorch implementation of the paper from Google: [MixConv: Mixed Depthwise Convolutional Kernels](https://arxiv.org/pdf/1907.09595.pdf)

This code mimics the implementation from the offical repo in Tensorflow (https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet)

### Dependencies  
Python 3.5+  
[PyTorch v1.0.0+](http://pytorch.org/)

### How to use
`python train_cifar.py --lr 0.016 --batch-size 256 -a mixnet-s --dtype cifar100 --optim adam --scheduler exp --epochs 650`

## Reproduce and Results
# CIFAR 100
| **Network** |  **Top 1**   | **#Params**       | **#Flops** |
| ----------- | ------------ | ------------------|------------|
| Mixnet-S    | in progress  | 2.7M (*this code*)| 3.2M (*this code*)       |
| Mixnet-M    | in progress  | 3.6M (*this code*)| 4.4M (*this code*)      |
| Mixnet-L    | in progress  | 5.8M (*this code*)| Bug issue (solved soon)|

# ImageNet
| **Network** |  **Top 1**   | **#Params**       | **#Flops** |
| ----------- | ------------ | ------------------|------------|
| Mixnet-S    | in progress  | 4.1M (*this code*)| 259M (*this code*)       |
| Mixnet-M    | in progress  | 5.0M (*this code*)| 360M (*this code*)       |
| Mixnet-L    | in progress  | 7.3M (*this code*)| 580M (*this code*)       |

### Discussion
Currently, the accuracy is very low compare with the numbers reported in the paper. So, welcome scientific, rigorous ,and helpful feedbacks to train MixConv proper in Pytorch. 
