import os
import sys
import time
import glob
import numpy as np
import torch
# import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import copy

from models import MixNet
from models import mixnet_builder
from models import utils

# from thop import profile
dtype = 'imagenet' # 'imagenet'
arch = 'mixnet-m'
input_size = 32
num_classes = 100
batch_size = 2

if dtype == 'imagenet':
    input_size = 224
    num_classes = 1000
    batch_size = 1
blocks_args, global_params = mixnet_builder.get_model_params(arch)
model = MixNet(input_size=input_size, num_classes=num_classes, blocks_args=blocks_args, global_params=global_params)
input = torch.randn(batch_size, 3, input_size, input_size)
# flops, params = profile(model, inputs=(input, ),)
out = model(input)
print('params= %fMB'%(model._num_params/1e6))
print('flops: %fM'%(model._num_flops/batch_size/1e6))
