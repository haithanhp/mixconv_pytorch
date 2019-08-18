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

from thop import profile

arch = 'mixnet-s'
blocks_args, global_params = mixnet_builder.get_model_params(arch)
model = MixNet(input_size=224, num_classes=1000, blocks_args=blocks_args, global_params=global_params)
input = torch.randn(1, 3, 224, 224)
# flops, params = profile(model, inputs=(input, ),)
out = model(input)
print('params= %fMB'%(model._num_params/1e6))
print('flops: %fM'%(model._num_flops/1e6))
