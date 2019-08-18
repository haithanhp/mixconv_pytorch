import math
import collections
import torch
from torch import nn
from torch.nn import functional as F

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'bn_momentum', 'bn_eps', 'dropout_rate', 'data_format',
    'num_classes', 'depth_multiplier', 'depth_divisor', 'min_depth', 
    'stem_size', 'feature_size',
])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'dw_ksize', 'expand_ksize', 'project_ksize', 'num_repeat', 'input_filters',
    'output_filters', 'expand_ratio', 'id_skip', 'strides', 'se_ratio',
    'swish', 'dilated',
])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

def swish(x):
    return x * torch.sigmoid(x)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        return x * self.sig(x)

def round_filters(filters, global_params):
  """Round number of filters based on depth multiplier."""
  multiplier = global_params.depth_multiplier
  divisor = global_params.depth_divisor
  min_depth = global_params.min_depth
  if not multiplier:
    return filters

  filters *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  return new_filters

class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, is_expand=False, is_reduce=False, is_project=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2
        self.is_expand = is_expand
        self.is_reduce = is_reduce
        self.is_project = is_project
        self._num_flops = 0
        # self.groups = groups
        self._num_params = kernel_size * kernel_size * in_channels * out_channels / groups
        # print('%dx%dx%dx%d,g%d,num:%d'%(kernel_size, kernel_size, in_channels, out_channels,groups,self._num_params))
        self._init_weights()

    def _init_weights(self):
        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        self.weight.data.normal_(0, math.sqrt(2.0 / n))
        if self.bias is not None:
            self.bias.data.zero_()

    def count_flops(self, x):
      # kernel_ops = self.weight.size()[2:].numel()  # Kw x Kh
      # bias_ops = 1 if self.bias is not None else 0

      # # N x Cout x H x W x  (Cin x Kw x Kh + bias)
      # total_ops = x.nelement() * (self.in_channels // self.groups * kernel_ops + bias_ops)

      # self.total_ops += torch.Tensor([int(total_ops)])
      # https://github.com/ShichenLiu/CondenseNet/blob/master/utils.py
      mutil_add = 1
      out_h = int((x.size()[2] + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)
      out_w = int((x.size()[3] + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1)
      self._num_flops = self.in_channels * self.out_channels * self.kernel_size[0] * self.kernel_size[1] * out_h * out_w / self.groups * mutil_add


    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        
        self.count_flops(x)
        
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def roundFilters(filters, global_params):
  """Round number of filters based on depth multiplier."""
  multiplier = global_params.depth_multiplier
  divisor = global_params.depth_divisor
  min_depth = global_params.min_depth
  if not multiplier:
    return filters

  filters *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  return new_filters

def splitFilters(channels, num_groups):
    split_channels = [channels//num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels