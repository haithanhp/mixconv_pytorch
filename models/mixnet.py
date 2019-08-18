import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
# from utils import swish
from .utils import *


# from utils import Conv2dSamePadding

NON_LINEARITY = {
    'ReLU': nn.ReLU(),
    'Swish': Swish(),
}

class GroupedConv2D(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, strides=1):
        super(GroupedConv2D, self).__init__()
        self._groups = len(kernel_size)
        self._convs = nn.ModuleList()
        self._num_params = 0
        self._num_flops = 0
        splits = splitFilters(out_filters, self._groups)
        inp_splits = splitFilters(in_filters, self._groups)
        for i in range(self._groups):
            # in_filters = in_filters if i==0 else splits[i]
            in_filters = inp_splits[i]
            self._convs.append(
                Conv2dSamePadding(in_channels=in_filters, 
                                out_channels=splits[i], 
                                groups=1, 
                                kernel_size=kernel_size[i],
                                stride=strides, 
                                bias=False)
            )
            self._num_params += self._convs[i]._num_params
    
    def forward(self, x):
        if len(self._convs)==1:
            x = self._convs[0](x)
            self._num_flops += self._convs[0]._num_flops
            return x
        filters = x.size(1)
        splits = splitFilters(filters, len(self._convs))
        x_splits = torch.split(x, splits, dim=1)
        x_outs = [c(x) for x, c in zip(x_splits, self._convs)]
        for c in self._convs:
            self._num_flops += c._num_flops
            
        x = torch.cat(x_outs, dim=1)
        return x
        

class MDConv(nn.Module):
    def __init__(self, filters, kernel_size, strides=1, dilated=False):
        super(MDConv, self).__init__()
        self._dilated = dilated
        self._convs = nn.ModuleList()
        self._groups = len(kernel_size)
        self._num_params = 0
        self._num_flops = 0
        splits = splitFilters(filters, self._groups)
        for i in range(self._groups):
            self._convs.append(
                Conv2dSamePadding(in_channels=splits[i],
                                out_channels=splits[i],
                                groups=splits[i],
                                kernel_size=kernel_size[i],
                                stride=strides,
                                bias=False)
            )
            self._num_params += self._convs[i]._num_params

    def forward(self, x):
        if self._groups == 1:
            x = self._convs[0](x)
            self._num_flops += self._convs[0]._num_flops
            return x

        filters = x.size(1)
        splits = splitFilters(filters, len(self._convs))
        x_splits = torch.split(x, splits, dim=1)
        x_outs = [c(x) for x, c in zip(x_splits, self._convs)]
        for c in self._convs:
            self._num_flops += c._num_flops

        x = torch.cat(x_outs, dim=1)
        return x


class MixnetBlock(nn.Module):
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_momentum = global_params.bn_momentum
        self._bn_eps = global_params.bn_eps
        self._data_format = global_params.data_format
        # if self._data_format == 'channel_first':
        #     self._channel_axis = 1
        #     self._spatial_dims = (2,3)
        # else:
        #     self._channel_axis = -1
        #     self._spatial_dims = (1,2)
        self._spatial_dims = (2,3)
        self._has_se = (self._block_args.se_ratio is not None) and (
            self._block_args.se_ratio > 0) and (self._block_args.se_ratio <= 1)
        non_linear = 'Swish' if self._block_args.swish else 'ReLU'
        self._act_fn = NON_LINEARITY[non_linear] #swish if self._block_args.swish else nn.ReLU()
        self._num_params = 0
        self._num_flops = 0

        # Build modules
        inp = self._block_args.input_filters
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        kExpand_size = self._block_args.expand_ksize

        if self._block_args.expand_ratio != 1:
            # Expansion component
            self._expand_conv = GroupedConv2D(inp,
                                            filters,
                                            kExpand_size)
            self._num_params += self._expand_conv._num_params

            self._bn0 = nn.BatchNorm2d(num_features=filters,
                                    momentum=self._bn_momentum, 
                                    eps=self._bn_eps)
            self._num_params += 4*filters
        
        # Depth-wise components
        kernel_size = self._block_args.dw_ksize
        self._depthwise_conv = MDConv(filters, kernel_size, 
                                    self._block_args.strides[0],
                                    dilated=self._block_args.dilated)
        self._num_params += self._depthwise_conv._num_params

        self._bn1 = nn.BatchNorm2d(num_features=filters,
                                    momentum=self._bn_momentum, 
                                    eps=self._bn_eps)
        self._num_params += 4*filters
        
        # Squeeze and Excite components
        if self._has_se:
            num_reduced_filters = max(
                    1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = GroupedConv2D(filters,
                                            num_reduced_filters,
                                            [1])
            self._num_params += self._se_reduce._num_params
            self._se_expand = GroupedConv2D(num_reduced_filters,
                                            filters,
                                            [1])
            self._num_params += self._se_expand._num_params
            self.sigmoid = nn.Sigmoid()

        # Output
        inp = filters
        filters = self._block_args.output_filters
        self._project_conv = GroupedConv2D(inp, 
                                            filters,
                                            self._block_args.project_ksize)
        self._num_params += self._project_conv._num_params
        self._bn2 = nn.BatchNorm2d(num_features=filters,
                                    momentum=self._bn_momentum, 
                                    eps=self._bn_eps)
        self._num_params += 4*filters
    
    def make_cuda_and_parallel(self):
        if self._block_args.expand_ratio != 1:
            self._expand_conv.make_cuda_and_parallel()
        
        self._depthwise_conv.make_cuda_and_parallel()

        if self._has_se:
            self._se_reduce.make_cuda_and_parallel()
            self._se_expand.make_cuda_and_parallel()
        
        self._project_conv.make_cuda_and_parallel()

    def forward(self, x):
        inputs = x.clone()
        
        if self._block_args.expand_ratio != 1:
            # print('do expand conv')
            x1 = self._expand_conv(x)
            self._num_flops += self._expand_conv._num_flops

            x2 = self._bn0(x1)
            t = x1[0]
            nelements = t.numel() 
            self._num_flops += 4*nelements

            x = self._act_fn(x2)
            # t = x2[0]
            # nelements = t.numel()
            # self._num_flops +=  nelements
            
        # print('do depthwise conv')
        x1 = self._depthwise_conv(x)
        self._num_flops += self._depthwise_conv._num_flops
        x2 = self._bn1(x1)
        t = x1[0]
        nelements = t.numel()
        self._num_flops += 4*nelements

        x = self._act_fn(x2)
        # t = x2[0]
        # nelements = t.numel()
        # self._num_flops +=  nelements
         
        # print('finish depthwise conv :', x.size())

        if self._has_se:
            # print('do squeeze and excite')
            se = torch.mean(x, self._spatial_dims, keepdim=True)
            s1 =  self._se_reduce(se)
            self._num_flops += self._se_reduce._num_flops
            s2 = self._act_fn(s1)
            # t = s1[0]
            # nelements=t.numel()
            # self._num_flops += nelements
            
            se = self._se_expand(s2)
            self._num_flops += self._se_expand._num_flops

            # print('finish squeeze and excite :', x.size())
            x = self.sigmoid(se) * x
            # t = se[0]
            # nelements = t.numel()
            # self._num_flops += nelements

        # print('do project conv')
        x1 = self._project_conv(x)
        self._num_flops += self._project_conv._num_flops
        x = self._bn2(x1)
        t = x1[0]
        nelements = t.numel()
        self._num_flops += nelements

        # print('finish project conv :', x.size())
        if self._block_args.id_skip:
            if all(s == 1 for s in self._block_args.strides) and self._block_args.input_filters == self._block_args.output_filters:
                x = inputs + x
                # t = x[0]
                # nelements = t.numel()
                # self._num_flops += nelements
        
        return x

class MixNet(nn.Module):
    def __init__(self, input_size=224, num_classes=1000, blocks_args=None, global_params=None):
        super(MixNet, self).__init__()
        if not isinstance(blocks_args, list):
            raise ValueError('blocks_args should be a list.')
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._relu = nn.ReLU()
        self._num_params = 0 
        self._num_flops = 0
        blocks = []
        for block_args in self._blocks_args:
            assert block_args.num_repeat > 0
            block_args = block_args._replace(
                input_filters=roundFilters(block_args.input_filters, self._global_params),
                output_filters=roundFilters(block_args.output_filters, self._global_params)
            )
            blocks.append(MixnetBlock(block_args, self._global_params))
            self._num_params += blocks[-1]._num_params

            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1,1])
            for _ in range(block_args.num_repeat - 1):
                blocks.append(MixnetBlock(block_args, self._global_params))
                self._num_params += blocks[-1]._num_params

        self._mix_blocks = nn.Sequential(*blocks)
        self._bn_momentum = global_params.bn_momentum
        self._bn_eps = global_params.bn_eps

        # Stem component
        stem_size = self._global_params.stem_size
        filters = roundFilters(stem_size, self._global_params)
        self._conv_stem = GroupedConv2D(3, 
                                        filters,
                                        [3],
                                        2)

        self._num_params += self._conv_stem._num_params
        self._bn0 = nn.BatchNorm2d(num_features=filters,
                                    momentum=self._bn_momentum, 
                                    eps=self._bn_eps)
        self._num_params += 4*filters

        # Head component
        feature_size = self._global_params.feature_size
        output_filters = self._blocks_args[-1].output_filters
        self._conv_head = GroupedConv2D(output_filters,
                                        feature_size,
                                        [1],
                                        1)
        self._num_params += self._conv_head._num_params
        self._bn1 = nn.BatchNorm2d(num_features=feature_size,
                                    momentum=self._bn_momentum, 
                                    eps=self._bn_eps)
        self._num_params += 4*feature_size

        self.avgpool = nn.AvgPool2d(input_size//32, stride=1)
        self.classifier = nn.Linear(feature_size, num_classes)

        if self._global_params.dropout_rate > 0:
            self.dropout = nn.Dropout(self._global_params.dropout_rate)
        else:
            self.dropout = None
        self._num_params += feature_size*num_classes

        self._initialize_weights()
    
    def forward(self,x):
        # print('do stem conv')
        x = self._conv_stem(x)
        self._num_flops += self._conv_stem._num_flops

        # print('finish stem conv x: ', x.size())
        x = self._bn0(x)
        t = x[0]
        nelements = t.numel()
        self._num_flops += nelements

        # print('do mix blocks')
        x = self._mix_blocks(x)
        # print('do conv head')
        for block in self._mix_blocks:
            self._num_flops += block._num_flops

        x = self._conv_head(x)
        self._num_flops += self._conv_head._num_flops

        x = self._bn1(x)
        t = x[0]
        nelements = t.numel()
        self._num_flops += nelements

        # print('do avg pooling')
        t = x.clone()
        x = self.avgpool(x)
        total_add = torch.prod(torch.Tensor([self.avgpool.kernel_size]))
        total_div = 1
        kernel_ops = total_add + total_div
        num_elements = t.numel()
        self._num_flops += kernel_ops * num_elements

        # print('do dropout')
        if self.dropout:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        t = x.clone()
        x = self.classifier(x)
        total_mul = self.classifier.in_features
        total_add = self.classifier.in_features - 1
        num_elements = x.numel()
        self._num_flops += (total_mul + total_add) * num_elements
        return x

    def _initialize_weights(self):
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2.0 / n))
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                init_range = 1.0 / np.sqrt(n)
                # m.weight.data.normal_(0, 0.01)
                m.weight.data.uniform_(init_range, init_range)
                m.bias.data.zero_()