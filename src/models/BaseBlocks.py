#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class DepthWiseConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, padding=0):
        super(DepthWiseConv2D, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
        self.act = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(in_ch)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.bn(self.act(out))
        out = self.point_conv(out)
        return out

class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input 
    planes with distinct spatial and time axes, by performing a 2D convolution over the 
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time 
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)


    def forward(self, x):

        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, 
                 kernel_size=(3, 3), stride=1, pooling=True, drop_out=True, use_softpool=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1, resA2, resA3), dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA

        if self.pooling:
            resB = self.dropout(resA) if self.drop_out else resA
            resB = self.pool(resB)
            return resB, resA
        else:
            resB = self.dropout(resA) if self.drop_out else resA
            return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3, 3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2, 2), dilation=2, padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA, skip), dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1, upE2, upE3), dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


class ResContextBlockDP(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlockDP, self).__init__()
        self.conv1 = DepthWiseConv2D(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = DepthWiseConv2D(out_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = DepthWiseConv2D(out_filters, out_filters, kernel_size=(3,3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)


    def forward(self, x):

        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output


class ResBlockDP(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlockDP, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = DepthWiseConv2D(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = DepthWiseConv2D(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = DepthWiseConv2D(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = DepthWiseConv2D(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = DepthWiseConv2D(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1,resA2,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA


        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlockDP(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlockDP, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = DepthWiseConv2D(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = DepthWiseConv2D(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = DepthWiseConv2D(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)


        self.conv4 = DepthWiseConv2D(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA,skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


class MetaKernel(nn.Module):
    def __init__(self, num_batch, feat_height, feat_width, coord_channels, fp16=False, num_frame=1):
        super(MetaKernel, self).__init__()
        self.num_batch = num_batch
        self.H = feat_height
        self.W = feat_width
        self.fp16 = fp16
        self.num_frame = num_frame
        self.channel_list = [16, 32]
        self.use_norm = False
        self.coord_channels = coord_channels

        # TODO: Best to put this part of the definition in __init__ function ?
        self.conv1 = nn.Conv2d(self.coord_channels,
                               self.channel_list[0],
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               padding=0,
                               dilation=1,
                               bias=True)
        if self.use_norm:
            self.bn1 = nn.BatchNorm2d(self.channel_list[0])
        # self.act1 = nn.ReLU()
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(self.channel_list[0],
                               self.channel_list[1],
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               padding=0,
                               dilation=1,
                               bias=True)
        self.conv1x1 = nn.Conv2d(in_channels=288, out_channels=32, kernel_size=1, stride=1, padding=0)

    def update_num_batch(self, num_batch):
        self.num_batch = num_batch

    @staticmethod
    def sampler_im2col(data, name=None, kernel=1, stride=1, pad=None, dilate=1):
        """ please refer to mx.symbol.im2col """
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilate, int):
            dilate = (dilate, dilate)
        if pad is None:
            assert kernel[0] % 2 == 1, "Specify pad for an even kernel size in function sampler_im2col"
            pad = ((kernel[0] - 1) * dilate[0] + 1) // 2
        if isinstance(pad, int):
            pad = (pad, pad)

        # https://mxnet.apache.org/versions/1.8.0/api/python/docs/api/symbol/symbol.html#mxnet.symbol.im2col
        # output = mx.symbol.im2col(
        #     name=name + "sampler",
        #     data=data,
        #     kernel=kernel,
        #     stride=stride,
        #     dilate=dilate,
        #     pad=pad)

        # https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        # torch._C._nn.im2col(input, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))
        output = F.unfold(data, kernel_size=kernel, 
                                dilation=dilate,
                                stride=stride, 
                                padding=pad)

        return output

    def sample_data(self, data, kernel_size):
        """
        data sample
        :param data: num_batch, num_channel_in, H, W
        :param kernel_size: int default=3
        :return: sample_output: num_batch, num_channel_in * kernel_size * kernel_size, H, W
        """
        sample_output = self.sampler_im2col(
            data=data,
            kernel=kernel_size,
            stride=1,
            pad=1,
            dilate=1)
        return sample_output

    def sample_coord(self, coord, kernel_size):
        """
        coord sample
        :param coord: num_batch, num_channel_in, H, W
        :param kernel_size: int default=3
        :return: coord_sample_data: num_batch, num_channel_in * kernel_size * kernel_size, H, W
        """
        coord_sample_data = self.sampler_im2col(
            data=coord,
            kernel=kernel_size,
            stride=1,
            pad=1,
            dilate=1)

        return coord_sample_data

    def relative_coord(self, sample_coord, center_coord, num_channel_in, kernel_size):
        """
        :param sample_coord: num_batch, num_channel_in * kernel_size * kernel_size, H, W
        :param center_coord: num_batch, num_channel_in, H, W
        :param num_channel_in: int
        :param kernel_size: int
        :return: rel_coord: num_batch, num_channel_in, kernel_size * kernel_size, H, W
        """
        sample_reshape = torch.reshape(
            sample_coord,
            shape=(
                self.num_batch,
                num_channel_in,
                kernel_size * kernel_size,
                self.H,
                self.W
            )
        )
        # ic(sample_reshape.size())

        center_coord_expand = torch.unsqueeze(center_coord, dim=2)  # expand_dims
        # ic(center_coord_expand.size())

        rel_coord = torch.subtract(sample_reshape, center_coord_expand)
        # ic(rel_coord.size())

        return rel_coord

    def mlp(self, data, in_channels, channel_list=None, b_mul=1, use_norm=False):
        """
        :param data: num_batch, num_channel_in * kernel_size * kernel_size, H, W
        :param in_channels: int
        :param norm: normalizer
        :param channel_list: List[int]
        :param b_mul: int default=1
        :param use_norm: bool default=False
        :return: mlp_output_reshape: num_batch, out_channels, kernel_size * kernel_size, H, W
        """
        assert isinstance(channel_list, list)
        # NOTE: Currently only supports two-layer Conv structure
        assert len(channel_list) == 2

        x = torch.reshape(
            data,
            shape=(self.num_batch * b_mul,
                in_channels,
                -1,
                self.W
            )
        )

        y = self.conv1(x)
        # ic(y.size())

        if use_norm:
            y = self.bn1(y)
        y = self.act1(y)
        y = self.conv2(y)
        # ic(y.size())

        mlp_output_reshape = torch.reshape(
            y,
            shape=(
                self.num_batch * b_mul,
                channel_list[-1],
                -1,
                self.H,
                self.W
            )
        )
        # ic(mlp_output_reshape.size())
        return mlp_output_reshape

    def forward(self, data, coord_data, data_channels, coord_channels, kernel_size=3):
        """
        # Without data mlp;
        # MLP: fc + norm + relu + fc;
        # Using normalized coordinates
        :param data: num_batch, num_channel_in, H, W
        :param coord_data: num_batch, 3, H, W
        :param data_channels: num_channel_in
        :param coord_channels: 3
        :param norm: normalizer
        :param conv1_filter: int
        :param kernel_size: int default=3
        :param kwargs:
        :return: conv1: num_batch, conv1_filter, H, W
        """

        coord_sample_data = self.sample_coord(coord_data, kernel_size)
        # ic(coord_sample_data.size())
        # ic(coord_data.size())

        rel_coord = self.relative_coord(
            coord_sample_data,
            coord_data,
            coord_channels,
            kernel_size)
        # ic(rel_coord.size())

        weights = self.mlp(
            rel_coord,
            in_channels=coord_channels,
            channel_list=self.channel_list)
        # ic(weights.size())

        data_sample = self.sample_data(data, kernel_size)
        # ic(data_sample.size())

        data_sample_reshape = torch.reshape(
            data_sample,
            shape=(
                self.num_batch,
                data_channels,
                kernel_size * kernel_size,
                self.H,
                self.W)
        )
        # ic(data_sample_reshape.size())

        output = data_sample_reshape * weights
        # ic(output.size())

        output_reshape = torch.reshape(
            output,
            shape=(
                self.num_batch,
                -1,
                self.H,
                self.W)
        )
        # ic(output_reshape.size())

        output = self.conv1x1(output_reshape)
        return output