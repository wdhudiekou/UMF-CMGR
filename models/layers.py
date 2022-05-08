# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

gpu_use = True


def construct_M(angle, scale_x, scale_y, center_x, center_y):
    alpha = torch.cos(angle)
    beta = torch.sin(angle)
    tx = center_x
    ty = center_y
    tmp0 = torch.cat((scale_x * alpha, beta), 1)
    tmp1 = torch.cat((-beta, scale_y * alpha), 1)
    theta = torch.cat((tmp0, tmp1), 0)
    t = torch.cat((tx, ty), 0)
    matrix = torch.cat((theta, t), 1)
    return theta, matrix




class ConstuctRotationLayer(nn.Module):
    def __init__(self):
        super(ConstuctRotationLayer, self).__init__()

    def forward(self, angle):
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        tmp0 = torch.cat((alpha, beta), 1)
        tmp1 = torch.cat((-beta, alpha), 1)
        theta = torch.cat((tmp0, tmp1), 0)
        t = torch.tensor([[0.], [0.]]).cuda()
        matrix = torch.cat((theta, t), 1)
        return theta, matrix


class ConstuctmatrixLayer(nn.Module):
    def __init__(self):
        super(ConstuctmatrixLayer, self).__init__()

    def forward(self, angle, scale_x, scale_y, center_x, center_y):
        theta, matrix = construct_M(angle, scale_x, scale_y, center_x, center_y)
        return theta, matrix


class AffineToFlow(nn.Module):

    def __init__(self, volsize):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(AffineToFlow, self).__init__()

        # Create sampling grid
        self.size = volsize

    def forward(self, matrix):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """

        flow = F.affine_grid(matrix.unsqueeze(0), [1, 1, self.size[0], self.size[1]], align_corners=True)
        shape = flow.shape[1:3]
        if len(shape) == 2:
            flow = flow[..., [1, 0]]
            flow = flow.permute(0, 3, 1, 2)

        for i in range(len(shape)):
            flow[:, i, ...] = (flow[:, i, ...].clone() / 2 + 0.5) * (shape[i] - 1)

        vectors = [torch.arange(0, s) for s in self.size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        flow_offset = flow - grid

        return flow_offset


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, volsize, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        size = volsize
        vectors = [ torch.arange(0, s) for s in size ]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...].clone() / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode='border', align_corners=True), new_locs


class PointSpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, volsize, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(PointSpatialTransformer, self).__init__()

        # Create sampling grid
        size = volsize
        vectors = [ torch.arange(0, s) for s in size ]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, point, flow, intep=False):
        """
        Push the src and flow through the spatial transform block
            :param point: [N, 2]
            :param flow: the output from the U-Net [*vol_shape, 2]
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...].clone() / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        new_point = point.clone().detach()

        if intep:
            for i in range(point.shape[1]):
                x_trunc, x_frac = new_point[0, i, 0].trunc(), new_point[0, i, 0].frac()
                y_trunc, y_frac = new_point[0, i, 1].trunc(), new_point[0, i, 1].frac()
                x0, y0 = x_trunc.long(), y_trunc.long()
                x1, y1 = (x_trunc+1).long(), y_trunc.long()
                x2, y2 = x_trunc.long(), (y_trunc+1).long()
                x3, y3 = (x_trunc+1).long(), (y_trunc+1).long()
                # dic ={'0': x_frac * y_frac, '1': (1-x_frac) * y_frac,
                #       '2': x_frac * (1-y_frac), '3': (1-x_frac) * (1-y_frac)}

                dic = {'0': x_frac * y_frac, '2': (1 - x_frac) * y_frac,
                       '1': x_frac * (1 - y_frac), '3': (1 - x_frac) * (1 - y_frac)}

                tmp_x = dic['0'] * new_locs[0, x0, y0, 0] + dic['1'] * new_locs[0, x1, y1, 0] +\
                               dic['2'] * new_locs[0, x2, y2, 0] + dic['3'] * new_locs[0, x3, y3, 0]
                tmp_y = dic['0'] * new_locs[0, x0, y0, 1] + dic['1'] * new_locs[0, x1, y1, 1] +\
                               dic['2'] * new_locs[0, x2, y2, 1] + dic['3'] * new_locs[0, x3, y3, 1]

                new_point[0, i, 1] = (tmp_x + 1) / 2 * 512
                new_point[0, i, 0] = (tmp_y + 1) / 2 * 512
        else:
            for i in range(point.shape[1]):
                x = min(new_point[0, i, 0].round().long(), 511)
                y = min(new_point[0, i, 1].round().long(), 511)
                new_point[0, i, 1] = (new_locs[0, x, y, 0] + 1) / 2 * 512
                new_point[0, i, 0] = (new_locs[0, x, y, 1] + 1) / 2 * 512

        return new_point


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """
    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out


def composition_flows(g1, g2):
    """
    warping an image twice, first with g1 then with g2
    :param g1, g2 is dense_flow/ offset
    :return:
    """
    transformer = SpatialTransformer(volsize=(512, 512))
    flow = g2 + transformer(g1, g2)
    return flow


def predict_flow(in_planes, d=3):
    dim = d
    conv_fn = getattr(nn, 'Conv%dd' % dim)
    return conv_fn(in_planes, dim, kernel_size=3, padding=1)


def conv2D(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))


def MatchCost(features_t, features_s):
    mc = torch.norm(features_t - features_s, p=1, dim=1) # torch.Size([1, 64, 64])
    mc = mc[..., np.newaxis] # np.newaxis: Extended dimension torch.Size([1, 64, 64, 1])
    return mc.permute(0, 3, 1, 2)
