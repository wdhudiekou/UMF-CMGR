# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import math
import torch.nn as nn
import torchvision
from models.layers import SpatialTransformer, ResizeTransform, PointSpatialTransformer

shape = (256, 256)


class LossFunction_Affine(nn.Module):
    def __init__(self):
        super(LossFunction_Affine, self).__init__()

    def forward(self, fixed, moved, matrix, transform_pred):
        angle, scale_x, scale_y, center_x, center_y = transform_pred
        hyper = {'sim': 1, 'det': 0.1}
        sim_loss = similarity_loss(fixed, moved) * hyper['sim']
        theta = matrix[:, :2]
        det_loss = determinant_loss(theta)

        scale_x_loss = torch.sum((scale_x - 1.0) ** 2)
        scale_y_loss = torch.sum((scale_y - 1.0) ** 2)

        loss = sim_loss + det_loss * hyper['det'] + scale_x_loss + scale_y_loss
        return loss, sim_loss, det_loss


class LossFunction_Dense(nn.Module):
    def __init__(self):
        super(LossFunction_Dense, self).__init__()
        self.gradient_loss = gradient_loss()
        self.multi_loss = multi_loss()
        self.feat_loss = VGGLoss()
        self.edge_loss = EdgeLoss()

    def forward(self, y, y_f, tgt, src, flow, flow1, flow2): # tgt: torch.Size([16, 1, 224, 224])

        hyper_ncc = 1
        hyper_grad = 10
        hyper_feat = 1
        # hyper_multi = 1
        # hyper_edge = 8
        # hyper_3 = 2
        # hyper_4 = 1
        # TODO: similarity loss
        # ncc_1 = similarity_loss(tgt, y)
        # ncc_2 = similarity_loss(src, y_f)
        ncc_1 = torch.nn.functional.l1_loss(tgt, y)
        ncc_2 = torch.nn.functional.l1_loss(src, y_f)
        ncc = ncc_1 + 0.2*ncc_2

        # TODO: feature loss
        feat_1 = self.feat_loss(y, tgt)
        feat_2 = self.feat_loss(y_f, src)
        feat = feat_1 + 0.2*feat_2

        # TODO: gradient loss
        grad = self.gradient_loss(flow)

        # TODO: multi-scale loss
        # multi_1 = self.multi_loss(src, tgt, flow1, flow2, hyper_3, hyper_4)
        # multi_2 = self.multi_loss(tgt, src, -flow1, -flow2, hyper_3, hyper_4)
        # multi = multi_1 + 0.2*multi_2

        # TODO: edge loss
        # edge_1 = self.edge_loss(y, tgt)
        # edge_2 = self.edge_loss(y_f, src)
        # edge = edge_1 + edge_2

        # TODO: total loss
        # loss = multi + hyper_ncc * ncc + hyper_grad * grad + hyper_feat * feat
        # return loss, multi, ncc, grad
        # loss = hyper_grad * grad + hyper_feat * feat
        loss = hyper_feat * feat + hyper_grad * grad
        return loss, feat, ncc, grad


class LossFunctionAddPoint(nn.Module):
    def __init__(self):
        super(LossFunctionAddPoint, self).__init__()
        self.gradient_loss = gradient_loss().cuda()
        self.multi_loss = multi_loss().cuda()

    def forward(self, y, tgt, src, flow, flow1, flow2, t_point, s_point, norm_value):
        hyper_1 = 10
        hyper_2 = 15
        hyper_3 = 3.2
        hyper_4 = 0.8
        ncc = similarity_loss(tgt, y)
        grad = self.gradient_loss(flow)
        multi = self.multi_loss(src, tgt, flow1, flow2, hyper_3, hyper_4)
        loss = multi + hyper_1 * ncc + hyper_2 * grad
        point_stn = PointSpatialTransformer(volsize=(512, 512))
        annotations_warped = point_stn(s_point, flow, intep=True)
        tre = compute_target_regist_error(t_point[0, ...], annotations_warped[0, ...])/norm_value
        loss = loss + 100 * tre
        return loss, ncc, grad, tre


class LossFunctionAddMask(nn.Module):
    def __init__(self):
        super(LossFunctionAddMask, self).__init__()
        self.gradient_loss = gradient_loss().cuda()
        self.multi_loss = multi_loss().cuda()

    def forward(self, y, tgt, src, flow, flow1, flow2, t_mask, s_mask):
        hyper_1 = 10
        hyper_2 = 15
        hyper_3 = 3.2
        hyper_4 = 0.8

        ncc = similarity_loss(tgt, y)
        flow = flow.mul(s_mask)
        grad = self.gradient_loss(flow)

        multi = self.multi_loss(src, tgt, flow1, flow2, hyper_3, hyper_4)
        loss = multi + hyper_1 * ncc + hyper_2 * grad
        return loss, ncc, grad


class gradient_loss(nn.Module):
    def __init__(self):
        super(gradient_loss, self).__init__()

    def forward(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0


class multi_loss(nn.Module):
    def __init__(self):
        super(multi_loss, self).__init__()

        inshape = shape
        down_shape2 = [int(d / 4) for d in inshape]
        down_shape1 = [int(d / 2) for d in inshape]
        self.spatial_transform_1 = SpatialTransformer(volsize=down_shape1)
        self.spatial_transform_2 = SpatialTransformer(volsize=down_shape2)
        self.resize_1 = ResizeTransform(2, len(inshape))
        self.resize_2 = ResizeTransform(4, len(inshape))
        self.feat_loss = VGGLoss()

    def forward(self, src, tgt, flow1, flow2, hyper_3, hyper_4):
        loss = 0.
        zoomed_x1 = self.resize_1(tgt) # torch.Size([16, 1, 112, 112])
        zoomed_x2 = self.resize_1(src) # torch.Size([16, 1, 112, 112])
        warped_zoomed_x2, _ = self.spatial_transform_1(zoomed_x2, flow1)
        # loss += hyper_3 * similarity_loss(warped_zoomed_x2, zoomed_x1)
        loss += hyper_3 * torch.nn.functional.l1_loss(warped_zoomed_x2, zoomed_x1)
        # loss += hyper_3 * self.feat_loss(warped_zoomed_x2, zoomed_x1)

        zoomed_x1 = self.resize_2(tgt) # torch.Size([16, 1, 56, 56])
        zoomed_x2 = self.resize_2(src) # torch.Size([16, 1, 56, 56])
        warped_zoomed_x2, _ = self.spatial_transform_2(zoomed_x2, flow2)
        # loss += hyper_4 * similarity_loss(warped_zoomed_x2, zoomed_x1)
        loss += hyper_4 * torch.nn.functional.l1_loss(warped_zoomed_x2, zoomed_x1)
        # loss += hyper_4 * self.feat_loss(warped_zoomed_x2, zoomed_x1)

        return loss


class LossFunction_LNCC(nn.Module):
    def __init__(self):
        super(LossFunction_LNCC, self).__init__()
        self.ncc_loss = ncc_loss().cuda()
        self.gradient_loss = gradient_loss().cuda()
        self.multi_loss = multi_loss_ncc().cuda()

    def forward(self, y, tgt, src, flow, flow1, flow2, hyper_1=10, hyper_2=15, hyper_3=3.2, hyper_4=0.8):
        ncc = self.ncc_loss(tgt, y)
        grad = self.gradient_loss(flow)
        multi = self.multi_loss(src, tgt, flow1, flow2, hyper_3, hyper_4)
        loss = multi + hyper_1 * ncc + hyper_2 * grad
        return loss, ncc, grad

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X) # torch.Size([1, 64, 256, 256])
        h_relu2 = self.slice2(h_relu1) # torch.Size([1, 128, 128, 128])
        h_relu3 = self.slice3(h_relu2) # torch.Size([1, 256, 64, 64])
        h_relu4 = self.slice4(h_relu3) # torch.Size([1, 512, 32, 32])
        h_relu5 = self.slice5(h_relu4) # torch.Size([1, 512, 16, 16])
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        if torch.cuda.is_available():
            self.vgg.cuda()
        self.vgg.eval()
        set_requires_grad(self.vgg, False)
        self.L1Loss = nn.L1Loss()
        self.criterion2 = nn.MSELoss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 , 1.0]

    def forward(self, x, y):
        contentloss = 0
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x_vgg = self.vgg(x)
        with torch.no_grad():
            y_vgg = self.vgg(y)

        contentloss += self.L1Loss(x_vgg[3], y_vgg[3].detach())

        return contentloss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class ncc_loss(nn.Module):
    def __init__(self):
        super(ncc_loss, self).__init__()

    def compute_local_sums(self, I, J, filt, stride, padding, win):
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
        J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
        I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
        J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
        IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        return I_var, J_var, cross

    def forward(self, I, J, win=[15]):
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        if win is None:
            win = [9] * ndims
        else:
            win = win * ndims

        sum_filt = torch.ones([1, 1, *win]).cuda()

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        I_var, J_var, cross = self.compute_local_sums(I, J, sum_filt, stride, padding, win)

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -1 * torch.mean(cc)


class multi_loss_ncc(nn.Module):
    def __init__(self):
        super(multi_loss_ncc, self).__init__()

        inshape = shape
        down_shape2 = [int(d / 4) for d in inshape]
        down_shape1 = [int(d / 2) for d in inshape]
        self.ncc_loss = ncc_loss()
        self.gradient_loss = gradient_loss()
        self.spatial_transform_1 = SpatialTransformer(volsize=down_shape1)
        self.spatial_transform_2 = SpatialTransformer(volsize=down_shape2)
        self.resize_1 = ResizeTransform(2, len(inshape))
        self.resize_2 = ResizeTransform(4, len(inshape))

    def forward(self, src, tgt, flow1, flow2, hyper_3, hyper_4):
        loss = 0.
        zoomed_x1 = self.resize_1(tgt)
        zoomed_x2 = self.resize_1(src)
        warped_zoomed_x2 = self.spatial_transform_1(zoomed_x2, flow1)
        loss += hyper_3 * self.ncc_loss(warped_zoomed_x2, zoomed_x1, win=[7])

        zoomed_x1 = self.resize_2(tgt)
        zoomed_x2 = self.resize_2(src)
        warped_zoomed_x2 = self.spatial_transform_2(zoomed_x2, flow2)
        loss += hyper_4 * self.ncc_loss(warped_zoomed_x2, zoomed_x1, win=[5])

        return loss


def similarity_loss(tgt, warped_img):

    sizes = np.prod(list(tgt.shape)[1:])
    flatten1 = torch.reshape(tgt, (-1, sizes))
    flatten2 = torch.reshape(warped_img, (-1, sizes))

    mean1 = torch.reshape(torch.mean(flatten1, dim=-1), (-1, 1))
    mean2 = torch.reshape(torch.mean(flatten2, dim=-1), (-1, 1))
    var1 = torch.mean((flatten1 - mean1) ** 2, dim=-1)
    var2 = torch.mean((flatten2 - mean2) ** 2, dim=-1)
    cov12 = torch.mean((flatten1 - mean1) * (flatten2 - mean2), dim=-1)
    pearson_r = cov12 / torch.sqrt((var1 + 1e-6) * (var2 + 1e-6))
    raw_loss = torch.sum(1 - pearson_r)

    return raw_loss


def orthogonal_loss(t):
    # C=A'A, a positive semi-definite matrix
    # should be close to I. For this, we require C
    # has eigen values close to 1
    c = torch.matmul(t, t)
    k = torch.linalg.eigvals(c)  # Get eigenvalues of C
    ortho_loss = torch.mean((k[0][0] - 1.0) ** 2) + torch.mean((k[0][1] - 1.0) ** 2)
    ortho_loss = ortho_loss.float()
    return ortho_loss


def determinant_loss(t):
    # Determinant Loss: determinant should be close to 1
    det_value = torch.det(t)
    det_loss = torch.sum((det_value - 1.0) ** 2)/2
    return det_loss


def compute_target_regist_error(points_ref, points_est):
    """ compute distance as between related points in two sets
    and make a statistic on those distances - mean, std, median, min, max

    :param ndarray points_ref: final landmarks in target image of  np.array<nb_points, dim>
    :param ndarray points_est: warped landmarks from source to target of np.array<nb_points, dim>
    :return tuple(ndarray,dict): (np.array<nb_points, 1>, dict)
    ([], {'overlap points': 0})
    """
    if not all(pts is not None and list(pts) for pts in [points_ref, points_est]):
        return [], {'overlap points': 0}

    lnd_sizes = [len(points_ref), len(points_est)]
    assert min(lnd_sizes) > 0, 'no common landmarks for metric'
    diffs = compute_tre(points_ref, points_est)
    return diffs


def compute_tre(points_1, points_2):
    """ computing Target Registration Error for each landmark pair

    :param ndarray points_1: set of points
    :param ndarray points_2: set of points
    :return ndarray: list of errors of size min nb of points
    array([ 0.21...,  0.70...,  0.44...,  0.34...,  0.41...,  0.41...])
    """
    points_1 = points_1[0, ...]
    points_2 = points_2[0, ...]
    nb_common = min([len(pts) for pts in [points_1, points_2]
                     if pts is not None])
    assert nb_common > 0, 'no common landmarks for metric'
    points_1 = points_1[:nb_common]
    points_2 = points_2[:nb_common]
    diffs = torch.sqrt(torch.sum(torch.square(points_1 - points_2)))
    return diffs