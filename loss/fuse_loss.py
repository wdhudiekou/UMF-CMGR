import kornia.losses
import torch
import torch.nn as nn
from torch import Tensor

from loss.ms_ssim import MSSSIM


class FuseLoss(nn.Module):

    def __init__(self, alpha=1, beta=10, base=10):
        super(FuseLoss, self).__init__()

        self.ms_ssim = MSSSIM()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

        self.alpha = alpha
        self.beta = beta
        self.base = base

        self._loss = self._normal_loss
        self._loss_fus = self._normal_loss_fus
        self._forward = self._single_forward

    def _mix_loss(self, input: Tensor, target: Tensor) -> Tensor: # TODO: Eqn (2)
        ms_ssim_loss = 1 - self.ms_ssim(input, target)
        l1_loss = self.l1_loss(input, target)
        return self.alpha * ms_ssim_loss + (1 - self.alpha) * l1_loss

    def _normal_loss(self, input: Tensor, target: Tensor) -> Tensor:
        ssim_loss = kornia.losses.ssim_loss(input, target, window_size=11)
        l1_loss = self.l1_loss(input, target)
        laplacian = kornia.filters.laplacian
        l2_loss = self.l2_loss(laplacian(input, 11), laplacian(target, 11))
        return ssim_loss + self.alpha * l1_loss + self.beta * l2_loss

    def _normal_loss_fus(self, input: Tensor, target: Tensor) -> Tensor:
        ssim_loss = kornia.losses.ssim_loss(input, target, window_size=11)
        l1_loss = self.l1_loss(input, target)
        return ssim_loss + self.alpha * l1_loss

    # TODO: Remove before publish
    def _fake_loss(self, im_f: Tensor, gt: Tensor) -> Tensor:
        return self._loss(im_f, gt)  # fusion -> ground truth

    def _fuse_loss(self, im_f, im_a, im_b, att_a=torch.tensor(1.0), att_b=torch.tensor(1.0)) -> Tensor:
        w_a, w_b = self._weight(att_a, att_b)
        return self._loss_fus(im_f, im_a * w_a + im_b * w_b)  # fusion -> weighted src images

    def _warp_loss(self, im_c, im_r, disp_pred, disp):
        l1_loss = self.l1_loss(disp_pred, disp)
        return self._loss(im_c, im_r) + 2 * l1_loss  # correction -> reference

    # TODO: Remove gt before publish
    def _sub_forward(self, fus, ir, vi, ir_att, vi_att, disp_pred, ir_predict, disp, gt):
        # fake_loss = self._fake_loss(fus, gt)
        fuse_loss = self._fuse_loss(fus, ir, vi, ir_att, vi_att)
        warp_loss = self._warp_loss(ir_predict, ir, disp_pred, disp)
        # return warp_loss, fake_loss
        return warp_loss, fuse_loss # wd+

    def _loop_forward(self, src, fus_list, att_list, ir_predict_list, gt) -> Tensor:
        loss = torch.tensor(0.).to(fus_list[-1].device)
        for x in range(len(fus_list) - 1):
            fus = fus_list[x + 1]
            ir, vi = src
            ir_att, vi_att = att_list[x]
            ir_predict = ir_predict_list[x]
            loss += self.base ** x * self._sub_forward(fus, ir, vi, ir_att, vi_att, ir_predict, gt)
        return loss

    def _single_forward(self, src, fus_list, att_list, reshape, disp, gt):
        fus = fus_list[-1]
        ir, vi = src
        ir_att, vi_att = att_list[-1]
        disp_pred, ir_predict = reshape[-1]
        warp_loss, fake_loss = self._sub_forward(fus, ir, vi, ir_att, vi_att, disp_pred, ir_predict, disp, gt)
        return warp_loss, fake_loss

    def forward(self, src, fus_list, att_list, reshape, disp, gt):
        return self._forward(src, fus_list, att_list, reshape, disp, gt)

    @staticmethod
    # def _weight(att_a: Tensor, att_b: Tensor, c=1e10) -> tuple[Tensor, Tensor]:
    def _weight(att_a: Tensor, att_b: Tensor, c=1e10):  # 1e10
        """
        transform attention map to weight map
        """
        # c = torch.tensor(c).to(att_a.device)
        # x_a, x_b = torch.pow(c, att_a), torch.pow(c, att_b)
        x_a, x_b = att_a, att_b
        w_a, w_b = x_a / (x_a + x_b), x_b / (x_a + x_b)
        return w_a, w_b
