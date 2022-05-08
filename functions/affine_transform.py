from typing import Tuple

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters.kernels import get_gaussian_kernel2d


class AffineTransform(nn.Module):
    """
    Add random affine transforms to a tensor image.
    Most functions are obtained from Kornia, difference:
    - gain the disp grid
    - no p and same_on_batch
    """

    def __init__(self, degrees=0, translate=0.1):
        super(AffineTransform, self).__init__()
        self.trs = kornia.augmentation.RandomAffine(degrees, (translate, translate), return_transform=True, p=1)

    def forward(self, input):
        # image shape
        batch_size, _, height, weight = input.shape
        # affine transform
        warped, affine_param = self.trs(input)  # [batch_size, 3, 3]
        affine_theta = self.param_to_theta(affine_param, weight, height)  # [batch_size, 2, 3]
        # base + disp = grid -> disp = grid - base
        base = kornia.utils.create_meshgrid(height, weight, device=input.device).to(input.dtype)
        grid = F.affine_grid(affine_theta, size=input.size(), align_corners=False)  # [batch_size, height, weight, 2]
        disp = grid - base
        return warped, -disp

    @staticmethod
    def param_to_theta(param, weight, height):
        """
        Convert affine transform matrix to theta in F.affine_grid
        :param param: affine transform matrix [batch_size, 3, 3]
        :param weight: image weight
        :param height: image height
        :return: theta in F.affine_grid [batch_size, 2, 3]
        """

        theta = torch.zeros(size=(param.shape[0], 2, 3)).to(param.device)  # [batch_size, 2, 3]

        theta[:, 0, 0] = param[:, 0, 0]
        theta[:, 0, 1] = param[:, 0, 1] * height / weight
        theta[:, 0, 2] = param[:, 0, 2] * 2 / weight + param[:, 0, 0] + param[:, 0, 1] - 1
        theta[:, 1, 0] = param[:, 1, 0] * weight / height
        theta[:, 1, 1] = param[:, 1, 1]
        theta[:, 1, 2] = param[:, 1, 2] * 2 / height + param[:, 1, 0] + param[:, 1, 1] - 1

        return theta
