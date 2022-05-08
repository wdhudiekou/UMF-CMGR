import kornia
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

'''
class Transform(nn.Module):
    """
    Predict affine grid for correcting distortion image.
    principle: im_distortion & im_reference -> transform grid -> im_correction
    """

    def __init__(self, feather_num=64):
        super(Transform, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2, feather_num, (3, 3), padding=(1, 1), bias=False), # 2->64
            nn.BatchNorm2d(feather_num, momentum=0.9, eps=1e-5),
            nn.ELU(),
            nn.Conv2d(feather_num, 2 * feather_num, (3, 3), padding=(1, 1), bias=False),# 64->128
            nn.BatchNorm2d(2 * feather_num, momentum=0.9, eps=1e-5),
            nn.ELU(),
            nn.Conv2d(2 * feather_num, 2, (3, 3), padding=(1, 1)), # 128->2
            nn.Tanh()
        )

        # self.conv.apply(self._weight_init)

    @staticmethod
    def _weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0., std=1e-5)
            nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, mean=0., std=1.)
            nn.init.constant_(m.bias, val=0.)

    def forward(self, distortion, reference) -> Tensor:
        """
        im_distortion & im_reference -> transform grid -> im_correction
        :return transform grid
        """
        batch, _, height, weight = distortion.shape

        # laplacian = kornia.filters.laplacian
        # edge_d, edge_r = laplacian(distortion, 11), laplacian(reference, 11)

        input = torch.cat([distortion, reference], dim=1)  # [batch_size, 2, height, weight]
        disp = self.conv(input).permute(0, 2, 3, 1)  # [batch_size, height, weight, 2]

        return disp
'''

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
        grid = torch.stack(grids) # torch.Size([2, 224, 224])
        grid = torch.unsqueeze(grid, 0)  # torch.Size([1, 2, 224, 224])
        grid = grid.type(torch.FloatTensor).cuda()
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow): # src: torch.Size([1, 1, 512, 512])
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """

        # print(src.shape, flow.shape, self.grid.shape) #torch.Size([16, 1, 224, 224]) torch.Size([16, 2, 224, 224]) torch.Size([1, 2, 224, 224])

        new_locs = self.grid + flow # torch.Size([1, 2, 224, 224])
        shape = flow.shape[2:] # torch.Size([224, 224])


        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...].clone() / (shape[i] - 1) - 0.5)


        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1) # torch.Size([16, 224, 224, 2])
            new_locs = new_locs[..., [1,0]] # torch.Size([16, 224, 224, 2])
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode='border', align_corners=True), new_locs

class Transform(nn.Module):

    def __init__(self, feather_num=64):
        super(Transform, self).__init__()

        self.downsampleby2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.upsampleby2   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
       		)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
       		)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
       		)
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
       		)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
       		)
        self.conv6 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
       		)
        self.conv7 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
       		)
        self.conv8 = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
       		)
        self.conv9 = nn.Sequential(
            nn.Conv2d(24, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            )
        self.conv10= nn.Sequential(
            nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1),
       		)
        self.spatial_transform = SpatialTransformer(volsize=(224, 224))  # 512, 512
        # print(self.spatial_transform.parameters())
        # for p in self.spatial_transform.parameters():
        #     print(p)
        # exit(00)

    def load_state_dict(self, state_dict, strict = False):
        state_dict.pop('spatial_transform.grid')
        super().load_state_dict(state_dict, strict)

    def forward(self, distortion, reference):
        input_1 	= distortion
        input_2 	= reference
        input_cat	= torch.cat((input_1, input_2), 1)
        skips       = []
        #
        # ENCODER block -------------------------------------------
        # conv1
        feats   	= self.conv1(input_cat)
        skips.append(feats)
        feats   	= self.downsampleby2(feats)
        # conv2
        feats   	= self.conv2(feats)
        skips.append(feats)
        feats   = self.downsampleby2(feats)
        # conv3
        feats   = self.conv3(feats)
        skips.append(feats)
        feats   = self.downsampleby2(feats)
        # conv4
        feats   = self.conv4(feats)
        skips.append(feats)
        feats   = self.downsampleby2(feats)
        # conv5
        feats   = self.conv5(feats) # torch.Size([8, 128, 14, 14])

        # DECODER block -------------------------------------------
        # conv6
        feats   =  self.upsampleby2(feats)
        feats   =  torch.cat((feats, skips.pop()), 1) # torch.Size([8, 128, 28, 28]), torch.Size([8, 64, 28, 28])
        feats   =  self.conv6(feats)
        # conv7
        feats   =  self.upsampleby2(feats)
        feats   =  torch.cat((feats, skips.pop()), 1) # torch.Size([8, 64, 56, 56]) torch.Size([8, 32, 56, 56])
        feats   =  self.conv7(feats)
        # conv8
        feats   =  self.upsampleby2(feats)
        feats   =  torch.cat((feats, skips.pop()), 1) # torch.Size([8, 32, 112, 112]) torch.Size([8, 16, 112, 112])
        feats   =  self.conv8(feats)
        # conv9
        feats   =  self.upsampleby2(feats)
        feats   =  torch.cat((feats, skips.pop()), 1) # torch.Size([8, 16, 224, 224]) torch.Size([8, 8, 224, 224])
        feats   =  self.conv9(feats) # torch.Size([8, 8, 224, 224])
        # conv10
        flow = self.conv10(feats) # torch.Size([8, 2, 224, 224])

        warped, disp_pre = self.spatial_transform(distortion, flow) # torch.Size([16, 1, 224, 224])
        f_warp, disp_pre_re = self.spatial_transform(reference, (-flow))

        return warped, f_warp, flow, disp_pre
        
