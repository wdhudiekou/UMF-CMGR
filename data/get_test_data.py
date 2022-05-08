import argparse
import pathlib
import warnings

import os
import cv2
import numpy as np
import kornia
import torch.backends.cudnn
import torch.cuda
import torch.utils.data
import torchvision
from torch import Tensor
from tqdm import tqdm

from functions.affine_transform import AffineTransform
from functions.elastic_transform import ElasticTransform


class getDeformableImages:
    """
    principle: ir -> ir_warp
    """
    def __init__(self):
        # hardware settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # deformable transforms
        self.elastic = ElasticTransform(kernel_size=101, sigma=16)
        self.affine  = AffineTransform(translate=0.01)

    @torch.no_grad()
    def __call__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path, dst: pathlib.Path):

        # get images list
        ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        # starting generate deformable infrared image
        loader = tqdm(zip(ir_list, vi_list))
        for ir_path, vi_path in loader:
            name = ir_path.name
            loader.set_description(f'warp: {name}')
            name_disp = name.split('.')[0] + '_disp.npy'

            # read images
            ir = self.imread(ir_path, unsqueeze=True).to(self.device)
            vi = self.imread(vi_path, unsqueeze=True).to(self.device)

            # get deformable images
            ir_affine, affine_disp = self.affine(ir)
            ir_elastic, elastic_disp = self.elastic(ir_affine)
            disp = affine_disp + elastic_disp  # cumulative disp grid [batch_size, height, weight, 2]
            ir_warp = ir_elastic

            _, _, h, w = ir_warp.shape
            grid = kornia.utils.create_meshgrid(h, w, device=ir_warp.device).to(ir_warp.dtype)
            grid = grid.permute(0, 3, 1, 2)
            disp = disp.permute(0, 3, 1, 2)
            new_grid = grid + disp

            # draw grid
            img_grid = self._draw_grid(ir.squeeze().cpu().numpy(), 24)

            new_grid = new_grid.permute(0, 2, 3, 1)
            warp_grid = torch.nn.functional.grid_sample(img_grid.unsqueeze(0), new_grid, padding_mode='border', align_corners=False)
            # raw image w/o warp
            ir_raw_grid  = 0.8 * ir + 0.2 * img_grid
            ir_raw_grid  = torch.clamp(ir_raw_grid, 0, 1)
            # warped grid & warped ir image
            ir_warp_grid = 0.8 * ir_warp + 0.2 * warp_grid
            ir_warp_grid = torch.clamp(ir_warp_grid, 0, 1)
            # disp
            disp_npy = disp.data.cpu().numpy()

            # save disp
            if not os.path.exists(dst):
                os.makedirs(dst)
            np.save(dst / name_disp, disp_npy)
            # save deformable images
            self.imsave(vi, dst / 'vi_gray', name)
            self.imsave(ir_warp, dst / 'ir_warp', name)
            self.imsave(warp_grid, dst / 'warp_grid', name)
            self.imsave(ir_warp_grid, dst / 'ir_warp_grid', name)
            self.imsave(ir_raw_grid, dst / 'ir_raw_grid', name)


    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE, unsqueeze=False):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts.unsqueeze(0) if unsqueeze else im_ts

    @staticmethod
    def imsave(im_s: [Tensor], dst: pathlib.Path, im_name: str = ''):
        """
        save images to path
        :param im_s: image(s)
        :param dst: if one image: path; if multiple images: folder path
        :param im_name: name of image
        """

        im_s = im_s if type(im_s) == list else [im_s]
        dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
        for im_ts, p in zip(im_s, dst):
            im_ts = im_ts.squeeze().cpu()
            p.parent.mkdir(parents=True, exist_ok=True)
            im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
            cv2.imwrite(str(p), im_cv)

    @staticmethod
    def _draw_grid(im_cv, grid_size: int = 10):
        im_gd_cv = np.full_like(im_cv, 255.0)
        im_gd_cv = cv2.cvtColor(im_gd_cv, cv2.COLOR_GRAY2BGR)
        color = (0, 0, 255)

        height, width = im_cv.shape
        for x in range(0, width - 1, grid_size):
            cv2.line(im_gd_cv, (x, 0), (x, height), color, 1, 1)
        for y in range(0, height - 1, grid_size):
            cv2.line(im_gd_cv, (0, y), (width, y), color, 1, 1)

        im_gd_ts = kornia.utils.image_to_tensor(im_gd_cv / 255.).type(torch.FloatTensor).cuda()
        return im_gd_ts


def hyper_args():
    """
    get hyper parameters from args
    """

    parser = argparse.ArgumentParser(description='Generating deformable testing data')
    # dataset
    parser.add_argument('--ir', default='../dataset/raw/ctest/Road/ir', type=pathlib.Path)
    parser.add_argument('--vi', default='../dataset/raw/ctest/Road/vi', type=pathlib.Path)
    parser.add_argument('--dst', default='../dataset/test/', help='fuse image save folder', type=pathlib.Path)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    args = hyper_args()
    data = getDeformableImages()
    data(ir_folder=args.ir, vi_folder=args.vi, dst=args.dst)
