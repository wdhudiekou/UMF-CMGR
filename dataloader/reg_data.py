import pathlib

import cv2
import numpy as np
import kornia.utils
import torch.utils.data
import torchvision.transforms.functional
from PIL import Image


class RegData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, it_folder: pathlib.Path, crop=lambda x: x):
        super(RegData, self).__init__()
        self.crop = crop
        # gain infrared and visible images list
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.it_list = [x for x in sorted(it_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]


    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        it_path = self.it_list[index]

        assert ir_path.name == it_path.name, f"Mismatch ir:{ir_path.name} vi:{it_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        it = self.imread(path=it_path, flags=cv2.IMREAD_GRAYSCALE)


        # crop same patch
        patch = torch.cat([ir, it], dim=0)
        patch = torchvision.transforms.functional.to_pil_image(patch)
        patch = self.crop(patch)
        patch = torchvision.transforms.functional.to_tensor(patch)
        ir, vi = torch.chunk(patch, 2, dim=0)


        return (ir, vi), (str(ir_path), str(it_path))

    def __len__(self):
        return len(self.ir_list)


    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class RegTestData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, it_folder: pathlib.Path, disp_folder: pathlib.Path):
        super(RegTestData, self).__init__()

        # gain images list
        self.ir_list   = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.it_list   = [x for x in sorted(it_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.disp_list = [x for x in sorted(disp_folder.glob('*')) if x.suffix in ['.npy']]


    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        it_path = self.it_list[index]
        disp_path = self.disp_list[index]

        assert ir_path.name == it_path.name, f"Mismatch ir:{ir_path.name} vi:{it_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE, unsqueeze=False)
        it = self.imread(path=it_path, flags=cv2.IMREAD_GRAYSCALE, unsqueeze=False)
        disp = torch.from_numpy(np.load(disp_path))


        return (ir, it, disp), (str(ir_path), str(it_path), str(disp_path))

    def __len__(self):
        return len(self.ir_list)


    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE, unsqueeze=False):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts.unsqueeze(0) if unsqueeze else im_ts




