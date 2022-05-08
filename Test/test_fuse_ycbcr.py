import sys

sys.path.append("..")

import argparse
import pathlib
import warnings
import statistics
import time

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

from dataloader.fuse_data_vsm import FuseTestDataYcbcr
from models.fusion_net import FusionNet


def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='Fuse Net eval process')
    # dataset
    parser.add_argument('--ir',      default='../dataset/raw/ctest/Road/ir_reg', type=pathlib.Path)
    parser.add_argument('--vi',      default='../dataset/raw/ctest/Road/vi', type=pathlib.Path)
    # checkpoint
    parser.add_argument('--ckpt', default='../cache/Fusion_only/220506_fusion_w_svm/fus_0200.pth', help='weight checkpoint', type=pathlib.Path) # weight/default.pth
    parser.add_argument('--dst', default='../results_Road/Fusion/220506_Deformable_2*Fe_10*Grad/', help='fuse image save folder', type=pathlib.Path)

    parser.add_argument('--dim', default=64, type=int, help='AFuse feather dim')
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")

    args = parser.parse_args()
    return args

def main(args):

    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    print("===> Building model")
    net = FusionNet(nfeats=args.dim).to(device)

    print("===> loading trained model '{}'".format(args.ckpt))
    model_state_dict = torch.load(args.ckpt)['net']
    net.load_state_dict(model_state_dict)

    print("===> Starting Testing")
    test(net, args.ir, args.vi, args.dst, device)


def test(net, ir_folder, vi_folder, dst, device):
    net.eval()
    ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
    vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    fus_time = []
    tqdm_loader = tqdm(zip(ir_list, vi_list))
    for ir_path, vi_path in tqdm_loader:
        file_name = ir_path.name
        ir = cv2.imread(str(ir_path), flags=cv2.IMREAD_GRAYSCALE)  # (256, 256)
        vi = cv2.imread(str(vi_path), flags=cv2.IMREAD_COLOR)

        # TODO: split RGB visible image into Ycbcr image
        vi_ycbcr = cv2.cvtColor(vi, cv2.COLOR_BGR2YCrCb)
        vi_y = vi_ycbcr[:, :, 0]
        vi_cb = vi_ycbcr[:, :, 1]
        vi_cr = vi_ycbcr[:, :, 2]

        ir = kornia.utils.image_to_tensor(ir / 255.).type(torch.FloatTensor).unsqueeze(0).cuda()  # torch.Size([1, 256, 256])
        vi = kornia.utils.image_to_tensor(vi / 255.).type(torch.FloatTensor).unsqueeze(0).cuda()  # torch.Size([1, 256, 256])
        vi_y = kornia.utils.image_to_tensor(vi_y / 255.).type(torch.FloatTensor).unsqueeze(0).cuda()  # torch.Size([1, 256, 256])

        # TODO: Fusion
        torch.cuda.synchronize() if str(device) == 'cuda' else None
        start = time.time()
        with torch.no_grad():
            fuse_out  = net(ir, vi_y)
        torch.cuda.synchronize() if str(device) == 'cuda' else None
        end = time.time()
        fus_time.append(end - start)

        # TODO: combine Ycbcr image into RGB image
        fuse_y = fuse_out.squeeze().cpu()
        fuse_y = kornia.utils.tensor_to_image(fuse_y) * 255.
        fuse_ycbcr = np.stack([fuse_y, vi_cb, vi_cr], axis=2).astype(np.uint8)
        fuse_bgr = cv2.cvtColor(fuse_ycbcr, cv2.COLOR_YCrCb2BGR)

        # TODO: save fused images
        imsave(fuse_out, dst / 'fused' / file_name)
        imsave(ir, dst / 'ir' / file_name)
        imsave(vi, dst / 'vi' / file_name)
        # TODO: save fused rgb images
        path_bgr = os.path.join(dst, 'fused_rgb/')
        if not os.path.exists(path_bgr):
            os.makedirs(path_bgr)
        filename_bgr = os.path.join(path_bgr, file_name)
        cv2.imwrite(filename_bgr, fuse_bgr)

    # statistics time record
    fuse_mean = statistics.mean(fus_time[1:])
    print('fuse time (average): {:.4f}'.format(fuse_mean))
    print('fps (equivalence): {:.4f}'.format(1. / fuse_mean))

    pass
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


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = hyper_args()
    main(args)