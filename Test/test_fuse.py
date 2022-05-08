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

from dataloader.fuse_data_vsm import FuseTestData
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

    print("===> Loading datasets")
    data = FuseTestData(args.ir, args.vi)
    test_data_loader = torch.utils.data.DataLoader(data, 1, True, pin_memory=True)

    print("===> Building model")
    net = FusionNet(nfeats=args.dim).to(device)

    print("===> loading trained model '{}'".format(args.ckpt))
    model_state_dict = torch.load(args.ckpt)['net']
    net.load_state_dict(model_state_dict)

    print("===> Starting Testing")
    test(net, test_data_loader, args.dst, device)


def test(net, test_data_loader, dst, device):
    net.eval()

    fus_time = []
    tqdm_loader = tqdm(test_data_loader, disable=True)
    for (ir, vi), (ir_path, vi_path) in tqdm_loader:
        name, ext = os.path.splitext(os.path.basename(ir_path[0]))
        file_name = name + ext
        ir = ir.cuda()
        vi = vi.cuda()


        # Fusion
        torch.cuda.synchronize() if str(device) == 'cuda' else None
        start = time.time()
        with torch.no_grad():
            fuse_out  = net(ir, vi)
        torch.cuda.synchronize() if str(device) == 'cuda' else None
        end = time.time()
        fus_time.append(end - start)

        # TODO: save fused images
        imsave(fuse_out, dst / 'fused' / file_name)
        imsave(ir, dst / 'ir' / file_name)
        imsave(vi, dst / 'vi' / file_name)

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