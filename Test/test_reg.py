import sys

sys.path.append("..")

import argparse
import pathlib
import warnings
import statistics
import time

import cv2
import numpy as np
import kornia
import torch.backends.cudnn
import torch.cuda
import torch.utils.data
from torch import Tensor
from tqdm import tqdm
import os

from models.deformable_net import DeformableNet
from dataloader.reg_data import RegTestData

def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='RegNet Net eval process')

    # dataset
    parser.add_argument('--it',   default='../dataset/raw/ctest/Road/it_edge', type=pathlib.Path)
    parser.add_argument('--ir', default='../dataset/raw/ctest/Road/ir_w',    type=pathlib.Path)
    parser.add_argument('--disp', default='../dataset/raw/ctest/Road/disp',    type=pathlib.Path)
    # checkpoint
    parser.add_argument('--ckpt', default='../cache/Reg_only/220507_Deformable_2*Fe_10*Grad/cp_0720.pth', help='weight checkpoint', type=pathlib.Path) # weight/default.pth
    parser.add_argument('--dst',  default='../results_Road/Reg/220507_Deformable_2*Fe_10*Grad/', help='fuse image save folder', type=pathlib.Path)

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
    data = RegTestData(args.ir, args.it, args.disp)
    test_data_loader = torch.utils.data.DataLoader(data, 1, True, pin_memory=True)

    print("===> Building model")
    net = DeformableNet().to(device)

    print("===> loading trained model '{}'".format(args.ckpt))
    model_state_dict = torch.load(args.ckpt)
    net.load_state_dict(model_state_dict)

    print("===> Starting Testing")
    test(net, test_data_loader, args.dst, device)


def test(net, test_data_loader, dst, device):
    net.eval()

    reg_time = []
    tqdm_loader = tqdm(test_data_loader, disable=True)

    for (ir, it, disp), (ir_path, it_path, disp_path) in tqdm_loader:
        name, ext = os.path.splitext(os.path.basename(ir_path[0]))
        file_name = name + ext
        ir = ir.cuda()
        it = it.cuda()
        disp = disp.squeeze(0).cuda() # torch.Size([1, 2, 256, 256])

        # Registration
        torch.cuda.synchronize() if str(device) == 'cuda' else None
        start = time.time()
        with torch.no_grad():
            ir_pred, f_warp, flow, int_flow1, int_flow2, disp_pred = net(it, ir)
        torch.cuda.synchronize() if str(device) == 'cuda' else None
        end = time.time()
        reg_time.append(end - start)

        _, _, h, w = ir.shape
        grid = kornia.utils.create_meshgrid(h, w, device=ir.device).to(ir.dtype)
        grid = grid.permute(0, 3, 1, 2)

        # TODO: Draw grid
        img_grid = _draw_grid(it.squeeze().cpu().numpy(), 24)

        # TODO: get warped grid & warped image
        new_grid = grid + disp
        new_grid = new_grid.permute(0, 2, 3, 1)

        warp_grid = torch.nn.functional.grid_sample(img_grid.unsqueeze(0), new_grid, padding_mode='border', align_corners=False)
        warp_combine = 0.8 * ir + 0.2 * warp_grid
        warp_combine = torch.clamp(warp_combine, 0, 1)

        # TODO: get registrated grid & registrated image
        pred_grid = torch.nn.functional.grid_sample(warp_grid, disp_pred, padding_mode='border', align_corners=True)
        pred_combine = 0.8 * ir_pred + 0.2 * pred_grid
        pred_combine = torch.clamp(pred_combine, 0, 1)

        # TODO: save registrated images
        imsave(ir, dst / 'ir', file_name)
        imsave(it, dst / 'it' / file_name)
        imsave(ir_pred, dst / 'ir_reg', file_name)
        imsave(img_grid, dst / 'grid', file_name)

        imsave(warp_grid, dst / 'warp_grid', file_name)
        imsave(pred_grid, dst / 'reg_grid', file_name)

        imsave(warp_combine, dst / 'ir_warp_grid', file_name)
        imsave(pred_combine, dst / 'ir_reg_grid', file_name)
        save_flow(flow, dst / 'ir_flow', file_name)
        save_flow(-disp, dst / 'disp', file_name)  # .permute(0, 3, 1, 2)

    # statistics time record
    reg_mean = statistics.mean(reg_time[1:])
    print('fuse time (average): {:.4f}'.format(reg_mean))
    print('fps (equivalence): {:.4f}'.format(1. / reg_mean))

    pass



def _draw_grid(im_cv, grid_size: int = 24):
    im_gd_cv = np.full_like(im_cv, 255.0)
    im_gd_cv = cv2.cvtColor(im_gd_cv, cv2.COLOR_GRAY2BGR)

    height, width = im_cv.shape
    color = (0, 0, 255)
    for x in range(0, width - 1, grid_size):
        cv2.line(im_gd_cv, (x, 0), (x, height), color, 1, 1) # (0, 0, 0)
    for y in range(0, height - 1, grid_size):
        cv2.line(im_gd_cv, (0, y), (width, y), color, 1, 1)
    im_gd_ts = kornia.utils.image_to_tensor(im_gd_cv / 255.).type(torch.FloatTensor).cuda()
    return im_gd_ts

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

def save_flow(flow: [Tensor], dst: pathlib.Path, im_name: str = ''):
    rgb_flow = flow2rgb(flow, max_value=None) # (3, 512, 512) type; numpy.ndarray
    im_s = rgb_flow if type(rgb_flow) == list else [rgb_flow]
    dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
    for im_ts, p in zip(im_s, dst):
        p.parent.mkdir(parents=True, exist_ok=True)
        im_cv = (im_ts * 255).astype(np.uint8).transpose(1, 2, 0)
        cv2.imwrite(str(p), im_cv)

def flow2rgb(flow_map: [Tensor], max_value: None):
    flow_map_np = flow_map.squeeze().detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    rgb_flow = rgb_map.clip(0, 1)
    return rgb_flow

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = hyper_args()
    main(args)