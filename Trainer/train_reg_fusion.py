import sys

sys.path.append("..")

import visdom
import pathlib
import warnings
import logging.config
import argparse, os

import torch.backends.cudnn
import torch.utils.data
import torchvision.transforms

from tqdm import tqdm
import torch.nn.functional
from functions.affine_transform import AffineTransform
from functions.elastic_transform import ElasticTransform
from dataloader.joint_data import JointTrainData
from models.deformable_net import DeformableNet
from models.fusion_net import FusionNet
from loss.reg_losses import LossFunction_Dense
from loss.fusion_loss import FusionLoss



def hyper_args():
# Training settings
    parser = argparse.ArgumentParser(description="PyTorch Corss-modality Registration")
    # dataset
    parser.add_argument('--ir', default='../dataset/raw/ctrain/Road/ir', type=pathlib.Path)
    parser.add_argument('--vi', default='../dataset/raw/ctrain/Road/vi', type=pathlib.Path)
    parser.add_argument('--it', default='../dataset/raw/ctrain/Road/it_edge', type=pathlib.Path)
    parser.add_argument('--ir_map', default='../dataset/raw/ctrain/Road/ir_map', type=pathlib.Path)
    parser.add_argument('--vi_map', default='../dataset/raw/ctrain/Road/vi_map', type=pathlib.Path)
    # train loss weights
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=20.0, type=float)
    parser.add_argument('--theta', default=5.0, type=float)
    # implement details
    parser.add_argument('--dim', default=64, type=int, help='AFuse feather dim')
    parser.add_argument("--batchsize", type=int, default=8, help="training batch size")
    parser.add_argument("--nEpochs", type=int, default=600, help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate. Default=1e-4")
    parser.add_argument("--step", type=int, default=1000, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--interval', default=20, help='record interval')
    # checkpoint
    parser.add_argument('--load_model_reg', type=str, default='../cache/trs_256/211122_Deformable_Fe_0.2*Fe_10*Grad/cp_0780.pth',
                        help="Location from which any pre-trained model needs to be loaded.")
    parser.add_argument('--load_model_fuse', type=str, default='../cache/Fusion_only/220507_fusion_w_svm/fus_0200.pth',
                        help="Location from which any pre-trained model needs to be loaded.")
    # save path of model
    parser.add_argument("--ckpt", default="../cache/Joint_reg_fusion/220508_cotrain_reg_fus", type=str, help="path to pretrained model (default: none)")

    args = parser.parse_args()
    return args


def main(args, visdom):

    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    log = logging.getLogger()

    epoch    = args.nEpochs
    interval = args.interval

    print("===> Creating Save Path of Checkpoints")
    cache = pathlib.Path(args.ckpt)
    if not os.path.exists(cache):
        os.makedirs(cache)

    print("===> Loading datasets")
    crop = torchvision.transforms.RandomResizedCrop(256)
    data = JointTrainData(args.ir, args.it, args.vi, args.ir_map, args.vi_map, crop)
    training_data_loader = torch.utils.data.DataLoader(data, args.batchsize, True, pin_memory=True)

    print("===> Building models")
    RegNet = DeformableNet().to(device)
    FuseNet = FusionNet(nfeats=args.dim).to(device)

    print("===> Defining Loss fuctions")
    criterion_reg = LossFunction_Dense().to(device)
    criterion_fus = FusionLoss(args.alpha, args.beta, args.theta).to(device)

    print("===> Setting Optimizers")
    optimizer_reg = torch.optim.Adam(params=RegNet.parameters(), lr=args.lr)
    optimizer_fus = torch.optim.Adam(params=FuseNet.parameters(), lr=args.lr)

    print("===> Building deformation")
    affine = AffineTransform(translate=0.01)
    elastic = ElasticTransform(kernel_size=101, sigma=16)

    # TODO: optionally copy weights from a checkpoint
    if args.load_model_reg is not None:
        print('Loading pre-trained RegNet checkpoint %s' % args.load_model_reg)
        log.info(f'Loading pre-trained checkpoint {str(args.load_model_reg)}')
        state = torch.load(str(args.load_model_reg))
        RegNet.load_state_dict(state)
    else:
        print("=> no model found at '{}'".format(args.load_model_reg))

    if args.load_model_fuse is not None:
        print('Loading pre-trained FuseNet checkpoint %s' % args.load_model_fuse)
        log.info(f'Loading pre-trained checkpoint {str(args.load_model_fuse)}')
        state = torch.load(args.load_model_fuse)#['net']
        FuseNet.load_state_dict(state)
    else:
        print("=> no model found at '{}'".format(args.load_model_fuse))

    # TODO: freeze parameter of RegNet
    for param in RegNet.parameters():
        param.requires_grad = False
    for param in FuseNet.parameters():
        param.requires_grad = True

    print("===> Starting Training")
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        tqdm_loader = tqdm(training_data_loader, disable=True)
        total_loss, reg_loss, fus_loss = Joint_train(args, tqdm_loader, optimizer_reg, optimizer_fus, RegNet, FuseNet, criterion_reg, criterion_fus, epoch, elastic, affine)
        dsp = f'epoch: [{epoch}/{args.nEpochs}] loss: {total_loss: .2f}'
        log.info(dsp)
        tqdm_loader.set_description(dsp)
        # TODO: visdom display
        visdom.line([total_loss], [epoch], win='loss-total', name='total', opts=dict(title='Total-loss'), update='append' if epoch else '')
        visdom.line([reg_loss], [epoch], win='loss-reg', name='reg', opts=dict(title='Reg-loss'), update='append' if epoch else '')
        visdom.line([fus_loss], [epoch], win='loss-fus', name='fus', opts=dict(title='Fuse-loss'), update='append' if epoch else '')
        # TODO: save checkpoint
        save_checkpoint(RegNet,  epoch, cache / f'reg_{epoch:04d}.pth') if epoch % interval == 0 else None
        save_checkpoint(FuseNet, epoch, cache / f'fus_{epoch:04d}.pth') if epoch % interval == 0 else None

def Joint_train(args, tqdm_loader, optimizer_reg, optimizer_fus, RegNet, FuseNet, criterion_reg, criterion_fus, epoch, elastic, affine):

    RegNet.train()
    FuseNet.train()
    # TODO: update learning rate of the optimizer
    lr_R = adjust_learning_rate(args, optimizer_reg, epoch - 1)
    lr_F = adjust_learning_rate(args, optimizer_fus, epoch - 1)
    print("Epoch={}, lr_R={}, lr_F={} ".format(epoch, lr_R, lr_F))

    loss_total, loss_reg, loss_fus = [], [], []
    for (ir, it, vi, ir_map, vi_map), _ in tqdm_loader:

        ir, it, vi     = ir.cuda(), it.cuda(), vi.cuda()
        ir_map, vi_map = ir_map.cuda(), vi_map.cuda()

        # TODO: generate warped ir images
        ir_affine, ir_affine_disp   = affine(ir)
        ir_elastic, ir_elastic_disp = elastic(ir_affine)
        disp_ir = ir_affine_disp + ir_elastic_disp  # cumulative disp grid [batch_size, height, weight, 2]
        ir_warp = ir_elastic

        ir_warp.detach_()
        disp_ir.detach_()

        # TODO: train registration
        ir_pred, ir_f_warp, ir_flow, ir_int_flow1, ir_int_flow2, ir_disp_pre = RegNet(it, ir_warp)
        reg_loss, vgg, ncc, grad = criterion_reg(ir_pred, ir_f_warp, it, ir_warp, ir_flow, ir_int_flow1, ir_int_flow2)

        # TODO: train fusion
        fuse_out  = FuseNet(ir_pred, vi)
        fuse_loss = criterion_fus(fuse_out, ir_pred, vi, ir_map, vi_map)
        # TODO: total loss
        loss = 1.0 * reg_loss + 1.0 * fuse_loss

        optimizer_reg.zero_grad()
        optimizer_fus.zero_grad()
        loss.backward()
        optimizer_reg.step()
        optimizer_fus.step()

        if tqdm_loader.n % 40 == 0:
            show = torch.stack([it[0], ir_warp[0], ir_pred[0], vi[0], fuse_out[0]])
            visdom.images(show, win='Reg+Fusion')

        loss_total.append(loss.item())
        loss_reg.append(reg_loss.item())
        loss_fus.append(fuse_loss.item())

    l = len(loss_total)
    return sum(loss_total) / l, sum(loss_reg) / l, sum(loss_fus) / l

def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def _warp_Dense_loss_unsupervised(criterion, im_pre, im_fwarp, im_fix, im_warp, flow, flow1, flow2):
    total_loss, multi, ncc, grad = criterion(im_pre, im_fwarp, im_fix, im_warp, flow, flow1, flow2)

    return multi, ncc, grad, total_loss

def save_checkpoint(net, epoch, cache):
    model_folder = cache
    model_out_path = str(model_folder / f'cp_{epoch:04d}.pth')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(net.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = hyper_args()
    visdom = visdom.Visdom(port=8097, env='Reg+Fusion')

    main(args, visdom)