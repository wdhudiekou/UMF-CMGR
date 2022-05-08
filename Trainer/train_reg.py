import sys

sys.path.append("..")

import visdom
import pathlib
import warnings
import argparse, os

import numpy
import torch.backends.cudnn
import torch.utils.data
import torchvision.transforms

from tqdm import tqdm
import torch.nn.functional
from functions.affine_transform import AffineTransform
from functions.elastic_transform import ElasticTransform
from dataloader.reg_data import RegData
from models.deformable_net import DeformableNet
from loss.reg_losses import LossFunction_Dense


def hyper_args():
# Training settings
    parser = argparse.ArgumentParser(description="PyTorch Corss-modality Registration")
    # dataset
    parser.add_argument('--ir', default='../dataset/raw/ctrain/Road/ir', type=pathlib.Path)
    parser.add_argument('--it', default='../dataset/raw/ctrain/Road/it_edge', type=pathlib.Path)
    parser.add_argument("--batchsize", type=int, default=16, help="training batch size")
    parser.add_argument("--nEpochs", type=int, default=800, help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate. Default=1e-4")
    parser.add_argument("--step", type=int, default=1200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument("--pretrained", default="../cache/Reg_only/220506_Deformable_2*Fe_10*Grad/cp_0800.pth", type=str, help="path to pretrained model (default: none)")
    parser.add_argument("--ckpt", default="../cache/Reg_only/220507_Deformable_2*Fe_10*Grad", type=str, help="path to pretrained model (default: none)")
    args = parser.parse_args()
    return args


def main(args, visdom):

    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    print("===> Creating Save Path of Checkpoints")
    cache = pathlib.Path(args.ckpt)

    print("===> Loading datasets")
    crop = torchvision.transforms.RandomResizedCrop(256)
    data = RegData(args.ir, args.it, crop)
    training_data_loader = torch.utils.data.DataLoader(data, args.batchsize, True, pin_memory=True)

    print("===> Building model")
    net = DeformableNet().cuda()
    criterion = LossFunction_Dense().cuda()

    print("===> Setting Optimizer")
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)

    print("===> Building deformation")
    affine = AffineTransform(translate=0.01)
    elastic = ElasticTransform(kernel_size=101, sigma=16)

    # TODO: optionally copy weights from a checkpoint
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading model '{}'".format(args.pretrained))
            model_state_dict = torch.load(args.pretrained)
            net.load_state_dict(model_state_dict)
        else:
            print("=> no model found at '{}'".format(args.pretrained))

    print("===> Training")
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        train(training_data_loader, optimizer, net, criterion, epoch, elastic, affine)
        if epoch % 20 == 0:
            save_checkpoint(net, epoch, cache)

def train(training_data_loader, optimizer, net, criterion, epoch, elastic, affine):

    net.train()
    tqdm_loader = tqdm(training_data_loader, disable=True)

    # TODO: update learning rate of the optimizer
    lr = adjust_learning_rate(optimizer, epoch - 1)
    print("Epoch={}, lr={}".format(epoch, lr))

    loss_rec = []
    loss_pos = []
    loss_neg = []
    loss_grad = []

    for (ir, it), (ir_path, it_path) in tqdm_loader:
        ir = ir.cuda() # torch.Size([16, 1, 256, 256])
        it = it.cuda()

        ir_affine, affine_disp = affine(ir)
        ir_elastic, elastic_disp = elastic(ir_affine)
        disp = affine_disp + elastic_disp
        ir_warp = ir_elastic

        ir_warp.detach_()
        disp.detach_()

        ir_pred, f_warp, flow, int_flow1, int_flow2, disp_pre = net(it, ir_warp)
        loss1, loss2, grad_loss, loss = _warp_Dense_loss_unsupervised(criterion, ir_pred, f_warp, it, ir_warp, flow,
                                                                      int_flow1, int_flow2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if tqdm_loader.n % 20 == 0:
            show = torch.stack([it[0], ir_warp[0], ir_pred[0], ir[0]])
            visdom.images(show, win='var')

        loss_rec.append(loss.item())
        loss_pos.append(loss1.item())
        loss_neg.append(loss2.item())
        loss_grad.append(grad_loss.item())

    loss_avg = numpy.mean(loss_rec)
    loss_pos = numpy.mean(loss_pos)
    loss_neg = numpy.mean(loss_neg)
    loss_grad = numpy.mean(loss_grad)

    visdom.line([loss_avg], [epoch], win='loss', name='Total-loss', opts=dict(title='Total-loss'), update='append' if epoch else '')
    visdom.line([loss_pos], [epoch], win='loss_1', name='L1-loss', opts=dict(title='Feats-loss'), update='append' if epoch else '')
    visdom.line([loss_neg], [epoch], win='loss_2', name='L2-loss', opts=dict(title='Pixel-loss'), update='append' if epoch else '')
    visdom.line([loss_grad], [epoch], win='loss_grad', name='Grad-loss', opts=dict(title='Grad-loss'), update='append' if epoch else '')

def adjust_learning_rate(optimizer, epoch):
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
    visdom = visdom.Visdom(port=8097, env='Reg')

    main(args, visdom)
