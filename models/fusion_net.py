import torch
import torch.nn as nn
import torch.nn.functional as F


class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out

class FuseModule(nn.Module):
  """ Interactive fusion module"""
  def __init__(self, in_dim=64):
    super(FuseModule, self).__init__()
    self.chanel_in = in_dim

    self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
    self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

    self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
    self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
    self.sig = nn.Sigmoid()

  def forward(self, x, prior):
    x_q = self.query_conv(x)
    prior_k = self.key_conv(prior)
    energy = x_q * prior_k
    attention = self.sig(energy)
    attention_x = x * attention
    attention_p = prior * attention

    x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
    x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

    p_gamma = self.gamma2(torch.cat((prior, attention_p), dim=1))
    prior_out = prior * p_gamma[:, [0], :, :] + attention_p * p_gamma[:, [1], :, :]

    return x_out, prior_out

class FusionNet(nn.Module):
  def __init__(self, nfeats=64):
    super(FusionNet, self).__init__()

    # head
    self.conv1_1 = nn.Sequential(
      nn.Conv2d(1, nfeats, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.2)
    )
    self.conv2_1 = nn.Sequential(
      nn.Conv2d(1, nfeats, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.2)
    )

    # body-densenet
    self.nChannels = nfeats
    self.nDenselayer = 3
    self.growthRate = nfeats
    Ir_path = []
    Vi_path = []
    for i in range(1):
      Ir_path.append(RDB(self.nChannels, self.nDenselayer, self.growthRate))
      Vi_path.append(RDB(self.nChannels, self.nDenselayer, self.growthRate))
    self.ir_path = nn.Sequential(*Ir_path)
    self.vi_path = nn.Sequential(*Vi_path)

    # body-fuse
    self.fuse = FuseModule()
    self.fuse_res = nn.Conv2d(nfeats * 2, nfeats, kernel_size=3, stride=1, padding=1)

    # tail
    self.out_conv = nn.Conv2d(nfeats, 1, kernel_size=3, stride=1, padding=1)
    self.act = nn.Tanh()


  def forward(self, ir, vi):
    # head
    ir_feat = self.conv1_1(ir)
    vi_feat = self.conv2_1(vi)

    # body-densenet
    ir_dfeats = self.ir_path(ir_feat)
    vi_dfeats = self.vi_path(vi_feat)

    # body-fuse
    fuse_feat_ir, fuse_feat_vi = self.fuse(ir_dfeats, vi_dfeats)
    fuse_feats = self.fuse_res(torch.cat((fuse_feat_ir, fuse_feat_vi), dim=1))

    # body-concat
    # fuse_feats = self.fuse_res(torch.cat((ir_dfeats, vi_dfeats), dim=1))

    # tail
    out = self.out_conv(fuse_feats)
    out = self.act(out)

    return out

def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()

if __name__ == '__main__':

    model = FusionNet(64).cuda()
    a = torch.randn(1, 1, 64, 64).cuda()
    b = model(a,a)
    print(b.shape)
    model =FusionNet(64).cuda()
    model.eval()
    print("Params(M): %.2f" % (params_count(model) / (1000 ** 2)))
    import time
    x = torch.Tensor(1, 1, 64, 64).cuda()

    N = 10
    with torch.no_grad():
        for _ in range(N):
            out = model(x, x)

        result = []
        for _ in range(N):
            torch.cuda.synchronize()
            st = time.time()
            for _ in range(N):
                out = model(x, x)
            torch.cuda.synchronize()
            result.append((time.time() - st)/N)
        print("Running Time: {:.3f}s\n".format(np.mean(result)))


