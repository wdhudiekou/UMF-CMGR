import torch
import torch.nn as nn
from models.layers import SpatialTransformer, ResizeTransform, conv_block, predict_flow, conv2D, MatchCost
import numpy as np

shape = (256, 256)

class DeformableNet(nn.Module):
    def __init__(self):
        super(DeformableNet, self).__init__()
        # int_steps = 7   #
        self.inshape = shape

        down_shape2 = [int(d / 4) for d in self.inshape] # [64, 64]
        down_shape1 = [int(d / 2) for d in self.inshape] # [128, 128]
        self.spatial_transform_f = SpatialTransformer(volsize=down_shape1)
        self.spatial_transform   = SpatialTransformer(volsize=self.inshape)

        # FeatureLearning/Encoder functions
        dim = 2
        self.enc = nn.ModuleList()
        self.enc.append(conv_block(dim, 1, 16, 2))  # 0 (dim, in_channels, out_channels, stride=1)
        self.enc.append(conv_block(dim, 16, 16, 1))  # 1
        self.enc.append(conv_block(dim, 16, 16, 1))  # 2
        self.enc.append(conv_block(dim, 16, 32, 2))  # 3
        self.enc.append(conv_block(dim, 32, 32, 1))  # 4
        self.enc.append(conv_block(dim, 32, 32, 1))  # 5


        # Dncoder functions
        od = 32 + 1
        self.conv2_0 = conv_block(dim, od, 48, 1) # [48, 32, 16]
        self.enc.append(self.conv2_0)
        self.conv2_1 = conv_block(dim, 48, 32, 1)
        self.enc.append(self.conv2_1)
        self.conv2_2 = conv_block(dim, 32, 16, 1)
        self.enc.append(self.conv2_2)
        self.predict_flow2a = predict_flow(16, 2)
        self.enc.append(self.predict_flow2a)

        self.dc_conv2_0 = conv2D(2, 48, kernel_size=3, stride=1, padding=1, dilation=1) # [48, 48, 32]
        self.enc.append(self.dc_conv2_0)
        self.dc_conv2_1 = conv2D(48, 48, kernel_size=3, stride=1, padding=2, dilation=2)
        self.enc.append(self.dc_conv2_1)
        self.dc_conv2_2 = conv2D(48, 32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.enc.append(self.dc_conv2_2)
        self.predict_flow2b = predict_flow(32, 2)
        self.enc.append(self.predict_flow2b)

        od = 1 + 16 + 16 + 2
        self.conv1_0 = conv_block(dim, od, 48, 1)
        self.enc.append(self.conv1_0)
        self.conv1_1 = conv_block(dim, 48, 32, 1)
        self.enc.append(self.conv1_1)
        self.conv1_2 = conv_block(dim, 32, 16, 1)
        self.enc.append(self.conv1_2)
        self.predict_flow1a = predict_flow(16, 2)
        self.enc.append(self.predict_flow1a)

        self.dc_conv1_0 = conv2D(2, 48, kernel_size=3, stride=1, padding=1, dilation=1)
        self.enc.append(self.dc_conv1_0)
        self.dc_conv1_1 = conv2D(48, 48, kernel_size=3, stride=1, padding=2, dilation=2)
        self.enc.append(self.dc_conv1_1)
        self.dc_conv1_2 = conv2D(48, 32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.enc.append(self.dc_conv1_2)
        self.predict_flow1b = predict_flow(32, 2)
        self.enc.append(self.predict_flow1b)

        self.resize = ResizeTransform(1 / 2, dim)
        # self.integrate2 = VecInt(down_shape2, int_steps)
        # self.integrate1 = VecInt(down_shape1, int_steps)

    def load_state_dict(self, state_dict, strict = False):
        state_dict.pop('spatial_transform.grid')
        state_dict.pop('spatial_transform_f.grid')
        super().load_state_dict(state_dict, strict)

    def forward(self, tgt, src, shape=None):
        if shape is not None:
            down_shape1 = [int(d / 2) for d in shape]
            self.spatial_transform_f = SpatialTransformer(volsize=down_shape1)
            self.spatial_transform   = SpatialTransformer(volsize=shape)
        ##################### Feature extraction #########################
        c11 = self.enc[2](self.enc[1](self.enc[0](src))) # torch.Size([16, 16, 128, 128])
        c21 = self.enc[2](self.enc[1](self.enc[0](tgt))) # torch.Size([16, 16, 128, 128])
        c12 = self.enc[5](self.enc[4](self.enc[3](c11))) # torch.Size([16, 32, 64, 64])
        c22 = self.enc[5](self.enc[4](self.enc[3](c21))) # torch.Size([16, 32, 64, 64])

        ##################### Estimation at scale-2 #######################
        corr2 = MatchCost(c22, c12)    # torch.Size([16,  1, 64, 64])
        x = torch.cat((corr2, c22), 1) # torch.Size([16, 33, 64, 64])
        x = self.conv2_0(x) # torch.Size([16, 48, 64, 64])
        x = self.conv2_1(x) # torch.Size([16, 32, 64, 64])
        x = self.conv2_2(x) # torch.Size([16, 16, 64, 64])
        flow2 = self.predict_flow2a(x) # torch.Size([16, 2, 64, 64]) flow2: flow field
        upfeat2 = self.resize(x) # torch.Size([16, 16, 128, 128])

        x = self.dc_conv2_0(flow2) # torch.Size([16, 48, 64, 64])
        x = self.dc_conv2_1(x) # torch.Size([16, 48, 64, 64])
        x = self.dc_conv2_2(x) # torch.Size([16, 32, 64, 64])

        refine_flow2 = self.predict_flow2b(x) + flow2 # torch.Size([16, 2, 64, 64])
        int_flow2 = refine_flow2
        # int_flow2 = self.integrate2(refine_flow2)
        up_int_flow2 = self.resize(int_flow2) # torch.Size([16, 2, 128, 128])
        features_s_warped, _ = self.spatial_transform_f(c11, up_int_flow2) # torch.Size([16, 16, 128, 128])


        ##################### Estimation at scale-1 #######################
        corr1 = MatchCost(c21, features_s_warped) # torch.Size([16, 1, 128, 128])
        x = torch.cat((corr1, c21, up_int_flow2, upfeat2), 1) # torch.Size([16, 35, 112, 112])
        x = self.conv1_0(x) # torch.Size([16, 48, 128, 128])
        x = self.conv1_1(x) # torch.Size([16, 32, 128, 128])
        x = self.conv1_2(x) # torch.Size([16, 16, 128, 128])
        flow1 = self.predict_flow1a(x) + up_int_flow2 # torch.Size([16, 2, 128, 128])

        x = self.dc_conv1_0(flow1) # torch.Size([16, 48, 128, 128])
        x = self.dc_conv1_1(x) # torch.Size([16, 48, 128, 128])
        x = self.dc_conv1_2(x) # torch.Size([16, 32, 128, 128])
        refine_flow1 = self.predict_flow1b(x) + flow1 # torch.Size([16, 2, 128, 128])
        int_flow1 = refine_flow1
        # int_flow1 = self.integrate1(refine_flow1)

        ##################### Upsample to scale-0 #######################
        flow = self.resize(int_flow1) # torch.Size([16, 2, 256, 256])
        m_warp, disp_pre = self.spatial_transform(src, flow) # torch.Size([16, 1, 256, 256]) torch.Size([16, 256, 256, 2])
        # wd+
        f_warp, _ = self.spatial_transform(tgt, (-flow)) # torch.Size([16, 1, 256, 256]) torch.Size([16, 256, 256, 2])

        return m_warp, f_warp, flow, int_flow1, int_flow2, disp_pre

def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()

if __name__ == '__main__':
    #
    model = DeformableNet().cuda()
    a = torch.randn(2, 1, 256, 256).cuda()
    b = torch.randn(2, 1, 256, 256).cuda()
    m_warp, f_warp, flow, int_flow1, int_flow2, disp_pre = model(a,b)
    print(m_warp.shape, f_warp.shape, flow.shape)
    print(int_flow1.shape, int_flow2.shape)


    # model =DeformableNet().cuda()
    # model.eval()
    # print("Params(M): %.2f" % (params_count(model) / (1000 ** 2)))


    # import time
    # x = torch.Tensor(1, 1, 64, 64).cuda()
    #
    # N = 10
    # with torch.no_grad():
    #     for _ in range(N):
    #         out = model(x, x)
    #
    #     result = []
    #     for _ in range(N):
    #         torch.cuda.synchronize()
    #         st = time.time()
    #         for _ in range(N):
    #             out = model(x, x)
    #         torch.cuda.synchronize()
    #         result.append((time.time() - st)/N)
    #     print("Running Time: {:.3f}s\n".format(np.mean(result)))
