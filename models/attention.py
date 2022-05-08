import torch
import torch.nn as nn
from torch import Tensor


class Attention(nn.Module):
    """
    Predict attention map for weight src image.
    principle: im_source & im_reference -> attention_source
    """

    def __init__(self):
        super(Attention, self).__init__()

        self.conv = nn.Conv2d(2, 1, (3, 3), padding=(1, 1), bias=False)

    def forward(self, source, reference):
        """
        im_source & im_reference -> attention_source
        :return: attention map
        """
        input = torch.cat([source, reference], dim=1)

        input_max, _ = torch.max(input, dim=1, keepdim=True)
        input_avg = torch.mean(input, dim=1, keepdim=True)

        spatial = torch.cat([input_max, input_avg], dim=1)
        attention_map = torch.sigmoid(self.conv(spatial))

        return attention_map
