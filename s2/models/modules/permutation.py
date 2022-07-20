import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.weight.requires_grad = False
        self.w_shape = w_shape

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        pixels = input.size(2) * input.size(3)
        # dlogdet = torch.slogdet(self.weight)[1] * pixels
        dlogdet = 1. * pixels
        if not reverse:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            # weight = self.weight.inverse().view(w_shape[0], w_shape[1], 1, 1)
            weight = self.weight.transpose(0, 1).view(w_shape[0], w_shape[1], 1, 1)
        return weight, dlogdet

    def forward(self, input, rev=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, self.logdet = self.get_weight(input, rev)
        z = F.conv2d(input, weight)
        return z

    def jacobian(self, rev=False):
        if not rev:
            return self.logdet
        else:
            return -self.logdet