import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
class SqueezeLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, rev=False):
        b, c, h, w = x.size()
        if not rev:
            x = x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.view(b, c * 4, h // 2, w // 2)
        else:
            x = x.view(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).contiguous()
            x = x.view(b, c // 4, h * 2, w * 2)

        return x
"""


class SqueezeLayer(nn.Module):
    '''
    The invertible spatial downsampling used in i-RevNet.
    Each group of four neighboring pixels is reordered into one pixel with four times
    the channels in a checkerboard-like pattern. See i-RevNet, Jacobsen 2018 et al.
    Reference from FrEIA(https://github.com/VLL-HD/FrEIA)
    '''

    def __init__(self):
        super(SqueezeLayer, self).__init__()
        
        self.downsample_kernel = torch.zeros(4, 1, 2, 2)

        self.downsample_kernel[0, 0, 0, 0] = 1
        self.downsample_kernel[1, 0, 0, 1] = 1
        self.downsample_kernel[2, 0, 1, 0] = 1
        self.downsample_kernel[3, 0, 1, 1] = 1

        self.downsample_kernel = nn.Parameter(self.downsample_kernel)
        self.downsample_kernel.requires_grad = False

    def forward(self, x, rev=False):
        channels = x.size(1)
        if not rev:
            output = F.conv2d(x, torch.cat([self.downsample_kernel] * channels, 0), stride=2, groups=channels)
            return output
        else:
            channels //= 4
            output = F.conv_transpose2d(x, torch.cat([self.downsample_kernel] * channels, 0), stride=2, groups=channels)
            return output

    
class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, rev=False):
        return self.last_jac
        
        


class Invertible2x2ConvLayer(nn.Module):
    def __init__(self, in_channels, keep_shape=False):
        super(Invertible2x2ConvLayer, self).__init__()

        self.in_channels = in_channels
        weights = np.linalg.qr(np.random.randn(4, 4))[0]
        self.weights = nn.Parameter(torch.tensor(weights).float())
        
        self.keep_shape = keep_shape
        if keep_shape:
            self.squeeze = SqueezeLayer()
            self.weights.requires_grad = False 

    def forward(self, x, rev=False):
        elements = x.size(2) * x.size(3)
        if not rev:
            w = torch.cat([self.weights.view(4, 1, 2, 2)] * self.in_channels, dim=0)
            x = F.conv2d(x, w, stride=2, groups=self.in_channels)
            if self.keep_shape:
                x = self.squeeze(x, rev=True)
        else:
            if self.keep_shape:
                x = self.squeeze(x, rev=False)
                w = self.weights
            else:
                try:
                    w = self.weights.inverse().transpose(0, 1).contiguous()
                except:
                    # torch.svd may have convergence issues for GPU and CPU.
                    w = self.weights + 1e-4 * self.weights.mean() * torch.rand(4, 4)
                    w = w.inverse().transpose(0, 1).contiguous()
            w = torch.cat([w.view(4, 1, 2, 2)] * self.in_channels, dim=0)
            x = F.conv_transpose2d(x, w, stride=2, groups=self.in_channels)
        
        if self.keep_shape:
            self.logdet = 1. * elements / 4 * self.in_channels
        else:
            try:
                self.logdet = self.weights.slogdet()[1] * elements / 4 * self.in_channels
            except:
                # torch.svd may have convergence issues for GPU and CPU.
                w = self.weights + 1e-4 * self.weights.mean() * torch.rand(4, 4)
                self.logdet = w.slogdet()[1] * elements / 4 * self.in_channels

        return x

    def jacobian(self, rev=False):
        if not rev:
            return self.logdet
        else:
            return -self.logdet

