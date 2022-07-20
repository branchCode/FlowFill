import torch
import torch.nn as nn

from models.modules.gaussian import GaussianDiag
from models.modules.flow_step import FlowStep
from models.modules.conv2d import Conv2dZeros
from models.modules.squeeze import HaarDownsampling, SqueezeLayer, Invertible2x2ConvLayer


class FlowBlock(nn.Module):
    def __init__(self, in_channels, K, n_trans=0, split=True, heat=0.7):
        super().__init__()
        
        self.heat = heat

        self.squeeze = HaarDownsampling(in_channels)
        # self.squeeze = Invertible2x2ConvLayer(in_channels)
        # self.squeeze = SqueezeLayer()
        self.flow_steps = nn.ModuleList()
        for i in range(K + n_trans):
            if i < n_trans:
                self.flow_steps.append(FlowStep(in_channels * 4, affine=False))
            else:
                self.flow_steps.append(FlowStep(in_channels * 4))

        self.split = split
        if split:
            self.prior = Conv2dZeros(in_channels * 2, in_channels * 4)


    def forward(self, x, logdet, conditional, rev=False):
        if not rev:
            # 1. Squeeze
            x = self.squeeze(x)
            logdet = logdet + self.squeeze.jacobian(rev)

            # 2. Transition steps and K flow steps
            for flow_step in self.flow_steps:
                x, logdet = flow_step(x, logdet=logdet, conditional=conditional)
            
            # 3. Split or not
            out, log_p, z_new = self._split(x)

            return out, logdet, log_p, z_new
        else:
            z, log_p = self._split(x, rev=True)

            for flow_step in reversed(self.flow_steps):
                z, logdet = flow_step(z, logdet=logdet, conditional=conditional, rev=True)

            z = self.squeeze(z, rev=True)
            logdet = logdet + self.squeeze.jacobian(rev)

            return z, logdet, log_p

    def _split(self, x, rev=False):
        if not rev:
            if self.split:
                out, z_new = x.chunk(2, dim=1)
                mean, log_std = self.prior(out).chunk(2, dim=1)
                log_p = GaussianDiag.logp(mean, log_std, z_new)
            else:
                out = x
                z_new = x
                log_p = GaussianDiag.logp(None, None, z_new)

            return out, log_p, z_new
        else:
            if self.split:
                z1 = x
                mean, log_std = self.prior(z1).chunk(2, dim=1)
                # eps = Gaussiandiag.sample_eps(mean.shape, eps_std=self.heat)
                # z2 = mean + log_std.exp() * eps
                z2 = GaussianDiag.sample(mean, log_std, eps_std=self.heat)

                z = torch.cat([z1, z2], dim=1)
                log_p = GaussianDiag.logp(mean, log_std, z2)
            else:
                z = x
                log_p = GaussianDiag.logp(None, None, z)

            return z, log_p