import torch
import torch.nn as nn

from models.modules.affine import AffineCouplingAndInjecting
from models.modules.normalization import ActNorm2d
from models.modules.permutation import InvertibleConv1x1
from models.modules.squeeze import Invertible2x2ConvLayer


class FlowStep(nn.Module):
    """One step of flow"""

    def __init__(self, n_channels, affine=True, actnorm_scale=1.0):
        super().__init__()

        # Actnorm
        self.actnorm = ActNorm2d(n_channels, actnorm_scale)

        # Permute
        self.invconv = InvertibleConv1x1(n_channels)
        
        # 2 x 2 convolution
        # self.spatial = Invertible2x2ConvLayer(n_channels, keep_shape=True)

        # Affine
        self.affine = affine
        if affine:
            self.affine1 = AffineCouplingAndInjecting(n_channels)
            self.affine2 = AffineCouplingAndInjecting(n_channels)

    def forward(self, x, logdet, conditional=None, rev=False):
        if not rev:
            # 1. actnorm
            x = self.actnorm(x, rev=False)

            # 2. permute
            x = self.invconv(x, rev=False)
            
            # 3. affine
            if self.affine:
                x = self.affine1(x, rev=False, conditional=conditional)
                
            # 4. 2x2 conv
            # x = self.spatial(x, rev=False)
            # Swap x1 and x2 for affine coupling on the other side
            x1, x2 = x.chunk(2, dim=1)
            x = torch.cat([x2, x1], dim=1)
            
            # 5. affine
            if self.affine:
                x = self.affine2(x, rev=False, conditional=conditional)
        else:
            if self.affine:
                x = self.affine2(x, rev=True, conditional=conditional)
            
            x1, x2 = x.chunk(2, dim=1)
            x = torch.cat([x2, x1], dim=1)
            # x = self.spatial(x, rev=True)
            
            if self.affine:
                x = self.affine1(x, rev=True, conditional=conditional)
            
            x = self.invconv(x, rev=True)

            x = self.actnorm(x, rev=True)

        logdet = logdet + self.jacobian(rev)

        return x, logdet

    def jacobian(self, rev=False):
        if self.affine is None:
            return self.actnorm.jacobian(rev) + self.invconv.jacobian(rev) # + self.spatial.jacobian(rev)
        return self.affine1.jacobian(rev) + self.affine2.jacobian(rev) + \
                self.actnorm.jacobian(rev) + self.invconv.jacobian(rev)  # + self.spatial.jacobian(rev)
