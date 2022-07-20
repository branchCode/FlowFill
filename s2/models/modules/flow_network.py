import torch
import torch.nn as nn

from models.modules.flow_block import FlowBlock
from models.modules.sa_gan import InpaintSANet


class ConditionalFlow(nn.Module):
    """Conditional flow network"""

    def __init__(self, in_channels, L, K, n_trans=0, heat=0.7, size=256):
        super().__init__()
        self.encoder = InpaintSANet()

        # L flow blocks
        self.blocks = nn.ModuleList()
        n_channels = in_channels
        for _ in range(L - 1):
            self.blocks.append(FlowBlock(n_channels, K=K, n_trans=n_trans, heat=heat))
            n_channels *= 2
            size //= 2
        
        self.blocks.append(FlowBlock(n_channels, K=K, n_trans=n_trans, heat=heat, split=False))

        self.z_l_shape = [n_channels * 4, size//2, size//2]

    def out_shapes(self):
        return self.z_l_shape


    def forward(self, x, imgs, masks, rev=False):
        ft, sagen_out = self.encoder(imgs=imgs, masks=masks)

        logdet = 0.
        if not rev:
            log_p_sum = 0.
            z_outs = []
            for block in self.blocks:
                x, logdet, log_p, z_new = block(x, logdet=logdet, conditional=ft)
                z_outs.append(z_new)
                log_p_sum += log_p

            return x, log_p_sum, logdet, z_outs
        else:
            z = x
            for block in reversed(self.blocks):
                z, _, _ = block(z, logdet=logdet, conditional=ft, rev=True)

            return z, sagen_out


