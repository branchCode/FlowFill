import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.conv2d import Conv2d, Conv2dZeros, GatedConv2d


class AffineCouplingAndInjecting(nn.Module):
    def __init__(self, n_channels, n_hidden_layers=1, injecting=True, hidden_channels=64, clamp=1.5):
        super().__init__()
        self.clamp = clamp
        self.in_channels = n_channels
        self.injecting = injecting
        self.channels_cond_ft = 518
        self.kernel_hidden = 1
        self.n_hidden_layers = n_hidden_layers
        self.hidden_channels = hidden_channels
        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        self.conv = nn.Conv2d(self.channels_cond_ft, self.channels_cond_ft, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.injecting_net = self.sub_net(in_channels=self.channels_cond_ft,
                                        out_channels=self.in_channels * 2,
                                        hidden_channels=self.hidden_channels,
                                        kernel_hidden=self.kernel_hidden,
                                        n_hidden_layers=self.n_hidden_layers)

        self.coupling_net = self.sub_net(in_channels=self.channels_for_nn + self.channels_cond_ft,
                                    out_channels=self.channels_for_co * 2,
                                    hidden_channels=self.hidden_channels,
                                    kernel_hidden=self.kernel_hidden,
                                    n_hidden_layers=self.n_hidden_layers)

    def forward(self, x, conditional, rev=False):
        assert x.size(1) == self.in_channels

        ft = self.lrelu(self.conv(F.interpolate(conditional, size=x.size(2), mode='bilinear')))

        if not rev:
            # Feature conditional
            if self.injecting:
                scale_ft, shift_ft, _ = self._feature_extract(ft)
                x = (x + shift_ft).mul(torch.exp(scale_ft))

            # Coupling conditional
            x1, x2 = self._split(x, type='split')
            scale, shift = self._feature_extract_aff(x1, ft)
            x2 = (x2 + shift).mul(torch.exp(scale))
            x = torch.cat([x1, x2], dim=1)
        else:
            x1, x2 = self._split(x, type='split')
            scale, shift = self._feature_extract_aff(x1, ft)
            x2 = x2.div(torch.exp(scale)) - shift
            x = torch.cat([x1, x2], dim=1)
            
            if self.injecting:
                scale_ft, shift_ft, _ = self._feature_extract(ft)
                x = x.div(torch.exp(scale_ft)) - shift_ft

        self.logdet = scale.view(scale.size(0), -1).sum(dim=-1)
        if self.injecting:
            self.logdet  += scale_ft.view(scale_ft.size(0), -1).sum(dim=-1)

        return x

    def jacobian(self, rev=False):
        if not rev:
            return self.logdet
        else:
            return -self.logdet

    def _feature_extract(self, ft):
        injecting_ft = self.injecting_net(ft)
        shift, scale = self._split(injecting_ft, type='cross')
        # scale = torch.sigmoid(scale + 2.) + self.affine_eps
        # scale = self.clamp * (torch.sigmoid(scale) * 2 - 1)
        scale = self.clamp * 0.636 * torch.atan(scale / self.clamp)

        return scale, shift, injecting_ft

    def _feature_extract_aff(self, x, ft):
        cond = torch.cat([x, ft], dim=1)
        shift, scale = self._split(self.coupling_net(cond), type='cross')
        # scale = self.clamp * (torch.sigmoid(scale) * 2 - 1)
        scale = self.clamp * 0.636 * torch.atan(scale / self.clamp)

        return scale, shift
        
    def sub_net(self, in_channels, out_channels, hidden_channels=64, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
            
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)

    def _split(self, x, type='split'):
        if type == 'split':
            x1, x2 = x.chunk(2, dim=1)
        elif type == 'cross':
            x1, x2 = x[:, 0::2, ...], x[:, 1::2, ...]
        
        return x1, x2
