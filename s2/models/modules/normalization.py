import torch
import torch.nn as nn



class ActNorm2d(nn.Module):
    """Activation normalization"""

    def __init__(self, n_channels, scale=1.):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.log_s = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.initialized = torch.tensor(0, dtype=torch.uint8)
        self.scale = torch.tensor(scale)
        self.n_channels = n_channels

    def _initialize(self, x):
        if not self.training:
            return
        assert x.size(1) == self.n_channels
        with torch.no_grad():
            bias = x.mean(dim=(0, 2, 3), keepdim=True) * -1.0
            vars = ((x + bias) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            log_s = (self.scale / (vars.sqrt() + 1e-6)).log()
            self.bias.data.copy_(bias.data)
            self.log_s.data.copy_(log_s.data)
            self.initialized.fill_(1)

    def forward(self, x, rev=False):
        if self.initialized.item() == 0:
            self._initialize(x)
        assert x.size(1) == self.n_channels

        if not rev:
            x = (x + self.bias) * torch.exp(self.log_s)
        else:
            x = x * torch.exp(-self.log_s) - self.bias

        self.logdet = self.log_s.sum() * x.size(2) * x.size(3)

        return x

    def jacobian(self, rev=False):
        if not rev:
            return self.logdet
        else:
            return -self.logdet