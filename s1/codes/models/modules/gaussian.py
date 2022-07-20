import torch
import torch.nn as nn
import numpy as np



class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        if mean is None and logs is None:
            return -0.5 * (x ** 2 + GaussianDiag.Log2PI)
        else:
            return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return likelihood.view(likelihood.size(0), -1).sum(dim=-1)

    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean), std=torch.ones_like(logs) * eps_std)
        return mean + torch.exp(logs) * eps

    @staticmethod
    def sample_eps(shape, eps_std=0.7, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        eps = torch.normal(mean=torch.zeros(shape), std=torch.ones(shape) * eps_std)
        return eps