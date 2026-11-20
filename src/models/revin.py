"""Reversible Instance Normalisation (RevIN, Kim et al., ICLR 2022) - hand-written.

Each input window's load channel is normalised by ITS OWN mean and standard deviation before the
network sees it; the network's 24 outputs are mapped back with the same statistics (+ a learnable
affine pair). The network therefore learns the SHAPE of the next day relative to the last week,
and level shifts between years (e.g. lower average load in 2017-18 than in 2002-14) no longer push
inputs outside the range seen in training. Uses only the past window -> no leakage.
"""
import torch
from torch import nn


class RevIN(nn.Module):
    def __init__(self, model: nn.Module, eps: float = 1e-5):
        super().__init__()
        self.model, self.eps = model, eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x_past, x_fut):
        load = x_past[..., :1]
        mu = load.mean(dim=1, keepdim=True)                        # [B, 1, 1]
        sd = load.std(dim=1, keepdim=True) + self.eps
        x = torch.cat([(load - mu) / sd * self.gamma + self.beta, x_past[..., 1:]], dim=-1)
        y = self.model(x, x_fut)                                   # [B, H] in normalised units
        y = (y - self.beta) / (self.gamma + self.eps)
        return y * sd[..., 0] + mu[..., 0]                         # back to global scale

    def __getattr__(self, name):                                   # expose inner model helpers
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
