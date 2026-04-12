from __future__ import annotations

import math

import torch
import torch.nn as nn


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size, src.shape[-1]), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf: int, cutoff: float):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.gamma = nn.Parameter(torch.tensor(10.0))
        self.cutoff = cutoff

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.unsqueeze(-1)
        rbf = torch.exp(-torch.abs(self.gamma) * (d - self.centers) ** 2)
        cutoff = 0.5 * (torch.cos(math.pi * distances / self.cutoff) + 1.0)
        cutoff = torch.where(distances < self.cutoff, cutoff, torch.zeros_like(cutoff))
        return rbf * cutoff.unsqueeze(-1)
