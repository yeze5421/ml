from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float


def mae_rmse(pred: torch.Tensor, target: torch.Tensor) -> RegressionMetrics:
    err = pred - target
    mae = torch.mean(torch.abs(err)).item()
    rmse = math.sqrt(torch.mean(err * err).item())
    return RegressionMetrics(mae=mae, rmse=rmse)
