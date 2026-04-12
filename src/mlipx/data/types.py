from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GraphBatch:
    z: torch.Tensor
    pos: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    edge_vec: torch.Tensor
    edge_dist: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor
    natoms: torch.Tensor
