from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mlipx.data.types import GraphBatch
from mlipx.models.painn import PaiNN


@dataclass
class EnergyScaler:
    mean: float
    std: float


class MLIPEnergyModel(nn.Module):
    def __init__(self, hidden_dim: int, n_interactions: int, n_rbf: int, cutoff: float, scaler: EnergyScaler):
        super().__init__()
        self.core = PaiNN(
            n_atom_embeddings=100,
            hidden_dim=hidden_dim,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
        )
        self.scaler = scaler

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        return self.core(batch)

    def predict_energy_forces(self, batch: GraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        pos = batch.pos.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            batch = GraphBatch(
                z=batch.z,
                pos=pos,
                batch=batch.batch,
                edge_index=batch.edge_index,
                edge_vec=(pos[batch.edge_index[1]] - pos[batch.edge_index[0]]),
                edge_dist=torch.norm(pos[batch.edge_index[1]] - pos[batch.edge_index[0]], dim=-1),
                energy=batch.energy,
                forces=batch.forces,
                natoms=batch.natoms,
            )
            en = self.forward(batch)
            forces = -torch.autograd.grad(en.sum(), pos, create_graph=False)[0]
        return en, forces
