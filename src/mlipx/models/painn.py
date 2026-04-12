"""PaiNN-style message passing network for atomistic potentials."""
from __future__ import annotations

import torch
import torch.nn as nn

from mlipx.data.types import GraphBatch

from .layers import GaussianRBF, scatter_add


class PaiNNInteraction(nn.Module):
    def __init__(self, hidden_dim: int, n_rbf: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )
        self.scalar_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_dist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        filters = self.filter_net(edge_rbf)
        f_q, f_mu_r, f_mu_v = torch.chunk(filters, chunks=3, dim=-1)

        q_msg = q[dst] * f_q
        dq = scatter_add(q_msg, src, q.shape[0])

        unit = edge_vec / edge_dist.unsqueeze(-1).clamp(min=1e-8)
        mu_dst = mu[dst]
        radial = f_mu_r.unsqueeze(-1) * unit.unsqueeze(1)
        mu_msg = f_mu_v.unsqueeze(-1) * mu_dst + radial
        dmu = torch.zeros_like(mu)
        dmu.index_add_(0, src, mu_msg)

        q = q + self.scalar_update(dq)
        mu = mu + dmu
        return q, mu


class PaiNN(nn.Module):
    def __init__(self, n_atom_embeddings: int, hidden_dim: int, n_interactions: int, n_rbf: int, cutoff: float):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_embeddings, hidden_dim)
        self.rbf = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
        self.interactions = nn.ModuleList([PaiNNInteraction(hidden_dim=hidden_dim, n_rbf=n_rbf) for _ in range(n_interactions)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        q = self.embedding(batch.z)
        mu = torch.zeros((q.shape[0], q.shape[1], 3), device=q.device, dtype=q.dtype)

        edge_rbf = self.rbf(batch.edge_dist)
        for block in self.interactions:
            q, mu = block(q, mu, batch.edge_index, edge_rbf, batch.edge_vec, batch.edge_dist)

        atom_e = self.readout(q).squeeze(-1)
        n_struct = int(batch.batch.max().item()) + 1
        total_e = torch.zeros((n_struct,), device=q.device, dtype=q.dtype)
        total_e.index_add_(0, batch.batch, atom_e)
        return total_e.unsqueeze(-1)
