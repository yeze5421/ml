import torch
import torch.nn as nn

from .config import MLIPConfig
from .descriptors import atomic_descriptors


class AtomicNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        atomic_e = self.net(x)
        return torch.sum(atomic_e, dim=1)


class TinyMLIP(nn.Module):
    def __init__(self, cfg: MLIPConfig):
        super().__init__()
        self.cfg = cfg
        self.atomic_net = AtomicNetwork(in_dim=len(cfg.radial_centers) + 1, hidden_dim=cfg.hidden_dim)

    def descriptors(self, positions: torch.Tensor) -> torch.Tensor:
        features = []
        for b in range(positions.shape[0]):
            feat_b = atomic_descriptors(
                positions[b],
                cutoff=self.cfg.cutoff,
                radial_centers=self.cfg.radial_centers,
                radial_width=self.cfg.radial_width,
            )
            features.append(feat_b)
        return torch.stack(features, dim=0)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.atomic_net(self.descriptors(positions))

    def predict_energy_and_forces(self, positions: torch.Tensor):
        positions = positions.clone().detach().requires_grad_(True)
        energy = self.forward(positions)
        forces = -torch.autograd.grad(energy.sum(), positions)[0]
        return energy, forces
