from typing import Tuple

import torch

from .physics import pairwise_distances, smooth_cutoff


def atomic_descriptors(
    positions: torch.Tensor,
    cutoff: float,
    radial_centers: Tuple[float, ...],
    radial_width: float,
) -> torch.Tensor:
    d = pairwise_distances(positions)
    n_atoms = d.shape[0]
    eye = torch.eye(n_atoms, device=d.device, dtype=torch.bool)
    feats = []

    for center in radial_centers:
        g = torch.exp(-((d - center) ** 2) / (2.0 * radial_width**2)) * smooth_cutoff(d, cutoff)
        g = torch.where(eye, torch.zeros_like(g), g)
        feats.append(torch.sum(g, dim=1, keepdim=True))

    inv_d = torch.where(eye, torch.zeros_like(d), 1.0 / (d + 1e-6))
    inv_d *= smooth_cutoff(d, cutoff)
    feats.append(torch.sum(inv_d, dim=1, keepdim=True))

    return torch.cat(feats, dim=1)
