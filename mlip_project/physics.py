import math

import torch


def pairwise_vectors(positions: torch.Tensor) -> torch.Tensor:
    return positions[:, None, :] - positions[None, :, :]


def pairwise_distances(positions: torch.Tensor) -> torch.Tensor:
    rij = pairwise_vectors(positions)
    return torch.sqrt(torch.sum(rij**2, dim=-1) + 1e-12)


def smooth_cutoff(r: torch.Tensor, rc: float) -> torch.Tensor:
    values = 0.5 * (torch.cos(math.pi * r / rc) + 1.0)
    return torch.where(r < rc, values, torch.zeros_like(r))


def lennard_jones_energy(positions: torch.Tensor, epsilon: float, sigma: float, cutoff: float) -> torch.Tensor:
    d = pairwise_distances(positions)
    mask = torch.triu(torch.ones_like(d), diagonal=1) > 0
    rij = d[mask]
    rij = rij[rij < cutoff]
    sr6 = (sigma / rij) ** 6
    sr12 = sr6**2
    return torch.sum(4.0 * epsilon * (sr12 - sr6))


def lennard_jones_forces(positions: torch.Tensor, epsilon: float, sigma: float, cutoff: float) -> torch.Tensor:
    positions = positions.clone().detach().requires_grad_(True)
    energy = lennard_jones_energy(positions, epsilon=epsilon, sigma=sigma, cutoff=cutoff)
    return -torch.autograd.grad(energy, positions)[0]
