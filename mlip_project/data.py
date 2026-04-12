import random
from typing import List, Tuple

import torch

from .config import MLIPConfig
from .physics import lennard_jones_energy, lennard_jones_forces


class ToyAtomicDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: MLIPConfig):
        self.items: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _ in range(cfg.n_samples):
            pos = sample_structure(cfg.n_atoms, cfg.box_size, cfg.min_dist)
            energy = lennard_jones_energy(pos, cfg.lj_epsilon, cfg.lj_sigma, cfg.cutoff)
            forces = lennard_jones_forces(pos, cfg.lj_epsilon, cfg.lj_sigma, cfg.cutoff)
            self.items.append((pos, energy.unsqueeze(0), forces))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def sample_structure(n_atoms: int, box_size: float, min_dist: float) -> torch.Tensor:
    points: List[torch.Tensor] = []
    trials = 0
    while len(points) < n_atoms and trials < 5000:
        candidate = torch.rand(3) * box_size
        if all(torch.norm(candidate - p) >= min_dist for p in points):
            points.append(candidate)
        trials += 1

    if len(points) < n_atoms:
        raise RuntimeError("Could not sample a valid structure. Reduce min_dist or n_atoms.")
    return torch.stack(points, dim=0)


def split_dataset(dataset: torch.utils.data.Dataset, train_ratio: float):
    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train
    return torch.utils.data.random_split(dataset, [n_train, n_val])


def collate_fn(batch):
    positions = torch.stack([item[0] for item in batch], dim=0)
    energies = torch.stack([item[1] for item in batch], dim=0)
    forces = torch.stack([item[2] for item in batch], dim=0)
    return positions, energies, forces


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
