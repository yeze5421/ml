from __future__ import annotations

from dataclasses import dataclass

import torch

from .dataset import ASEDataset, StructureSample
from .graph import build_radius_graph
from .types import GraphBatch


@dataclass
class NormalizationStats:
    energy_mean: float
    energy_std: float


class MLIPDataModule:
    def __init__(
        self,
        dataset: ASEDataset,
        cutoff: float,
        max_neighbors: int,
        batch_size: int,
        num_workers: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        normalize_energy_per_atom: bool,
        seed: int,
    ):
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("train/val/test ratios must sum to 1.0")
        self.dataset = dataset
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize_energy_per_atom = normalize_energy_per_atom
        self.seed = seed

        n_total = len(dataset)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        generator = torch.Generator().manual_seed(seed)
        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test], generator=generator
        )
        self.stats = self.compute_train_stats()

    def _energy_target(self, sample: StructureSample) -> float:
        e = sample.energy.item()
        if self.normalize_energy_per_atom:
            e /= float(sample.z.numel())
        return e

    def compute_train_stats(self) -> NormalizationStats:
        targets = []
        for idx in self.train_set.indices:
            sample = self.dataset[idx]
            targets.append(self._energy_target(sample))
        mean = float(torch.tensor(targets).mean())
        std = float(torch.tensor(targets).std(unbiased=False).clamp(min=1e-8))
        return NormalizationStats(energy_mean=mean, energy_std=std)

    def collate(self, batch: list[StructureSample]) -> GraphBatch:
        z_list, pos_list, energy_list, forces_list, natoms = [], [], [], [], []
        batch_index = []
        offset = 0
        for i, item in enumerate(batch):
            n = item.z.numel()
            natoms.append(n)
            z_list.append(item.z)
            pos_list.append(item.pos)
            forces_list.append(item.forces)
            et = self._energy_target(item)
            et = (et - self.stats.energy_mean) / self.stats.energy_std
            energy_list.append(torch.tensor([et], dtype=torch.float32))
            batch_index.append(torch.full((n,), i, dtype=torch.long))
            offset += n

        z = torch.cat(z_list, dim=0)
        pos = torch.cat(pos_list, dim=0)
        forces = torch.cat(forces_list, dim=0)
        batch_vec = torch.cat(batch_index, dim=0)
        energy = torch.stack(energy_list, dim=0)
        edge_index, edge_vec, edge_dist = build_radius_graph(
            pos=pos,
            batch=batch_vec,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
        )
        return GraphBatch(
            z=z,
            pos=pos,
            batch=batch_vec,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_dist=edge_dist,
            energy=energy,
            forces=forces,
            natoms=torch.tensor(natoms, dtype=torch.long),
        )

    def _loader(self, split, shuffle: bool) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            split,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )

    def train_loader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_loader(self):
        return self._loader(self.val_set, shuffle=False)

    def test_loader(self):
        return self._loader(self.test_set, shuffle=False)
