from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from ase import Atoms
from ase.io import read


@dataclass
class StructureSample:
    z: torch.Tensor
    pos: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor
    cell: torch.Tensor
    pbc: torch.Tensor


class ASEDataset(torch.utils.data.Dataset):
    """ASE extxyz dataset for atomistic ML."""

    def __init__(self, path: str, energy_key: str = "energy", force_key: str = "forces"):
        self.frames: Sequence[Atoms] = read(path, index=":")
        if len(self.frames) == 0:
            raise ValueError(f"No structures found in dataset: {path}")
        self.energy_key = energy_key
        self.force_key = force_key

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> StructureSample:
        atoms = self.frames[idx]
        z = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=torch.float32)

        energy = atoms.info.get(self.energy_key)
        if energy is None:
            try:
                energy = atoms.get_potential_energy()
            except Exception as exc:  # noqa: BLE001
                raise KeyError(f"Missing energy for sample {idx}") from exc

        forces = atoms.arrays.get(self.force_key)
        if forces is None:
            try:
                forces = atoms.get_forces()
            except Exception as exc:  # noqa: BLE001
                raise KeyError(f"Missing forces for sample {idx}") from exc

        cell = torch.tensor(np.asarray(atoms.cell.array), dtype=torch.float32)
        pbc = torch.tensor(np.asarray(atoms.pbc, dtype=np.int64), dtype=torch.bool)
        return StructureSample(
            z=z,
            pos=pos,
            energy=torch.tensor([float(energy)], dtype=torch.float32),
            forces=torch.tensor(forces, dtype=torch.float32),
            cell=cell,
            pbc=pbc,
        )
