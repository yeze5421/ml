import pytest
pytest.importorskip("torch")
pytest.importorskip("ase")
pytest.importorskip("numpy")

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import write

from mlipx.data.datamodule import MLIPDataModule
from mlipx.data.dataset import ASEDataset


def build_tmp_dataset(path: Path, n_samples: int = 6):
    frames = []
    for _ in range(n_samples):
        atoms = Atoms("H2O", positions=np.random.rand(3, 3), cell=np.eye(3) * 6.0, pbc=False)
        atoms.calc = EMT()
        atoms.info["energy"] = atoms.get_potential_energy()
        atoms.arrays["forces"] = atoms.get_forces()
        frames.append(atoms)
    write(path, frames, format="extxyz")


@pytest.mark.parametrize("batch_size", [2])
def test_loader_and_graph(batch_size, tmp_path):
    data_path = tmp_path / "tiny.extxyz"
    build_tmp_dataset(data_path)
    ds = ASEDataset(str(data_path))
    dm = MLIPDataModule(
        dataset=ds,
        cutoff=5.0,
        max_neighbors=16,
        batch_size=batch_size,
        num_workers=0,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        normalize_energy_per_atom=True,
        seed=123,
    )
    batch = next(iter(dm.train_loader()))
    assert batch.z.ndim == 1
    assert batch.pos.shape[1] == 3
    assert batch.edge_index.shape[0] == 2
    assert batch.energy.ndim == 2
