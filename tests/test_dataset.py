import pytest

torch = pytest.importorskip("torch")

from mlip_project.config import MLIPConfig
from mlip_project.data import ToyAtomicDataset


def test_dataset_shapes():
    cfg = MLIPConfig(n_samples=8, n_atoms=4)
    ds = ToyAtomicDataset(cfg)
    positions, energy, forces = ds[0]
    assert positions.shape == (4, 3)
    assert energy.shape == (1,)
    assert forces.shape == (4, 3)
