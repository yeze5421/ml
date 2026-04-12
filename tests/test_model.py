import pytest

torch = pytest.importorskip("torch")

from mlip_project.config import MLIPConfig
from mlip_project.model import TinyMLIP


def test_model_energy_force_shapes():
    cfg = MLIPConfig(n_atoms=4, n_samples=8)
    model = TinyMLIP(cfg)
    positions = torch.rand(2, 4, 3)
    energy, forces = model.predict_energy_and_forces(positions)
    assert energy.shape == (2, 1)
    assert forces.shape == (2, 4, 3)
