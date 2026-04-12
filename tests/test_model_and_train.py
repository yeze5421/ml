import pytest
pytest.importorskip("torch")
pytest.importorskip("ase")
pytest.importorskip("numpy")

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import write

from mlipx.config import load_config
from mlipx.data.datamodule import MLIPDataModule
from mlipx.data.dataset import ASEDataset
from mlipx.models.model import EnergyScaler, MLIPEnergyModel
from mlipx.training.engine import Trainer


def _make_dataset(path, n=8):
    frames = []
    for _ in range(n):
        atoms = Atoms("H2", positions=np.random.rand(2, 3), cell=np.eye(3) * 5.0, pbc=False)
        atoms.calc = EMT()
        atoms.info["energy"] = atoms.get_potential_energy()
        atoms.arrays["forces"] = atoms.get_forces()
        frames.append(atoms)
    write(path, frames, format="extxyz")


def test_forward_and_force(tmp_path):
    path = tmp_path / "d.extxyz"
    _make_dataset(path, n=6)
    ds = ASEDataset(str(path))
    dm = MLIPDataModule(ds, 5.0, 16, 2, 0, 0.8, 0.1, 0.1, True, 42)
    batch = next(iter(dm.train_loader()))
    model = MLIPEnergyModel(hidden_dim=32, n_interactions=2, n_rbf=16, cutoff=5.0, scaler=EnergyScaler(dm.stats.energy_mean, dm.stats.energy_std))
    e, f = model.predict_energy_forces(batch)
    assert e.shape[0] == batch.energy.shape[0]
    assert f.shape == batch.forces.shape


def test_smoke_train(tmp_path):
    path = tmp_path / "d2.extxyz"
    _make_dataset(path, n=10)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
name: smoke
data:
  path: {path}
  format: extxyz
  energy_key: energy
  force_key: forces
  cutoff: 5.0
  max_neighbors: 16
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  batch_size: 2
  num_workers: 0
  normalize_energy_per_atom: true
model:
  hidden_dim: 32
  n_interactions: 2
  n_rbf: 16
  cutoff: 5.0
train:
  seed: 42
  device: cpu
  epochs: 1
  lr: 0.001
  weight_decay: 1.0e-6
  energy_weight: 1.0
  force_weight: 5.0
  grad_clip: 5.0
  amp: false
  scheduler: cosine
  scheduler_tmax: 1
  early_stopping_patience: 5
  output_dir: {tmp_path}/artifacts
  resume_checkpoint: null
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    dm = MLIPDataModule(ASEDataset(cfg.data.path), cfg.data.cutoff, cfg.data.max_neighbors, cfg.data.batch_size, cfg.data.num_workers, cfg.data.train_ratio, cfg.data.val_ratio, cfg.data.test_ratio, cfg.data.normalize_energy_per_atom, cfg.train.seed)
    trainer = Trainer(cfg, dm)
    trainer.fit()
