from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from mlipx.config import ExperimentConfig
from mlipx.data.datamodule import MLIPDataModule
from mlipx.data.dataset import ASEDataset, StructureSample
from mlipx.models.model import EnergyScaler, MLIPEnergyModel


@dataclass
class CalculatorContext:
    cfg: ExperimentConfig
    dm: MLIPDataModule
    model: MLIPEnergyModel
    device: torch.device


class MLIPXCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, context: CalculatorContext, **kwargs):
        super().__init__(**kwargs)
        self.ctx = context

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        sample = StructureSample(
            z=torch.tensor(atoms.numbers, dtype=torch.long),
            pos=torch.tensor(atoms.positions, dtype=torch.float32),
            energy=torch.tensor([0.0], dtype=torch.float32),
            forces=torch.zeros((len(atoms), 3), dtype=torch.float32),
            cell=torch.tensor(np.asarray(atoms.cell.array), dtype=torch.float32),
            pbc=torch.tensor(np.asarray(atoms.pbc, dtype=np.int64), dtype=torch.bool),
        )
        batch = self.ctx.dm.collate([sample])
        batch = type(batch)(**{k: getattr(batch, k).to(self.ctx.device) for k in batch.__dataclass_fields__.keys()})
        self.ctx.model.eval()
        en_n, forces = self.ctx.model.predict_energy_forces(batch)
        en = en_n * self.ctx.dm.stats.energy_std + self.ctx.dm.stats.energy_mean
        self.results["energy"] = float(en.item())
        self.results["forces"] = forces.detach().cpu().numpy()


def build_calculator(cfg: ExperimentConfig, checkpoint: str) -> MLIPXCalculator:
    ds = ASEDataset(path=cfg.data.path, energy_key=cfg.data.energy_key, force_key=cfg.data.force_key)
    dm = MLIPDataModule(
        dataset=ds,
        cutoff=cfg.data.cutoff,
        max_neighbors=cfg.data.max_neighbors,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        normalize_energy_per_atom=cfg.data.normalize_energy_per_atom,
        seed=cfg.train.seed,
    )
    device = torch.device("cuda" if (cfg.train.device == "auto" and torch.cuda.is_available()) else cfg.train.device)
    model = MLIPEnergyModel(
        hidden_dim=cfg.model.hidden_dim,
        n_interactions=cfg.model.n_interactions,
        n_rbf=cfg.model.n_rbf,
        cutoff=cfg.model.cutoff,
        scaler=EnergyScaler(mean=dm.stats.energy_mean, std=dm.stats.energy_std),
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    ctx = CalculatorContext(cfg=cfg, dm=dm, model=model, device=device)
    return MLIPXCalculator(context=ctx)
