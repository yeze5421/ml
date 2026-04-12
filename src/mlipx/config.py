"""Configuration dataclasses and YAML loading utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    path: str
    format: str = "extxyz"
    energy_key: str = "energy"
    force_key: str = "forces"
    cutoff: float = 5.0
    max_neighbors: int = 64
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    batch_size: int = 8
    num_workers: int = 0
    normalize_energy_per_atom: bool = True


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    n_interactions: int = 4
    n_rbf: int = 32
    cutoff: float = 5.0


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "auto"
    epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 1e-6
    energy_weight: float = 1.0
    force_weight: float = 50.0
    grad_clip: float = 5.0
    amp: bool = False
    scheduler: str = "cosine"
    scheduler_tmax: int = 20
    early_stopping_patience: int = 10
    output_dir: str = "artifacts/run_default"
    resume_checkpoint: str | None = None


@dataclass
class ExperimentConfig:
    name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def _build_cfg(raw: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        name=raw.get("name", "mlipx"),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _build_cfg(raw)
