from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class MLIPConfig:
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"

    # dataset
    n_atoms: int = 5
    box_size: float = 4.0
    n_samples: int = 600
    train_ratio: float = 0.8
    min_dist: float = 0.85

    # physics labels
    cutoff: float = 3.0
    lj_epsilon: float = 0.5
    lj_sigma: float = 1.0

    # descriptor + model
    radial_centers: Tuple[float, ...] = (0.4, 0.7, 1.0, 1.3, 1.6)
    radial_width: float = 0.25
    hidden_dim: int = 64

    # optimization
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 50

    # output
    out_dir: Path = field(default_factory=lambda: Path("artifacts"))
    checkpoint_name: str = "tiny_mlip.pt"

    @property
    def checkpoint_path(self) -> Path:
        return self.out_dir / self.checkpoint_name
