from __future__ import annotations

import argparse

from mlipx.config import load_config
from mlipx.data.datamodule import MLIPDataModule
from mlipx.data.dataset import ASEDataset
from mlipx.training.engine import Trainer
from mlipx.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLIP model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr

    set_seed(cfg.train.seed)
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
    trainer = Trainer(cfg=cfg, datamodule=dm)
    trainer.fit()


if __name__ == "__main__":
    main()
