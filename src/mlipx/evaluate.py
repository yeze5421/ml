from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from mlipx.config import load_config
from mlipx.data.datamodule import MLIPDataModule
from mlipx.data.dataset import ASEDataset
from mlipx.models.model import EnergyScaler, MLIPEnergyModel
from mlipx.utils.metrics import mae_rmse


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained MLIP model")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--out", default="artifacts/predictions.csv")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
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
    ckpt = torch.load(args.checkpoint, map_location=device)
    scaler = EnergyScaler(mean=dm.stats.energy_mean, std=dm.stats.energy_std)
    model = MLIPEnergyModel(
        hidden_dim=cfg.model.hidden_dim,
        n_interactions=cfg.model.n_interactions,
        n_rbf=cfg.model.n_rbf,
        cutoff=cfg.model.cutoff,
        scaler=scaler,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = dm.test_loader() if args.split == "test" else dm.val_loader()
    pred_e, true_e, pred_f, true_f = [], [], [], []
    for batch in loader:
        batch = type(batch)(**{k: getattr(batch, k).to(device) for k in batch.__dataclass_fields__.keys()})
        e_pred_n, f_pred = model.predict_energy_forces(batch)
        e_pred = e_pred_n * dm.stats.energy_std + dm.stats.energy_mean
        e_true = batch.energy * dm.stats.energy_std + dm.stats.energy_mean
        pred_e.append(e_pred.detach().cpu())
        true_e.append(e_true.detach().cpu())
        pred_f.append(f_pred.detach().cpu())
        true_f.append(batch.forces.detach().cpu())

    pred_e_t = torch.cat(pred_e, dim=0)
    true_e_t = torch.cat(true_e, dim=0)
    pred_f_t = torch.cat(pred_f, dim=0)
    true_f_t = torch.cat(true_f, dim=0)

    e_metrics = mae_rmse(pred_e_t, true_e_t)
    f_metrics = mae_rmse(pred_f_t, true_f_t)
    print(f"{args.split} | E_MAE={e_metrics.mae:.6f} E_RMSE={e_metrics.rmse:.6f} | F_MAE={f_metrics.mae:.6f} F_RMSE={f_metrics.rmse:.6f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true_energy", "pred_energy"])
        for i, (t, p) in enumerate(zip(true_e_t.squeeze(-1).tolist(), pred_e_t.squeeze(-1).tolist())):
            w.writerow([i, t, p])


if __name__ == "__main__":
    main()
