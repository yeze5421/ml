from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from mlipx.config import ExperimentConfig
from mlipx.data.datamodule import MLIPDataModule
from mlipx.data.types import GraphBatch
from mlipx.models.model import EnergyScaler, MLIPEnergyModel
from mlipx.utils.metrics import mae_rmse


class Trainer:
    def __init__(self, cfg: ExperimentConfig, datamodule: MLIPDataModule):
        self.cfg = cfg
        self.dm = datamodule
        self.device = torch.device("cuda" if (cfg.train.device == "auto" and torch.cuda.is_available()) else cfg.train.device)
        scaler = EnergyScaler(mean=self.dm.stats.energy_mean, std=self.dm.stats.energy_std)
        self.model = MLIPEnergyModel(
            hidden_dim=cfg.model.hidden_dim,
            n_interactions=cfg.model.n_interactions,
            n_rbf=cfg.model.n_rbf,
            cutoff=cfg.model.cutoff,
            scaler=scaler,
        ).to(self.device)
        self.opt = AdamW(self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg.train.scheduler_tmax)
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.out_dir = Path(cfg.train.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_val = float("inf")
        self.bad_epochs = 0
        self.start_epoch = 1

        if cfg.train.resume_checkpoint:
            self._load_checkpoint(cfg.train.resume_checkpoint)

    def _to_device(self, b: GraphBatch) -> GraphBatch:
        return GraphBatch(
            z=b.z.to(self.device),
            pos=b.pos.to(self.device),
            batch=b.batch.to(self.device),
            edge_index=b.edge_index.to(self.device),
            edge_vec=b.edge_vec.to(self.device),
            edge_dist=b.edge_dist.to(self.device),
            energy=b.energy.to(self.device),
            forces=b.forces.to(self.device),
            natoms=b.natoms.to(self.device),
        )

    def _compute_loss(self, batch: GraphBatch) -> tuple[torch.Tensor, dict[str, float]]:
        pred_energy_norm, pred_forces = self.model.predict_energy_forces(batch)
        loss_e = F.mse_loss(pred_energy_norm, batch.energy)
        pred_forces_norm = pred_forces / self.dm.stats.energy_std
        target_forces_norm = batch.forces / self.dm.stats.energy_std
        loss_f = F.mse_loss(pred_forces_norm, target_forces_norm)
        total = self.cfg.train.energy_weight * loss_e + self.cfg.train.force_weight * loss_f

        pred_e_phys = pred_energy_norm * self.dm.stats.energy_std + self.dm.stats.energy_mean
        target_e_phys = batch.energy * self.dm.stats.energy_std + self.dm.stats.energy_mean
        e_metrics = mae_rmse(pred_e_phys, target_e_phys)
        f_metrics = mae_rmse(pred_forces, batch.forces)
        logs = {
            "loss": float(total.item()),
            "e_mae": e_metrics.mae,
            "e_rmse": e_metrics.rmse,
            "f_mae": f_metrics.mae,
            "f_rmse": f_metrics.rmse,
        }
        return total, logs

    def _run_epoch(self, loader, train: bool) -> dict[str, float]:
        self.model.train(train)
        totals = {"loss": 0.0, "e_mae": 0.0, "e_rmse": 0.0, "f_mae": 0.0, "f_rmse": 0.0}
        n_batches = 0
        for b in loader:
            batch = self._to_device(b)
            if train:
                self.opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    loss, logs = self._compute_loss(batch)
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                self.grad_scaler.step(self.opt)
                self.grad_scaler.update()
            else:
                loss, logs = self._compute_loss(batch)
            for k in totals:
                totals[k] += logs[k]
            n_batches += 1
        for k in totals:
            totals[k] /= max(1, n_batches)
        return totals

    def _save_checkpoint(self, epoch: int, is_best: bool):
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": asdict(self.cfg),
            "stats": asdict(self.dm.stats),
        }
        torch.save(ckpt, self.out_dir / "last.pt")
        if is_best:
            torch.save(ckpt, self.out_dir / "best.pt")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = int(ckpt["epoch"]) + 1

    def fit(self) -> None:
        history = []
        for epoch in range(self.start_epoch, self.cfg.train.epochs + 1):
            train_logs = self._run_epoch(self.dm.train_loader(), train=True)
            val_logs = self._run_epoch(self.dm.val_loader(), train=False)
            self.scheduler.step()
            is_best = val_logs["loss"] < self.best_val
            if is_best:
                self.best_val = val_logs["loss"]
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            self._save_checkpoint(epoch, is_best=is_best)
            row = {"epoch": epoch, "train": train_logs, "val": val_logs}
            history.append(row)
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_logs['loss']:.4f} val_loss={val_logs['loss']:.4f} | "
                f"E_MAE={val_logs['e_mae']:.4f} E_RMSE={val_logs['e_rmse']:.4f} | "
                f"F_MAE={val_logs['f_mae']:.4f} F_RMSE={val_logs['f_rmse']:.4f}"
            )
            if self.bad_epochs >= self.cfg.train.early_stopping_patience:
                print("Early stopping triggered")
                break

        with (self.out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
