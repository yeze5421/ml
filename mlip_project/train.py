from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from .config import MLIPConfig
from .data import ToyAtomicDataset, collate_fn, seed_everything, split_dataset
from .model import TinyMLIP


def build_device_and_dtype(cfg: MLIPConfig):
    device = torch.device(cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = getattr(torch, cfg.dtype)
    return device, dtype


def train(cfg: MLIPConfig):
    seed_everything(cfg.seed)
    device, dtype = build_device_and_dtype(cfg)

    dataset = ToyAtomicDataset(cfg)
    train_set, val_set = split_dataset(dataset, cfg.train_ratio)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model = TinyMLIP(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for positions, energies, _ in train_loader:
            positions = positions.to(device=device, dtype=dtype)
            energies = energies.to(device=device, dtype=dtype)
            pred = model(positions)
            loss = mse(pred, energies)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * positions.shape[0]

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss, val_mae = 0.0, 0.0
        with torch.no_grad():
            for positions, energies, _ in val_loader:
                positions = positions.to(device=device, dtype=dtype)
                energies = energies.to(device=device, dtype=dtype)
                pred = model(positions)
                loss = mse(pred, energies)
                val_loss += loss.item() * positions.shape[0]
                val_mae += torch.sum(torch.abs(pred - energies)).item()

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae})

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_mae={val_mae:.6f}")

    save_checkpoint(model, cfg, history)
    return model, val_set, history


def save_checkpoint(model: TinyMLIP, cfg: MLIPConfig, history):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(cfg),
        "history": history,
    }
    torch.save(payload, cfg.checkpoint_path)
    print(f"Saved checkpoint to: {cfg.checkpoint_path}")


def load_checkpoint(path: Path, map_location: str = "cpu"):
    return torch.load(path, map_location=map_location)
