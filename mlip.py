import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# Tiny MLIP from scratch (toy version)
# ------------------------------------------------------------
# What this file does:
# 1. Generate a synthetic atomic dataset using Lennard-Jones energy
# 2. Build simple radial descriptors for each atom
# 3. Predict total energy with a small neural network
# 4. Obtain forces by automatic differentiation
#
# This is a teaching demo, not a production MLIP.
# It is intentionally small and readable.
# ============================================================

# ------------------------
# Reproducibility
# ------------------------
random.seed(42)
torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


@dataclass
class Config:
    n_atoms: int = 5
    box_size: float = 4.0
    n_samples: int = 800
    train_ratio: float = 0.8
    cutoff: float = 3.0
    sigma_values: Tuple[float, ...] = (0.4, 0.7, 1.0, 1.3, 1.6)
    hidden_dim: int = 64
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 80
    lj_epsilon: float = 0.5
    lj_sigma: float = 1.0
    min_dist: float = 0.85  # avoid atom overlap


CFG = Config()


# ------------------------
# Physics: Lennard-Jones toy label
# ------------------------
def pairwise_vectors(positions: torch.Tensor) -> torch.Tensor:
    # positions: [N, 3]
    return positions[:, None, :] - positions[None, :, :]


def pairwise_distances(positions: torch.Tensor) -> torch.Tensor:
    rij = pairwise_vectors(positions)
    dij = torch.sqrt(torch.sum(rij**2, dim=-1) + 1e-12)
    return dij


def smooth_cutoff(r: torch.Tensor, rc: float) -> torch.Tensor:
    # cosine cutoff
    x = 0.5 * (torch.cos(math.pi * r / rc) + 1.0)
    return torch.where(r < rc, x, torch.zeros_like(r))


def lennard_jones_energy(positions: torch.Tensor, epsilon: float, sigma: float, cutoff: float) -> torch.Tensor:
    # total energy for one structure
    d = pairwise_distances(positions)
    mask = torch.triu(torch.ones_like(d), diagonal=1) > 0
    rij = d[mask]
    valid = rij < cutoff
    rij = rij[valid]
    sr6 = (sigma / rij) ** 6
    sr12 = sr6**2
    e_pair = 4.0 * epsilon * (sr12 - sr6)
    return torch.sum(e_pair)


def sample_structure(n_atoms: int, box_size: float, min_dist: float) -> torch.Tensor:
    pts: List[torch.Tensor] = []
    max_trials = 5000
    trials = 0
    while len(pts) < n_atoms and trials < max_trials:
        cand = torch.rand(3) * box_size
        if len(pts) == 0:
            pts.append(cand)
        else:
            ok = True
            for p in pts:
                if torch.norm(cand - p) < min_dist:
                    ok = False
                    break
            if ok:
                pts.append(cand)
        trials += 1
    if len(pts) < n_atoms:
        raise RuntimeError("Could not sample a valid structure. Try reducing min_dist.")
    return torch.stack(pts, dim=0)


# ------------------------
# Descriptor: simple radial Gaussian sums
# ------------------------
def atomic_descriptors(positions: torch.Tensor, cutoff: float, sigma_values: Tuple[float, ...]) -> torch.Tensor:
    # output: [N, n_features]
    d = pairwise_distances(positions)  # [N, N]
    n_atoms = d.shape[0]
    eye = torch.eye(n_atoms, device=d.device, dtype=torch.bool)
    desc_list = []

    for center in sigma_values:
        # Gaussian in distance, summed over neighbors
        g = torch.exp(-((d - center) ** 2) / (2.0 * 0.25**2)) * smooth_cutoff(d, cutoff)
        g = torch.where(eye, torch.zeros_like(g), g)
        desc_list.append(torch.sum(g, dim=1, keepdim=True))

    # also add inverse-distance-like feature
    inv_d = torch.where(eye, torch.zeros_like(d), 1.0 / (d + 1e-6))
    inv_d = inv_d * smooth_cutoff(d, cutoff)
    desc_list.append(torch.sum(inv_d, dim=1, keepdim=True))

    return torch.cat(desc_list, dim=1)


# ------------------------
# Dataset
# ------------------------
class ToyAtomicDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Config):
        self.items = []
        for _ in range(cfg.n_samples):
            pos = sample_structure(cfg.n_atoms, cfg.box_size, cfg.min_dist)
            energy = lennard_jones_energy(pos, cfg.lj_epsilon, cfg.lj_sigma, cfg.cutoff)
            self.items.append((pos, energy.unsqueeze(0)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    positions = torch.stack([item[0] for item in batch], dim=0)  # [B, N, 3]
    energies = torch.stack([item[1] for item in batch], dim=0)   # [B, 1]
    return positions, energies


# ------------------------
# Tiny MLIP model
# ------------------------
class AtomicNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F]
        atomic_e = self.net(x)          # [B, N, 1]
        total_e = torch.sum(atomic_e, dim=1)  # [B, 1]
        return total_e


class TinyMLIP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        in_dim = len(cfg.sigma_values) + 1
        self.atomic_net = AtomicNetwork(in_dim, cfg.hidden_dim)

    def descriptors(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: [B, N, 3]
        feats = []
        for b in range(positions.shape[0]):
            feat_b = atomic_descriptors(positions[b], self.cfg.cutoff, self.cfg.sigma_values)
            feats.append(feat_b)
        return torch.stack(feats, dim=0)  # [B, N, F]

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        feats = self.descriptors(positions)
        return self.atomic_net(feats)

    def predict_energy_forces(self, positions: torch.Tensor):
        # positions: [B, N, 3]
        positions = positions.clone().detach().requires_grad_(True)
        energy = self.forward(positions)
        forces = -torch.autograd.grad(energy.sum(), positions, create_graph=False)[0]
        return energy, forces


# ------------------------
# Training and evaluation
# ------------------------
def split_dataset(dataset, train_ratio=0.8):
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train
    return torch.utils.data.random_split(dataset, [n_train, n_val])


def mae(x: torch.Tensor, y: torch.Tensor) -> float:
    return torch.mean(torch.abs(x - y)).item()


def train_model(cfg: Config):
    dataset = ToyAtomicDataset(cfg)
    train_set, val_set = split_dataset(dataset, cfg.train_ratio)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = TinyMLIP(cfg).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for positions, energies in train_loader:
            positions = positions.to(DEVICE, dtype=DTYPE)
            energies = energies.to(DEVICE, dtype=DTYPE)

            pred = model(positions)
            loss = criterion(pred, energies)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * positions.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for positions, energies in val_loader:
                positions = positions.to(DEVICE, dtype=DTYPE)
                energies = energies.to(DEVICE, dtype=DTYPE)
                pred = model(positions)
                loss = criterion(pred, energies)
                val_loss += loss.item() * positions.size(0)
                val_mae += torch.sum(torch.abs(pred - energies)).item()

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | val_MAE={val_mae:.6f}")

    return model, val_set


def demo_prediction(model: TinyMLIP, sample_item):
    positions, true_energy = sample_item
    positions = positions.unsqueeze(0).to(DEVICE, dtype=DTYPE)
    true_energy = true_energy.to(DEVICE, dtype=DTYPE)

    pred_energy, forces = model.predict_energy_forces(positions)

    print("\n===== Demo prediction =====")
    print(f"True energy : {true_energy.item():.6f}")
    print(f"Pred energy : {pred_energy.item():.6f}")
    print(f"Forces shape: {tuple(forces.shape)}")
    print("First atom force:", forces[0, 0].detach().cpu().numpy())


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    model, val_set = train_model(CFG)
    demo_prediction(model, val_set[0])
