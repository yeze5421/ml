import argparse
from pathlib import Path

from .config import MLIPConfig
from .train import train


def parse_args():
    p = argparse.ArgumentParser(description="Train a toy machine-learning interatomic potential.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=600)
    p.add_argument("--n-atoms", type=int, default=5)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", type=str, default="artifacts")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MLIPConfig(
        epochs=args.epochs,
        n_samples=args.n_samples,
        n_atoms=args.n_atoms,
        device=args.device,
        out_dir=Path(args.out_dir),
    )

    model, val_set, _ = train(cfg)
    positions, true_energy, _ = val_set[0]
    energy, forces = model.predict_energy_and_forces(positions.unsqueeze(0))

    print("\nDemo")
    print(f"True energy: {true_energy.item():.6f}")
    print(f"Pred energy: {energy.item():.6f}")
    print(f"Forces shape: {tuple(forces.shape)}")


if __name__ == "__main__":
    main()
