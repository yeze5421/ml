from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import read

from mlipx.ase_ext.calculator import build_calculator
from mlipx.config import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Inference with trained MLIP")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="single structure file or extxyz trajectory")
    p.add_argument("--output", default="artifacts/infer_results.txt")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    calc = build_calculator(cfg, args.checkpoint)
    frames = read(args.input, index=":")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, atoms in enumerate(frames):
            atoms.calc = calc
            e = atoms.get_potential_energy()
            forces = atoms.get_forces()
            f.write(f"frame={i} energy={e:.8f} f_norm={float((forces**2).sum()**0.5):.8f}\n")
    print(f"Saved inference results to {out}")


if __name__ == "__main__":
    main()
