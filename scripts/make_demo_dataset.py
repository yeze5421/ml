#!/usr/bin/env python
from __future__ import annotations

import argparse
import random

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/demo.extxyz")
    p.add_argument("--n-samples", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    frames = []
    for _ in range(args.n_samples):
        n = random.randint(2, 6)
        symbols = random.choices(["H", "C", "N", "O"], k=n)
        pos = np.random.uniform(0.0, 3.5, size=(n, 3))
        atoms = Atoms(symbols=symbols, positions=pos, cell=np.eye(3) * 8.0, pbc=False)
        atoms.calc = EMT()
        atoms.info["energy"] = float(atoms.get_potential_energy())
        atoms.arrays["forces"] = atoms.get_forces()
        frames.append(atoms)

    write(args.out, frames, format="extxyz")
    print(f"wrote {len(frames)} samples to {args.out}")


if __name__ == "__main__":
    main()
