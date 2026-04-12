#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/minimal.yaml}
CKPT=${2:-artifacts/minimal/best.pt}
INP=${3:-data/demo.extxyz}
python -m mlipx.infer --config "$CFG" --checkpoint "$CKPT" --input "$INP" --output artifacts/infer.txt
