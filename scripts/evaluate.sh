#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/minimal.yaml}
CKPT=${2:-artifacts/minimal/best.pt}
python -m mlipx.evaluate --config "$CFG" --checkpoint "$CKPT" --split test --out artifacts/test_predictions.csv
