#!/usr/bin/env bash
set -euo pipefail
python -m mlip_project.cli --epochs 30 --n-samples 600 --n-atoms 5 --device auto
