#!/usr/bin/env bash
set -euo pipefail
python -m mlipx.train --config ${1:-configs/minimal.yaml}
