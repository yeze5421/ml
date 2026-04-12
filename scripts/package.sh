#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-mlip_project.zip}
zip -r "$OUT" . \
  -x '.git/*' '__pycache__/*' '*.pyc' '.pytest_cache/*' '*.zip'

echo "Created archive: $OUT"
