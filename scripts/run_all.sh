#!/usr/bin/env bash
# Full pilot once data/inshop/{Eval/list_eval_partition.txt,img/} and the CLIP checkpoint are reachable.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
$PY -m pytest -q
$PY scripts/pilot.py preflight
$PY scripts/pilot.py benchmark --n 256
$PY scripts/pilot.py audit --workers "$(nproc)"
$PY scripts/pilot.py embed --batch-size 64 --workers "$(nproc)"
$PY scripts/pilot.py evaluate            # full official queries vs full gallery
$PY scripts/pilot.py examples --per-group 10
