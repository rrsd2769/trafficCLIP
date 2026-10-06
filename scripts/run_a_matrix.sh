#!/bin/bash
# Full A training matrix: 10-class, 20-class and the broken-conv1 ablation,
# three seeds each. Sequential, since MPS is a single GPU on a fanless machine.
#
# `bash scripts/progress.sh` reads logs/a_progress.json, which A/train.py
# writes every epoch.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=.
PY=.venv/bin/python
PROGRESS=logs/a_progress.json
TOTAL_RUNS=9   # 3 configs x 3 seeds

echo "[$(date)] Starting A/10-class run"
$PY A/train.py --npz data/processed/ustc_10class.npz --classes 10 --epochs 200 \
    --seeds 62 63 64 --progress-file "$PROGRESS" --matrix-offset 0 --matrix-total-runs $TOTAL_RUNS

echo "[$(date)] Starting A/20-class run"
$PY A/train.py --npz data/processed/ustc_20class.npz --classes 20 --epochs 200 \
    --seeds 62 63 64 --progress-file "$PROGRESS" --matrix-offset 3 --matrix-total-runs $TOTAL_RUNS

echo "[$(date)] Starting A/20-class broken-conv1 ablation"
$PY A/train.py --npz data/processed/ustc_20class.npz --classes 20 --epochs 200 \
    --seeds 62 63 64 --broken-conv1 --progress-file "$PROGRESS" --matrix-offset 6 --matrix-total-runs $TOTAL_RUNS

echo "[$(date)] A training matrix complete"
