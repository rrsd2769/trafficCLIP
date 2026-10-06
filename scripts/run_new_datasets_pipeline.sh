#!/bin/bash
set -e
PY="PYTHONPATH=. .venv/bin/python"

echo "=== Cache build: 5G, 3-class ==="
PYTHONPATH=. .venv/bin/python A/cache_features.py --npz data/processed/g5_3class.npz
PYTHONPATH=. .venv/bin/python A/cache_features_full.py --npz data/processed/g5_3class.npz

echo "=== Track A: 5G, 3-class ==="
PYTHONPATH=. .venv/bin/python A/train.py --npz data/processed/g5_3class.npz --classes 3 \
  --out results/comparison/A_5G_3class.csv

echo "=== Track A-Full: 5G, 3-class ==="
PYTHONPATH=. .venv/bin/python A/train_full.py --npz data/processed/g5_3class.npz --classes 3 \
  --out results/comparison/A_Full_5G_3class.csv

echo "=== Track C: 5G, 3-class (all 4 models) ==="
PYTHONPATH=. .venv/bin/python C/run_matrix.py --npz data/processed/g5_3class.npz --classes 3

echo "=== Cache build: IoT-23, 9-class ==="
PYTHONPATH=. .venv/bin/python A/cache_features.py --npz data/processed/iot23_9class.npz
PYTHONPATH=. .venv/bin/python A/cache_features_full.py --npz data/processed/iot23_9class.npz

echo "=== Track A: IoT-23, 9-class ==="
PYTHONPATH=. .venv/bin/python A/train.py --npz data/processed/iot23_9class.npz --classes 9 \
  --out results/comparison/A_IoT23_9class.csv

echo "=== Track A-Full: IoT-23, 9-class ==="
PYTHONPATH=. .venv/bin/python A/train_full.py --npz data/processed/iot23_9class.npz --classes 9 \
  --out results/comparison/A_Full_IoT23_9class.csv

echo "=== Track C: IoT-23, 9-class (all 4 models) ==="
PYTHONPATH=. .venv/bin/python C/run_matrix.py --npz data/processed/iot23_9class.npz --classes 9

echo "NEW_DATASETS_PIPELINE_DONE"
