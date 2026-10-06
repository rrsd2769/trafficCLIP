#!/bin/bash
# Prints a progress bar for the A training matrix, reading the JSON
# A/train.py writes every epoch.
set -euo pipefail
cd "$(dirname "$0")/.."

PROGRESS_FILE="${1:-logs/a_progress.json}"

if [ ! -f "$PROGRESS_FILE" ]; then
    echo "No progress file yet at $PROGRESS_FILE -- has the run started?"
    exit 0
fi

.venv/bin/python - "$PROGRESS_FILE" <<'EOF'
import json
import sys

path = sys.argv[1]
with open(path) as f:
    p = json.load(f)

pct = p["overall_percent"]
filled = int(pct / 2)  # 50-char bar
bar = "#" * filled + "-" * (50 - filled)

eta_min = p["eta_seconds"] / 60
print(f"[{bar}] {pct:5.1f}%")
print(f"Run:   {p['run_label']} seed={p['seed']}")
print(f"Epoch: {p['epoch']}/{p['epochs']}  (val_acc={p['val_acc']:.4f} val_f1={p['val_f1']:.4f})")
print(f"Speed: {p['sec_per_epoch']:.2f}s/epoch  ETA: {eta_min:.1f} min remaining")
print(f"Updated: {p['updated_at']}")
EOF
