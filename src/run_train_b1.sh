#!/bin/bash

echo "STARTING ABLATION TRAIN SWEEP"

# Original model with original prompt baseline with lambda = 1.0
python src/train_runner.py --model_version original --lambda_cl 1.0

# Original model with stats prompt baseline with lambda = 1.0
python src/train_runner.py --model_version original --use_stats_prompts --lambda_cl 1.0

echo "ABLATION TRAIN SWEEP COMPLETE"
