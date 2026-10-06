#!/bin/bash

echo "STARTING ABLATION TRAIN SWEEP WITH Dynamic Scheduling"

# Run Training
python src/train_runner.py --model_version optimized --lambda_cl 5.0
python src/train_runner.py --model_version optimized --use_stats_prompts --lambda_cl 5.0
python src/train_runner.py --model_version optimized --use_stats --stats_input_dim 3 --lambda_cl 5.0

echo "ABLATION TRAIN SWEEP COMPLETE"