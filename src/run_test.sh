#!/bin/bash

# Define the same Lambda values used in training 0.0 0.5 1.0 2.0 5.0
LAMBDAS="0.0 0.5 1.0 2.0 5.0"

echo "STARTING ABLATION TEST SWEEP"

# Test Optimized Model with Statistical Prompts (Phase 2 & 3)
for l in $LAMBDAS
do
   echo "Evaluating Statistical Prompts | Lambda: $l"
   python src/test_runner.py --model_version optimized --use_stats_prompts --lambda_cl $l --num_runs 3 --seed 62
done

# Test Optimized Model with Original Prompts (Phase 3 Architecture Only)
for l in $LAMBDAS
do
   echo "Evaluating Original Prompts | Lambda: $l"
   python src/test_runner.py --model_version optimized --lambda_cl $l --num_runs 3 --seed 62
done

# Test Optimized Model with Original Prompts and Statistical Features (Phase 2 & 3)
for l in $LAMBDAS
do
   echo "Evaluating Original Prompts | Lambda: $l with statsistical features"
   python src/test_runner.py --model_version optimized --use_stats --stats_input_dim 3 --lambda_cl $l --num_runs 3 --seed 62
done

echo "Evaluating Original Model with Original Prompts | 1.0 Lambda"
python src/test_runner.py --model_version original --lambda_cl 1.0 --num_runs 3 --seed 62
echo "Evaluating Original Model with Statistical Prompts | 1.0 Lambda"
python src/test_runner.py --model_version original --use_stats_prompts --lambda_cl 1.0 --num_runs 3 --seed 62
echo "ABLATION TEST SWEEP COMPLETE"