#!/bin/bash

echo "STARTING GRAD-CAM EXPLAINABILITY DIAGNOSTIC"

LAMBDAS="0.0 0.5 1.0 2.0 5.0"

# Generate explanations for Original Prompts variant
# for l in $LAMBDAS
# do
#     echo "Generating Grad-CAM Explanations | Lambda: $l"
#     python src/explainability_runner.py  --lambda_cl $l --model_version optimized --conflicts BitTorrent Gmail Gmail Skype 
# done

# # Generate explanations for Statistical Prompts variant
# for l in $LAMBDAS
# do
#     echo "Generating Grad-CAM Explanations | Lambda: $l"
#     python src/explainability_runner.py  --lambda_cl $l --model_version optimized  --use_stats_prompts  --conflicts BitTorrent Gmail Gmail Skype 
# done

# Generate explanations for Original Prompts with statistical features variant
for l in $LAMBDAS
do
    echo "Generating Grad-CAM Explanations | Lambda: $l"
    python src/explainability_runner.py  --lambda_cl $l --model_version optimized  --use_stats  --stats_input_dim 3 --conflicts BitTorrent Gmail Gmail Skype 
done

echo "GRAD-CAM EXPLAINABILITY DIAGNOSTIC COMPLETE"