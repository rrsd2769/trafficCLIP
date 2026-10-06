#!/bin/bash

echo "STARTING ABLATION TRAIN SWEEP (STATS PROMPTS)"

# Initialize counter
count=1

# Sweep through five Lambda values
for l in 0.0 0.5 1.0 2.0 5.0
do
    echo "------------------------------------------------"
    echo "Training Stats Prompts with Lambda: $l (Run #$count)"
    echo "------------------------------------------------"

    # Run Training with the --use_stats_prompts flag
    python src/train_runner.py --model_version optimized --use_stats_prompts --lambda_cl $l

    # Check if running in Google Colab environment
    if [ -d "/content" ]; then
        EXP_ZIP="experiments_run${count}_lambda${l}.zip"
        RES_ZIP="results_run${count}_lambda${l}.zip"

        # Zip the outputs from the exact source folders
        echo "Zipping results for run #$count..."
        zip -r "$EXP_ZIP" /content/TrafficCLIP-Optimized/experiments
        zip -r "$RES_ZIP" /content/TrafficCLIP-Optimized/results

        # Move the newly created zips to Google Drive
        echo "Moving to Google Drive..."
        mv "$EXP_ZIP" /content/drive/MyDrive/
        mv "$RES_ZIP" /content/drive/MyDrive/
    fi    

    # Increment counter for next run
    count=$((count + 1))
done

echo "ABLATION TRAIN SWEEP (STATS PROMPTS) COMPLETE"