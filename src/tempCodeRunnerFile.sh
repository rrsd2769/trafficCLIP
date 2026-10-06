timized Model with Original Prompts and Statistical Features (Phase 2 & 3)
for l in $LAMBDAS
do
   echo "Evaluating Original Prompts | Lambda: $l with statsistical features"
   python src/test_runner.py --model_version optimized --use_stats --stats_input_dim 3 --lambda_cl $l --num_runs 3 --seed 62
done

echo "Evaluating Original Model with Original Prompts | 1.0 Lambda"
python src/test_runner.py --model_version original --lambda_cl 1.0 --num_runs 3 --seed 62
echo "Evaluating Original Model with Statistical Prompts | 1.0 Lambda"
python src/t