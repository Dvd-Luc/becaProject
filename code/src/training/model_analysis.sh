# Predict each fold's validation clips with that fold's model (after birdnet_training.sh).
# Output: models/preds/fold_K/*.BirdNET.results.csv, all scores >= 0.01,
#         read by src/tools/confusion_matrix.py outputs/folds models/preds
# Run from code/.
for f in 0 1 2 3 4; do
  python -m birdnet_analyzer.analyze outputs/folds/fold_$f/val \
    -o models/preds/fold_$f -c models/fold_$f.tflite \
    --min_conf 0.01 --rtype csv
done