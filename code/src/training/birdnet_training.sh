# Train one BirdNET classifier per cross-validation fold (single 5-fold run).
# Input:  outputs/folds/fold_{0..4}/{train,val}/<class>/  (make_folds.py)
# Output: models/fold_K.tflite + BirdNET metrics, validated on fold_K/val
# Then:   model_analysis.sh, then src/tools/confusion_matrix.py outputs/folds models/preds
# Run from code/.
for f in 0 1 2 3 4; do
  python -m birdnet_analyzer.train outputs/folds/fold_$f/train \
    -o models/fold_$f.tflite --test_data outputs/folds/fold_$f/val
done