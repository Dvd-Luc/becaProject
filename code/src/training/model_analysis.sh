for f in 0 1 2 3 4; do
  python -m birdnet_analyzer.analyze outputs/folds/fold_$f/val \
    -o models/preds/fold_$f -c models/fold_$f.tflite \
    --min_conf 0.01 --rtype csv
done