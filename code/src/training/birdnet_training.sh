for f in 0 1 2 3 4; do
  python -m birdnet_analyzer.train outputs/folds/fold_$f/train \
    -o models/fold_$f.tflite --test_data outputs/folds/fold_$f/val
done