"""Build folds and train one BirdNET classifier per (fraction, subset seed, fold).

    python run_learning_curve.py --clips-csv outputs/clips/metadata.csv \
        --xc-csv data/xc_gallinago_gallinago/metadata.csv

Layout it creates (resumable: existing folds and models are skipped):
    outputs/lc/frac_0.25/seed_0/fold_0/{train,val}/<class>/*.wav
    outputs/lc/frac_0.25/seed_0/fold_stats.csv
    models/lc/frac_0.25/seed_0/fold_0.tflite   (+ BirdNET's own *Metrics.csv next to it)
    models/lc/frac_0.25/seed_0/fold_0.log      (trainer output)

For fraction 1.0 only one subset seed is run (the subset is the whole pool).
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """CLI entry point: for each fraction x subset seed, make the folds and train every fold.

    Folds whose fold_stats.csv exists and models whose .tflite exists are skipped.

    Raises:
        subprocess.CalledProcessError: If make_folds.py or BirdNET training fails.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--clips-csv", required=True)
    parser.add_argument("--xc-csv", required=True)
    parser.add_argument("--work-dir", default="outputs/lc")
    parser.add_argument("--models-dir", default="models/lc")
    parser.add_argument("--make-folds", default="src/training/make_folds.py")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--subset-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="fold assignment, same for all runs")
    parser.add_argument("--keep-labels", nargs="+", default=None)
    parser.add_argument("--drop-labels", nargs="+", default=[])
    args = parser.parse_args()

    work, models = Path(args.work_dir), Path(args.models_dir)
    for frac in args.fractions:
        seeds = args.subset_seeds[:1] if frac >= 1.0 else args.subset_seeds
        for subset_seed in seeds:
            tag = Path(f"frac_{frac:g}") / f"seed_{subset_seed}"
            folds_dir, model_dir = work / tag, models / tag
            if not (folds_dir / "fold_stats.csv").exists():
                print(f"== folds for {tag}")
                extra = []
                if args.keep_labels:
                    extra += ["--keep-labels", *args.keep_labels]
                if args.drop_labels:
                    extra += ["--drop-labels", *args.drop_labels]
                subprocess.run([sys.executable, args.make_folds,
                                "--clips-csv", args.clips_csv, "--xc-csv", args.xc_csv,
                                "--out-dir", str(folds_dir), "--folds", str(args.folds),
                                "--train-fraction", str(frac), "--seed", str(args.seed),
                                "--subset-seed", str(subset_seed)] + extra, check=True)
            model_dir.mkdir(parents=True, exist_ok=True)
            for fold in range(args.folds):
                model = model_dir / f"fold_{fold}.tflite"
                if model.exists():
                    continue
                print(f"== training {tag} fold_{fold}")
                with open(model_dir / f"fold_{fold}.log", "w") as log:
                    subprocess.run([sys.executable, "-m", "birdnet_analyzer.train",
                                    str(folds_dir / f"fold_{fold}" / "train"), "-o", str(model),
                                    "--test_data", str(folds_dir / f"fold_{fold}" / "val")],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                if not list(model_dir.glob(f"fold_{fold}*_evaluation.csv")):
                    print(f"   warning: no metrics CSV next to {model}; "
                          f"check {model_dir / f'fold_{fold}.log'} and `ls {model_dir}`")


if __name__ == "__main__":
    main()
