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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clips-csv", required=True)
    ap.add_argument("--xc-csv", required=True)
    ap.add_argument("--work-dir", default="outputs/lc")
    ap.add_argument("--models-dir", default="models/lc")
    ap.add_argument("--make-folds", default="src/training/make_folds.py")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--subset-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42, help="fold assignment, same for all runs")
    ap.add_argument("--keep-labels", nargs="+", default=None)
    ap.add_argument("--drop-labels", nargs="+", default=[])
    a = ap.parse_args()

    work, models = Path(a.work_dir), Path(a.models_dir)
    for frac in a.fractions:
        seeds = a.subset_seeds[:1] if frac >= 1.0 else a.subset_seeds
        for s in seeds:
            tag = Path(f"frac_{frac:g}") / f"seed_{s}"
            fdir, mdir = work / tag, models / tag
            if not (fdir / "fold_stats.csv").exists():
                print(f"== folds for {tag}")
                extra = []
                if a.keep_labels:
                    extra += ["--keep-labels", *a.keep_labels]
                if a.drop_labels:
                    extra += ["--drop-labels", *a.drop_labels]
                subprocess.run([sys.executable, a.make_folds,
                                "--clips-csv", a.clips_csv, "--xc-csv", a.xc_csv,
                                "--out-dir", str(fdir), "--folds", str(a.folds),
                                "--train-fraction", str(frac), "--seed", str(a.seed),
                                "--subset-seed", str(s)] + extra, check=True)
            mdir.mkdir(parents=True, exist_ok=True)
            for f in range(a.folds):
                model = mdir / f"fold_{f}.tflite"
                if model.exists():
                    continue
                print(f"== training {tag} fold_{f}")
                with open(mdir / f"fold_{f}.log", "w") as log:
                    subprocess.run([sys.executable, "-m", "birdnet_analyzer.train",
                                    str(fdir / f"fold_{f}" / "train"), "-o", str(model),
                                    "--test_data", str(fdir / f"fold_{f}" / "val")],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                if not list(mdir.glob(f"fold_{f}*_evaluation.csv")):
                    print(f"   warning: no metrics CSV next to {model}; "
                          f"check {mdir / f'fold_{f}.log'} and `ls {mdir}`")


if __name__ == "__main__":
    main()