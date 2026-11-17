"""Collect the metrics of all learning-curve runs, save them as CSV, and plot.

    python plot_learning_curve.py --models-dir models/lc --work-dir outputs/lc --out-dir results/lc

Writes
  learning_curve_long.csv      one row per (fraction, subset_seed, fold, class, metric)
  learning_curve_summary.csv   mean / sd over folds per (class, metric, fraction)
  learning_curve_<metric>.png  one panel per class; faint dots = folds, line = mean +- sd

Every run is read from  <models-dir>/frac_X/seed_S/fold_K*_evaluation.csv  (BirdNET's table,
first column = class). Scores of different subset seeds are averaged within a fold first,
so the spread shown is the spread between folds (= between validation recordists).
"""
import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def num(x):
    """float(x), or None if x is not a number."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_stats(path):
    """fold -> (train clips, train recordists) from fold_stats.csv."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["part"] == "train" and r["class"] == "ALL":
                out[int(r["fold"])] = (int(r["clips"]), int(r["recordists"]))
    return out


def main():
    """CLI entry point: collect all fold metrics, write the CSVs and plots, print the paired gap."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models-dir", default="models/lc")
    ap.add_argument("--work-dir", default="outputs/lc")
    ap.add_argument("--out-dir", default="results/lc")
    ap.add_argument("--keep-labels", nargs="+", default=None,
                    help="only use these classes (all others are ignored)")
    ap.add_argument("--drop-labels", nargs="+", default=[],
                    help="classes to ignore, e.g. --drop-labels Other")
    ap.add_argument("--metrics", nargs="+", default=["F1 Score (opt)", "AUPRC", "F1 Score (0.5)"])
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    long_rows = []
    for p in sorted(Path(a.models_dir).rglob("*_evaluation.csv")):
        m = re.search(r"frac_([\d.]+)[/\\]seed_(\d+)", p.as_posix())
        fm = re.search(r"fold_(\d+)", p.name)
        if not (m and fm):
            continue
        frac, seed, fold = float(m.group(1)), int(m.group(2)), int(fm.group(1))
        stats = read_stats(Path(a.work_dir) / f"frac_{m.group(1)}" / f"seed_{seed}" / "fold_stats.csv")
        n_clips, n_rec = stats.get(fold, (None, None))
        with open(p, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            key = reader.fieldnames[0]
            for r in reader:
                cls = r[key].strip()
                cls = "Macro" if cls.upper().startswith("OVERALL") else cls
                for col in reader.fieldnames[1:]:
                    v = num(r[col])
                    if v is not None:
                        long_rows.append([frac, seed, fold, n_rec, n_clips, cls, col, v])
    if not long_rows:
        raise SystemExit(f"no metrics CSV found under {a.models_dir} (looked for *_evaluation.csv)")

    with open(out / "learning_curve_long.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fraction", "subset_seed", "fold", "train_recordists", "train_clips",
                    "class", "metric", "value"])
        w.writerows(long_rows)

    # average over subset seeds within a fold, then summarise over folds
    cell = defaultdict(list)                    # (class, metric, frac, fold) -> values
    xs = defaultdict(list)                      # frac -> n train recordists
    for frac, seed, fold, n_rec, _, cls, met, v in long_rows:
        cell[(cls, met, frac, fold)].append(v)
        if n_rec is not None:
            xs[frac].append(n_rec)
    per_fold = defaultdict(dict)                # (class, metric, frac) -> {fold: mean}
    for (cls, met, frac, fold), vals in cell.items():
        per_fold[(cls, met, frac)][fold] = statistics.mean(vals)
    x_of = {fr: statistics.mean(v) for fr, v in xs.items()} or {}

    with open(out / "learning_curve_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "metric", "fraction", "train_recordists", "n_folds", "mean", "sd"])
        for (cls, met, frac), d in sorted(per_fold.items()):
            v = list(d.values())
            w.writerow([cls, met, frac, f"{x_of.get(frac, float('nan')):.1f}", len(v),
                        f"{statistics.mean(v):.4f}",
                        f"{statistics.stdev(v):.4f}" if len(v) > 1 else ""])

    classes = sorted({k[0] for k in per_fold}, key=lambda c: (c == "Macro", c))
    for met in a.metrics:
        if not any(k[1] == met for k in per_fold):
            print(f"metric '{met}' not found, skipping")
            continue
        fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 3.6), sharey=True)
        axes = [axes] if len(classes) == 1 else list(axes)
        for ax, cls in zip(axes, classes):
            fracs = sorted(fr for (c, m, fr) in per_fold if c == cls and m == met)
            xv = [x_of.get(fr, fr) for fr in fracs]
            means, sds = [], []
            for fr, x in zip(fracs, xv):
                vals = per_fold[(cls, met, fr)]
                ax.scatter([x] * len(vals), list(vals.values()), s=12, alpha=0.35, color="gray")
                v = list(vals.values())
                means.append(statistics.mean(v))
                sds.append(statistics.stdev(v) if len(v) > 1 else 0)
            ax.errorbar(xv, means, yerr=sds, marker="o", capsize=3)
            ax.set_title(cls)
            ax.set_xlabel("training recordists" if x_of else "train fraction")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(met)
        fig.tight_layout()
        slug = re.sub(r"[^\w]+", "_", met).strip("_")
        fig.savefig(out / f"learning_curve_{slug}.png", dpi=150)
        plt.close(fig)

    # paired gap between the two largest fractions, fold by fold
    met = a.metrics[0]
    fr_all = sorted({k[2] for k in per_fold})
    if len(fr_all) >= 2:
        lo, hi = fr_all[-2], fr_all[-1]
        print(f"\nPaired gap {met}: fraction {hi:g} minus {lo:g}, per fold")
        for cls in classes:
            a_, b_ = per_fold.get((cls, met, lo), {}), per_fold.get((cls, met, hi), {})
            diffs = [b_[f] - a_[f] for f in a_ if f in b_]
            if diffs:
                sd = statistics.stdev(diffs) if len(diffs) > 1 else float("nan")
                print(f"  {cls:<10} {statistics.mean(diffs):+.3f} +- {sd:.3f}   "
                      f"({sum(d > 0 for d in diffs)}/{len(diffs)} folds improve)")
    print(f"\nsaved to {out}/")


if __name__ == "__main__":
    main()