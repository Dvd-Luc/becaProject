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


def num(text):
    """Parse a number.

    Args:
        text (Any): Value to convert.

    Returns:
        float | None: float(x), or None if x is not a number.
    """
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_stats(path):
    """Training size per fold, from fold_stats.csv.

    Args:
        path (Path): fold_stats.csv written by make_folds.py.

    Returns:
        dict[int, tuple[int, int]]: Fold -> (training clips, training recordists).
    """
    stats = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["part"] == "train" and row["class"] == "ALL":
                stats[int(row["fold"])] = (int(row["clips"]), int(row["recordists"]))
    return stats


def main():
    """CLI entry point: collect all fold metrics, write the CSVs and plots, print the paired gap."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--models-dir", default="models/lc")
    parser.add_argument("--work-dir", default="outputs/lc")
    parser.add_argument("--out-dir", default="results/lc")
    parser.add_argument("--keep-labels", nargs="+", default=None,
                        help="only use these classes (all others are ignored)")
    parser.add_argument("--drop-labels", nargs="+", default=[],
                        help="classes to ignore, e.g. --drop-labels Other")
    parser.add_argument("--metrics", nargs="+", default=["F1 Score (opt)", "AUPRC", "F1 Score (0.5)"])
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows = []
    for eval_path in sorted(Path(args.models_dir).rglob("*_evaluation.csv")):
        path_match = re.search(r"frac_([\d.]+)[/\\]seed_(\d+)", eval_path.as_posix())
        fold_match = re.search(r"fold_(\d+)", eval_path.name)
        if not (path_match and fold_match):
            continue
        frac, seed, fold = float(path_match.group(1)), int(path_match.group(2)), int(fold_match.group(1))
        run_dir = Path(args.work_dir) / f"frac_{path_match.group(1)}" / f"seed_{seed}"
        stats = read_stats(run_dir / "fold_stats.csv")
        n_clips, n_rec = stats.get(fold, (None, None))
        with open(eval_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            key = reader.fieldnames[0]
            for row in reader:
                cls = row[key].strip()
                cls = "Macro" if cls.upper().startswith("OVERALL") else cls
                for col in reader.fieldnames[1:]:
                    value = num(row[col])
                    if value is not None:
                        long_rows.append([frac, seed, fold, n_rec, n_clips, cls, col, value])
    if not long_rows:
        raise SystemExit(f"no metrics CSV found under {args.models_dir} (looked for *_evaluation.csv)")

    with open(out_dir / "learning_curve_long.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fraction", "subset_seed", "fold", "train_recordists", "train_clips",
                         "class", "metric", "value"])
        writer.writerows(long_rows)

    # average over subset seeds within a fold, then summarise over folds
    cell = defaultdict(list)                    # (class, metric, frac, fold) -> values
    n_rec_by_frac = defaultdict(list)                      # frac -> n train recordists
    for frac, seed, fold, n_rec, _, cls, metric, value in long_rows:
        cell[(cls, metric, frac, fold)].append(value)
        if n_rec is not None:
            n_rec_by_frac[frac].append(n_rec)
    per_fold = defaultdict(dict)                # (class, metric, frac) -> {fold: mean}
    for (cls, metric, frac, fold), vals in cell.items():
        per_fold[(cls, metric, frac)][fold] = statistics.mean(vals)
    x_of_frac = {frac: statistics.mean(n_recs) for frac, n_recs in n_rec_by_frac.items()} or {}

    with open(out_dir / "learning_curve_summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "metric", "fraction", "train_recordists", "n_folds", "mean", "sd"])
        for (cls, metric, frac), fold_means in sorted(per_fold.items()):
            values = list(fold_means.values())
            writer.writerow([cls, metric, frac, f"{x_of_frac.get(frac, float('nan')):.1f}", len(values),
                             f"{statistics.mean(values):.4f}",
                             f"{statistics.stdev(values):.4f}" if len(values) > 1 else ""])

    classes = sorted({key[0] for key in per_fold}, key=lambda c: (c == "Macro", c))
    for metric in args.metrics:
        if not any(key[1] == metric for key in per_fold):
            print(f"metric '{metric}' not found, skipping")
            continue
        fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 3.6), sharey=True)
        axes = [axes] if len(classes) == 1 else list(axes)
        for ax, cls in zip(axes, classes):
            fracs = sorted(frac for (key_cls, key_metric, frac) in per_fold
                           if key_cls == cls and key_metric == metric)
            x_values = [x_of_frac.get(frac, frac) for frac in fracs]
            means, sds = [], []
            for frac, x_val in zip(fracs, x_values):
                fold_scores = per_fold[(cls, metric, frac)]
                ax.scatter([x_val] * len(fold_scores), list(fold_scores.values()),
                           s=12, alpha=0.35, color="gray")
                values = list(fold_scores.values())
                means.append(statistics.mean(values))
                sds.append(statistics.stdev(values) if len(values) > 1 else 0)
            ax.errorbar(x_values, means, yerr=sds, marker="o", capsize=3)
            ax.set_title(cls)
            ax.set_xlabel("training recordists" if x_of_frac else "train fraction")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(metric)
        fig.tight_layout()
        slug = re.sub(r"[^\w]+", "_", metric).strip("_")
        fig.savefig(out_dir / f"learning_curve_{slug}.png", dpi=150)
        plt.close(fig)

    # paired gap between the two largest fractions, fold by fold
    metric = args.metrics[0]
    all_fracs = sorted({key[2] for key in per_fold})
    if len(all_fracs) >= 2:
        lo, hi = all_fracs[-2], all_fracs[-1]
        print(f"\nPaired gap {metric}: fraction {hi:g} minus {lo:g}, per fold")
        for cls in classes:
            lo_scores, hi_scores = per_fold.get((cls, metric, lo), {}), per_fold.get((cls, metric, hi), {})
            diffs = [hi_scores[fold] - lo_scores[fold] for fold in lo_scores if fold in hi_scores]
            if diffs:
                sd = statistics.stdev(diffs) if len(diffs) > 1 else float("nan")
                print(f"  {cls:<10} {statistics.mean(diffs):+.3f} +- {sd:.3f}   "
                      f"({sum(diff > 0 for diff in diffs)}/{len(diffs)} folds improve)")
    print(f"\nsaved to {out_dir}/")


if __name__ == "__main__":
    main()
