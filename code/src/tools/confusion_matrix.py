"""Pooled and per-fold confusion matrices from BirdNET analyze outputs.

    python src/tools/confusion_matrix.py outputs/folds models/preds [confusion_matrices.csv]

rows = true class (val folder name), columns = predicted class (top-1 prediction).
Clips with no prediction above --min_conf are counted as "none".

The output must be a .csv file, and nothing is written when no fold_* folder is found.
All the work happens in main(), so importing this module (e.g. pdoc) has no side effect.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path


def read_preds(pred_root, fold):
    """Top-1 prediction per validation clip from BirdNET analyze outputs.

    Args:
        pred_root (Path): Folder with one prediction folder per fold (model_analysis.sh).
        fold (str): Fold folder name, e.g. "fold_0".

    Returns:
        dict[str, tuple[str, float]]: Clip file name -> (best label, confidence).
    """
    best = {}
    for csv_path in sorted((pred_root / fold).rglob("*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not {"File", "Common name", "Confidence"} <= set(reader.fieldnames or []):
                continue                      # e.g. BirdNET_analysis_params.csv
            for row in reader:
                name = Path(row["File"]).name
                conf = float(row["Confidence"])
                if name not in best or conf > best[name][1]:
                    best[name] = (row["Common name"], conf)
    return best


def confusion(val_root, pred_root, folds):
    """Count (true class, predicted class) pairs per fold.

    Args:
        val_root (Path): Folds folder (fold_K/val/<class>/*.wav, from make_folds.py).
        pred_root (Path): Predictions folder (fold_K/*.csv, from model_analysis.sh).
        folds (list[str]): Fold folder names.

    Returns:
        tuple[dict[str, Counter], Counter]: (per-fold counts, pooled counts), keyed by
            (true class, predicted class); "none" when a clip has no prediction.
    """
    per_fold, total = {}, Counter()
    for fold in folds:
        best = read_preds(pred_root, fold)
        counts = Counter()
        for cls_dir in sorted((val_root / fold / "val").iterdir()):
            if not cls_dir.is_dir():
                continue
            for clip in cls_dir.glob("*.wav"):
                counts[(cls_dir.name, best.get(clip.name, ("none", 0))[0])] += 1
        per_fold[fold] = counts
        total.update(counts)
        print(f"{fold}: {len(best)} predicted clips, {sum(counts.values())} validation clips")
    return per_fold, total


def main():
    """CLI entry point: print the pooled matrix, write per-fold and pooled matrices to CSV.

    Raises:
        SystemExit: If the output is not a .csv file or no fold_* folder is found.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("val_root", type=Path, help="folds folder, e.g. outputs/folds")
    parser.add_argument("pred_root", type=Path, help="predictions folder, e.g. models/preds")
    parser.add_argument("out_csv", type=Path, nargs="?", default=Path("confusion_matrices.csv"),
                        help="output CSV (default: confusion_matrices.csv)")
    args = parser.parse_args()
    if args.out_csv.suffix.lower() != ".csv":
        raise SystemExit(f"output must be a .csv file, got {args.out_csv} (refusing to overwrite it)")
    folds = sorted(pred.name for pred in args.val_root.glob("fold_*"))
    if not folds:
        raise SystemExit(f"no fold_* folder in {args.val_root}; nothing written")

    per_fold, total = confusion(args.val_root, args.pred_root, folds)
    true_cls = sorted({true for true, _ in total})
    pred_cls = sorted({pred for _, pred in total})

    print("\nPOOLED (rows = true, columns = predicted)")
    print(f"{'':<12}" + "".join(f"{pred:>10}" for pred in pred_cls))
    for true in true_cls:
        print(f"{true:<12}" + "".join(f"{total[(true, pred)]:>10}" for pred in pred_cls))

    # one row per (fold, true class), plus a pooled block
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fold", "true"] + pred_cls + ["total", "recall"])
        for name, counts in [*per_fold.items(), ("pooled", total)]:
            for true in true_cls:
                row = [counts[(true, pred)] for pred in pred_cls]
                n_true = sum(row)
                recall = f"{counts[(true, true)] / n_true:.4f}" if n_true else ""
                writer.writerow([name, true] + row + [n_true, recall])
    print(f"\nsaved {args.out_csv}")


if __name__ == "__main__":
    main()
