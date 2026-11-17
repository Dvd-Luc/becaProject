"""Pooled and per-fold confusion matrices from BirdNET analyze outputs.

    python confusion_matrix.py outputs/folds models/preds [confusion_matrices.csv]

rows = true class (val folder name), columns = predicted class (top-1 prediction).
Clips with no prediction above --min_conf are counted as "none".
"""
import csv
import sys
from collections import Counter
from pathlib import Path

val_root, pred_root = Path(sys.argv[1]), Path(sys.argv[2])
out_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("confusion_matrices.csv")
folds = sorted(p.name for p in val_root.glob("fold_*"))


def read_preds(fold):
    """clip filename -> (best label, confidence) from analyze CSVs."""
    best = {}
    for f in sorted((pred_root / fold).rglob("*.csv")):
        with open(f, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not {"File", "Common name", "Confidence"} <= set(reader.fieldnames or []):
                continue                      # e.g. BirdNET_analysis_params.csv
            for r in reader:
                name = Path(r["File"]).name
                conf = float(r["Confidence"])
                if name not in best or conf > best[name][1]:
                    best[name] = (r["Common name"], conf)
    return best


per_fold = {}
total = Counter()
for fold in folds:
    best = read_preds(fold)
    m = Counter()
    for cls_dir in sorted((val_root / fold / "val").iterdir()):
        if not cls_dir.is_dir():
            continue
        for clip in cls_dir.glob("*.wav"):
            pred = best.get(clip.name, ("none", 0))[0]
            m[(cls_dir.name, pred)] += 1
    per_fold[fold] = m
    total.update(m)
    n_val = sum(m.values())
    print(f"{fold}: {len(best)} predicted clips, {n_val} validation clips")

true_cls = sorted({t for t, _ in total})
pred_cls = sorted({p for _, p in total})

# console output of the pooled matrix
print("\nPOOLED (rows = true, columns = predicted)")
print(f"{'':<12}" + "".join(f"{p:>10}" for p in pred_cls))
for t in true_cls:
    print(f"{t:<12}" + "".join(f"{total[(t, p)]:>10}" for p in pred_cls))

# CSV: one row per (fold, true class), plus a pooled block
out_csv.parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["fold", "true"] + pred_cls + ["total", "recall"])
    for name, m in [*per_fold.items(), ("pooled", total)]:
        for t in true_cls:
            row = [m[(t, p)] for p in pred_cls]
            n = sum(row)
            recall = f"{m[(t, t)] / n:.4f}" if n else ""
            w.writerow([name, t] + row + [n, recall])
print(f"\nsaved {out_csv}")