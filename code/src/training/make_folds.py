"""Split the 3 s clips into cross-validation folds, grouped by recordist.

All clips from one recordist (and therefore from the same recordings, sites and
microphones) end up on the same side of every split, so validation scores measure
how the model does on recordings it has never heard from that source.

    python make_folds.py --clips-csv clips/metadata.csv --xc-csv xc.../metadata.csv \
                         --out-dir folds --folds 5

creates   folds/fold_0/train/<class>/*.wav   and   folds/fold_0/val/<class>/*.wav, ...
and       folds/fold_stats.csv   (clips and recordists per fold / part / class)

--train-fraction 0.5 keeps only half of the training recordists in each fold:
run it for 0.25, 0.5, 0.75 and 1.0 to draw a learning curve.

  --seed         controls the FOLD ASSIGNMENT only (keep it fixed so every fraction
                 is validated on the same recordists).
  --subset-seed  controls WHICH training recordists are kept. For a given subset
                 seed the subsets are nested: the 25% set is contained in the 50%
                 set, which is contained in the 75% set. Change it to average the
                 curve over several random subsets.

Files are hard-linked (no extra disk space); use --mode copy if linking fails.
"""
import argparse
import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

MIN_VAL, MIN_TRAIN = 10, 30      # warn below these clip counts per class


def place(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    elif mode == "symlink":
        os.symlink(Path(src).resolve(), dst)
        return
    shutil.copy2(src, dst)


def assign_groups(groups, k, rng):
    """Greedy balanced assignment of groups to k folds, class by class."""
    totals = Counter()
    for c in groups.values():
        totals.update(c)
    order = sorted(groups, key=lambda g: (-sum(n / totals[c] for c, n in groups[g].items()),
                                          rng.random()))
    load = [Counter() for _ in range(k)]
    fold_of = {}
    for g in order:
        def cost(f):        # change in squared distance to the ideal share 1/k of each class
            return sum(((load[f][c] + groups[g][c]) / totals[c] - 1 / k) ** 2
                       - (load[f][c] / totals[c] - 1 / k) ** 2 for c in totals)
        best = min(range(k), key=cost)
        fold_of[g] = best
        load[best].update(groups[g])
    for f in range(k):      # safety net: never leave a fold empty
        if f not in fold_of.values():
            sizes = Counter(fold_of.values())
            donor = max(sizes, key=sizes.get)
            g = min((g for g in fold_of if fold_of[g] == donor), key=lambda g: sum(groups[g].values()))
            fold_of[g] = f
    return fold_of


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clips-csv", required=True)
    ap.add_argument("--xc-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    ap.add_argument("--seed", type=int, default=42, help="fold assignment (keep fixed)")
    ap.add_argument("--subset-seed", type=int, default=0,
                    help="choice of training recordists when --train-fraction < 1")
    ap.add_argument("--file-col", default="file-name")
    ap.add_argument("--rec-col", default="rec")
    ap.add_argument("--keep-labels", nargs="+", default=None,
                    help="only use these classes (all others are ignored)")
    ap.add_argument("--drop-labels", nargs="+", default=[],
                    help="classes to ignore, e.g. --drop-labels Other")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    who = {}
    with open(a.xc_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            who[Path(r[a.file_col]).stem] = (r.get(a.rec_col) or "").strip()

        clips = []                                   # (group, label, clip path)
    with open(a.clips_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            stem = Path(r["recording"]).stem
            clips.append((who.get(stem) or f"file:{stem}", r["label"], r["clip"]))

    present = {lab for _, lab, _ in clips}
    if a.keep_labels:
        unknown = set(a.keep_labels) - present
        if unknown:
            raise SystemExit(f"--keep-labels not found in metadata: {sorted(unknown)}; "
                             f"available: {sorted(present)}")
        clips = [c for c in clips if c[1] in a.keep_labels]
    clips = [c for c in clips if c[1] not in a.drop_labels]
    if not clips:
        raise SystemExit("no clips left after label filtering")

    groups = defaultdict(Counter)
    for g, lab, _ in clips:
        groups[g][lab] += 1
    if a.folds > len(groups):
        raise SystemExit(f"{a.folds} folds requested but only {len(groups)} recordists")
    fold_of = assign_groups(groups, a.folds, rng)   # the only use of --seed
    labels = sorted({lab for _, lab, _ in clips})
    print(f"{len(clips)} clips, {len(groups)} recordists, classes: {labels}")

    out = Path(a.out_dir)
    stats = [["fold", "part", "class", "clips", "recordists"]]
    for f in range(a.folds):
        val_groups = {g for g, k in fold_of.items() if k == f}
        pool = sorted(g for g in groups if g not in val_groups)
        train_groups = pool
        if a.train_fraction < 1.0:
            # one shuffle per (subset seed, fold); taking a prefix makes the subsets nested
            order = pool[:]
            random.Random(f"{a.subset_seed}-{f}").shuffle(order)
            n = max(1, round(a.train_fraction * len(pool)))
            train_groups = order[:n]
        train_set = set(train_groups)
        cnt = {"train": Counter(), "val": Counter()}
        rec = {"train": defaultdict(set), "val": defaultdict(set)}
        for g, lab, path in clips:
            part = "val" if g in val_groups else "train" if g in train_set else None
            if part:
                place(path, out / f"fold_{f}" / part / lab / Path(path).name, a.mode)
                cnt[part][lab] += 1
                rec[part][lab].add(g)
        print(f"\nfold_{f}: {len(val_groups)} val recordists, {len(train_groups)} train recordists")
        stats.append([f, "train", "ALL", sum(cnt["train"].values()), len(train_groups)])
        stats.append([f, "val", "ALL", sum(cnt["val"].values()), len(val_groups)])
        for lab in labels:
            for part in ("train", "val"):
                stats.append([f, part, lab, cnt[part][lab], len(rec[part][lab])])
            warn = ""
            if cnt["val"][lab] < MIN_VAL:
                warn += "   <-- few validation clips"
            if cnt["train"][lab] < MIN_TRAIN:
                warn += "   <-- few training clips"
            print(f"  {lab:<30} train {cnt['train'][lab]:>5} ({len(rec['train'][lab]):>2} rec)"
                  f"   val {cnt['val'][lab]:>5} ({len(rec['val'][lab]):>2} rec){warn}")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "fold_stats.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(stats)

    print("\nTrain one model per fold, for example:")
    print(f"  python -m birdnet_analyzer.train {out}/fold_0/train "
          f"-o models/fold_0.tflite --test_data {out}/fold_0/val")


if __name__ == "__main__":
    main()