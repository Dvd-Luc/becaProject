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
    """Put src at dst; does nothing if dst exists.

    Args:
        src (str | Path): Existing clip.
        dst (Path): Destination (parent folders are created).
        mode (str): "hardlink" (falls back to copy), "symlink" or "copy".
    """
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


def assign_groups(groups, n_folds, rng):
    """Greedy balanced assignment of groups to k folds, class by class.

    Args:
        groups (dict[str, Counter]): Recordist -> clips per class.
        n_folds (int): Number of folds.
        rng (random.Random): Breaks ties in the processing order.

    Returns:
        dict[str, int]: Recordist -> fold index; no fold is left empty.
    """
    totals = Counter()
    for class_counts in groups.values():
        totals.update(class_counts)
    def share(group):
        """Sum over classes of the fraction of each class's clips that this group holds.

        Args:
            group (str): Recordist.

        Returns:
            float: Larger for groups that weigh more in the rare classes.
        """
        return sum(n_clips / totals[cls] for cls, n_clips in groups[group].items())

    order = sorted(groups, key=lambda group: (-share(group), rng.random()))
    load = [Counter() for _ in range(n_folds)]
    fold_of = {}
    for group in order:
        def cost(fold):        # change in squared distance to the ideal share 1/k of each class
            """Change in squared distance to the ideal share 1/k of each class if g goes to fold f.

            Args:
                fold (int): Candidate fold.

            Returns:
                float: The cost (lower is better).
            """
            return sum(((load[fold][cls] + groups[group][cls]) / totals[cls] - 1 / n_folds) ** 2
                       - (load[fold][cls] / totals[cls] - 1 / n_folds) ** 2 for cls in totals)
        best = min(range(n_folds), key=cost)
        fold_of[group] = best
        load[best].update(groups[group])
    for fold in range(n_folds):      # safety net: never leave a fold empty
        if fold not in fold_of.values():
            sizes = Counter(fold_of.values())
            donor = max(sizes, key=sizes.get)
            group = min((group for group in fold_of if fold_of[group] == donor),
                        key=lambda group: sum(groups[group].values()))
            fold_of[group] = fold
    return fold_of


def main():
    """CLI entry point: group clips by recordist, assign folds, fill fold_K/{train,val}, write stats."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--clips-csv", required=True)
    parser.add_argument("--xc-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--seed", type=int, default=42, help="fold assignment (keep fixed)")
    parser.add_argument("--subset-seed", type=int, default=0,
                        help="choice of training recordists when --train-fraction < 1")
    parser.add_argument("--file-col", default="file-name")
    parser.add_argument("--rec-col", default="rec")
    parser.add_argument("--keep-labels", nargs="+", default=None,
                        help="only use these classes (all others are ignored)")
    parser.add_argument("--drop-labels", nargs="+", default=[],
                        help="classes to ignore, e.g. --drop-labels Other")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    recordist_of = {}
    with open(args.xc_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            recordist_of[Path(row[args.file_col]).stem] = (row.get(args.rec_col) or "").strip()

        clips = []                                   # (group, label, clip path)
    with open(args.clips_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stem = Path(row["recording"]).stem
            clips.append((recordist_of.get(stem) or f"file:{stem}", row["label"], row["clip"]))

    present = {label for _, label, _ in clips}
    if args.keep_labels:
        unknown = set(args.keep_labels) - present
        if unknown:
            raise SystemExit(f"--keep-labels not found in metadata: {sorted(unknown)}; "
                             f"available: {sorted(present)}")
        clips = [clip for clip in clips if clip[1] in args.keep_labels]
    clips = [clip for clip in clips if clip[1] not in args.drop_labels]
    if not clips:
        raise SystemExit("no clips left after label filtering")

    groups = defaultdict(Counter)
    for group, label, _ in clips:
        groups[group][label] += 1
    if args.folds > len(groups):
        raise SystemExit(f"{args.folds} folds requested but only {len(groups)} recordists")
    fold_of = assign_groups(groups, args.folds, rng)   # the only use of --seed
    labels = sorted({label for _, label, _ in clips})
    print(f"{len(clips)} clips, {len(groups)} recordists, classes: {labels}")

    out_dir = Path(args.out_dir)
    stats = [["fold", "part", "class", "clips", "recordists"]]
    for fold in range(args.folds):
        val_groups = {group for group, group_fold in fold_of.items() if group_fold == fold}
        pool = sorted(group for group in groups if group not in val_groups)
        train_groups = pool
        if args.train_fraction < 1.0:
            # one shuffle per (subset seed, fold); taking a prefix makes the subsets nested
            order = pool[:]
            random.Random(f"{args.subset_seed}-{fold}").shuffle(order)
            n_train = max(1, round(args.train_fraction * len(pool)))
            train_groups = order[:n_train]
        train_set = set(train_groups)
        counts = {"train": Counter(), "val": Counter()}
        recordists = {"train": defaultdict(set), "val": defaultdict(set)}
        for group, label, path in clips:
            part = "val" if group in val_groups else "train" if group in train_set else None
            if part:
                place(path, out_dir / f"fold_{fold}" / part / label / Path(path).name, args.mode)
                counts[part][label] += 1
                recordists[part][label].add(group)
        print(f"\nfold_{fold}: {len(val_groups)} val recordists, {len(train_groups)} train recordists")
        stats.append([fold, "train", "ALL", sum(counts["train"].values()), len(train_groups)])
        stats.append([fold, "val", "ALL", sum(counts["val"].values()), len(val_groups)])
        for label in labels:
            for part in ("train", "val"):
                stats.append([fold, part, label, counts[part][label], len(recordists[part][label])])
            warn = ""
            if counts["val"][label] < MIN_VAL:
                warn += "   <-- few validation clips"
            if counts["train"][label] < MIN_TRAIN:
                warn += "   <-- few training clips"
            n_train_rec, n_val_rec = len(recordists["train"][label]), len(recordists["val"][label])
            print(f"  {label:<30} train {counts['train'][label]:>5} ({n_train_rec:>2} rec)"
                  f"   val {counts['val'][label]:>5} ({n_val_rec:>2} rec){warn}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fold_stats.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(stats)

    print("\nTrain one model per fold, for example:")
    print(f"  python -m birdnet_analyzer.train {out_dir}/fold_0/train "
          f"-o models/fold_0.tflite --test_data {out_dir}/fold_0/val")


if __name__ == "__main__":
    main()
