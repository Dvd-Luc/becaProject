"""Summarise your annotations (and, optionally, the training clips cut from them).

    python label_stats.py --labels-dir labels --xc-csv xc_gallinago_gallinago/metadata.csv
    python label_stats.py --labels-dir labels --xc-csv .../metadata.csv --clips-csv clips/metadata.csv

For every class: number of labels, total and typical duration, how many different
recordings and recordists they come from, and how concentrated they are in one file.
With --clips-csv, the same for the 3 s clips made by make_training_clips.py.
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def read_labels(path):
    """Read an Audacity label file.

    Args:
        path (str | Path): Label file.

    Returns:
        list[tuple[float, float, str]]: (start s, end s, label).
    """
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("\\"):
            continue
        p = line.split("\t")
        if len(p) >= 3:
            out.append((float(p[0].replace(",", ".")), float(p[1].replace(",", ".")), p[2].strip()))
    return out


def recordist_map(xc_csv, file_col, rec_col):
    """Recording stem -> recordist name, from the Xeno-canto metadata CSV.

    Args:
        xc_csv (str | Path): XC metadata.csv.
        file_col (str): File-name column.
        rec_col (str): Recordist column.

    Returns:
        dict[str, str]: Stem -> recordist ("unknown" when empty).
    """
    m = {}
    with open(xc_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m[Path(r[file_col]).stem] = (r.get(rec_col) or "").strip() or "unknown"
    return m


def table(title, rows, header):
    """Print rows as an aligned text table (first column left-aligned).

    Args:
        title (str): Line printed above the table.
        rows (list[list]): Table rows.
        header (list[str]): Column names.
    """
    print(f"\n{title}")
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = "  ".join(f"{{:>{w}}}" if i else f"{{:<{w}}}" for i, w in enumerate(widths))
    print(fmt.format(*header))
    for r in rows:
        print(fmt.format(*r))


def main():
    """CLI entry point: print per-class label statistics, and per-class clip counts with --clips-csv."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--xc-csv", required=True, help="Xeno-canto metadata.csv (for recordists)")
    ap.add_argument("--clips-csv", help="metadata.csv written by make_training_clips.py")
    ap.add_argument("--file-col", default="file-name")
    ap.add_argument("--rec-col", default="rec", help="CSV column with the recordist")
    ap.add_argument("--tau", type=float, default=3.0, help="long-call threshold (s)")
    a = ap.parse_args()

    who = recordist_map(a.xc_csv, a.file_col, a.rec_col)
    per_cat = defaultdict(list)          # cat -> [(duration, stem)]
    n_files = n_empty = 0
    files_with, recs_all = set(), set()
    for lab in sorted(Path(a.labels_dir).glob("*.txt")):
        labels = read_labels(lab)
        n_files += 1
        if not labels:
            n_empty += 1
            continue
        files_with.add(lab.stem)
        recs_all.add(who.get(lab.stem, "unknown"))
        for s, e, c in labels:
            per_cat[c].append((e - s, lab.stem))
    print(f"label files: {n_files} ({n_empty} empty = no calls), "
          f"files with labels: {len(files_with)}, recordists: {len(recs_all)}")

    rows = []
    for c, items in sorted(per_cat.items()):
        d = [x[0] for x in items]
        stems = [x[1] for x in items]
        per_file = defaultdict(int)
        for st in stems:
            per_file[st] += 1
        rows.append([c, len(d), len(per_file), len({who.get(s, "unknown") for s in stems}),
                     f"{sum(d):.0f}", f"{statistics.median(d):.2f}", f"{min(d):.2f}",
                     f"{max(d):.2f}", f"{100 * sum(x > a.tau for x in d) / len(d):.0f}",
                     f"{100 * max(per_file.values()) / len(d):.0f}"])
    table("Labels per class", rows,
          ["class", "labels", "files", "recordists", "total_s", "median_s", "min_s",
           "max_s", f">{a.tau:g}s_%", "top_file_%"])
    for r in rows:
        if float(r[-1]) > 25:
            print(f"  note: {r[0]}: {r[-1]}% of its labels come from a single file")
        if int(r[3]) < 20:
            print(f"  note: {r[0]}: only {r[3]} different recordists")

    if a.clips_csv:
        clips = defaultdict(list)        # label -> [(kind, recording)]
        with open(a.clips_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                clips[r["label"]].append((r["kind"], Path(r["recording"]).stem))
        rows = []
        for c, items in sorted(clips.items()):
            stems = [x[1] for x in items]
            rows.append([c, len(items), sum(k == "short" for k, _ in items),
                         sum(k == "long" for k, _ in items), len(set(stems)),
                         len({who.get(s, "unknown") for s in stems})])
        table("3 s clips per class", rows,
              ["class", "clips", "from_short", "from_long", "files", "recordists"])


if __name__ == "__main__":
    main()