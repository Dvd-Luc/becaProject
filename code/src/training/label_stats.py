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
    labels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("\\"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            start, end = float(parts[0].replace(",", ".")), float(parts[1].replace(",", "."))
            labels.append((start, end, parts[2].strip()))
    return labels


def recordist_map(xc_csv, file_col, rec_col):
    """Recording stem -> recordist name, from the Xeno-canto metadata CSV.

    Args:
        xc_csv (str | Path): XC metadata.csv.
        file_col (str): File-name column.
        rec_col (str): Recordist column.

    Returns:
        dict[str, str]: Stem -> recordist ("unknown" when empty).
    """
    recordist_of = {}
    with open(xc_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            recordist_of[Path(row[file_col]).stem] = (row.get(rec_col) or "").strip() or "unknown"
    return recordist_of


def table(title, rows, header):
    """Print rows as an aligned text table (first column left-aligned).

    Args:
        title (str): Line printed above the table.
        rows (list[list]): Table rows.
        header (list[str]): Column names.
    """
    print(f"\n{title}")
    widths = [max(len(str(cell)) for cell in col) for col in zip(header, *rows)]
    fmt = "  ".join(f"{{:>{width}}}" if i else f"{{:<{width}}}" for i, width in enumerate(widths))
    print(fmt.format(*header))
    for row in rows:
        print(fmt.format(*row))


def main():
    """CLI entry point: print per-class label statistics, and per-class clip counts with --clips-csv."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--xc-csv", required=True, help="Xeno-canto metadata.csv (for recordists)")
    parser.add_argument("--clips-csv", help="metadata.csv written by make_training_clips.py")
    parser.add_argument("--file-col", default="file-name")
    parser.add_argument("--rec-col", default="rec", help="CSV column with the recordist")
    parser.add_argument("--tau", type=float, default=3.0, help="long-call threshold (s)")
    args = parser.parse_args()

    recordist_of = recordist_map(args.xc_csv, args.file_col, args.rec_col)
    per_cat = defaultdict(list)          # cat -> [(duration, stem)]
    n_files = n_empty = 0
    files_with, recs_all = set(), set()
    for label_path in sorted(Path(args.labels_dir).glob("*.txt")):
        labels = read_labels(label_path)
        n_files += 1
        if not labels:
            n_empty += 1
            continue
        files_with.add(label_path.stem)
        recs_all.add(recordist_of.get(label_path.stem, "unknown"))
        for start, end, cat in labels:
            per_cat[cat].append((end - start, label_path.stem))
    print(f"label files: {n_files} ({n_empty} empty = no calls), "
          f"files with labels: {len(files_with)}, recordists: {len(recs_all)}")

    rows = []
    for cat, items in sorted(per_cat.items()):
        durations = [item[0] for item in items]
        stems = [item[1] for item in items]
        per_file = defaultdict(int)
        for stem in stems:
            per_file[stem] += 1
        n_recordists = len({recordist_of.get(stem, "unknown") for stem in stems})
        pct_long = 100 * sum(duration > args.tau for duration in durations) / len(durations)
        pct_top_file = 100 * max(per_file.values()) / len(durations)
        rows.append([cat, len(durations), len(per_file), n_recordists,
                     f"{sum(durations):.0f}", f"{statistics.median(durations):.2f}",
                     f"{min(durations):.2f}", f"{max(durations):.2f}",
                     f"{pct_long:.0f}", f"{pct_top_file:.0f}"])
    table("Labels per class", rows,
          ["class", "labels", "files", "recordists", "total_s", "median_s", "min_s",
           "max_s", f">{args.tau:g}s_%", "top_file_%"])
    for row in rows:
        if float(row[-1]) > 25:
            print(f"  note: {row[0]}: {row[-1]}% of its labels come from a single file")
        if int(row[3]) < 20:
            print(f"  note: {row[0]}: only {row[3]} different recordists")

    if args.clips_csv:
        clips = defaultdict(list)        # label -> [(kind, recording)]
        with open(args.clips_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                clips[row["label"]].append((row["kind"], Path(row["recording"]).stem))
        rows = []
        for cat, items in sorted(clips.items()):
            stems = [item[1] for item in items]
            rows.append([cat, len(items), sum(kind == "short" for kind, _ in items),
                         sum(kind == "long" for kind, _ in items), len(set(stems)),
                         len({recordist_of.get(stem, "unknown") for stem in stems})])
        table("3 s clips per class", rows,
              ["class", "clips", "from_short", "from_long", "files", "recordists"])


if __name__ == "__main__":
    main()
