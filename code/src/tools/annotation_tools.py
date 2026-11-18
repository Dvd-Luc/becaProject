"""Organise recordings and track Audacity annotation progress.

Folder layout expected
    metadata.csv            CSV with one row per recording
    recordings/             all audio files (names given by the CSV)
    labels/                 Audacity label exports, <recording stem>.txt

Commands
    split   copy/move/symlink recordings into one subfolder per quality
    status  rebuild annotation_status.csv from the labels/ folder and print a summary
    next    list the next recordings still to annotate; with --queue-dir, also
            refresh a batch folder of copies of them

A recording counts as
    todo      no labels/<stem>.txt yet
    done      labels/<stem>.txt exists and contains at least one label
    no_calls  labels/<stem>.txt exists but is empty (listened to, nothing to label)

Examples
    python annotation_tools.py split  --csv metadata.csv --audio-dir recordings --out-dir by_quality
    python annotation_tools.py status --csv metadata.csv --labels-dir labels
    python annotation_tools.py next   --csv metadata.csv --labels-dir labels --n 10 --type drum
    python annotation_tools.py next   --csv metadata.csv --labels-dir labels --audio-dir recordings \
                                      --queue-dir to_label --n 25
(the subcommand comes first, then all the options)
"""
import argparse
import csv
import os
import shutil
from collections import Counter
from pathlib import Path

MANIFEST = ".queue_manifest.txt"   # names of the files placed in the batch folder


def load_rows(path):
    """Rows of a CSV file as dicts.

    Args:
        path (str | Path): CSV file with a header row.

    Returns:
        list[dict[str, str]]: One dict per row, keyed by column name.
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_label_names(path):
    """Label texts in an Audacity export (ignores frequency-range lines).

    Args:
        path (str | Path): Label file.

    Returns:
        list[str]: Label names, in file order.
    """
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("\\"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            names.append(parts[2].strip())
    return names


# ---------- split ----------
def cmd_split(args):
    """`split`: place each recording in <out-dir>/<quality>/ (copy, move or symlink).

    Args:
        args (argparse.Namespace): csv, file_col, q_col, audio_dir, out_dir, mode.
    """
    rows = load_rows(args.csv)
    counts, missing = Counter(), []
    for row in rows:
        name, quality = row[args.file_col].strip(), row[args.q_col].strip().upper() or "unknown"
        src = Path(args.audio_dir) / name
        if not src.exists():
            missing.append(name)
            continue
        dest_dir = Path(args.out_dir) / quality
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        if dest.exists() or dest.is_symlink():
            continue
        if args.mode == "copy":
            shutil.copy2(src, dest)
        elif args.mode == "move":
            shutil.move(src, dest)
        else:
            os.symlink(src.resolve(), dest)
        counts[quality] += 1
    print("placed:", dict(counts))
    if missing:
        print(f"{len(missing)} files listed in the CSV were not found, e.g. {missing[:3]}")


# ---------- status ----------
def build_status(args):
    """One status row per CSV recording: todo / done / no_calls, with its label names.

    Args:
        args (argparse.Namespace): csv, file_col, q_col, type_col, labels_dir.

    Returns:
        list[dict]: Rows with recording, q, type, status, n_labels, label_names.
    """
    rows = load_rows(args.csv)
    status_rows = []
    for row in rows:
        name = row[args.file_col].strip()
        label_path = Path(args.labels_dir) / f"{Path(name).stem}.txt"
        if not label_path.exists():
            status, names = "todo", []
        else:
            names = read_label_names(label_path)
            status = "done" if names else "no_calls"
        status_rows.append({"recording": name,
                            "q": row.get(args.q_col, "").strip().upper(),
                            "type": row.get(args.type_col, ""),
                            "status": status, "n_labels": len(names),
                            "label_names": ";".join(sorted(set(names)))})
    return status_rows


def cmd_status(args):
    """`status`: write the status CSV and print progress and label counts per class.

    Args:
        args (argparse.Namespace): Status options (status_file, allowed, ...).
    """
    status_rows = build_status(args)
    with open(args.status_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(status_rows[0]))
        writer.writeheader()
        writer.writerows(status_rows)

    print(f"{len(status_rows)} recordings")
    for key in ("q", "status"):
        print(f"by {key}:", dict(Counter(row[key] for row in status_rows)))
    by_qs = Counter((row["q"], row["status"]) for row in status_rows)
    print("quality x status:",
          {f"{quality}/{status}": count for (quality, status), count in sorted(by_qs.items())})

    total = Counter()
    for label_path in Path(args.labels_dir).glob("*.txt"):
        total.update(read_label_names(label_path))
    print("labels per class:", dict(total))
    if args.allowed:
        allowed = set(args.allowed)
        unknown_names = {name: count for name, count in total.items() if name not in allowed}
        print("UNKNOWN label names (typos?):", unknown_names if unknown_names else "none")
    print(f"saved {args.status_file}")


# ---------- next ----------
def cmd_next(args):
    """`next`: list the next recordings to annotate, best quality first.

    With --queue-dir, the batch folder is emptied of the files it placed earlier
    (tracked in .queue_manifest.txt) and refilled with copies of the next batch.

    Args:
        args (argparse.Namespace): Next options (n, type, queue_dir, audio_dir, ...).

    Raises:
        SystemExit: If --queue-dir is given without --audio-dir.
    """
    todo = [rec for rec in build_status(args) if rec["status"] == "todo"]
    if args.type:
        todo = [rec for rec in todo if args.type.lower() in rec["type"].lower()]
    todo.sort(key=lambda rec: (rec["q"], rec["recording"]))   # A before B

    if not args.queue_dir:
        print(f"{len(todo)} left to annotate; next {args.n}:")
        for rec in todo[: args.n]:
            print(f"  {rec['recording']}  q={rec['q']}  type={rec['type']}")
        return

    if not args.audio_dir:
        raise SystemExit("--audio-dir is required together with --queue-dir")
    queue = Path(args.queue_dir)
    queue.mkdir(parents=True, exist_ok=True)
    manifest = queue / MANIFEST

    # choose the next recordings still to annotate (that exist on disk)
    batch, missing = [], []
    for rec in todo:
        if len(batch) >= args.n:
            break
        if not (Path(args.audio_dir) / rec["recording"]).exists():
            missing.append(rec["recording"])
            continue
        batch.append(rec)
    wanted = {rec["recording"] for rec in batch}

    # empty the batch folder: remove only what this command put there earlier
    # (listed in the manifest) plus old symlinks; anything else is left alone.
    # Files that stay in the new batch are kept as they are.
    managed = set(manifest.read_text(encoding="utf-8").split("\n")) if manifest.exists() else set()
    kept = []
    for item in queue.iterdir():
        if item.name == MANIFEST:
            continue
        if item.is_symlink() or (item.name in managed and item.name not in wanted):
            item.unlink()
        elif item.name not in managed:
            kept.append(item.name)
    if kept:
        print(f"note: {len(kept)} files in {queue} were not placed by this command "
              f"and were left alone, e.g. {kept[:3]}")

    # fill it with copies
    for rec in batch:
        dest = queue / rec["recording"]
        if not dest.exists():
            shutil.copy2(Path(args.audio_dir) / rec["recording"], dest)
    manifest.write_text("\n".join(sorted(wanted)), encoding="utf-8")

    print(f"{len(todo)} left to annotate; {len(batch)} copied to {queue}:")
    for rec in batch:
        print(f"  {rec['recording']}  q={rec['q']}  type={rec['type']}")
    if missing:
        print(f"{len(missing)} todo files not found in {args.audio_dir}, skipped "
              f"(e.g. {missing[:3]})")


def main():
    """CLI entry point: parse the subcommand and its options, then run it."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--csv", required=True)
    common.add_argument("--file-col", default="file-name", help="CSV column with the file name")
    common.add_argument("--q-col", default="q", help="CSV column with the quality")
    common.add_argument("--type-col", default="type", help="CSV column with the sound type")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub_parser = sub.add_parser("split", parents=[common])
    sub_parser.add_argument("--audio-dir", required=True)
    sub_parser.add_argument("--out-dir", required=True)
    sub_parser.add_argument("--mode", choices=["copy", "move", "symlink"], default="copy")
    sub_parser.set_defaults(fn=cmd_split)

    sub_parser = sub.add_parser("status", parents=[common])
    sub_parser.add_argument("--labels-dir", required=True)
    sub_parser.add_argument("--status-file", default="annotation_status.csv")
    sub_parser.add_argument("--allowed", nargs="*", help="valid label names, to catch typos")
    sub_parser.set_defaults(fn=cmd_status)

    sub_parser = sub.add_parser("next", parents=[common])
    sub_parser.add_argument("--labels-dir", required=True)
    sub_parser.add_argument("--n", type=int, default=10, help="batch size")
    sub_parser.add_argument("--type", help="only recordings whose CSV 'type' contains this text")
    sub_parser.add_argument("--queue-dir", help="batch folder: created if missing, emptied "
                            "(only files it placed itself) and refilled with copies of the next "
                            "recordings to annotate")
    sub_parser.add_argument("--audio-dir", help="folder with the audio files (needed with --queue-dir)")
    sub_parser.set_defaults(fn=cmd_next)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
