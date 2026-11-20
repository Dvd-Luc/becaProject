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
    sample-test
            draw and freeze the XC test sample: recordings not yet annotated, by
            recordists that never appear in the annotated (training) recordings.
            Settings: config section xc_test. XC 'type' tags (often wrong) only enrich
            the strata; the truth comes from annotating the sample later.

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
    (run from code/ for sample-test, which reads config/project.yaml)
    python src/tools/annotation_tools.py sample-test --csv data/xc_gallinago_gallinago/metadata.csv \
                                      --labels-dir data/annotation_labels --dry-run
    python src/tools/annotation_tools.py sample-test --csv data/xc_gallinago_gallinago/metadata.csv \
                                      --labels-dir data/annotation_labels
(the subcommand comes first, then all the options)

Writes:

    status       <--status-file> (default annotation_status.csv)
    next         <--queue-dir>/ copies + .queue_manifest.txt
    sample-test  <xc_test.dir>/sample.csv (frozen: reused as is, --force to redraw)
                 <xc_test.dir>/sample_settings.json (settings used; a change stops the run)
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import load

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


# ---------- sample-test ----------
def length_seconds(text):
    """Duration of an XC 'length' field ("m:ss" or "h:mm:ss") in seconds.

    Args:
        text (str): XC length, e.g. "1:34".

    Returns:
        float: Seconds; infinity when the field is empty or malformed.
    """
    try:
        seconds = 0.0
        for part in text.strip().split(":"):
            seconds = seconds * 60 + float(part)
        return seconds
    except ValueError:
        return float("inf")


def draw_test_sample(rows, used_recordists, test_cfg, args):
    """Seeded stratified draw of unseen-recordist recordings, capped per recordist.

    Strata are filled in config order; a recording is picked at most once.

    Args:
        rows (list[dict]): XC metadata rows still to annotate (status todo).
        used_recordists (set[str]): Recordists of the annotated (training) recordings.
        test_cfg (dict): xc_test config section (seed, strata, max_per_recordist,
            max_length_s, max_recordist_recordings).
        args (argparse.Namespace): group_col, type_col.

    Returns:
        list[dict]: Sampled metadata rows, each with an added 'stratum' column.
    """
    rng = random.Random(test_cfg["seed"])
    # a sampled recordist is lost for training: skip prolific ones to keep the training pool
    pool_size = Counter(row[args.group_col].strip() for row in rows)
    eligible = [row for row in rows
                if row[args.group_col].strip() not in used_recordists
                and pool_size[row[args.group_col].strip()] <= test_cfg["max_recordist_recordings"]
                and length_seconds(row.get("length", "")) <= test_cfg["max_length_s"]]
    per_recordist, picked, sample = Counter(), set(), []
    for stratum, stratum_cfg in test_cfg["strata"].items():
        keyword = stratum_cfg["keyword"].lower()
        candidates = [row for row in eligible if keyword in row[args.type_col].lower()]
        rng.shuffle(candidates)
        n_taken = 0
        for row in candidates:
            if n_taken >= stratum_cfg["n"]:
                break
            name, recordist = row[args.file_col], row[args.group_col].strip()
            if name in picked or per_recordist[recordist] >= test_cfg["max_per_recordist"]:
                continue
            sample.append({"stratum": stratum, **row})
            picked.add(name)
            per_recordist[recordist] += 1
            n_taken += 1
        if n_taken < stratum_cfg["n"]:
            print(f"warning: stratum {stratum}: only {n_taken} of {stratum_cfg['n']} recordings "
                  f"available")
    return sample


def cmd_sample_test(args):
    """`sample-test`: draw and freeze the XC test sample from unseen recordists.

    Args:
        args (argparse.Namespace): csv, labels_dir, config, group_col, dry_run, force, ...

    Raises:
        SystemExit: If the settings changed under an existing sample (without --force),
            or if the frozen sample shares a recordist with the annotated recordings.
    """
    test_cfg = load(args.config)["xc_test"]
    out_dir = Path(test_cfg["dir"])
    sample_path, settings_path = out_dir / "sample.csv", out_dir / "sample_settings.json"
    settings = {key: test_cfg[key] for key in ("seed", "strata", "max_per_recordist", "max_length_s",
                                               "max_recordist_recordings")}
    settings.update(csv=str(args.csv), labels_dir=str(args.labels_dir))

    rows = load_rows(args.csv)
    status = {rec["recording"]: rec["status"] for rec in build_status(args)}
    used_recordists = {row[args.group_col].strip() for row in rows
                       if status[row[args.file_col].strip()] != "todo"}
    todo_rows = [row for row in rows if status[row[args.file_col].strip()] == "todo"]

    if sample_path.exists() and not args.force:
        old_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if old_settings != settings:
            raise SystemExit(f"settings changed since {sample_path} was drawn:\n  was {old_settings}\n"
                             f"  now {settings}\nrestore them, or --force to redraw the sample")
        sample = load_rows(sample_path)
        print(f"frozen sample reused: {sample_path}")
    else:
        sample = draw_test_sample(todo_rows, used_recordists, test_cfg, args)

    overlap = {row[args.group_col].strip() for row in sample} & used_recordists
    if overlap:
        raise SystemExit(f"test sample shares recordists with the annotated recordings: "
                         f"{sorted(overlap)} (were they annotated after the draw?)")
    sample_recordists = Counter(row[args.group_col].strip() for row in sample)
    print(f"{len(used_recordists)} recordists in annotated recordings (excluded); "
          f"{len(todo_rows)} recordings still to annotate")
    print(f"sample: {len(sample)} recordings, {len(sample_recordists)} recordists "
          f"(max {max(sample_recordists.values(), default=0)} per recordist), "
          f"0 shared with training")
    print("per stratum:", dict(Counter(row["stratum"] for row in sample)))
    n_locked = sum(row[args.group_col].strip() in sample_recordists for row in todo_rows) - len(sample)
    print(f"{n_locked} other unannotated recordings by these recordists become unavailable "
          f"for training")
    total_s = sum(length_seconds(row.get("length", "")) for row in sample)
    print(f"audio to annotate: {total_s / 60:.0f} min")

    if args.dry_run:
        print("dry run: nothing written")
        return
    if sample_path.exists() and not args.force:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample[0]))
        writer.writeheader()
        writer.writerows(sample)
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"wrote {sample_path} and {settings_path.name}")


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

    sub_parser = sub.add_parser("sample-test", parents=[common])
    sub_parser.add_argument("--labels-dir", required=True,
                            help="training labels: their recordists are excluded")
    sub_parser.add_argument("--config", default="config/project.yaml", help="reads section xc_test")
    sub_parser.add_argument("--group-col", default="rec", help="CSV column with the recordist")
    sub_parser.add_argument("--dry-run", action="store_true", help="show the draw, write nothing")
    sub_parser.add_argument("--force", action="store_true", help="redraw over an existing sample")
    sub_parser.set_defaults(fn=cmd_sample_test)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
