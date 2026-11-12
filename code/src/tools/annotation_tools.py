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
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_label_names(path):
    """Label texts in an Audacity export (ignores frequency-range lines)."""
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("\\"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            names.append(parts[2].strip())
    return names


# ---------- split ----------
def cmd_split(a):
    rows = load_rows(a.csv)
    counts, missing = Counter(), []
    for r in rows:
        name, q = r[a.file_col].strip(), r[a.q_col].strip().upper() or "unknown"
        src = Path(a.audio_dir) / name
        if not src.exists():
            missing.append(name)
            continue
        dest_dir = Path(a.out_dir) / q
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        if dest.exists() or dest.is_symlink():
            continue
        if a.mode == "copy":
            shutil.copy2(src, dest)
        elif a.mode == "move":
            shutil.move(src, dest)
        else:
            os.symlink(src.resolve(), dest)
        counts[q] += 1
    print("placed:", dict(counts))
    if missing:
        print(f"{len(missing)} files listed in the CSV were not found, e.g. {missing[:3]}")


# ---------- status ----------
def build_status(a):
    rows = load_rows(a.csv)
    out = []
    for r in rows:
        name = r[a.file_col].strip()
        lab = Path(a.labels_dir) / f"{Path(name).stem}.txt"
        if not lab.exists():
            status, names = "todo", []
        else:
            names = read_label_names(lab)
            status = "done" if names else "no_calls"
        out.append({"recording": name,
                    "q": r.get(a.q_col, "").strip().upper(),
                    "type": r.get(a.type_col, ""),
                    "status": status, "n_labels": len(names),
                    "label_names": ";".join(sorted(set(names)))})
    return out


def cmd_status(a):
    out = build_status(a)
    with open(a.status_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    print(f"{len(out)} recordings")
    for key in ("q", "status"):
        print(f"by {key}:", dict(Counter(o[key] for o in out)))
    by_qs = Counter((o["q"], o["status"]) for o in out)
    print("quality x status:", {f"{q}/{s}": n for (q, s), n in sorted(by_qs.items())})

    total = Counter()
    for lab in Path(a.labels_dir).glob("*.txt"):
        total.update(read_label_names(lab))
    print("labels per class:", dict(total))
    if a.allowed:
        allowed = set(a.allowed)
        bad = {n: c for n, c in total.items() if n not in allowed}
        print("UNKNOWN label names (typos?):", bad if bad else "none")
    print(f"saved {a.status_file}")


# ---------- next ----------
def cmd_next(a):
    todo = [o for o in build_status(a) if o["status"] == "todo"]
    if a.type:
        todo = [o for o in todo if a.type.lower() in o["type"].lower()]
    todo.sort(key=lambda o: (o["q"], o["recording"]))   # A before B

    if not a.queue_dir:
        print(f"{len(todo)} left to annotate; next {a.n}:")
        for o in todo[: a.n]:
            print(f"  {o['recording']}  q={o['q']}  type={o['type']}")
        return

    if not a.audio_dir:
        raise SystemExit("--audio-dir is required together with --queue-dir")
    queue = Path(a.queue_dir)
    queue.mkdir(parents=True, exist_ok=True)
    manifest = queue / MANIFEST

    # choose the next recordings still to annotate (that exist on disk)
    batch, missing = [], []
    for o in todo:
        if len(batch) >= a.n:
            break
        if not (Path(a.audio_dir) / o["recording"]).exists():
            missing.append(o["recording"])
            continue
        batch.append(o)
    wanted = {o["recording"] for o in batch}

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
    for o in batch:
        dest = queue / o["recording"]
        if not dest.exists():
            shutil.copy2(Path(a.audio_dir) / o["recording"], dest)
    manifest.write_text("\n".join(sorted(wanted)), encoding="utf-8")

    print(f"{len(todo)} left to annotate; {len(batch)} copied to {queue}:")
    for o in batch:
        print(f"  {o['recording']}  q={o['q']}  type={o['type']}")
    if missing:
        print(f"{len(missing)} todo files not found in {a.audio_dir}, skipped "
              f"(e.g. {missing[:3]})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--csv", required=True)
    common.add_argument("--file-col", default="file-name", help="CSV column with the file name")
    common.add_argument("--q-col", default="q", help="CSV column with the quality")
    common.add_argument("--type-col", default="type", help="CSV column with the sound type")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split", parents=[common])
    s.add_argument("--audio-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--mode", choices=["copy", "move", "symlink"], default="copy")
    s.set_defaults(fn=cmd_split)

    s = sub.add_parser("status", parents=[common])
    s.add_argument("--labels-dir", required=True)
    s.add_argument("--status-file", default="annotation_status.csv")
    s.add_argument("--allowed", nargs="*", help="valid label names, to catch typos")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("next", parents=[common])
    s.add_argument("--labels-dir", required=True)
    s.add_argument("--n", type=int, default=10, help="batch size")
    s.add_argument("--type", help="only recordings whose CSV 'type' contains this text")
    s.add_argument("--queue-dir", help="batch folder: created if missing, emptied "
                   "(only files it placed itself) and refilled with copies of the next "
                   "recordings to annotate")
    s.add_argument("--audio-dir", help="folder with the audio files (needed with --queue-dir)")
    s.set_defaults(fn=cmd_next)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()