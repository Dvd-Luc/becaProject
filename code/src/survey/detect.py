"""Run the BirdNET classifier on the survey clips, then turn windows into events.

    python src/survey/detect.py --dry-run          # what would be analysed, run nothing
    python src/survey/detect.py                    # analyse new clips + rebuild tables
    python src/survey/detect.py --split train      # only clips of one split
    python src/survey/detect.py --post-only        # re-threshold / re-merge, no BirdNET
    python src/survey/detect.py --min-conf 0.35    # other event threshold than the config
    python src/survey/detect.py --gen 1            # model generation 1 -> generations/gen_1/detections

Two steps:
1. BirdNET (detection.model) runs on every clip of survey.dir/manifest.csv that is cut
   and has no raw result yet. It keeps windows >= detection.raw_min_conf, so step 2 can
   be redone with another min_conf without re-running the model. Resumable: a clip with
   a raw CSV is skipped, so run it again after each new drive is sampled.
   Clips in clips/ that are not in the manifest (stale) are ignored.
2. Windows >= detection.min_conf are merged per clip and label into events when the gap
   between them is <= detection.merge_gap_s (optionally capped at max_events_per_clip,
   highest confidence kept).

Writes, under detection.dir:
    raw/<clip_id>.BirdNET.results.csv   BirdNET output, one row per window and label
    raw/run_params.csv                  model + overlap + raw_min_conf of the raw results
    events.csv                          one row per event, with the clip's manifest fields
    clip_summary.csv                    one row per analysed clip x label (zeros included)

If the model, overlap or raw_min_conf in the config changes, the raw results no longer
match: the script stops; delete raw/ (or use --force) to re-run everything.

Run from code/.
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import load

MANIFEST_KEYS = ["clip_id", "site", "split", "period", "drive"]
EVENT_HEADER = MANIFEST_KEYS + ["label", "start_s", "end_s", "n_windows",
                                "max_conf", "mean_conf"]
SUMMARY_HEADER = MANIFEST_KEYS + ["label", "n_events", "n_windows", "max_conf"]
PARAM_KEYS = ["model", "overlap", "raw_min_conf"]


def read_dicts(path):
    """Rows of a CSV file as dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, header, rows):
    """Write a CSV file with a header row, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def raw_path(raw_dir, cid):
    """Path of BirdNET's raw result file for one clip."""
    return raw_dir / f"{cid}.BirdNET.results.csv"


def check_params(raw_dir, params, force):
    """Raw results are only valid for the model/overlap/raw_min_conf they were made with."""
    p = raw_dir / "run_params.csv"
    if p.exists():
        old = read_dicts(p)[0]
        if old != params:
            diff = {k: (old.get(k), params[k]) for k in PARAM_KEYS if old.get(k) != params[k]}
            if not force:
                raise SystemExit(f"raw results in {raw_dir} were made with other settings "
                                 f"(old, new): {diff}\ndelete {raw_dir} or use --force")
            print(f"--force: re-running all clips, settings changed: {diff}")
            for f in raw_dir.glob("*.BirdNET.results.csv"):
                f.unlink()
    write_rows(p, PARAM_KEYS, [[params[k] for k in PARAM_KEYS]])


def run_birdnet(todo, raw_dir, d, threads):
    """Analyse the clips through a folder of links, so only manifest clips are seen."""
    with tempfile.TemporaryDirectory(prefix="detect_") as tmp:
        for r in todo:
            (Path(tmp) / f"{r['clip_id']}.wav").symlink_to(Path(r["clip_path"]).resolve())
        cmd = [sys.executable, "-m", "birdnet_analyzer.analyze", tmp,
               "-o", str(raw_dir), "-c", d["model"],
               "--min_conf", str(d["raw_min_conf"]), "--overlap", str(d["overlap"]),
               "--rtype", "csv"]
        if threads:
            cmd += ["-t", str(threads)]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


def merge_windows(windows, gap):
    """windows: [(start, end, conf)] of one clip and label -> events."""
    events = []
    for s, e, c in sorted(windows):
        if events and s - events[-1]["end"] <= gap:
            ev = events[-1]
            ev["end"] = max(ev["end"], e)
            ev["confs"].append(c)
        else:
            events.append({"start": s, "end": e, "confs": [c]})
    return events


def post(clips, raw_dir, d, labels):
    """Turn raw BirdNET windows into events and per-clip summaries.

    Keeps windows >= detection.min_conf, merges them per clip and label when the gap
    is <= merge_gap_s, optionally caps the events per clip (highest confidence kept).
    Returns (event rows, summary rows); every clip x label gets a summary row, zeros included.
    """
    min_conf, gap, cap = d["min_conf"], d["merge_gap_s"], d.get("max_events_per_clip")
    events_out, summary_out = [], []
    for r in clips:
        windows = defaultdict(list)
        for w in read_dicts(raw_path(raw_dir, r["clip_id"])):
            c = float(w["Confidence"])
            if c >= min_conf:
                label = w["Common name"] or w["Scientific name"]
                windows[label].append((float(w["Start (s)"]), float(w["End (s)"]), c))
        clip_events = []
        for label, ws in windows.items():
            for ev in merge_windows(ws, gap):
                clip_events.append((label, ev))
        if cap:
            clip_events = sorted(clip_events, key=lambda x: -max(x[1]["confs"]))[:cap]
        key = [r[k] for k in MANIFEST_KEYS]
        for label, ev in sorted(clip_events, key=lambda x: x[1]["start"]):
            cs = ev["confs"]
            events_out.append(key + [label, f"{ev['start']:.1f}", f"{ev['end']:.1f}",
                                     len(cs), f"{max(cs):.4f}", f"{sum(cs) / len(cs):.4f}"])
        for label in labels:
            evs = [ev for lab, ev in clip_events if lab == label]
            nw = sum(len(ev["confs"]) for ev in evs)
            mx = max((max(ev["confs"]) for ev in evs), default=None)
            summary_out.append(key + [label, len(evs), nw, "" if mx is None else f"{mx:.4f}"])
    return events_out, summary_out


def main():
    """CLI entry point: run BirdNET on the clips still to analyse, then rebuild the tables."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="config/project.yaml")
    ap.add_argument("--split", help="only clips of this split (e.g. train, heldout)")
    ap.add_argument("--threads", type=int, help="BirdNET CPU threads (default: BirdNET's)")
    ap.add_argument("--gen", type=int, help="use generations.dir/gen_K/model.tflite, "
                    "write to gen_K/detections/")
    ap.add_argument("--min-conf", type=float,
                    help="event threshold for this run (default: detection.min_conf)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--post-only", action="store_true", help="skip BirdNET, rebuild tables")
    ap.add_argument("--force", action="store_true",
                    help="re-run all clips if model/overlap/raw_min_conf changed")
    a = ap.parse_args()
    cfg = load(a.config)
    s, d = cfg["survey"], cfg["detection"]
    if a.gen is not None:                  # a model generation from make_generation.py
        gd = Path(cfg["generations"]["dir"]) / f"gen_{a.gen}"
        d["model"], d["dir"] = str(gd / "model.tflite"), str(gd / "detections")
    if a.min_conf is not None:
        d["min_conf"] = a.min_conf
    if d["raw_min_conf"] > d["min_conf"]:
        raise SystemExit("detection.raw_min_conf must be <= detection.min_conf")
    model = Path(d["model"])
    labels_file = model.with_name(model.stem + "_Labels.txt")
    if not model.exists() or not labels_file.exists():
        raise SystemExit(f"model or labels not found: {model}, {labels_file}")
    labels = [l.strip() for l in labels_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    survey_dir, out = Path(s["dir"]), Path(d["dir"])
    raw_dir = out / "raw"
    manifest = read_dicts(survey_dir / "manifest.csv")
    if a.split:
        manifest = [r for r in manifest if r["split"] == a.split]
    cut = [r for r in manifest if Path(r["clip_path"]).exists()]
    todo = [r for r in cut if not raw_path(raw_dir, r["clip_id"]).exists()]
    listed = {Path(r["clip_path"]).name for r in read_dicts(survey_dir / "manifest.csv")}
    stale = [p for p in (survey_dir / "clips").glob("*.wav") if p.name not in listed]
    print(f"manifest: {len(manifest)} clips{f' ({a.split})' if a.split else ''}, "
          f"{len(cut)} cut, {len(cut) - len(todo)} already analysed, {len(todo)} to analyse")
    if stale:
        print(f"note: {len(stale)} files in {survey_dir / 'clips'} are not in the manifest "
              f"(ignored), e.g. {stale[0].name}")
    if a.dry_run:
        return

    params = {"model": str(model), "overlap": str(d["overlap"]),
              "raw_min_conf": str(d["raw_min_conf"])}
    if not a.post_only:
        raw_dir.mkdir(parents=True, exist_ok=True)
        check_params(raw_dir, params, a.force)
        todo = [r for r in cut if not raw_path(raw_dir, r["clip_id"]).exists()]
        if todo:
            run_birdnet(todo, raw_dir, d, a.threads)
        failed = [r["clip_id"] for r in todo if not raw_path(raw_dir, r["clip_id"]).exists()]
        if failed:
            print(f"warning: no BirdNET result for {len(failed)} clips, e.g. {failed[:3]}")

    done = [r for r in cut if raw_path(raw_dir, r["clip_id"]).exists()]
    events, summary = post(done, raw_dir, d, labels)
    write_rows(out / "events.csv", EVENT_HEADER, events)
    write_rows(out / "clip_summary.csv", SUMMARY_HEADER, summary)
    n_pos = defaultdict(int)
    for row in summary:
        n_pos[row[5]] += row[6] > 0
    print(f"{len(done)} clips analysed, {len(events)} events at min_conf {d['min_conf']}; "
          f"clips with >= 1 event: {dict(n_pos)}")
    print(f"wrote {out / 'events.csv'}, {out / 'clip_summary.csv'}")


if __name__ == "__main__":
    main()
