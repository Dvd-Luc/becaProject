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
    """Rows of a CSV file as dicts.

    Args:
        path (str | Path): CSV file with a header row.

    Returns:
        list[dict[str, str]]: One dict per row, keyed by column name.
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, header, rows):
    """Write a CSV file with a header row, creating parent folders.

    Args:
        path (Path): Output CSV file.
        header (list[str]): Column names.
        rows (list[list]): Rows, in header order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def raw_path(raw_dir, clip_id):
    """Path of BirdNET's raw result file for one clip.

    Args:
        raw_dir (Path): detection.dir/raw.
        clip_id (str): Clip id (manifest clip_id).

    Returns:
        Path: raw_dir/<cid>.BirdNET.results.csv.
    """
    return raw_dir / f"{clip_id}.BirdNET.results.csv"


def check_params(raw_dir, params, force):
    """Make sure the raw results were made with the current settings, then record them.

    Args:
        raw_dir (Path): detection.dir/raw.
        params (dict[str, str]): model, overlap and raw_min_conf of this run.
        force (bool): If the settings changed, delete the raw results instead of stopping.

    Raises:
        SystemExit: If the settings changed and force is False.
    """
    params_path = raw_dir / "run_params.csv"
    if params_path.exists():
        old = read_dicts(params_path)[0]
        if old != params:
            diff = {key: (old.get(key), params[key]) for key in PARAM_KEYS if old.get(key) != params[key]}
            if not force:
                raise SystemExit(f"raw results in {raw_dir} were made with other settings "
                                 f"(old, new): {diff}\ndelete {raw_dir} or use --force")
            print(f"--force: re-running all clips, settings changed: {diff}")
            for path in raw_dir.glob("*.BirdNET.results.csv"):
                path.unlink()
    write_rows(params_path, PARAM_KEYS, [[params[key] for key in PARAM_KEYS]])


def run_birdnet(todo, raw_dir, detection_cfg, threads):
    """Analyse the clips through a folder of links, so only manifest clips are seen.

    Args:
        todo (list[dict]): Manifest rows of the clips to analyse.
        raw_dir (Path): Output folder for BirdNET's CSVs.
        detection_cfg (dict): The detection config section (model, raw_min_conf, overlap).
        threads (int | None): BirdNET CPU threads (None: BirdNET's default).

    Raises:
        subprocess.CalledProcessError: If BirdNET fails.
    """
    with tempfile.TemporaryDirectory(prefix="detect_") as tmp:
        for clip in todo:
            (Path(tmp) / f"{clip['clip_id']}.wav").symlink_to(Path(clip["clip_path"]).resolve())
        cmd = [sys.executable, "-m", "birdnet_analyzer.analyze", tmp,
               "-o", str(raw_dir), "-c", detection_cfg["model"],
               "--min_conf", str(detection_cfg["raw_min_conf"]), "--overlap", str(detection_cfg["overlap"]),
               "--rtype", "csv"]
        if threads:
            cmd += ["-t", str(threads)]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


def merge_windows(windows, gap):
    """Merge the windows of one clip and label into events.

    Args:
        windows (list[tuple[float, float, float]]): (start s, end s, confidence).
        gap (float): Maximum gap in seconds between two windows of one event.

    Returns:
        list[dict]: Events with start (float), end (float) and confs (list[float]).
    """
    events = []
    for start, end, conf in sorted(windows):
        if events and start - events[-1]["end"] <= gap:
            event = events[-1]
            event["end"] = max(event["end"], end)
            event["confs"].append(conf)
        else:
            events.append({"start": start, "end": end, "confs": [conf]})
    return events


def post(clips, raw_dir, detection_cfg, labels):
    """Turn raw BirdNET windows into events and per-clip summaries.

    Keeps windows >= detection.min_conf, merges them per clip and label when the gap is
    <= merge_gap_s, optionally caps the events per clip (highest confidence kept).

    Args:
        clips (list[dict]): Manifest rows of the analysed clips.
        raw_dir (Path): detection.dir/raw.
        detection_cfg (dict): The detection config section (min_conf, merge_gap_s, max_events_per_clip).
        labels (list[str]): Model classes; each gets a summary row per clip.

    Returns:
        tuple[list[list], list[list]]: (event rows in EVENT_HEADER order,
            summary rows in SUMMARY_HEADER order, zeros included).
    """
    min_conf, gap = detection_cfg["min_conf"], detection_cfg["merge_gap_s"]
    max_events = detection_cfg.get("max_events_per_clip")
    events_out, summary_out = [], []
    for clip in clips:
        windows = defaultdict(list)
        for window in read_dicts(raw_path(raw_dir, clip["clip_id"])):
            conf = float(window["Confidence"])
            if conf >= min_conf:
                label = window["Common name"] or window["Scientific name"]
                windows[label].append((float(window["Start (s)"]), float(window["End (s)"]), conf))
        clip_events = []
        for label, label_windows in windows.items():
            for event in merge_windows(label_windows, gap):
                clip_events.append((label, event))
        if max_events:
            clip_events = sorted(clip_events, key=lambda item: -max(item[1]["confs"]))[:max_events]
        key = [clip[col] for col in MANIFEST_KEYS]
        for label, event in sorted(clip_events, key=lambda item: item[1]["start"]):
            confs = event["confs"]
            events_out.append(key + [label, f"{event['start']:.1f}", f"{event['end']:.1f}",
                                     len(confs), f"{max(confs):.4f}", f"{sum(confs) / len(confs):.4f}"])
        for label in labels:
            label_events = [event for event_label, event in clip_events if event_label == label]
            n_windows = sum(len(event["confs"]) for event in label_events)
            max_conf = max((max(event["confs"]) for event in label_events), default=None)
            max_conf_text = "" if max_conf is None else f"{max_conf:.4f}"
            summary_out.append(key + [label, len(label_events), n_windows, max_conf_text])
    return events_out, summary_out


def main():
    """CLI entry point: run BirdNET on the clips still to analyse, then rebuild the tables."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--split", help="only clips of this split (e.g. train, heldout)")
    parser.add_argument("--threads", type=int, help="BirdNET CPU threads (default: BirdNET's)")
    parser.add_argument("--gen", type=int, help="use generations.dir/gen_K/model.tflite, "
                        "write to gen_K/detections/")
    parser.add_argument("--min-conf", type=float,
                        help="event threshold for this run (default: detection.min_conf)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--post-only", action="store_true", help="skip BirdNET, rebuild tables")
    parser.add_argument("--force", action="store_true",
                        help="re-run all clips if model/overlap/raw_min_conf changed")
    args = parser.parse_args()
    cfg = load(args.config)
    survey_cfg, detection_cfg = cfg["survey"], cfg["detection"]
    if args.gen is not None:                  # a model generation from make_generation.py
        gen_dir = Path(cfg["generations"]["dir"]) / f"gen_{args.gen}"
        detection_cfg["model"] = str(gen_dir / "model.tflite")
        detection_cfg["dir"] = str(gen_dir / "detections")
    if args.min_conf is not None:
        detection_cfg["min_conf"] = args.min_conf
    if detection_cfg["raw_min_conf"] > detection_cfg["min_conf"]:
        raise SystemExit("detection.raw_min_conf must be <= detection.min_conf")
    model = Path(detection_cfg["model"])
    labels_file = model.with_name(model.stem + "_Labels.txt")
    if not model.exists() or not labels_file.exists():
        raise SystemExit(f"model or labels not found: {model}, {labels_file}")
    labels = [line.strip() for line in labels_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    survey_dir, detection_dir = Path(survey_cfg["dir"]), Path(detection_cfg["dir"])
    raw_dir = detection_dir / "raw"
    manifest = read_dicts(survey_dir / "manifest.csv")
    if args.split:
        manifest = [clip for clip in manifest if clip["split"] == args.split]
    cut = [clip for clip in manifest if Path(clip["clip_path"]).exists()]
    todo = [clip for clip in cut if not raw_path(raw_dir, clip["clip_id"]).exists()]
    listed = {Path(clip["clip_path"]).name for clip in read_dicts(survey_dir / "manifest.csv")}
    stale = [path for path in (survey_dir / "clips").glob("*.wav") if path.name not in listed]
    print(f"manifest: {len(manifest)} clips{f' ({args.split})' if args.split else ''}, "
          f"{len(cut)} cut, {len(cut) - len(todo)} already analysed, {len(todo)} to analyse")
    if stale:
        print(f"note: {len(stale)} files in {survey_dir / 'clips'} are not in the manifest "
              f"(ignored), e.g. {stale[0].name}")
    if args.dry_run:
        return

    params = {"model": str(model), "overlap": str(detection_cfg["overlap"]),
              "raw_min_conf": str(detection_cfg["raw_min_conf"])}
    if not args.post_only:
        raw_dir.mkdir(parents=True, exist_ok=True)
        check_params(raw_dir, params, args.force)
        todo = [clip for clip in cut if not raw_path(raw_dir, clip["clip_id"]).exists()]
        if todo:
            run_birdnet(todo, raw_dir, detection_cfg, args.threads)
        failed = [clip["clip_id"] for clip in todo if not raw_path(raw_dir, clip["clip_id"]).exists()]
        if failed:
            print(f"warning: no BirdNET result for {len(failed)} clips, e.g. {failed[:3]}")

    done = [clip for clip in cut if raw_path(raw_dir, clip["clip_id"]).exists()]
    events, summary = post(done, raw_dir, detection_cfg, labels)
    write_rows(detection_dir / "events.csv", EVENT_HEADER, events)
    write_rows(detection_dir / "clip_summary.csv", SUMMARY_HEADER, summary)
    n_pos = defaultdict(int)
    for row in summary:
        n_pos[row[5]] += row[6] > 0
    print(f"{len(done)} clips analysed, {len(events)} events at min_conf {detection_cfg['min_conf']}; "
          f"clips with >= 1 event: {dict(n_pos)}")
    print(f"wrote {detection_dir / 'events.csv'}, {detection_dir / 'clip_summary.csv'}")


if __name__ == "__main__":
    main()
