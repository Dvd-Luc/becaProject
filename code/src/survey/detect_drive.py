"""Run a model generation on every field recording, read straight from the plugged-in drive.

    python src/survey/detect_drive.py --drive D1 --dry-run        # files, hours, done / to do
    python src/survey/detect_drive.py --drive D1 --limit 20       # speed test on 20 files
    python src/survey/detect_drive.py --drive D1                  # the whole drive (resumable)
    python src/survey/detect_drive.py --drive D2 --root /mnt/e --threads 12 --batch-size 16
    python src/survey/detect_drive.py --post-only --min-conf 0.35 # new threshold on the saved scores, model not re-run

Plug one drive, run it with that drive's name (a key of survey.drives). Nothing is
copied: BirdNET reads the WAV files on the drive and only CSVs are written here.

1. File list: on the first run of a drive, its recordings are listed with the same
   survey.audio_globs / filename_pattern as sample_clips.py, with the duration read from
   each file header. The list is frozen in files/<drive>_files.csv and reused as is.
   Only sites of survey.dir/sites.csv are kept (their split is copied); if the sites
   found belong to another drive in sites.csv, the run stops (wrong drive plugged in).
2. BirdNET (generations.dir/gen_K/model.tflite, K = drive_detection.gen) runs site by
   site. A file with a raw result is skipped, so stop any time (Ctrl-C, unplug) and run
   again to continue. Files without a result after BirdNET are listed and retried on
   the next run. A progress bar shows the files done and the time left; BirdNET's own
   output goes to raw/birdnet_<drive>.log. BirdNET runs up to `threads` files at once,
   so the bar moves in jumps. Each worker needs about 1 GB of RAM: if one is killed for
   lack of memory, BirdNET waits forever, so after drive_detection.stall_min minutes
   without a finished file the run is stopped (lower --threads and run again).
3. Tables are rebuilt from every drive done so far: windows >= drive_detection.min_conf
   are merged per file and label into events (gap <= merge_gap_s).

Writes, under drive_detection.dir:
    files/<drive>_files.csv             frozen file list: file_id, site, split, drive,
                                        start, duration_s, rel_path
    raw/<file_id>.BirdNET.results.csv   BirdNET output, one row per window >= raw_min_conf
    raw/run_params.csv                  model + overlap + raw_min_conf of the raw results
    raw/birdnet_<drive>.log             BirdNET's console output
    events.csv                          one row per event; event_time = file start + start_s
    file_summary.csv                    one row per analysed file x label (zeros included),
                                        with duration_s, for rates per hour / site / date

If the model, overlap or raw_min_conf changes, the raw results no longer match: the
script stops; use --force (re-runs every file) or another drive_detection.dir.

Run from code/.
"""
import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detect import birdnet_command, check_params, link_audio, post, raw_path, read_dicts, write_rows
from sample_clips import scan

from common.config import load

FILE_HEADER = ["file_id", "site", "split", "drive", "start", "duration_s", "rel_path"]
KEYS = ["file_id", "site", "split", "drive", "start", "duration_s"]
EVENT_HEADER = ["file_id", "site", "split", "drive", "start", "label", "start_s", "end_s",
                "event_time", "n_windows", "max_conf", "mean_conf"]
SUMMARY_HEADER = KEYS + ["label", "n_events", "n_windows", "max_conf"]
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
POLL_S = 5


def build_file_list(survey_cfg, drive, root, sites):
    """List one drive's recordings, with split and duration.

    Args:
        survey_cfg (dict): The survey config section (audio_globs, filename_pattern).
        drive (str): Drive name, a key of survey.drives.
        root (Path): Mount point of the drive.
        sites (dict[str, dict]): survey.dir/sites.csv rows by site name.

    Returns:
        list[list[str]]: Rows in FILE_HEADER order, sorted by site and start.

    Raises:
        SystemExit: If no recording is found, or the sites belong to another drive.
    """
    recordings = scan(survey_cfg, drive, root)
    if not recordings:
        raise SystemExit(f"no recording found under {root}: drive not mounted, or wrong --root?")
    found = sorted({rec["site"] for rec in recordings})
    other_drive = [site for site in found if site in sites and sites[site]["drive"] != drive]
    if other_drive:
        raise SystemExit(f"{len(other_drive)} sites under {root} belong to another drive in sites.csv "
                         f"(e.g. {other_drive[0]}: {sites[other_drive[0]]['drive']}): wrong drive plugged in?")
    unknown = [site for site in found if site not in sites]
    if unknown:
        print(f"warning: {len(unknown)} sites are not in sites.csv (skipped): {unknown}")
    rows, unreadable = [], []
    for rec in tqdm(recordings, desc="reading file headers", unit="file", leave=False):
        if rec["site"] not in sites:
            continue
        try:
            duration_s = sf.info(root / rec["rel"]).duration
        except RuntimeError:
            unreadable.append(rec["rel"])
            continue
        file_id = f"{rec['site']}_{Path(rec['rel']).stem}"
        rows.append([file_id, rec["site"], sites[rec["site"]]["split"], drive,
                     rec["dt"].strftime(TIME_FORMAT), f"{duration_s:.1f}", rec["rel"]])
    if unreadable:
        print(f"warning: {len(unreadable)} unreadable files (left out): {unreadable[:5]}")
    ids = Counter(row[0] for row in rows)
    duplicates = [file_id for file_id, count in ids.items() if count > 1]
    if duplicates:
        raise SystemExit(f"{len(duplicates)} file ids are not unique, e.g. {duplicates[0]}")
    return sorted(rows, key=lambda row: (row[1], row[4]))


def stop_group(process):
    """Stop a BirdNET run and all its worker processes, then wait for it.

    Args:
        process (subprocess.Popen): BirdNET, started with start_new_session=True.
    """
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def analyse_site(site, todo, raw_dir, detection_cfg, threads, batch_size, log_path, bar, stall_min):
    """Run BirdNET on one site's files and advance the progress bar as results appear.

    If no file finishes for stall_min minutes (typically a worker killed for lack of
    memory, which leaves BirdNET waiting forever), BirdNET is stopped.

    Args:
        site (str): Site name (shown on the bar).
        todo (list[dict]): File rows still to analyse, with an abs_path key.
        raw_dir (Path): Output folder for BirdNET's CSVs.
        detection_cfg (dict): Model, raw_min_conf and overlap.
        threads (int | None): BirdNET CPU threads.
        batch_size (int | None): BirdNET batch size.
        log_path (Path): BirdNET's console output is appended here.
        bar (tqdm): Progress bar over files.
        stall_min (float): Minutes without a finished file before BirdNET is stopped.

    Raises:
        SystemExit: If BirdNET fails (the end of its log is shown), stalls, or Ctrl-C.
    """
    bar.set_postfix_str(site[:40])
    with tempfile.TemporaryDirectory(prefix="detect_drive_") as tmp:
        link_audio(todo, tmp, id_key="file_id", path_key="abs_path")
        cmd = birdnet_command(tmp, raw_dir, detection_cfg, threads, batch_size)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n# {datetime.now():{TIME_FORMAT}} {site}: {len(todo)} files\n# {' '.join(cmd)}\n")
            log.flush()
            # own process group: stopping it also stops BirdNET's worker processes
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            pending = list(todo)
            last_progress = time.time()
            try:
                while True:
                    finished = process.poll() is not None
                    done_now = [row for row in pending if raw_path(raw_dir, row["file_id"]).exists()]
                    if done_now:
                        bar.update(len(done_now))
                        done_ids = {row["file_id"] for row in done_now}
                        pending = [row for row in pending if row["file_id"] not in done_ids]
                        last_progress = time.time()
                    if finished:
                        break
                    if time.time() - last_progress > stall_min * 60:
                        stop_group(process)
                        bar.close()
                        raise SystemExit(f"BirdNET stalled: no file finished in {stall_min} min (a worker killed "
                                         f"for lack of memory?). Lower --threads (or give it more memory), then "
                                         f"run again: the {len(pending)} unfinished files of {site} are redone.")
                    time.sleep(POLL_S)
            except KeyboardInterrupt:
                stop_group(process)
                bar.close()
                raise SystemExit("stopped: run the same command again to continue") from None
    if process.returncode != 0:
        bar.close()
        tail = log_path.read_text(encoding="utf-8").splitlines()[-15:]
        raise SystemExit(f"BirdNET failed on {site} (exit {process.returncode}); end of {log_path}:\n"
                         + "\n".join(tail))


def build_tables(out_dir, raw_dir, detection_cfg, labels):
    """Rebuild events.csv and file_summary.csv from every frozen drive list.

    Args:
        out_dir (Path): drive_detection.dir.
        raw_dir (Path): out_dir/raw.
        detection_cfg (dict): min_conf, merge_gap_s (max_events_per_clip optional).
        labels (list[str]): Model classes.

    Returns:
        tuple[list[dict], list[list], list[list]]: (analysed file rows, event rows in
            EVENT_HEADER order, summary rows in SUMMARY_HEADER order).
    """
    files = []
    for list_path in sorted((out_dir / "files").glob("*_files.csv")):
        files += read_dicts(list_path)
    done = [row for row in files if raw_path(raw_dir, row["file_id"]).exists()]
    events, summary = post(done, raw_dir, detection_cfg, labels, keys=KEYS)
    event_rows = []
    for row in events:
        file_id, site, split, drive, start, _duration, label, start_s, end_s, *conf = row
        event_time = datetime.strptime(start, TIME_FORMAT) + timedelta(seconds=float(start_s))
        event_rows.append([file_id, site, split, drive, start, label, start_s, end_s,
                           event_time.strftime(TIME_FORMAT), *conf])
    write_rows(out_dir / "events.csv", EVENT_HEADER, event_rows)
    write_rows(out_dir / "file_summary.csv", SUMMARY_HEADER, summary)
    return done, event_rows, summary


def main():
    """CLI entry point: list the drive's files, analyse the missing ones, rebuild the tables."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--drive", help="drive name, a key of survey.drives (not needed with --post-only)")
    parser.add_argument("--root", help="mount point of the drive (default: survey.drives.<drive>)")
    parser.add_argument("--gen", type=int, help="model generation (default: drive_detection.gen)")
    parser.add_argument("--threads", type=int, help="BirdNET CPU threads (default: drive_detection.threads)")
    parser.add_argument("--batch-size", type=int, help="BirdNET batch size (default: drive_detection.batch_size)")
    parser.add_argument("--limit", type=int, help="analyse at most N files in this run (speed test)")
    parser.add_argument("--min-conf", type=float, help="event threshold (default: drive_detection.min_conf)")
    parser.add_argument("--dry-run", action="store_true", help="show the files and what is left, write nothing")
    parser.add_argument("--post-only", action="store_true", help="model not re-run: rebuild the tables from the saved scores")
    parser.add_argument("--force", action="store_true",
                        help="re-run every file if model / overlap / raw_min_conf changed")
    args = parser.parse_args()
    cfg = load(args.config)
    survey_cfg, detection_cfg = cfg["survey"], dict(cfg["drive_detection"])
    if args.gen is not None:
        detection_cfg["gen"] = args.gen
    if args.min_conf is not None:
        detection_cfg["min_conf"] = args.min_conf
    if detection_cfg["raw_min_conf"] > detection_cfg["min_conf"]:
        raise SystemExit("drive_detection.raw_min_conf must be <= min_conf")
    model = Path(cfg["generations"]["dir"]) / f"gen_{detection_cfg['gen']}" / "model.tflite"
    labels_file = model.with_name(model.stem + "_Labels.txt")
    if not model.exists() or not labels_file.exists():
        raise SystemExit(f"model or labels not found: {model}, {labels_file}")
    detection_cfg["model"] = str(model)
    labels = [line.strip() for line in labels_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_dir = Path(detection_cfg["dir"])
    raw_dir = out_dir / "raw"
    threads = args.threads or detection_cfg.get("threads")
    batch_size = args.batch_size or detection_cfg.get("batch_size")

    if args.post_only:
        done, events, _summary = build_tables(out_dir, raw_dir, detection_cfg, labels)
        hours = sum(float(row["duration_s"]) for row in done) / 3600
        print(f"{len(done)} files ({hours:.0f} h) analysed, {len(events)} events at min_conf "
              f"{detection_cfg['min_conf']}: {dict(Counter(row[5] for row in events))}")
        print(f"wrote {out_dir / 'events.csv'}, {out_dir / 'file_summary.csv'}")
        return

    if not args.drive or args.drive not in survey_cfg["drives"]:
        raise SystemExit(f"--drive must be one of {list(survey_cfg['drives'])}")
    root = Path(args.root or survey_cfg["drives"][args.drive])
    sites = {row["site"]: row for row in read_dicts(Path(survey_cfg["dir"]) / "sites.csv")}
    list_path = out_dir / "files" / f"{args.drive}_files.csv"
    if list_path.exists():
        files = read_dicts(list_path)
        print(f"file list: {list_path} (frozen, {len(files)} files)")
    else:
        if not root.is_dir() or not any(root.iterdir()):
            raise SystemExit(f"{root} is empty: plug in drive {args.drive}, or give --root")
        rows = build_file_list(survey_cfg, args.drive, root, sites)
        files = [dict(zip(FILE_HEADER, row, strict=True)) for row in rows]
        if args.dry_run:
            print(f"file list: {len(files)} files found (not frozen: --dry-run)")
        else:
            write_rows(list_path, FILE_HEADER, rows)
            print(f"file list: {len(files)} files, frozen in {list_path}")

    for row in files:
        row["abs_path"] = str(root / row["rel_path"])
    todo = [row for row in files if not raw_path(raw_dir, row["file_id"]).exists()]
    hours_all = sum(float(row["duration_s"]) for row in files) / 3600
    hours_todo = sum(float(row["duration_s"]) for row in todo) / 3600
    print(f"drive {args.drive}: {len({row['site'] for row in files})} sites, {len(files)} files "
          f"({hours_all:.0f} h), {len(files) - len(todo)} done, {len(todo)} to do ({hours_todo:.0f} h)")
    if args.dry_run:
        todo_ids = {row["file_id"] for row in todo}
        per_site = defaultdict(lambda: [0, 0, 0.0, ""])
        for row in files:
            site_count = per_site[row["site"]]
            site_count[0] += 1
            site_count[1] += row["file_id"] in todo_ids
            site_count[2] += float(row["duration_s"]) / 3600
            site_count[3] = row["split"]
        print(f"    {'site':45s} {'split':8s} {'files':>6s} {'to do':>6s} {'hours':>6s}")
        for site, (n_files, n_todo, hours, split) in sorted(per_site.items()):
            print(f"    {site[:45]:45s} {split:8s} {n_files:6d} {n_todo:6d} {hours:6.0f}")
        print(f"model gen_{detection_cfg['gen']}, threads {threads}, batch size {batch_size}; "
              f"run --limit 20 first to measure the speed")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    params = {"model": str(model), "overlap": str(detection_cfg["overlap"]),
              "raw_min_conf": str(detection_cfg["raw_min_conf"])}
    check_params(raw_dir, params, args.force)
    todo = [row for row in files if not raw_path(raw_dir, row["file_id"]).exists()]
    missing = [row for row in todo if not Path(row["abs_path"]).exists()]
    if missing:
        print(f"warning: {len(missing)} files to do are not under {root} (skipped), e.g. {missing[0]['rel_path']}")
        if len(missing) == len(todo):
            raise SystemExit(f"none of the files to do is under {root}: plug in drive {args.drive}, or give --root")
    missing_ids = {row["file_id"] for row in missing}
    todo = [row for row in todo if row["file_id"] not in missing_ids]
    if args.limit:
        todo = todo[:args.limit]

    started = time.time()
    if todo:
        by_site = defaultdict(list)
        for row in todo:
            by_site[row["site"]].append(row)
        print(f"analysing {len(todo)} files with gen_{detection_cfg['gen']} "
              f"(threads {threads}, batch size {batch_size}); BirdNET log: {raw_dir / f'birdnet_{args.drive}.log'}")
        bar = tqdm(total=len(todo), unit="file", dynamic_ncols=True)
        for site, site_todo in by_site.items():
            analyse_site(site, site_todo, raw_dir, detection_cfg, threads, batch_size,
                         raw_dir / f"birdnet_{args.drive}.log", bar, detection_cfg.get("stall_min", 15))
        bar.close()
    elapsed = time.time() - started

    failed = [row["rel_path"] for row in todo if not raw_path(raw_dir, row["file_id"]).exists()]
    if failed:
        print(f"warning: no BirdNET result for {len(failed)} files (retried next run), e.g. {failed[:3]}")
    n_new = len(todo) - len(failed)
    left = [row for row in files if not raw_path(raw_dir, row["file_id"]).exists()]
    print(f"drive {args.drive}: {n_new} files analysed in this run, {len(files) - len(left)}/{len(files)} done")
    if n_new:
        failed_set = set(failed)
        hours_new = sum(float(row["duration_s"]) for row in todo if row["rel_path"] not in failed_set) / 3600
        hours_left = sum(float(row["duration_s"]) for row in left) / 3600
        per_hour = elapsed / hours_new
        print(f"speed: {per_hour:.0f} s per hour of audio ({hours_new:.1f} h in {elapsed / 60:.1f} min); "
              f"{len(left)} files ({hours_left:.0f} h) left on this drive, about {per_hour * hours_left / 3600:.1f} h")

    done, events, _summary = build_tables(out_dir, raw_dir, detection_cfg, labels)
    hours = sum(float(row["duration_s"]) for row in done) / 3600
    print(f"all drives: {len(done)} files ({hours:.0f} h) analysed, {len(events)} events at min_conf "
          f"{detection_cfg['min_conf']}: {dict(Counter(row[5] for row in events))}")
    print(f"wrote {out_dir / 'events.csv'}, {out_dir / 'file_summary.csv'}")


if __name__ == "__main__":
    main()
