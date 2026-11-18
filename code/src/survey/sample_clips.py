"""Sample 3 x 20 min clips per site, one drive at a time: scan, split, freeze, cut.

    python src/survey/sample_clips.py --drive D1 --dry-run   # show this drive's plan, write nothing
    python src/survey/sample_clips.py --drive D1             # plan (once) + cut missing clips
    python src/survey/sample_clips.py --drive D2 --root /mnt/d   # other mount point than config

Plug each drive once and run it with that drive's name (a key of survey.drives).
Sites are split into train / heldout / ... with survey.split.fractions, applied within
each drive, so no drive needs to know about the others.

Writes, under survey.dir:
    plan/<drive>_sites.csv, plan/<drive>_manifest.csv   frozen per drive, reused as is
    sites.csv, manifest.csv                             all drives done so far, merged
    clips/<clip_id>.wav

Run from code/.
"""
import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import load

SITE_HEADER = ["site", "drive", "n_files", "first_date", "last_date", "split"]
SYSTEM_DIRS = {"$RECYCLE.BIN", "System Volume Information"}   # Windows, never sites
MANIFEST_HEADER = ["clip_id", "site", "split", "period", "drive", "rel_path",
                   "offset_s", "duration_s", "clip_path"]


def scan(survey_cfg, drive, root):
    """List the recordings of one drive.

    Globs survey.audio_globs under root, takes the site from the first folder and the
    date/time from survey.filename_pattern. Files that do not match the pattern are
    reported and skipped.

    Args:
        survey_cfg (dict): The survey config section.
        drive (str): Drive name, a key of survey.drives.
        root (Path): Mount point of the drive.

    Returns:
        list[dict]: One dict per recording: site (str), drive (str), rel (str, path
            relative to root), dt (datetime of the recording start).
    """
    name_pattern = re.compile(survey_cfg["filename_pattern"])
    rows, seen, unmatched = [], set(), []
    for glob_pattern in survey_cfg["audio_globs"]:
        for path in sorted(root.glob(glob_pattern)):
            rel = path.relative_to(root)
            if rel in seen or rel.parts[0] in SYSTEM_DIRS:
                continue
            seen.add(rel)
            match = name_pattern.search(path.name)
            if not match:
                unmatched.append(rel.as_posix())
                continue
            dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
            rows.append({"site": rel.parts[0], "drive": drive,
                         "rel": rel.as_posix(), "dt": dt})
    if unmatched:
        print(f"warning: {len(unmatched)} files did not match the pattern (skipped):")
        for rel_path in unmatched:
            print(f"    {rel_path}")
    return rows


def split_sites(sites, split_cfg, drive):
    """Shuffle this drive's sites and cut them by cumulative rounded fractions.

    Args:
        sites (list[str]): Site names on this drive.
        split_cfg (dict): survey.split config: seed (int), fractions (dict[str, float]).
        drive (str): Drive name, part of the shuffle seed.

    Returns:
        dict[str, str]: Site name -> split name (e.g. "train", "heldout").

    Raises:
        SystemExit: If the fractions do not sum to 1.
    """
    fractions = split_cfg["fractions"]
    if abs(sum(fractions.values()) - 1) > 1e-6:
        raise SystemExit(f"survey.split.fractions must sum to 1, got {fractions}")
    order = sorted(sites)
    random.Random(f"{split_cfg['seed']}-{drive}").shuffle(order)
    split_of, cumulative, start = {}, 0.0, 0
    for split_name, fraction in fractions.items():
        cumulative += fraction
        end = round(cumulative * len(order))
        for site in order[start:end]:
            split_of[site] = split_name
        start = end
    return split_of


def plan(survey_cfg, rows, drive, root):
    """Split this drive's sites and choose one clip per site and period.

    Each site's date range is cut into survey.sampling.periods equal parts; in each part
    a random day, one of its files and a random clip_minutes offset are drawn (seeded by
    site and period, so the choice is reproducible).

    Args:
        survey_cfg (dict): The survey config section.
        rows (list[dict]): Recordings returned by scan().
        drive (str): Drive name.
        root (Path): Mount point of the drive (to read file durations).

    Returns:
        tuple[list[list], list[list]]: (site rows in SITE_HEADER order,
            manifest rows in MANIFEST_HEADER order).
    """
    by_site = defaultdict(list)
    for rec in rows:
        by_site[rec["site"]].append(rec)
    sites = sorted(by_site)
    split_of = split_sites(sites, survey_cfg["split"], drive)

    sampling_cfg = survey_cfg["sampling"]
    clip_seconds = sampling_cfg["clip_minutes"] * 60
    site_rows, manifest = [], []
    for site in sites:
        site_recs = sorted(by_site[site], key=lambda rec: rec["dt"])
        by_day = defaultdict(list)
        for rec in site_recs:
            by_day[rec["dt"].date()].append(rec)
        days = sorted(by_day)
        first_day, n_days = days[0], (days[-1] - days[0]).days + 1
        split = split_of[site]
        site_rows.append([site, drive, len(site_recs), days[0], days[-1], split])
        n_periods = sampling_cfg["periods"]

        def period_of(day):
            """Index of the period that contains a day.

            Args:
                day (datetime.date): A recording day of this site.

            Returns:
                int: 0 .. n_periods - 1.
            """
            return min(n_periods - 1, (day - first_day).days * n_periods // n_days)

        for period in range(n_periods):
            candidate_days = [day for day in days if period_of(day) == period]
            if not candidate_days:
                print(f"warning: {site}: no recordings in period {period}")
                continue
            rng = random.Random(f"{sampling_cfg['seed']}-{site}-{period}")
            files = by_day[rng.choice(candidate_days)]
            rec = (files[0] if sampling_cfg["file"] == "pre" else files[-1] if sampling_cfg["file"] == "post"
                   else rng.choice(files))
            info = sf.info(root / rec["rel"])
            duration_s = info.frames / info.samplerate
            if duration_s > clip_seconds:
                offset_s = rng.randrange(0, int(duration_s - clip_seconds) + 1)
            else:
                offset_s = 0
            length = min(clip_seconds, duration_s)
            clip_id = f"{site}_{rec['dt']:%Y%m%d_%H%M%S}_{offset_s}"
            manifest.append([clip_id, site, split, period, drive, rec["rel"], offset_s,
                             f"{length:.1f}", f"{survey_cfg['dir']}/clips/{clip_id}.wav"])
    return site_rows, manifest


def read_rows(path):
    """Rows of a CSV file as lists, header skipped.

    Args:
        path (Path): CSV file.

    Returns:
        list[list[str]]: Data rows.
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))[1:]


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


def check_duplicates(survey_cfg, plan_dir, drive, sites):
    """Warn if some of these sites are already planned on another drive.

    Args:
        survey_cfg (dict): The survey config section.
        plan_dir (Path): Folder of the frozen per-drive plans.
        drive (str): The drive being planned.
        sites (list[str]): Its site names.
    """
    for other in survey_cfg["drives"]:
        sites_path = plan_dir / f"{other}_sites.csv"
        if other == drive or not sites_path.exists():
            continue
        duplicates = sorted(set(sites) & {row[0] for row in read_rows(sites_path)})
        if duplicates:
            print(f"warning: sites {duplicates[:5]} are already planned on drive {other} "
                  f"(wrong --drive, or a site split over two drives?)")


def cut(manifest, root):
    """Cut the manifest's clips that are not on disk yet.

    Missing source files are reported (wrong drive plugged in, or wrong --root).

    Args:
        manifest (list[list[str]]): Manifest rows (MANIFEST_HEADER order).
        root (Path): Mount point of the drive.

    Returns:
        int: Number of clips present after the run.
    """
    done = missing = 0
    for clip_id, site, split, period, drive, rel, offset_s, length, clip_path in manifest:
        dst = Path(clip_path)
        if dst.exists():
            done += 1
            continue
        src = root / rel
        if not src.exists():
            missing += 1
            print(f"missing: {src}")
            continue
        info = sf.info(src)
        sample_rate = info.samplerate
        audio, _ = sf.read(src, start=int(offset_s) * sample_rate, frames=int(float(length) * sample_rate),
                           dtype="float32")
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, audio, sample_rate, subtype=info.subtype)
        done += 1
        print("cut", clip_id)
    if missing:
        print(f"warning: {missing} source files missing: wrong drive plugged in, or wrong --root?")
    return done


def merge(survey_cfg, plan_dir, survey_dir):
    """Rebuild sites.csv and manifest.csv from all frozen per-drive plans; print a summary.

    Args:
        survey_cfg (dict): The survey config section.
        plan_dir (Path): Folder of the frozen per-drive plans.
        survey_dir (Path): survey.dir, where sites.csv and manifest.csv are written.
    """
    sites, manifest, drives_done = [], [], []
    for drive in survey_cfg["drives"]:
        sites_path, manifest_path = plan_dir / f"{drive}_sites.csv", plan_dir / f"{drive}_manifest.csv"
        if sites_path.exists() and manifest_path.exists():
            drives_done.append(drive)
            sites += read_rows(sites_path)
            manifest += read_rows(manifest_path)
    write_rows(survey_dir / "sites.csv", SITE_HEADER, sites)
    write_rows(survey_dir / "manifest.csv", MANIFEST_HEADER, manifest)
    n_clips = sum(Path(row[8]).exists() for row in manifest)
    splits = Counter(row[5] for row in sites)
    print(f"all drives so far: {len(drives_done)}/{len(survey_cfg['drives'])} planned "
          f"({', '.join(drives_done)}), {len(sites)} sites {dict(splits)}, "
          f"{n_clips}/{len(manifest)} clips cut")


def main():
    """CLI entry point: plan this drive once (or reuse its frozen plan), cut clips, merge."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--drive", required=True, help="drive name, a key of survey.drives")
    parser.add_argument("--root", help="mount point of the drive (default: survey.drives[drive])")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    survey_cfg = load(args.config)["survey"]
    if args.drive not in survey_cfg["drives"]:
        raise SystemExit(f"unknown drive {args.drive}; known: {list(survey_cfg['drives'])}")
    root = Path(args.root or survey_cfg["drives"][args.drive])
    if not root.is_dir():
        raise SystemExit(f"drive {args.drive} not found at {root} (plugged in? use --root)")
    survey_dir = Path(survey_cfg["dir"])
    plan_dir = survey_dir / "plan"
    sites_path, manifest_path = plan_dir / f"{args.drive}_sites.csv", plan_dir / f"{args.drive}_manifest.csv"

    if manifest_path.exists() and sites_path.exists():
        site_rows, manifest = read_rows(sites_path), read_rows(manifest_path)
        print(f"drive {args.drive}: reusing frozen plan, {len(site_rows)} sites, {len(manifest)} clips")
    else:
        rows = scan(survey_cfg, args.drive, root)
        if not rows:
            raise SystemExit(f"no recordings found on {root} with {survey_cfg['audio_globs']}")
        site_rows, manifest = plan(survey_cfg, rows, args.drive, root)
        splits = Counter(row[5] for row in site_rows)
        print(f"drive {args.drive}: {len(rows)} files, {len(site_rows)} sites {dict(splits)}, "
              f"{len(manifest)} clips")
        check_duplicates(survey_cfg, plan_dir, args.drive, [row[0] for row in site_rows])
        if args.dry_run:
            return
        write_rows(sites_path, SITE_HEADER, site_rows)
        write_rows(manifest_path, MANIFEST_HEADER, manifest)
    if args.dry_run:
        return

    done = cut(manifest, root)
    print(f"drive {args.drive}: {done}/{len(manifest)} clips cut")
    merge(survey_cfg, plan_dir, survey_dir)


if __name__ == "__main__":
    main()
