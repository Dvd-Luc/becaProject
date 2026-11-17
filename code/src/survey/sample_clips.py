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


def scan(s, drive, root):
    """List the recordings of one drive.

    Globs survey.audio_globs under root, takes the site from the first folder and the
    date/time from survey.filename_pattern. Returns [{site, drive, rel, dt}];
    files that do not match the pattern are reported and skipped.
    """
    pat = re.compile(s["filename_pattern"])
    rows, seen, bad = [], set(), []
    for g in s["audio_globs"]:
        for p in sorted(root.glob(g)):
            rel = p.relative_to(root)
            if rel in seen or rel.parts[0] in SYSTEM_DIRS:
                continue
            seen.add(rel)
            m = pat.search(p.name)
            if not m:
                bad.append(rel.as_posix())
                continue
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            rows.append({"site": rel.parts[0], "drive": drive,
                         "rel": rel.as_posix(), "dt": dt})
    if bad:
        print(f"warning: {len(bad)} files did not match the pattern (skipped):")
        for b in bad:
            print(f"    {b}")
    return rows


def split_sites(sites, sp, drive):
    """Shuffle this drive's sites and cut them by cumulative rounded fractions."""
    fr = sp["fractions"]
    if abs(sum(fr.values()) - 1) > 1e-6:
        raise SystemExit(f"survey.split.fractions must sum to 1, got {fr}")
    order = sorted(sites)
    random.Random(f"{sp['seed']}-{drive}").shuffle(order)
    out, cum, start = {}, 0.0, 0
    for name, f in fr.items():
        cum += f
        end = round(cum * len(order))
        for site in order[start:end]:
            out[site] = name
        start = end
    return out


def plan(s, rows, drive, root):
    """Split this drive's sites and choose one clip per site and period.

    Each site's date range is cut into survey.sampling.periods equal parts; in each part
    a random day, one of its files and a random clip_minutes offset are drawn
    (seeded by site and period, so the choice is reproducible).
    Returns (site rows, manifest rows) in SITE_HEADER / MANIFEST_HEADER order.
    """
    by_site = defaultdict(list)
    for r in rows:
        by_site[r["site"]].append(r)
    sites = sorted(by_site)
    split_of = split_sites(sites, s["split"], drive)

    sm = s["sampling"]
    clip_s = sm["clip_minutes"] * 60
    site_rows, manifest = [], []
    for site in sites:
        rs = sorted(by_site[site], key=lambda r: r["dt"])
        by_day = defaultdict(list)
        for r in rs:
            by_day[r["dt"].date()].append(r)
        days = sorted(by_day)
        d0, span = days[0], (days[-1] - days[0]).days + 1
        split = split_of[site]
        site_rows.append([site, drive, len(rs), days[0], days[-1], split])
        part = lambda d: min(sm["periods"] - 1, (d - d0).days * sm["periods"] // span)
        for k in range(sm["periods"]):
            cands = [d for d in days if part(d) == k]
            if not cands:
                print(f"warning: {site}: no recordings in period {k}")
                continue
            rng = random.Random(f"{sm['seed']}-{site}-{k}")
            files = by_day[rng.choice(cands)]
            r = (files[0] if sm["file"] == "pre" else files[-1] if sm["file"] == "post"
                 else rng.choice(files))
            info = sf.info(root / r["rel"])
            dur = info.frames / info.samplerate
            off = rng.randrange(0, int(dur - clip_s) + 1) if dur > clip_s else 0
            length = min(clip_s, dur)
            cid = f"{site}_{r['dt']:%Y%m%d_%H%M%S}_{off}"
            manifest.append([cid, site, split, k, drive, r["rel"], off,
                             f"{length:.1f}", f"{s['dir']}/clips/{cid}.wav"])
    return site_rows, manifest


def read_rows(path):
    """Rows of a CSV file as lists, header skipped."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))[1:]


def write_rows(path, header, rows):
    """Write a CSV file with a header row, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def check_duplicates(s, plan_dir, drive, sites):
    """Warn if some of these sites are already planned on another drive."""
    for other in s["drives"]:
        p = plan_dir / f"{other}_sites.csv"
        if other == drive or not p.exists():
            continue
        dup = sorted(set(sites) & {r[0] for r in read_rows(p)})
        if dup:
            print(f"warning: sites {dup[:5]} are already planned on drive {other} "
                  f"(wrong --drive, or a site split over two drives?)")


def cut(manifest, root):
    """Cut the manifest's clips from the drive that are not on disk yet.

    Returns the number of clips present after the run; missing source files are
    reported (wrong drive plugged in, or wrong --root).
    """
    done = missing = 0
    for cid, site, split, k, drive, rel, off, length, clip in manifest:
        dst = Path(clip)
        if dst.exists():
            done += 1
            continue
        src = root / rel
        if not src.exists():
            missing += 1
            print(f"missing: {src}")
            continue
        info = sf.info(src)
        sr = info.samplerate
        x, _ = sf.read(src, start=int(off) * sr, frames=int(float(length) * sr),
                       dtype="float32")
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, x, sr, subtype=info.subtype)
        done += 1
        print("cut", cid)
    if missing:
        print(f"warning: {missing} source files missing: wrong drive plugged in, or wrong --root?")
    return done


def merge(s, plan_dir, out):
    """Rebuild sites.csv and manifest.csv from all frozen per-drive plans and print a summary."""
    sites, manifest, drives_done = [], [], []
    for d in s["drives"]:
        sp, mp = plan_dir / f"{d}_sites.csv", plan_dir / f"{d}_manifest.csv"
        if sp.exists() and mp.exists():
            drives_done.append(d)
            sites += read_rows(sp)
            manifest += read_rows(mp)
    write_rows(out / "sites.csv", SITE_HEADER, sites)
    write_rows(out / "manifest.csv", MANIFEST_HEADER, manifest)
    n_clips = sum(Path(r[8]).exists() for r in manifest)
    splits = Counter(r[5] for r in sites)
    print(f"all drives so far: {len(drives_done)}/{len(s['drives'])} planned "
          f"({', '.join(drives_done)}), {len(sites)} sites {dict(splits)}, "
          f"{n_clips}/{len(manifest)} clips cut")


def main():
    """CLI entry point: plan this drive once (or reuse its frozen plan), cut clips, merge."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="config/project.yaml")
    ap.add_argument("--drive", required=True, help="drive name, a key of survey.drives")
    ap.add_argument("--root", help="mount point of the drive (default: survey.drives[drive])")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    s = load(a.config)["survey"]
    if a.drive not in s["drives"]:
        raise SystemExit(f"unknown drive {a.drive}; known: {list(s['drives'])}")
    root = Path(a.root or s["drives"][a.drive])
    if not root.is_dir():
        raise SystemExit(f"drive {a.drive} not found at {root} (plugged in? use --root)")
    out = Path(s["dir"])
    plan_dir = out / "plan"
    spath, mpath = plan_dir / f"{a.drive}_sites.csv", plan_dir / f"{a.drive}_manifest.csv"

    if mpath.exists() and spath.exists():
        site_rows, manifest = read_rows(spath), read_rows(mpath)
        print(f"drive {a.drive}: reusing frozen plan, {len(site_rows)} sites, {len(manifest)} clips")
    else:
        rows = scan(s, a.drive, root)
        if not rows:
            raise SystemExit(f"no recordings found on {root} with {s['audio_globs']}")
        site_rows, manifest = plan(s, rows, a.drive, root)
        splits = Counter(r[5] for r in site_rows)
        print(f"drive {a.drive}: {len(rows)} files, {len(site_rows)} sites {dict(splits)}, "
              f"{len(manifest)} clips")
        check_duplicates(s, plan_dir, a.drive, [r[0] for r in site_rows])
        if a.dry_run:
            return
        write_rows(spath, SITE_HEADER, site_rows)
        write_rows(mpath, MANIFEST_HEADER, manifest)
    if a.dry_run:
        return

    done = cut(manifest, root)
    print(f"drive {a.drive}: {done}/{len(manifest)} clips cut")
    merge(s, plan_dir, out)


if __name__ == "__main__":
    main()
