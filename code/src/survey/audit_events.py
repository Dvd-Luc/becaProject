"""Audit the full-survey events: judge a random sample blind, estimate their precision.

    python src/survey/audit_events.py sample --drive D4 --dry-run   # strata and populations only
    python src/survey/audit_events.py sample --drive D4             # draw and freeze the sample
    python src/survey/audit_events.py cut --drive D4 --root /mnt/d/2026_Becacoustic_LPO_BFC_part1 --limit 3
    python src/survey/audit_events.py cut --drive D4 --root /mnt/d/2026_Becacoustic_LPO_BFC_part1
    python src/survey/review.py --queue audit --drive D4            # judge (blind GUI)
    python src/survey/audit_events.py report                        # precision, all drives judged so far

detect_drive.py gives thousands of events per drive. This script picks a few hundred,
cuts their audio so review.py can show them, and turns the verdicts into the
precision of all the events, with confidence intervals.

sample  Reads the drive's rows of drive_detection.dir/events.csv and puts each event in
        a stratum: predicted class x max_conf bin (review.audit_bins). It draws
        review.audit_per_stratum events at random per stratum (a smaller stratum is
        taken whole), seeded by review.seed and the drive name. The queue and the
        stratum populations are frozen and reused as they are; --force redraws, and
        only while nothing is judged yet.
cut     Writes each sampled event with review.context_s on each side as a small WAV
        snippet, read from the drive (or any copy of it, --root). Existing snippets
        are skipped. Run it where the audio is: the laptop with the drive plugged in,
        or the cluster with $SCRATCH/beca/audio/<drive>.
report  For every audit queue judged so far: an event is correct when the verdict is
        its predicted class; Unsure is left out. Precision per class x bin, per class
        and over all events, each stratum weighted by population / judged. The 95 %
        CI comes from a bootstrap that resamples sites within each drive (events cluster
        by site). It also gives the confusion of predicted class x verdict, as numbers
        judged and as estimated numbers of events.

Judging is blind: review.py shows neither the predicted class nor the score. The
sample covers train and heldout sites; only train-site verdicts may ever train a model.

Writes:
    <review.dir>/audit_<drive>_queue.csv     frozen sample, review.py queue columns + drive,
                                             rel_path, offset_s (snippet start in the
                                             file), snippet_s (snippet length)
    <review.dir>/audit_<drive>_strata.csv    drive, stratum, population, sampled, n_sites
    <drive_detection.dir>/audit/snippets/<item_id>.wav   audio of each sampled event
    <out>/audit_full_gen<K>.csv              precision per class x bin, class, all
    <out>/audit_full_gen<K>_confusion.csv    predicted x verdict: judged, estimated events
(review.py --queue audit writes <review.dir>/audit_<drive>_judgments.csv.)

Run from code/.
"""
import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review import QUEUE_HEADER, UNSURE, load_judgments, read_dicts, write_dicts

from common.config import load

AUDIT_HEADER = QUEUE_HEADER + ["drive", "rel_path", "offset_s", "snippet_s"]
STRATA_HEADER = ["drive", "stratum", "population", "sampled", "n_sites"]
RESULT_HEADER = ["scope", "label", "bin", "population", "sampled", "judged", "unsure",
                 "correct", "precision", "ci_low", "ci_high", "strata_missing"]
CONFUSION_HEADER = ["predicted", "verdict", "judged", "est_events"]


def bin_names(bins):
    """Names of the max_conf bins, e.g. "0.5-0.7".

    Args:
        bins (list[float]): Bin edges (review.audit_bins).

    Returns:
        list[str]: One name per bin.
    """
    return [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]


def bin_of(conf, bins):
    """Index of the bin holding conf; the last bin includes its upper edge.

    Args:
        conf (float): An event's max_conf.
        bins (list[float]): Bin edges.

    Returns:
        int | None: Bin index, None if conf is outside the bins.
    """
    for i in range(len(bins) - 1):
        if bins[i] <= conf < bins[i + 1] or (i == len(bins) - 2 and conf == bins[-1]):
            return i
    return None


def model_labels(cfg):
    """Class names of the model that made the full-survey events.

    Args:
        cfg (dict): The whole project config.

    Returns:
        list[str]: Labels of generations.dir/gen_K/model_Labels.txt, K = drive_detection.gen.
    """
    gen_dir = Path(cfg["generations"]["dir"]) / f"gen_{cfg['drive_detection']['gen']}"
    text = (gen_dir / "model_Labels.txt").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def audit_paths(cfg, drive):
    """Paths of one drive's audit files.

    Args:
        cfg (dict): The whole project config.
        drive (str): Drive name.

    Returns:
        tuple[Path, Path, Path, Path]: (queue CSV, strata CSV, judgments CSV, snippet folder).
    """
    review_dir = Path(cfg["review"]["dir"])
    snippet_dir = Path(cfg["drive_detection"]["dir"]) / "audit" / "snippets"
    return (review_dir / f"audit_{drive}_queue.csv", review_dir / f"audit_{drive}_strata.csv",
            review_dir / f"audit_{drive}_judgments.csv", snippet_dir)


# ---------------------------------------------------------------- sample
def stratify(events, labels, bins):
    """Group events by stratum (class x max_conf bin).

    Args:
        events (list[dict]): events.csv rows of one drive.
        labels (list[str]): Model classes.
        bins (list[float]): Bin edges.

    Returns:
        dict[str, list[dict]]: "Label|lo-hi" -> events, every stratum present (maybe empty).
    """
    names = bin_names(bins)
    strata = {f"{label}|{name}": [] for label in labels for name in names}
    outside = 0
    for event in events:
        bin_idx = bin_of(float(event["max_conf"]), bins)
        if bin_idx is None or event["label"] not in labels:
            outside += 1
            continue
        strata[f"{event['label']}|{names[bin_idx]}"].append(event)
    if outside:
        print(f"warning: {outside} events outside the audit bins {bins} or classes (not sampled)")
    return strata


def queue_row(event, file_row, snippet_dir, context_s):
    """Queue row of one sampled event, with its snippet window.

    Args:
        event (dict): events.csv row.
        file_row (dict): files/<drive>_files.csv row of the event's file.
        snippet_dir (Path): Folder of the snippets.
        context_s (float): Seconds of audio kept on each side of the event.

    Returns:
        dict[str, str]: Row keyed by AUDIT_HEADER; start_s / end_s are snippet seconds.
    """
    start, end = float(event["start_s"]), float(event["end_s"])
    offset = max(0.0, start - context_s)
    snippet_end = min(float(file_row["duration_s"]), end + context_s)
    item_id = f"audit_{event['file_id']}_{event['label']}_{start:.0f}"
    return {"item_id": item_id, "kind": "audit", "clip_id": event["file_id"], "site": event["site"],
            "split": event["split"], "period": "", "clip_path": str(snippet_dir / f"{item_id}.wav"),
            "start_s": f"{start - offset:.1f}", "end_s": f"{end - offset:.1f}",
            "predicted": event["label"], "max_conf": event["max_conf"], "stratum": "",
            "drive": event["drive"], "rel_path": file_row["rel_path"],
            "offset_s": f"{offset:.1f}", "snippet_s": f"{snippet_end - offset:.1f}"}


def run_sample(cfg, drive, dry_run, force):
    """Draw and freeze one drive's audit sample, or check the frozen one.

    Args:
        cfg (dict): The whole project config.
        drive (str): Drive name.
        dry_run (bool): Print the strata, write nothing.
        force (bool): Redraw an existing sample (refused once judgments exist).

    Raises:
        SystemExit: If the drive has no events, the settings changed under a frozen
            sample, or --force is used after judging started.
    """
    review_cfg, detection_cfg = cfg["review"], cfg["drive_detection"]
    queue_path, strata_path, judgments_path, snippet_dir = audit_paths(cfg, drive)
    bins, per_stratum = review_cfg["audit_bins"], review_cfg["audit_per_stratum"]
    out_dir = Path(detection_cfg["dir"])
    events = [event for event in read_dicts(out_dir / "events.csv") if event["drive"] == drive]
    if not events:
        raise SystemExit(f"no {drive} event in {out_dir / 'events.csv'}: run detect_drive.py --drive {drive} first")
    strata = stratify(events, model_labels(cfg), bins)

    if queue_path.exists() and not force:
        frozen = read_dicts(strata_path)
        changed = [row["stratum"] for row in frozen
                   if len(strata.get(row["stratum"], [])) != int(row["population"])]
        if sorted(row["stratum"] for row in frozen) != sorted(strata) or changed:
            raise SystemExit(f"the {drive} events or audit_bins changed since the sample was frozen "
                             f"({len(changed)} strata differ, e.g. {changed[:2]}). Restore the settings "
                             f"(drive_detection.min_conf, review.audit_bins), or redraw with --force "
                             f"(only before judging).")
        n_judged = len(load_judgments(judgments_path))
        print(f"sample: {queue_path} (frozen, {len(read_dicts(queue_path))} items, {n_judged} judged)")
        print_strata(frozen)
        return
    if queue_path.exists() and force and load_judgments(judgments_path):
        raise SystemExit(f"{judgments_path} has verdicts: the {drive} sample can no longer be redrawn")

    files = {row["file_id"]: row for row in read_dicts(out_dir / "files" / f"{drive}_files.csv")}
    rng = random.Random(f"{review_cfg['seed']}-audit-{drive}")
    rows, strata_rows = [], []
    for name, members in sorted(strata.items()):
        members = sorted(members, key=lambda event: (event["file_id"], event["label"], float(event["start_s"])))
        picked = rng.sample(members, min(per_stratum, len(members)))
        strata_rows.append({"drive": drive, "stratum": name, "population": len(members),
                            "sampled": len(picked), "n_sites": len({event["site"] for event in members})})
        for event in picked:
            row = queue_row(event, files[event["file_id"]], snippet_dir, review_cfg["context_s"])
            row["stratum"] = name
            rows.append(row)
    rng.shuffle(rows)                       # any session of --n N gets a mix
    print_strata(strata_rows)
    snippet_min = sum(float(row["snippet_s"]) for row in rows) / 60
    if dry_run:
        print(f"{len(rows)} events would be sampled ({snippet_min:.0f} min of snippets); nothing written (--dry-run)")
        return
    write_dicts(queue_path, AUDIT_HEADER, rows)
    write_dicts(strata_path, STRATA_HEADER, strata_rows)
    print(f"froze {len(rows)} events ({snippet_min:.0f} min of snippets): {queue_path}, {strata_path}")
    print(f"next: audit_events.py cut --drive {drive} --root <where the {drive} audio is>")


def print_strata(strata_rows):
    """Print the strata table.

    Args:
        strata_rows (list[dict]): Rows keyed by STRATA_HEADER.
    """
    print(f"    {'stratum':22s} {'population':>10s} {'sampled':>8s} {'sites':>6s}")
    for row in strata_rows:
        print(f"    {row['stratum']:22s} {int(row['population']):10d} {int(row['sampled']):8d} {int(row['n_sites']):6d}")
    print(f"    {'total':22s} {sum(int(row['population']) for row in strata_rows):10d} "
          f"{sum(int(row['sampled']) for row in strata_rows):8d}")


# ---------------------------------------------------------------- cut
def run_cut(cfg, drive, root, limit, dry_run, force):
    """Write the audio snippet of each sampled event.

    Args:
        cfg (dict): The whole project config.
        drive (str): Drive name.
        root (Path): Drive mount point or copy (rel_path is relative to it).
        limit (int | None): Cut at most this many snippets (test).
        dry_run (bool): Count what would be cut, write nothing.
        force (bool): Re-cut existing snippets.

    Raises:
        SystemExit: If the sample is not frozen yet or no source file is under root.
    """
    queue_path, _strata_path, _judgments_path, snippet_dir = audit_paths(cfg, drive)
    if not queue_path.exists():
        raise SystemExit(f"{queue_path} not found: run audit_events.py sample --drive {drive} first")
    rows = read_dicts(queue_path)
    todo = [row for row in rows if force or not Path(row["clip_path"]).exists()]
    missing = [row for row in todo if not (root / row["rel_path"]).exists()]
    print(f"{drive}: {len(rows)} snippets, {len(rows) - len(todo)} already cut, {len(todo)} to cut")
    if missing:
        print(f"warning: {len(missing)} source files not under {root}, e.g. {missing[0]['rel_path']}")
        if len(missing) == len(todo):
            raise SystemExit(f"none of the source files is under {root}: plug in {drive}, or give --root")
    missing_ids = {row["item_id"] for row in missing}
    todo = [row for row in todo if row["item_id"] not in missing_ids][:limit]
    if dry_run or not todo:
        if dry_run:
            print(f"{len(todo)} snippets would be cut into {snippet_dir}; nothing written (--dry-run)")
        return
    snippet_dir.mkdir(parents=True, exist_ok=True)
    for row in todo:
        source = root / row["rel_path"]
        sample_rate = sf.info(source).samplerate
        first = int(round(float(row["offset_s"]) * sample_rate))
        audio, _ = sf.read(source, start=first, frames=int(round(float(row["snippet_s"]) * sample_rate)),
                           dtype="int16", always_2d=True)
        sf.write(row["clip_path"], audio, sample_rate, subtype="PCM_16")
    print(f"cut {len(todo)} snippets into {snippet_dir}")


# ---------------------------------------------------------------- report
def load_audits(cfg):
    """Judged items and stratum populations of every audit queue.

    Args:
        cfg (dict): The whole project config.

    Returns:
        tuple[list[dict], dict, dict]: (judged items with drive, stratum, site, predicted,
            verdict; (drive, stratum) -> population; (drive, stratum) -> sampled).
    """
    review_dir = Path(cfg["review"]["dir"])
    items, population, sampled = [], {}, {}
    for queue_path in sorted(review_dir.glob("audit_*_queue.csv")):
        drive = queue_path.name[len("audit_"):-len("_queue.csv")]
        _queue, strata_path, judgments_path, _snippets = audit_paths(cfg, drive)
        for row in read_dicts(strata_path):
            population[(drive, row["stratum"])] = int(row["population"])
            sampled[(drive, row["stratum"])] = int(row["sampled"])
        verdicts = load_judgments(judgments_path)
        for row in read_dicts(queue_path):
            if row["item_id"] in verdicts:
                items.append({"drive": drive, "stratum": row["stratum"], "site": row["site"],
                              "predicted": row["predicted"], "verdict": verdicts[row["item_id"]]})
    return items, population, sampled


def precision(items, keys, population):
    """Population-weighted precision over some strata.

    Each stratum contributes its judged precision times its population; strata with
    no usable verdict are left out (and reported as missing).

    Args:
        items (list[dict]): Judged items (Unsure already removed).
        keys (list[tuple[str, str]]): The (drive, stratum) keys making up the scope.
        population (dict): (drive, stratum) -> number of events.

    Returns:
        tuple[float, int]: (precision or nan, number of non-empty strata without verdicts).
    """
    correct, judged = Counter(), Counter()
    for item in items:
        key = (item["drive"], item["stratum"])
        judged[key] += 1
        correct[key] += item["verdict"] == item["predicted"]
    used = [key for key in keys if judged[key]]
    missing = sum(1 for key in keys if population[key] and not judged[key])
    total = sum(population[key] for key in used)
    if not total:
        return float("nan"), missing
    return sum(population[key] * correct[key] / judged[key] for key in used) / total, missing


def bootstrap(items, keys, population, n_boot, rng):
    """95 % CI of precision, resampling sites within each drive.

    Args:
        items (list[dict]): Judged items (Unsure removed).
        keys (list[tuple[str, str]]): Strata of the scope.
        population (dict): (drive, stratum) -> number of events.
        n_boot (int): Bootstrap replicates.
        rng (numpy.random.Generator): Seeded generator.

    Returns:
        tuple[float, float]: (2.5 %, 97.5 %) percentiles, nan if not computable.
    """
    by_site = defaultdict(list)
    for item in items:
        by_site[(item["drive"], item["site"])].append(item)
    sites_of = defaultdict(list)
    for drive, site in by_site:
        sites_of[drive].append(site)
    values = []
    for _ in range(n_boot):
        replicate = []
        for drive, sites in sites_of.items():
            for i in rng.integers(0, len(sites), len(sites)):
                replicate += by_site[(drive, sites[i])]
        value, _missing = precision(replicate, keys, population)
        if not np.isnan(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def run_report(cfg, out_dir):
    """Precision of the full-survey events from every audit judged so far.

    Args:
        cfg (dict): The whole project config.
        out_dir (Path): Folder of the result CSVs.

    Raises:
        SystemExit: If no audit sample exists.
    """
    review_cfg = cfg["review"]
    items, population, sampled = load_audits(cfg)
    if not population:
        raise SystemExit("no audit sample: run audit_events.py sample --drive <drive> first")
    usable = [item for item in items if item["verdict"] != UNSURE]
    rng = np.random.default_rng(review_cfg["seed"])
    n_boot = review_cfg["audit_boot"]
    labels = model_labels(cfg)
    names = bin_names(review_cfg["audit_bins"])
    scopes = [("class|bin", label, name, [key for key in population if key[1] == f"{label}|{name}"])
              for label in labels for name in names]
    scopes += [("class", label, "", [key for key in population if key[1].split("|")[0] == label])
               for label in labels]
    scopes.append(("all", "", "", list(population)))

    result_rows = []
    for scope, label, name, keys in scopes:
        key_set = set(keys)
        in_scope = [item for item in items if (item["drive"], item["stratum"]) in key_set]
        usable_in = [item for item in in_scope if item["verdict"] != UNSURE]
        value, missing = precision(usable_in, keys, population)
        low, high = bootstrap(usable_in, keys, population, n_boot, rng) if usable_in else (float("nan"),) * 2
        result_rows.append({"scope": scope, "label": label, "bin": name,
                            "population": sum(population[key] for key in keys),
                            "sampled": sum(sampled[key] for key in keys), "judged": len(in_scope),
                            "unsure": len(in_scope) - len(usable_in),
                            "correct": sum(item["verdict"] == item["predicted"] for item in usable_in),
                            "precision": f"{value:.3f}", "ci_low": f"{low:.3f}", "ci_high": f"{high:.3f}",
                            "strata_missing": missing})

    judged_per_stratum = Counter((item["drive"], item["stratum"]) for item in usable)
    confusion, estimate = Counter(), defaultdict(float)
    for item in usable:
        key = (item["drive"], item["stratum"])
        confusion[(item["predicted"], item["verdict"])] += 1
        estimate[(item["predicted"], item["verdict"])] += population[key] / judged_per_stratum[key]
    confusion_rows = [{"predicted": predicted, "verdict": verdict, "judged": count,
                       "est_events": f"{estimate[(predicted, verdict)]:.1f}"}
                      for (predicted, verdict), count in sorted(confusion.items())]

    gen = cfg["drive_detection"]["gen"]
    result_path = out_dir / f"audit_full_gen{gen}.csv"
    confusion_path = out_dir / f"audit_full_gen{gen}_confusion.csv"
    write_dicts(result_path, RESULT_HEADER, result_rows)
    write_dicts(confusion_path, CONFUSION_HEADER, confusion_rows)

    drives = sorted({drive for drive, _stratum in population})
    print(f"audit of the gen {gen} full-survey events, drives {drives}: {len(items)} judged "
          f"of {sum(sampled.values())} sampled ({len(items) - len(usable)} Unsure, left out)")
    print(f"    {'scope':10s} {'label':9s} {'bin':8s} {'events':>7s} {'judged':>7s} {'precision':>9s}  95 % CI")
    for row in result_rows:
        note = f"   ({row['strata_missing']} strata not judged yet)" if row["strata_missing"] else ""
        print(f"    {row['scope']:10s} {row['label']:9s} {row['bin']:8s} {row['population']:7d} "
              f"{row['judged']:7d} {row['precision']:>9s}  [{row['ci_low']}, {row['ci_high']}]{note}")
    errors = [row for row in confusion_rows if row["predicted"] != row["verdict"]]
    if errors:
        print("errors (predicted -> verdict: judged, estimated events):")
        for row in errors:
            print(f"    {row['predicted']:9s} -> {row['verdict']:9s} {row['judged']:4d} {row['est_events']:>7s}")
    print(f"wrote {result_path}, {confusion_path}")


def main():
    """CLI entry point: sample, cut or report."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["sample", "cut", "report"])
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--drive", help="drive name, a key of survey.drives (sample, cut)")
    parser.add_argument("--root", help="cut: where the drive's audio is (default: survey.drives.<drive>)")
    parser.add_argument("--limit", type=int, help="cut: at most N snippets (test)")
    parser.add_argument("--out", default="results/survey", help="report: folder of the result CSVs")
    parser.add_argument("--dry-run", action="store_true", help="show what would happen, write nothing")
    parser.add_argument("--force", action="store_true",
                        help="sample: redraw (only before judging); cut: re-cut existing snippets")
    args = parser.parse_args()
    cfg = load(args.config)
    for key in ("audit_bins", "audit_per_stratum", "audit_boot"):
        if key not in cfg["review"]:
            raise SystemExit(f"review.{key} missing from {args.config}")
    if args.command == "report":
        run_report(cfg, Path(args.out))
        return
    drives = cfg["survey"]["drives"]
    if args.drive not in drives:
        raise SystemExit(f"--drive must be one of {list(drives)}")
    if args.command == "sample":
        run_sample(cfg, args.drive, args.dry_run, args.force)
    else:
        run_cut(cfg, args.drive, Path(args.root or drives[args.drive]), args.limit, args.dry_run, args.force)


if __name__ == "__main__":
    main()
