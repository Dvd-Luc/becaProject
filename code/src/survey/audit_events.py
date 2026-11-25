"""Audit the full-survey events: judge samples blind, estimate precision, check presence per site.

    python src/survey/audit_events.py sample --drive D1 --dry-run   # strata and populations only
    python src/survey/audit_events.py sample --drive D1             # draw and freeze the sample
    python src/survey/audit_events.py sample --drive D2 --per-stratum 5   # more events per stratum
    python src/survey/audit_events.py cut --drive D1 --root /mnt/d --limit 3   # test
    python src/survey/audit_events.py cut --drive D1 --root /mnt/d  # audit + presence candidates
    python src/survey/review.py --queue audit --drive D1            # judge the sample (blind GUI)
    python src/survey/audit_events.py presence --drive D1 --dry-run # site-by-site check, what to judge
    python src/survey/audit_events.py presence --drive D1           # freeze it (snippets already cut)
    python src/survey/review.py --queue presence --drive D1         # judge the presence check (blind)
    python src/survey/audit_events.py report                        # precision + presence, all drives

detect_drive.py gives thousands of events per drive. This script picks a few hundred,
cuts their audio so review.py can show them, and turns the verdicts into the precision
of all the events and the list of sites where each class is confirmed.

sample    Puts each of the drive's events (drive_detection.dir/events.csv) in a stratum
          and draws a random sample per stratum, seeded by review.seed and the drive.
          review.audit_strata picks the strata:
            site_bin  class x site x max_conf bin (review.audit_bins),
                      review.audit_per_site_bin per stratum: every site is heard at
                      low and high scores, so precision can be compared between
                      sites and bins (used from D1, D2 on)
            site      class x site, review.audit_per_site per stratum (scores not
                      spread: mostly low ones are heard)
            bin       class x max_conf bin, review.audit_per_stratum per stratum
                      (used for D4)
          --per-stratum N overrides the number for one drive (the strata file keeps
          what was drawn). A smaller stratum is taken whole. The queue and the stratum
          populations are frozen and reused as they are (the stratum kind too, whatever
          the config says later); --force redraws, and only while nothing is judged yet.
presence  For each class of review.presence_labels and each site with events of it: if
          no verdict of the drive's audit or presence queues confirms the class there,
          the best events not judged yet (by max_conf) are queued, so that up to
          review.presence_per_site events per site and class get judged. A site is then
          "present" (at least one confirmed), "not confirmed" (all judged events were
          something else) or "not judged". Frozen like the sample.
cut       Writes each queued event (audit and presence queues of the drive) with
          review.context_s on each side as a small WAV snippet, read from the drive or any
          copy of it (--root). Existing snippets are skipped. Run it where the audio is:
          the laptop with the drive plugged in, or the cluster ($SCRATCH/beca/audio/<drive>).
          Before the presence queue exists, it also cuts the presence candidates (the
          best presence_per_site events per class x site outside the audit queue), so
          one cut, right after sample, covers both queues: the drive is needed once.
report    Precision from every audit queue judged so far: an event is correct when the
          verdict is its predicted class; Unsure is left out. Per class x bin (bin strata),
          per class, per class over the events still kept by drive_detection.min_event_conf
          ("kept"), and over all events; each stratum weighted by population / judged.
          95 % CI from a bootstrap that resamples sites within each drive. Also: judged
          and correct per class x site (unweighted, Wilson 95 % CI), the confusion of
          predicted class x verdict, and the presence status of every site x class.

Judging is blind: review.py shows neither the predicted class nor the score. The audit
covers train and heldout sites; only train-site verdicts may ever train a model.

Writes:
    <review.dir>/audit_<drive>_queue.csv     frozen sample, review.py queue columns + drive,
                                             rel_path, offset_s (snippet start in the
                                             file), snippet_s (snippet length)
    <review.dir>/audit_<drive>_strata.csv    kind, drive, stratum, population, sampled, n_sites
    <review.dir>/presence_<drive>_queue.csv  frozen presence check, same columns
    <drive_detection.dir>/audit/snippets/<item_id>.wav   audio of each queued event
    <out>/audit_full_gen<K>.csv              precision per class x bin, class, kept, all
    <out>/audit_full_gen<K>_sites.csv        judged / correct per class x site
    <out>/audit_full_gen<K>_confusion.csv    predicted x verdict: judged, estimated events
    <out>/presence_full_gen<K>.csv           site x class: events, judged, confirmed, status
(review.py writes <review.dir>/{audit,presence}_<drive>_judgments.csv.)

Run from code/.
"""
import argparse
import math
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
STRATA_HEADER = ["kind", "drive", "stratum", "population", "sampled", "n_sites"]
RESULT_HEADER = ["scope", "label", "bin", "population", "sampled", "judged", "unsure",
                 "correct", "precision", "ci_low", "ci_high", "strata_missing"]
SITE_HEADER = ["drive", "site", "split", "label", "events", "judged", "correct", "share",
               "ci_low", "ci_high"]
CONFUSION_HEADER = ["predicted", "verdict", "judged", "est_events"]
PRESENCE_HEADER = ["drive", "site", "split", "label", "events", "max_conf", "judged",
                   "confirmed", "status"]
QUEUE_KINDS = ("audit", "presence")


def bin_names(bins):
    """Names of the max_conf bins, e.g. "0.5-0.7".

    Args:
        bins (list[float]): Bin edges (review.audit_bins).

    Returns:
        list[str]: One name per bin.
    """
    return [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]


def stratum_parts(name, bins):
    """Split a stratum name into class, site and bin.

    Args:
        name (str): "Label|lo-hi" (bin), "Label|site" (site) or "Label|site|lo-hi" (site_bin).
        bins (list[float]): Bin edges (review.audit_bins).

    Returns:
        tuple[str, str, str]: (label, site or "", bin name or "").
    """
    parts = name.split("|")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if parts[1] in bin_names(bins):
        return parts[0], "", parts[1]
    return parts[0], parts[1], ""


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


def class_floor(cfg, label):
    """Lowest best-window score an event of this class can have in events.csv.

    Args:
        cfg (dict): The whole project config.
        label (str): Class name.

    Returns:
        float: max(drive_detection.min_conf, min_event_conf of the class).
    """
    detection_cfg = cfg["drive_detection"]
    return max(detection_cfg["min_conf"], (detection_cfg.get("min_event_conf") or {}).get(label, 0.0))


def queue_paths(cfg, kind, drive):
    """Paths of one drive's queue files.

    Args:
        cfg (dict): The whole project config.
        kind (str): "audit" or "presence".
        drive (str): Drive name.

    Returns:
        tuple[Path, Path, Path]: (queue CSV, judgments CSV, strata CSV; the last is only
            written for audit queues).
    """
    review_dir = Path(cfg["review"]["dir"])
    return (review_dir / f"{kind}_{drive}_queue.csv", review_dir / f"{kind}_{drive}_judgments.csv",
            review_dir / f"{kind}_{drive}_strata.csv")


def snippet_dir_of(cfg):
    """Folder of the audio snippets.

    Args:
        cfg (dict): The whole project config.

    Returns:
        Path: drive_detection.dir/audit/snippets.
    """
    return Path(cfg["drive_detection"]["dir"]) / "audit" / "snippets"


def drive_events(cfg, drive):
    """The drive's rows of the full-survey events.csv.

    Args:
        cfg (dict): The whole project config.
        drive (str): Drive name.

    Returns:
        list[dict]: events.csv rows of that drive.

    Raises:
        SystemExit: If the drive has no events.
    """
    path = Path(cfg["drive_detection"]["dir"]) / "events.csv"
    events = [event for event in read_dicts(path) if event["drive"] == drive]
    if not events:
        raise SystemExit(f"no {drive} event in {path}: run detect_drive.py --drive {drive} first")
    return events


def event_key(row):
    """Identity of an event, the same in events.csv and in queue rows.

    Args:
        row (dict): events.csv row, or queue row (clip_id = file_id, file seconds from
            offset_s + start_s).

    Returns:
        tuple[str, str, int]: (file_id, label, start second rounded).
    """
    if "file_id" in row:
        return row["file_id"], row["label"], round(float(row["start_s"]))
    return row["clip_id"], row["predicted"], round(float(row["offset_s"]) + float(row["start_s"]))


# ---------------------------------------------------------------- sample
def stratify(events, labels, kind, bins):
    """Group events by stratum: class x max_conf bin, class x site, or class x site x bin.

    Args:
        events (list[dict]): events.csv rows of one drive.
        labels (list[str]): Model classes.
        kind (str): "bin", "site" or "site_bin".
        bins (list[float]): Bin edges (bin and site_bin strata).

    Returns:
        dict[str, list[dict]]: "Label|lo-hi", "Label|site" or "Label|site|lo-hi" -> events.
            With bin strata every stratum is present (maybe empty); otherwise only
            non-empty ones.
    """
    names = bin_names(bins)
    strata = {f"{label}|{name}": [] for label in labels for name in names} if kind == "bin" else {}
    outside = 0
    for event in events:
        if event["label"] not in labels:
            outside += 1
            continue
        if kind == "site":
            strata.setdefault(f"{event['label']}|{event['site']}", []).append(event)
            continue
        bin_idx = bin_of(float(event["max_conf"]), bins)
        if bin_idx is None:
            outside += 1
            continue
        if kind == "site_bin":
            strata.setdefault(f"{event['label']}|{event['site']}|{names[bin_idx]}", []).append(event)
            continue
        strata[f"{event['label']}|{names[bin_idx]}"].append(event)
    if outside:
        print(f"warning: {outside} events outside the audit bins {bins} or classes (not sampled)")
    return strata


def queue_row(prefix, event, file_row, snippet_dir, context_s, stratum):
    """Queue row of one event, with its snippet window.

    Args:
        prefix (str): "audit" or "presence" (start of the item id).
        event (dict): events.csv row.
        file_row (dict): files/<drive>_files.csv row of the event's file.
        snippet_dir (Path): Folder of the snippets.
        context_s (float): Seconds of audio kept on each side of the event.
        stratum (str): Stratum (or site group) stored in the queue.

    Returns:
        dict[str, str]: Row keyed by AUDIT_HEADER; start_s / end_s are snippet seconds.
    """
    start, end = float(event["start_s"]), float(event["end_s"])
    offset = max(0.0, start - context_s)
    snippet_end = min(float(file_row["duration_s"]), end + context_s)
    item_id = f"{prefix}_{event['file_id']}_{event['label']}_{start:.0f}"
    return {"item_id": item_id, "kind": prefix, "clip_id": event["file_id"], "site": event["site"],
            "split": event["split"], "period": "", "clip_path": str(snippet_dir / f"{item_id}.wav"),
            "start_s": f"{start - offset:.1f}", "end_s": f"{end - offset:.1f}",
            "predicted": event["label"], "max_conf": event["max_conf"], "stratum": stratum,
            "drive": event["drive"], "rel_path": file_row["rel_path"],
            "offset_s": f"{offset:.1f}", "snippet_s": f"{snippet_end - offset:.1f}"}


def frozen_kind(strata_rows):
    """Stratum kind of a frozen sample (files from before the kind column are bin strata).

    Args:
        strata_rows (list[dict]): Rows of audit_<drive>_strata.csv.

    Returns:
        str: "bin", "site" or "site_bin".
    """
    return strata_rows[0].get("kind") or "bin" if strata_rows else "bin"


def check_frozen(cfg, drive, strata, frozen, bins):
    """Stop if the events changed under a frozen sample.

    Strata lying wholly below a class's current floor (drive_detection.min_event_conf,
    e.g. Call bins under 0.9 after the threshold was raised) are expected to be empty
    now and are not compared.

    Args:
        cfg (dict): The whole project config.
        drive (str): Drive name.
        strata (dict[str, list]): Strata recomputed from the current events.
        frozen (list[dict]): Frozen strata rows.
        bins (list[float]): Bin edges.

    Raises:
        SystemExit: If a compared stratum's population differs.
    """
    upper = dict(zip(bin_names(bins), bins[1:], strict=True))
    changed = []
    for row in frozen:
        label, _site, bin_name = stratum_parts(row["stratum"], bins)
        if bin_name and upper[bin_name] <= class_floor(cfg, label):
            continue
        if len(strata.get(row["stratum"], [])) != int(row["population"]):
            changed.append(row["stratum"])
    if changed:
        raise SystemExit(f"the {drive} events changed since the sample was frozen ({len(changed)} strata "
                         f"differ, e.g. {changed[:2]}). Restore the settings (drive_detection.min_conf, "
                         f"min_event_conf, review.audit_bins), or redraw with --force (only before judging).")


def run_sample(cfg, drive, dry_run, force, per_stratum=None):
    """Draw and freeze one drive's audit sample, or check the frozen one.

    Args:
        cfg (dict): The whole project config.
        drive (str): Drive name.
        dry_run (bool): Print the strata, write nothing.
        force (bool): Redraw an existing sample (refused once judgments exist).
        per_stratum (int | None): Events drawn per stratum; None = the config value of
            the stratum kind.

    Raises:
        SystemExit: If the drive has no events, the events changed under a frozen
            sample, or --force is used after judging started.
    """
    review_cfg = cfg["review"]
    queue_path, judgments_path, strata_path = queue_paths(cfg, "audit", drive)
    bins = review_cfg["audit_bins"]
    events = drive_events(cfg, drive)
    labels = model_labels(cfg)

    if queue_path.exists() and not force:
        frozen = read_dicts(strata_path)
        check_frozen(cfg, drive, stratify(events, labels, frozen_kind(frozen), bins), frozen, bins)
        n_judged = len(load_judgments(judgments_path))
        print(f"sample: {queue_path} (frozen, {frozen_kind(frozen)} strata, "
              f"{len(read_dicts(queue_path))} items, {n_judged} judged)")
        print_strata(frozen)
        return
    if queue_path.exists() and force and load_judgments(judgments_path):
        raise SystemExit(f"{judgments_path} has verdicts: the {drive} sample can no longer be redrawn")

    kind = review_cfg.get("audit_strata", "bin")
    config_key = {"bin": "audit_per_stratum", "site": "audit_per_site", "site_bin": "audit_per_site_bin"}
    if kind not in config_key:
        raise SystemExit(f"review.audit_strata must be one of {list(config_key)}, not {kind}")
    if per_stratum is None:
        if config_key[kind] not in review_cfg:
            raise SystemExit(f"review.{config_key[kind]} missing from the config (needed by {kind} strata)")
        per_stratum = review_cfg[config_key[kind]]
    strata = stratify(events, labels, kind, bins)
    files = {row["file_id"]: row for row in read_dicts(Path(cfg["drive_detection"]["dir"]) / "files" / f"{drive}_files.csv")}
    rng = random.Random(f"{review_cfg['seed']}-audit-{drive}")
    rows, strata_rows = [], []
    for name, members in sorted(strata.items()):
        members = sorted(members, key=lambda event: (event["file_id"], event["label"], float(event["start_s"])))
        picked = rng.sample(members, min(per_stratum, len(members)))
        strata_rows.append({"kind": kind, "drive": drive, "stratum": name, "population": len(members),
                            "sampled": len(picked), "n_sites": len({event["site"] for event in members})})
        rows += [queue_row("audit", event, files[event["file_id"]], snippet_dir_of(cfg),
                           review_cfg["context_s"], name) for event in picked]
    rng.shuffle(rows)                       # any session of --n N gets a mix
    print_strata(strata_rows)
    snippet_min = sum(float(row["snippet_s"]) for row in rows) / 60
    if dry_run:
        print(f"{len(rows)} events would be sampled ({kind} strata, up to {per_stratum} per stratum, "
              f"{snippet_min:.0f} min of snippets); "
              f"nothing written (--dry-run)")
        return
    write_dicts(queue_path, AUDIT_HEADER, rows)
    write_dicts(strata_path, STRATA_HEADER, strata_rows)
    print(f"froze {len(rows)} events ({kind} strata, up to {per_stratum} per stratum, "
          f"{snippet_min:.0f} min of snippets): {queue_path}, {strata_path}")
    print(f"next: audit_events.py cut --drive {drive} --root <where the {drive} audio is>")


def print_strata(strata_rows):
    """Print the strata table.

    Args:
        strata_rows (list[dict]): Rows keyed by STRATA_HEADER.
    """
    width = max([22] + [len(short(row["stratum"])) for row in strata_rows])
    print(f"    {'stratum':{width}s} {'population':>10s} {'sampled':>8s} {'sites':>6s}")
    for row in strata_rows:
        print(f"    {short(row['stratum']):{width}s} {int(row['population']):10d} {int(row['sampled']):8d} "
              f"{int(row['n_sites']):6d}")
    print(f"    {'total':{width}s} {sum(int(row['population']) for row in strata_rows):10d} "
          f"{sum(int(row['sampled']) for row in strata_rows):8d}")


def short(name):
    """Site or stratum name without the long survey prefix, for printing.

    Args:
        name (str): Site or stratum name.

    Returns:
        str: The name without "2026_Beca_SMMICRO_" / "BECA_SMMICRO_".
    """
    return name.replace("2026_Beca_SMMICRO_", "").replace("BECA_SMMICRO_", "")


# ---------------------------------------------------------------- presence
def judged_events(cfg, drive):
    """Verdicts already given on the drive's events, in any of its queues.

    Args:
        cfg (dict): The whole project config.
        drive (str): Drive name.

    Returns:
        dict[tuple, str]: event_key -> verdict (Unsure included).
    """
    verdicts = {}
    for kind in QUEUE_KINDS:
        queue_path, judgments_path, _strata = queue_paths(cfg, kind, drive)
        if not queue_path.exists():
            continue
        judged = load_judgments(judgments_path)
        for row in read_dicts(queue_path):
            if row["item_id"] in judged:
                verdicts[event_key(row)] = judged[row["item_id"]]
    return verdicts


def queued_events(cfg, drive, kinds=QUEUE_KINDS):
    """Events already in the drive's queues, judged or not.

    Args:
        cfg (dict): The whole project config.
        drive (str): Drive name.
        kinds (tuple[str]): Queue kinds to look at.

    Returns:
        set[tuple]: event_key of every queued event.
    """
    keys = set()
    for kind in kinds:
        queue_path = queue_paths(cfg, kind, drive)[0]
        if queue_path.exists():
            keys |= {event_key(row) for row in read_dicts(queue_path)}
    return keys


def presence_candidates(cfg, events, audit_queued):
    """Events that the presence check may queue, per class x site, best first.

    Args:
        cfg (dict): The whole project config.
        events (list[dict]): events.csv rows of one drive.
        audit_queued (set[tuple]): event_key of the drive's audit queue.

    Returns:
        dict[tuple[str, str], tuple[list[dict], list[dict]]]: (label, site) -> (all events
            of that class there, events outside the audit queue sorted by max_conf, best
            first), for the classes of review.presence_labels.
    """
    by_cell = defaultdict(list)
    for event in events:
        if event["label"] in cfg["review"]["presence_labels"]:
            by_cell[(event["label"], event["site"])].append(event)
    return {cell: (members, sorted((event for event in members if event_key(event) not in audit_queued),
                                   key=lambda event: -float(event["max_conf"])))
            for cell, members in sorted(by_cell.items())}


def run_presence(cfg, drive, dry_run, force):
    """Queue the best events of each site x class not confirmed yet, and freeze the queue.

    Args:
        cfg (dict): The whole project config.
        drive (str): Drive name.
        dry_run (bool): Print what would be queued, write nothing.
        force (bool): Rebuild an existing presence queue (refused once judgments exist).

    Raises:
        SystemExit: If the drive has no events, or --force is used after judging started.
    """
    review_cfg = cfg["review"]
    queue_path, judgments_path, _strata = queue_paths(cfg, "presence", drive)
    if queue_path.exists() and not force:
        rows = read_dicts(queue_path)
        print(f"presence: {queue_path} (frozen, {len(rows)} items, {len(load_judgments(judgments_path))} judged)")
        return
    if queue_path.exists() and force and load_judgments(judgments_path):
        raise SystemExit(f"{judgments_path} has verdicts: the {drive} presence queue can no longer be rebuilt")
    events = drive_events(cfg, drive)
    audit_verdicts = judged_events(cfg, drive)          # audit verdicts (presence not built yet)
    audit_queued = queued_events(cfg, drive, kinds=("audit",))
    unjudged_audit = len(audit_queued - set(audit_verdicts))
    if unjudged_audit:
        print(f"warning: {unjudged_audit} {drive} audit events are not judged yet; judge them first "
              f"for a shorter presence queue")
    files = {row["file_id"]: row for row in read_dicts(Path(cfg["drive_detection"]["dir"]) / "files" / f"{drive}_files.csv")}
    per_site = review_cfg["presence_per_site"]
    rows, report = [], []
    for (label, site), (members, candidates) in presence_candidates(cfg, events, audit_queued).items():
        verdicts = [audit_verdicts[event_key(event)] for event in members if event_key(event) in audit_verdicts]
        confirmed = sum(verdict == label for verdict in verdicts)
        judged = sum(verdict != UNSURE for verdict in verdicts)
        if confirmed:
            report.append((label, site, len(members), "confirmed by the audit", 0))
            continue
        take = candidates[:max(0, per_site - judged)]
        rows += [queue_row("presence", event, files[event["file_id"]], snippet_dir_of(cfg),
                           review_cfg["context_s"], label) for event in take]
        report.append((label, site, len(members), f"{judged} judged in the audit, none confirmed", len(take)))
    print(f"    {'class':9s} {'site':38s} {'events':>6s} {'queued':>6s}  status")
    for label, site, n_events, status, n_take in report:
        print(f"    {label:9s} {short(site)[:38]:38s} {n_events:6d} {n_take:6d}  {status}")
    rng = random.Random(f"{review_cfg['seed']}-presence-{drive}")
    rng.shuffle(rows)                       # blind: no order by site or score
    if dry_run:
        print(f"{len(rows)} events would be queued; nothing written (--dry-run)")
        return
    write_dicts(queue_path, AUDIT_HEADER, rows)
    uncut = sum(not Path(row["clip_path"]).exists() for row in rows)
    print(f"froze {len(rows)} events: {queue_path} ({len(rows) - uncut} snippets already cut)")
    if uncut:
        print(f"next: audit_events.py cut --drive {drive} --root <where the {drive} audio is>")
    else:
        print(f"next: review.py --queue presence --drive {drive}")


# ---------------------------------------------------------------- cut
def run_cut(cfg, drive, root, limit, dry_run, force):
    """Write the audio snippet of each event of the drive's audit and presence queues.

    While the drive has no frozen presence queue, the presence candidates are cut too
    (the best review.presence_per_site events per class x site outside the audit
    queue): the presence queue built after judging only ever takes events among them,
    with the same snippet paths, so the drive is needed once.

    Args:
        cfg (dict): The whole project config.
        drive (str): Drive name.
        root (Path): Drive mount point or copy (rel_path is relative to it).
        limit (int | None): Cut at most this many snippets (test).
        dry_run (bool): Count what would be cut, write nothing.
        force (bool): Re-cut existing snippets.

    Raises:
        SystemExit: If no queue is frozen yet or no source file is under root.
    """
    rows, counts = [], {}
    for kind in QUEUE_KINDS:
        queue_path = queue_paths(cfg, kind, drive)[0]
        if queue_path.exists():
            queue_rows = read_dicts(queue_path)
            rows += queue_rows
            counts[kind] = len(queue_rows)
    if not rows:
        raise SystemExit(f"no {drive} queue: run audit_events.py sample (or presence) --drive {drive} first")
    snippet_dir = snippet_dir_of(cfg)
    if "presence" not in counts:
        review_cfg = cfg["review"]
        files = {row["file_id"]: row for row in
                 read_dicts(Path(cfg["drive_detection"]["dir"]) / "files" / f"{drive}_files.csv")}
        cells = presence_candidates(cfg, drive_events(cfg, drive), queued_events(cfg, drive, kinds=("audit",)))
        candidates = [queue_row("presence", event, files[event["file_id"]], snippet_dir,
                                review_cfg["context_s"], label)
                      for (label, _site), (_members, best) in cells.items()
                      for event in best[:review_cfg["presence_per_site"]]]
        rows += candidates
        counts["presence candidates"] = len(candidates)
    todo = [row for row in rows if force or not Path(row["clip_path"]).exists()]
    missing = [row for row in todo if not (root / row["rel_path"]).exists()]
    parts = " + ".join(f"{n} {kind}" for kind, n in counts.items())
    print(f"{drive}: {len(rows)} snippets ({parts}), {len(rows) - len(todo)} already cut, {len(todo)} to cut")
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
def drives_with(cfg, kind):
    """Drives that have a frozen queue of this kind.

    Args:
        cfg (dict): The whole project config.
        kind (str): "audit" or "presence".

    Returns:
        list[str]: Drive names.
    """
    review_dir = Path(cfg["review"]["dir"])
    return sorted(path.name[len(kind) + 1:-len("_queue.csv")] for path in review_dir.glob(f"{kind}_*_queue.csv"))


def load_audits(cfg):
    """Judged items and stratum populations of every audit queue.

    Args:
        cfg (dict): The whole project config.

    Returns:
        tuple[list[dict], dict, dict]: (judged items with drive, stratum, site, split,
            predicted, verdict; (drive, stratum) -> population; (drive, stratum) -> sampled).
    """
    items, population, sampled = [], {}, {}
    for drive in drives_with(cfg, "audit"):
        queue_path, judgments_path, strata_path = queue_paths(cfg, "audit", drive)
        for row in read_dicts(strata_path):
            population[(drive, row["stratum"])] = int(row["population"])
            sampled[(drive, row["stratum"])] = int(row["sampled"])
        verdicts = load_judgments(judgments_path)
        for row in read_dicts(queue_path):
            if row["item_id"] in verdicts:
                items.append({"drive": drive, "stratum": row["stratum"], "site": row["site"],
                              "split": row["split"], "predicted": row["predicted"],
                              "max_conf": float(row["max_conf"]), "verdict": verdicts[row["item_id"]]})
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


def wilson(successes, n):
    """Wilson 95 % interval of a proportion.

    Args:
        successes (int): Number of successes.
        n (int): Number of trials.

    Returns:
        tuple[float, float]: (low, high), nan when n is 0.
    """
    if not n:
        return float("nan"), float("nan")
    z, share = 1.96, successes / n
    centre = (share + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def precision_rows(cfg, items, population, sampled, rng):
    """Precision rows for every scope: class x bin, class, kept, all.

    Args:
        cfg (dict): The whole project config.
        items (list[dict]): Judged audit items (Unsure included).
        population (dict): (drive, stratum) -> number of events.
        sampled (dict): (drive, stratum) -> number sampled.
        rng (numpy.random.Generator): Seeded generator.

    Returns:
        list[dict]: Rows keyed by RESULT_HEADER.
    """
    review_cfg = cfg["review"]
    labels = model_labels(cfg)
    bins = review_cfg["audit_bins"]
    names = bin_names(bins)
    lower = dict(zip(names, bins[:-1], strict=True))
    parts = {key: stratum_parts(key[1], bins) for key in population}

    def kept(key):
        """True if the stratum's events are still in events.csv under the class floor.

        Args:
            key (tuple[str, str]): (drive, stratum).

        Returns:
            bool: False for a bin stratum wholly below the class floor.
        """
        label, _site, bin_name = parts[key]
        return not bin_name or lower[bin_name] >= class_floor(cfg, label)

    scopes = [("class|bin", label, name, [key for key in population if parts[key][::2] == (label, name)])
              for label in labels for name in names]
    scopes = [scope for scope in scopes if scope[3]]
    scopes += [("class", label, "", [key for key in population if key[1].split("|")[0] == label])
               for label in labels]
    scopes += [("kept", label, f">={class_floor(cfg, label)}",
                [key for key in population if key[1].split("|")[0] == label and kept(key)])
               for label in labels if any(not kept(key) for key in population if key[1].split("|")[0] == label)]
    scopes.append(("all", "", "", list(population)))
    scopes.append(("all kept", "", "", [key for key in population if kept(key)]))
    rows = []
    for scope, label, name, keys in scopes:
        key_set = set(keys)
        in_scope = [item for item in items if (item["drive"], item["stratum"]) in key_set]
        usable = [item for item in in_scope if item["verdict"] != UNSURE]
        value, missing = precision(usable, keys, population)
        low, high = (bootstrap(usable, keys, population, review_cfg["audit_boot"], rng)
                     if usable else (float("nan"), float("nan")))
        rows.append({"scope": scope, "label": label, "bin": name,
                     "population": sum(population[key] for key in keys),
                     "sampled": sum(sampled[key] for key in keys), "judged": len(in_scope),
                     "unsure": len(in_scope) - len(usable),
                     "correct": sum(item["verdict"] == item["predicted"] for item in usable),
                     "precision": f"{value:.3f}", "ci_low": f"{low:.3f}", "ci_high": f"{high:.3f}",
                     "strata_missing": missing})
    return rows


def site_rows(cfg, items):
    """Judged and correct audit events per drive x site x class (unweighted).

    Only events still above the class floor count, so a site is judged on the events
    that remain in events.csv.

    Args:
        cfg (dict): The whole project config.
        items (list[dict]): Judged audit items.

    Returns:
        list[dict]: Rows keyed by SITE_HEADER.
    """
    events = Counter()
    for drive in drives_with(cfg, "audit"):
        for event in drive_events(cfg, drive):
            events[(drive, event["site"], event["label"])] += 1
    cells = defaultdict(list)
    for item in items:
        if item["verdict"] != UNSURE and item["max_conf"] >= class_floor(cfg, item["predicted"]):
            cells[(item["drive"], item["site"], item["split"], item["predicted"])].append(item)
    rows = []
    for (drive, site, split, label), cell in sorted(cells.items()):
        correct = sum(item["verdict"] == label for item in cell)
        low, high = wilson(correct, len(cell))
        rows.append({"drive": drive, "site": site, "split": split, "label": label,
                     "events": events[(drive, site, label)], "judged": len(cell), "correct": correct,
                     "share": f"{correct / len(cell):.2f}", "ci_low": f"{low:.2f}", "ci_high": f"{high:.2f}"})
    return rows


def presence_rows(cfg):
    """Presence status of every drive x site x class of review.presence_labels.

    Uses every verdict of the drive (audit and presence queues).

    Args:
        cfg (dict): The whole project config.

    Returns:
        list[dict]: Rows keyed by PRESENCE_HEADER.
    """
    rows = []
    for drive in sorted(set(drives_with(cfg, "audit")) | set(drives_with(cfg, "presence"))):
        verdicts = judged_events(cfg, drive)
        cells = defaultdict(list)
        for event in drive_events(cfg, drive):
            if event["label"] in cfg["review"]["presence_labels"]:
                cells[(event["site"], event["split"], event["label"])].append(event)
        for (site, split, label), members in sorted(cells.items()):
            judged = [verdicts[event_key(event)] for event in members
                      if verdicts.get(event_key(event), UNSURE) != UNSURE]
            confirmed = sum(verdict == label for verdict in judged)
            status = "present" if confirmed else "not confirmed" if judged else "not judged"
            rows.append({"drive": drive, "site": site, "split": split, "label": label,
                         "events": len(members), "max_conf": f"{max(float(event['max_conf']) for event in members):.3f}",
                         "judged": len(judged), "confirmed": confirmed, "status": status})
    return rows


def run_report(cfg, out_dir):
    """Precision of the full-survey events and presence per site, from every verdict so far.

    Args:
        cfg (dict): The whole project config.
        out_dir (Path): Folder of the result CSVs.

    Raises:
        SystemExit: If no audit sample exists.
    """
    items, population, sampled = load_audits(cfg)
    if not population:
        raise SystemExit("no audit sample: run audit_events.py sample --drive <drive> first")
    rng = np.random.default_rng(cfg["review"]["seed"])
    result_rows = precision_rows(cfg, items, population, sampled, rng)
    per_site = site_rows(cfg, items)
    usable = [item for item in items if item["verdict"] != UNSURE]
    judged_per_stratum = Counter((item["drive"], item["stratum"]) for item in usable)
    confusion, estimate = Counter(), defaultdict(float)
    for item in usable:
        key = (item["drive"], item["stratum"])
        confusion[(item["predicted"], item["verdict"])] += 1
        estimate[(item["predicted"], item["verdict"])] += population[key] / judged_per_stratum[key]
    confusion_rows = [{"predicted": predicted, "verdict": verdict, "judged": count,
                       "est_events": f"{estimate[(predicted, verdict)]:.1f}"}
                      for (predicted, verdict), count in sorted(confusion.items())]
    presence = presence_rows(cfg)

    gen = cfg["drive_detection"]["gen"]
    paths = {"precision": out_dir / f"audit_full_gen{gen}.csv",
             "sites": out_dir / f"audit_full_gen{gen}_sites.csv",
             "confusion": out_dir / f"audit_full_gen{gen}_confusion.csv",
             "presence": out_dir / f"presence_full_gen{gen}.csv"}
    write_dicts(paths["precision"], RESULT_HEADER, result_rows)
    write_dicts(paths["sites"], SITE_HEADER, per_site)
    write_dicts(paths["confusion"], CONFUSION_HEADER, confusion_rows)
    write_dicts(paths["presence"], PRESENCE_HEADER, presence)

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
    weak = [row for row in per_site if int(row["correct"]) < int(row["judged"])]
    if weak:
        print("sites with errors (events kept; correct / judged):")
        for row in weak:
            print(f"    {row['drive']} {short(row['site'])[:36]:36s} {row['label']:9s} "
                  f"{row['correct']}/{row['judged']}  [{row['ci_low']}, {row['ci_high']}]")
    if presence:
        status = Counter((row["label"], row["status"]) for row in presence)
        print("presence (sites): " + ", ".join(f"{label} {state} {n}" for (label, state), n in sorted(status.items())))
    print("wrote " + ", ".join(str(path) for path in paths.values()))


def main():
    """CLI entry point: sample, presence, cut or report."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["sample", "presence", "cut", "report"])
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--drive", help="drive name, a key of survey.drives (sample, presence, cut)")
    parser.add_argument("--root", help="cut: where the drive's audio is (default: survey.drives.<drive>)")
    parser.add_argument("--limit", type=int, help="cut: at most N snippets (test)")
    parser.add_argument("--per-stratum", type=int,
                        help="sample: events drawn per stratum (overrides review.audit_per_site_bin / "
                             "audit_per_site / audit_per_stratum for this drive)")
    parser.add_argument("--out", default="results/survey", help="report: folder of the result CSVs")
    parser.add_argument("--dry-run", action="store_true", help="show what would happen, write nothing")
    parser.add_argument("--force", action="store_true",
                        help="sample / presence: rebuild (only before judging); cut: re-cut existing snippets")
    args = parser.parse_args()
    cfg = load(args.config)
    needed = ("audit_bins", "audit_per_stratum", "audit_per_site", "audit_boot",
              "presence_labels", "presence_per_site")
    absent = [key for key in needed if key not in cfg["review"]]
    if absent:
        raise SystemExit(f"review.{absent[0]} missing from {args.config} (needed: {', '.join(needed)})")
    if args.command == "report":
        run_report(cfg, Path(args.out))
        return
    drives = cfg["survey"]["drives"]
    if args.drive not in drives:
        raise SystemExit(f"--drive must be one of {list(drives)}")
    if args.command == "sample":
        run_sample(cfg, args.drive, args.dry_run, args.force, args.per_stratum)
    elif args.command == "presence":
        run_presence(cfg, args.drive, args.dry_run, args.force)
    else:
        run_cut(cfg, args.drive, Path(args.root or drives[args.drive]), args.limit, args.dry_run, args.force)


if __name__ == "__main__":
    main()
