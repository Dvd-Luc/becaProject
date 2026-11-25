"""Compare model generations on a fixed test set: the blind survey windows, or the XC test.

    python src/survey/evaluate.py --gens 0 1 --dry-run
    python src/survey/evaluate.py --gens 0 1
    python src/survey/evaluate.py --gens 0 1 2 --boot 1000
    python src/survey/evaluate.py --gens 0 1 --testset xc
    python src/survey/evaluate.py --gens 2 --testset audit      # check: gen 2 rescored = its max_conf
    python src/survey/evaluate.py --gens 2 3 --testset audit

Every generation scores exactly the same 3 s windows, so the numbers are comparable:
  1. the test windows are cut once (kept)
  2. each generations.dir/gen_K/model.tflite scores them (BirdNET, min_conf 0.01); the
     scores are cached in one CSV per generation (delete it to re-score)
  3. per class, plus "any" (any snipe sound vs nothing): average precision (AP) and
     precision / recall at generations.eval_thresholds, with 95% bootstrap intervals
     and the paired difference to the first generation.

--testset survey (default): the blind heldout windows judged with review.py --queue test.
  The windows were sampled per v0-score stratum, so each judged window stands for
  population / n_judged windows of its stratum (test_strata.csv); without weights, rare
  high-score windows would be over-represented. Bootstrap: windows resampled within
  strata. Unsure verdicts are left out; Noise windows are negatives for every class.
  Also false_alarms_per_h@t: windows judged NOT of a class that score >= t for it (for
  "any": Noise windows whose best score is >= t), weighted, per hour of heldout audio
  (all stratum populations x 3 s). It needs no snipe at the sites, so it measures field
  false alarms (e.g. a false Drumming = false breeding evidence) even where a class is
  absent and its recall cannot be measured.

--testset xc: the XC test sample (annotation_tools.py sample-test), recordists never
  used for training, annotated with annotator.py --section annotator_xc_test. Only
  recordings with a label file count (annotate in any order, re-run any time). Each
  recording is cut into a grid of 3 s windows. A window is positive for a class when a
  label of that class overlaps it by >= min(TAU s, label length), the rule of
  make_training_clips.py; a smaller overlap makes it ambiguous for that class (left out
  of that class only). Unweighted. Bootstrap: recordists resampled, with all their
  windows (windows of one recording are not independent). XC recordings are focal and
  clean: this measures generalisation to new recordists, not field precision.

--testset audit: the heldout-site events of the full-survey audit (audit_events.py,
  judged blind with review.py --queue audit), gen_2's events still kept by the class
  floors (so gen_2 keeps all of them: it is the reference). Never trained on (only
  train-site verdicts train). Each event is cut into its 3 s windows on its recording's
  grid (from the snippets). A generation keeps an event if its best window inside the
  event scores >= the class floor (drive_detection.min_conf, min_event_conf: FIXED,
  never tuned on this set) for the predicted class. Per class and "all": true events
  kept and precision of the kept events (weighted by audit stratum population /
  judged), false events removed (share); bootstrap over heldout sites. It measures how
  a generation filters gen_2's events, not recall (that is the XC test). Unsure and the
  presence verdicts (chosen by score, not at random) are left out.

Writes:
  survey  <review.dir>/test_windows/<item_id>.wav, generations.dir/gen_K/test_scores.csv,
          results/survey/compare.csv, results/survey/pr_<class>.png
  xc      <xc_test.dir>/windows/<window_id>.wav, generations.dir/gen_K/xc_scores.csv,
          results/survey/compare_xc.csv, results/survey/pr_xc_<class>.png
  audit   <review.dir>/audit_windows/<item_id>_w<k>.wav, generations.dir/gen_K/audit_scores.csv,
          results/survey/compare_audit.csv
  (gen_K/<cache>_params.csv: BirdNET's analysis parameters of the last scoring run)

Run from code/.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import load
from training.make_training_clips import overlap, read_labels

WIN = 3.0
TAU = 1.0                      # s, as make_training_clips.py --tau
warnings.filterwarnings("ignore", "All-NaN slice")   # precision is NaN when nothing passes
NOISE, UNSURE = "Noise", "Unsure"
ANY = "any"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed order


def read_dicts(path):
    """Rows of a CSV file as dicts.

    Args:
        path (str | Path): CSV file with a header row.

    Returns:
        list[dict[str, str]]: One dict per row, keyed by column name.
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cut_window(src, start_s, dst, subtype=None):
    """Write the 3 s window of src starting at start_s to dst (kept if it exists).

    Args:
        src (str | Path): Source audio file.
        start_s (float): Window start, seconds.
        dst (Path): Output wav file.
        subtype (str | None): Output subtype (None: the source's, e.g. for wav sources).

    Returns:
        bool: True if the window was cut now.
    """
    if dst.exists():
        return False
    info = sf.info(src)
    audio, sample_rate = sf.read(src, start=int(start_s * info.samplerate),
                                 frames=int(WIN * info.samplerate), dtype="float32")
    sf.write(dst, audio, sample_rate, subtype=subtype or info.subtype)
    return True


# ---------- test sets ----------
def load_survey(cfg, args):
    """The judged blind survey windows (review.py --queue test), cut once.

    Args:
        cfg (dict): Whole config (review section).
        args (argparse.Namespace): review_dir, dry_run.

    Returns:
        dict: ids (list[str]), wavs (dict id -> Path), positives (list[set[str]]),
            ambiguous (list[set[str]]), weights (numpy.ndarray), resample
            (callable rng -> index array), cache (score CSV name), legacy (old score
            folder name), suffix (output name suffix).

    Raises:
        SystemExit: If there are no test judgments yet.
    """
    review_dir = Path(args.review_dir or cfg["review"]["dir"])
    queue = {row["item_id"]: row for row in read_dicts(review_dir / "test_queue.csv")}
    judgments_path = review_dir / "test_judgments.csv"
    if not judgments_path.exists():
        raise SystemExit(f"no test judgments yet: {judgments_path}")
    verdict_of = {row["item_id"]: row["verdict"] for row in read_dicts(judgments_path)}
    items = [queue[item_id] for item_id in verdict_of if item_id in queue and verdict_of[item_id] != UNSURE]
    population = {row["stratum"]: int(row["population"])
                  for row in read_dicts(review_dir / "test_strata.csv")}
    n_judged = Counter(item["stratum"] for item in items)
    weights = np.array([population[item["stratum"]] / n_judged[item["stratum"]] for item in items])
    strata = np.array([item["stratum"] for item in items])
    verdicts = [verdict_of[item["item_id"]] for item in items]
    print(f"{len(items)} judged test windows (unsure left out): {dict(Counter(verdicts))}")
    print("judged per stratum:",
          {stratum: f"{n_judged[stratum]}/{population[stratum]}" for stratum in population})

    windows_dir = review_dir / "test_windows"
    wavs = {item["item_id"]: windows_dir / f"{item['item_id']}.wav" for item in items}
    if not args.dry_run:
        windows_dir.mkdir(parents=True, exist_ok=True)
        n_cut = sum(cut_window(item["clip_path"], float(item["start_s"]), wavs[item["item_id"]])
                    for item in items)
        if n_cut:
            print(f"cut {n_cut} test windows into {windows_dir}")

    groups = [np.where(strata == stratum)[0] for stratum in np.unique(strata)]

    def resample(rng):
        """One stratified bootstrap sample: windows resampled within each stratum.

        Args:
            rng (numpy.random.Generator): Random generator.

        Returns:
            numpy.ndarray: Window indices.
        """
        return np.concatenate([rng.choice(indices, len(indices)) for indices in groups])

    hours = sum(population.values()) * WIN / 3600       # all heldout windows, judged or not
    return {"ids": [item["item_id"] for item in items], "wavs": wavs,
            "positives": [set() if verdict == NOISE else {verdict} for verdict in verdicts],
            "ambiguous": [set() for _ in items], "weights": weights, "resample": resample,
            "cache": "test_scores", "legacy": "test_scores", "suffix": "", "hours": hours}


def load_xc(cfg, args):
    """The annotated recordings of the XC test sample, as a grid of 3 s windows.

    Args:
        cfg (dict): Whole config (xc_test, annotator_xc_test sections).
        args (argparse.Namespace): dry_run.

    Returns:
        dict: Same keys as load_survey.

    Raises:
        SystemExit: If the sample is missing or nothing is annotated yet.
    """
    test_cfg = cfg["xc_test"]
    test_dir = Path(test_cfg["dir"])
    sample_path = test_dir / "sample.csv"
    if not sample_path.exists():
        raise SystemExit(f"{sample_path} not found (annotation_tools.py sample-test)")
    annot_cfg = cfg["annotator_xc_test"]
    audio_dir, labels_dir = Path(annot_cfg["audio_dir"]), Path(annot_cfg["labels_dir"])
    file_col, group_col = annot_cfg.get("file_col", "file-name"), annot_cfg["group_cols"][0]
    sample = read_dicts(sample_path)
    annotated = [row for row in sample
                 if (labels_dir / f"{Path(row[file_col]).stem}.txt").exists()]
    if not annotated:
        raise SystemExit(f"no annotated recording of the XC test sample yet in {labels_dir}")

    windows_dir = test_dir / "windows"
    if not args.dry_run:
        windows_dir.mkdir(parents=True, exist_ok=True)
    ids, wavs, positives, ambiguous, recordists = [], {}, [], [], []
    confirmed = defaultdict(Counter)       # stratum -> label -> recordings having it
    n_cut, missing = 0, []
    for row in annotated:
        audio_path = audio_dir / row[file_col]
        if not audio_path.exists():
            missing.append(row[file_col])
            continue
        labels = read_labels(labels_dir / f"{Path(row[file_col]).stem}.txt")
        confirmed[row["stratum"]].update({label for _, _, label in labels} | {"(recordings)"})
        n_windows = int(sf.info(audio_path).duration // WIN)
        for k in range(n_windows):
            win_start = k * WIN
            window_id = f"{Path(row[file_col]).stem}_{win_start:07.1f}s"
            positive, partial = set(), set()
            for start, end, label in labels:
                shared = overlap(win_start, win_start + WIN, start, end)
                if shared >= min(TAU, end - start) - 1e-9:
                    positive.add(label)
                elif shared > 0:
                    partial.add(label)
            ids.append(window_id)
            wavs[window_id] = windows_dir / f"{window_id}.wav"
            positives.append(positive)
            ambiguous.append(partial - positive)
            recordists.append(row[group_col].strip())
            if not args.dry_run:
                n_cut += cut_window(audio_path, win_start, wavs[window_id], subtype="PCM_16")
    if missing:
        print(f"warning: {len(missing)} annotated recordings not found in {audio_dir}, "
              f"e.g. {missing[:3]}")
    print(f"XC test: {len(annotated)}/{len(sample)} sampled recordings annotated, "
          f"{len(set(recordists))} recordists, {len(ids)} windows")
    print("positive windows per class:",
          dict(Counter(label for positive in positives for label in positive)))
    print("recordings with each label, per sampling stratum (XC tags vs annotation):")
    for stratum, counts in confirmed.items():
        n_recordings = counts.pop("(recordings)")
        print(f"  {stratum}: {n_recordings} annotated; "
              + ", ".join(f"{label} {count}" for label, count in sorted(counts.items())))
    if n_cut:
        print(f"cut {n_cut} windows into {windows_dir}")

    recordist_array = np.array(recordists)
    groups = [np.where(recordist_array == recordist)[0] for recordist in np.unique(recordist_array)]

    def resample(rng):
        """One cluster bootstrap sample: recordists resampled, with all their windows.

        Args:
            rng (numpy.random.Generator): Random generator.

        Returns:
            numpy.ndarray: Window indices.
        """
        return np.concatenate([groups[k] for k in rng.choice(len(groups), len(groups))])

    return {"ids": ids, "wavs": wavs, "positives": positives, "ambiguous": ambiguous,
            "weights": np.ones(len(ids)), "resample": resample,
            "cache": "xc_scores", "legacy": None, "suffix": "_xc", "hours": None}


def class_floor(cfg, label):
    """Lowest best-window score a kept full-survey event of this class may have.

    Args:
        cfg (dict): Whole config (drive_detection section).
        label (str): Class name.

    Returns:
        float: max(drive_detection.min_conf, min_event_conf of the class).
    """
    detection_cfg = cfg["drive_detection"]
    return max(detection_cfg["min_conf"], (detection_cfg.get("min_event_conf") or {}).get(label, 0.0))


def load_audit(cfg, args):
    """Heldout-site events of the full-survey audit, as 3 s windows on their file's grid.

    Args:
        cfg (dict): Whole config (review, drive_detection sections).
        args (argparse.Namespace): review_dir, dry_run.

    Returns:
        dict: events (list of dicts: item_id, drive, site, predicted, true, weight,
            windows, max_conf), wavs (window id -> Path), cache, legacy, suffix.

    Raises:
        SystemExit: If no audit is judged or a snippet is missing.
    """
    review_dir = Path(args.review_dir or cfg["review"]["dir"])
    events, wavs = [], {}
    windows_dir = review_dir / "audit_windows"
    n_cut, missing = 0, []
    for queue_path in sorted(review_dir.glob("audit_D*_queue.csv")):
        drive = queue_path.name[len("audit_"):-len("_queue.csv")]
        judgments_path = review_dir / f"audit_{drive}_judgments.csv"
        if not judgments_path.exists():
            continue
        verdict_of = {row["item_id"]: row["verdict"] for row in read_dicts(judgments_path)}
        population = {row["stratum"]: int(row["population"])
                      for row in read_dicts(review_dir / f"audit_{drive}_strata.csv")}
        judged = [row for row in read_dicts(queue_path) if verdict_of.get(row["item_id"], UNSURE) != UNSURE]
        n_judged = Counter(row["stratum"] for row in judged)          # weights over all splits
        for row in judged:
            if row["split"] == "train" or float(row["max_conf"]) < class_floor(cfg, row["predicted"]):
                continue                     # train site, or an event the survey no longer keeps
            offset = float(row["offset_s"])
            first = round((float(row["start_s"]) + offset) / WIN)
            last = max(first + 1, round((float(row["end_s"]) + offset) / WIN))
            window_ids = []
            for index in range(first, last):
                window_id = f"{row['item_id']}_w{index}"
                window_ids.append(window_id)
                wavs[window_id] = windows_dir / f"{window_id}.wav"
                if not args.dry_run:
                    if not Path(row["clip_path"]).exists():
                        missing.append(row["clip_path"])
                        break
                    windows_dir.mkdir(parents=True, exist_ok=True)
                    n_cut += cut_window(row["clip_path"], index * WIN - offset, wavs[window_id])
            events.append({"item_id": row["item_id"], "drive": drive, "site": row["site"],
                           "predicted": row["predicted"], "true": verdict_of[row["item_id"]] == row["predicted"],
                           "weight": population[row["stratum"]] / n_judged[row["stratum"]],
                           "windows": window_ids, "max_conf": float(row["max_conf"])})
    if missing:
        raise SystemExit(f"{len(missing)} audit snippets missing, e.g. {missing[0]}: audit_events.py cut")
    if not events:
        raise SystemExit(f"no judged heldout audit event in {review_dir} (audit_events.py, review.py --queue audit)")
    print(f"heldout audit events (Unsure left out): {len(events)} at {len({e['site'] for e in events})} sites, "
          f"{len(wavs)} windows; " + ", ".join(
              f"{label} {sum(e['true'] for e in events if e['predicted'] == label)} true / "
              f"{sum(not e['true'] for e in events if e['predicted'] == label)} false"
              for label in sorted({e['predicted'] for e in events})))
    if n_cut:
        print(f"cut {n_cut} windows into {windows_dir}")
    return {"events": events, "wavs": wavs, "cache": "audit_scores", "legacy": None, "suffix": "_audit"}


def run_audit(cfg, args, test, models, gens_dir, out_dir):
    """Event-level comparison of generations on the heldout audit events.

    An event is kept by a generation if its best window, inside the event, scores at
    least the class floor (drive_detection.min_conf / min_event_conf) for the predicted
    class. Per class and over all: share of the true events kept (weighted by stratum
    population / judged), share of the false events removed, and the precision of the
    events kept; 95 % CI from a bootstrap over heldout sites.

    Args:
        cfg (dict): Whole config.
        args (argparse.Namespace): gens, boot, threads.
        test (dict): From load_audit.
        models (dict[int, Path]): gen -> model.tflite.
        gens_dir (Path): generations.dir.
        out_dir (Path): Output folder.
    """
    events = test["events"]
    scores = {gen: score(path, test["wavs"], path.parent, test["cache"], test["legacy"], args.threads)
              for gen, path in models.items()}
    best = {gen: np.array([max((scores[gen][w].get(e["predicted"], 0.0) for w in e["windows"]), default=0.0)
                           for e in events]) for gen in args.gens}
    ref = args.gens[0]
    gap = np.abs(best[ref] - np.array([e["max_conf"] for e in events]))
    print(f"check, gen_{ref} rescored vs the full-survey max_conf: {int((gap > 0.01).sum())} of {len(events)} "
          f"events differ by > 0.01 (median gap {np.median(gap):.4f})")
    floors = np.array([class_floor(cfg, e["predicted"]) for e in events])
    true = np.array([e["true"] for e in events], float)
    weight = np.array([e["weight"] for e in events])
    labels = sorted({e["predicted"] for e in events})
    scopes = {label: np.array([e["predicted"] == label for e in events]) for label in labels}
    scopes["all"] = np.ones(len(events), bool)
    sites = np.array([e["site"] for e in events])
    site_groups = [np.where(sites == site)[0] for site in np.unique(sites)]

    def metrics(idx, gen):
        """Retention, removal, precision of kept events over a resample.

        Args:
            idx (numpy.ndarray): Event indices (with repeats).
            gen (int): Generation.

        Returns:
            dict[str, list[float]]: scope -> [true kept (weighted), false removed (share),
                precision of kept (weighted)].
        """
        kept = (best[gen] >= floors).astype(float)
        out = {}
        for name, mask in scopes.items():
            m = mask[idx]
            t, k, w = true[idx][m], kept[idx][m], weight[idx][m]
            true_w, kept_w = (w * t).sum(), (w * k).sum()
            n_false = (1 - t).sum()
            out[name] = [(w * t * k).sum() / true_w if true_w else np.nan,
                         ((1 - t) * (1 - k)).sum() / n_false if n_false else np.nan,
                         (w * t * k).sum() / kept_w if kept_w else np.nan]
        return out

    names = ["true_kept", "false_removed", "precision_kept"]
    everything = np.arange(len(events))
    point = {gen: metrics(everything, gen) for gen in args.gens}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([site_groups[k] for k in rng.choice(len(site_groups), len(site_groups))])
             for _ in range(args.boot)]
    boot = {gen: [metrics(idx, gen) for idx in boots] for gen in args.gens}
    rows = []
    for name in scopes:
        mask = scopes[name]
        for gen in args.gens:
            for j, metric in enumerate(names):
                row = {"gen": gen, "class": name, "metric": metric, "n_true": int(true[mask].sum()),
                       "n_false": int((1 - true[mask]).sum()), "value": point[gen][name][j]}
                if args.boot:
                    values = np.array([b[name][j] for b in boot[gen]])
                    row["lo"], row["hi"] = np.nanpercentile(values, [2.5, 97.5])
                    if gen != ref:
                        delta = values - np.array([b[name][j] for b in boot[ref]])
                        row[f"diff_vs_gen{ref}"] = point[gen][name][j] - point[ref][name][j]
                        row["diff_lo"], row["diff_hi"] = np.nanpercentile(delta, [2.5, 97.5])
                rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"compare{test['suffix']}.csv"
    header = ["gen", "class", "metric", "n_true", "n_false", "value", "lo", "hi",
              f"diff_vs_gen{ref}", "diff_lo", "diff_hi"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: (f"{value:.4f}" if isinstance(value, float) else value) for col, value in row.items()})
    print(f"\nheldout audit events, kept if best window >= {', '.join(f'{label} {class_floor(cfg, label):g}' for label in labels)}"
          f" [95% CI, sites resampled]")
    for name in scopes:
        for metric in names:
            cells = []
            for gen in args.gens:
                row = next(r for r in rows if r["gen"] == gen and r["class"] == name and r["metric"] == metric)
                cells.append(f"gen_{gen} {row['value']:.3f} [{row.get('lo', np.nan):.3f}, {row.get('hi', np.nan):.3f}]")
            print(f"  {name:9s} {metric:15s} " + "   ".join(cells))
    print(f"wrote {path}")


# ---------- scoring ----------
def parse_result(path):
    """Best score per label in the first 3 s segment of one BirdNET result file.

    Args:
        path (Path): <window>.BirdNET.results.csv.

    Returns:
        dict[str, float]: label -> score (labels below min_conf absent).
    """
    window_scores = defaultdict(float)
    for window in read_dicts(path):
        if float(window["Start (s)"]) < 0.5:              # the one 3 s segment
            label = window["Common name"] or window["Scientific name"]
            window_scores[label] = max(window_scores[label], float(window["Confidence"]))
    return dict(window_scores)


def score(model, wavs, gen_dir, cache_name, legacy_name, threads):
    """Scores of one model on the test windows, cached in gen_dir/<cache_name>.csv.

    Windows missing from the cache are taken from the old per-window result folder
    (gen_dir/<legacy_name>/) when present, else scored with BirdNET (a temporary folder
    of links to them, removed afterwards).

    Args:
        model (Path): The generation's model.tflite.
        wavs (dict[str, Path]): Window id -> wav file.
        gen_dir (Path): generations.dir/gen_K.
        cache_name (str): Cache CSV stem (columns window_id, label, score).
        legacy_name (str | None): Old per-window result folder name, if any.
        threads (int | None): BirdNET CPU threads (None: BirdNET's default).

    Returns:
        dict[str, dict[str, float]]: window id -> {label: score}; labels below 0.01 absent.

    Raises:
        SystemExit: If BirdNET produced no result for a window.
    """
    cache_path = gen_dir / f"{cache_name}.csv"
    cached = {}
    if cache_path.exists():
        for row in read_dicts(cache_path):
            window_scores = cached.setdefault(row["window_id"], {})
            if row["label"]:
                window_scores[row["label"]] = float(row["score"])
    n_cached = len(cached)
    legacy_dir = gen_dir / legacy_name if legacy_name else None
    if legacy_dir and legacy_dir.is_dir():
        for window_id in wavs:
            result_path = legacy_dir / f"{window_id}.BirdNET.results.csv"
            if window_id not in cached and result_path.exists():
                cached[window_id] = parse_result(result_path)
        params = legacy_dir / "BirdNET_analysis_params.csv"
        if params.exists() and not (gen_dir / f"{cache_name}_params.csv").exists():
            shutil.copy2(params, gen_dir / f"{cache_name}_params.csv")
    todo = [window_id for window_id in wavs if window_id not in cached]
    if todo:
        tmp_dir = gen_dir / f"{cache_name}_tmp"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        (tmp_dir / "in").mkdir(parents=True)
        for window_id in todo:
            os.symlink(wavs[window_id].resolve(), tmp_dir / "in" / f"{window_id}.wav")
        cmd = [sys.executable, "-m", "birdnet_analyzer.analyze", str(tmp_dir / "in"),
               "-o", str(tmp_dir / "out"), "-c", str(model), "--min_conf", "0.01",
               "--overlap", "0", "--rtype", "csv"]
        if threads:
            cmd += ["-t", str(threads)]
        print(f"scoring {len(todo)} windows with {model}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        for window_id in todo:
            result_path = tmp_dir / "out" / f"{window_id}.BirdNET.results.csv"
            if not result_path.exists():
                raise SystemExit(f"no BirdNET result for {window_id} in {tmp_dir / 'out'}")
            cached[window_id] = parse_result(result_path)
        params = tmp_dir / "out" / "BirdNET_analysis_params.csv"
        if params.exists():
            shutil.copy2(params, gen_dir / f"{cache_name}_params.csv")
        shutil.rmtree(tmp_dir)
    if len(cached) != n_cached:
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["window_id", "label", "score"])
            for window_id in sorted(cached):
                if not cached[window_id]:
                    writer.writerow([window_id, "", ""])      # scored, nothing above 0.01
                for label, value in sorted(cached[window_id].items()):
                    writer.writerow([window_id, label, repr(value)])
        print(f"{len(cached) - n_cached} windows added to {cache_path}")
    return {window_id: cached[window_id] for window_id in wavs}


# ---------- metrics ----------
def ap_pr(y, scores, weights, thresholds):
    """Weighted average precision, and precision / recall per threshold.

    Args:
        y (numpy.ndarray): 1.0 for positive windows, 0.0 otherwise.
        scores (numpy.ndarray): Model scores.
        weights (numpy.ndarray): Window weights (0 leaves a window out).
        thresholds (list[float]): Score thresholds.

    Returns:
        tuple[float, list[tuple[float, float]]]: (AP, [(precision, recall)] per
            threshold); NaN where undefined (no positives, nothing above a threshold).
    """
    pos = (weights * y).sum()
    if pos == 0:
        return np.nan, [(np.nan, np.nan)] * len(thresholds)
    order = np.argsort(-scores, kind="stable")
    tp, fp = np.cumsum((weights * y)[order]), np.cumsum((weights * (1 - y))[order])
    precision = tp / np.maximum(tp + fp, 1e-12)
    ap = float((precision * (weights * y)[order]).sum() / pos)
    pr_points = []
    for threshold in thresholds:
        above = scores >= threshold
        selected = (weights * above).sum()
        true_pos = (weights * y * above).sum()
        pr_points.append((true_pos / selected if selected else np.nan, true_pos / pos))
    return ap, pr_points


def pr_curve(y, scores, weights):
    """Weighted recall and precision along the ranking by score, for plotting.

    Args:
        y (numpy.ndarray): 1.0 for positive windows, 0.0 otherwise.
        scores (numpy.ndarray): Model scores.
        weights (numpy.ndarray): Window weights.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: (recall, precision), one value per window.
    """
    order = np.argsort(-scores, kind="stable")
    tp, fp = np.cumsum((weights * y)[order]), np.cumsum((weights * (1 - y))[order])
    return tp / max((weights * y).sum(), 1e-12), tp / np.maximum(tp + fp, 1e-12)


def main():
    """CLI entry point: load the test set, score it with each generation, compute metrics, plot."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--gens", type=int, nargs="+", required=True)
    parser.add_argument("--testset", choices=["survey", "xc", "audit"], default="survey")
    parser.add_argument("--review-dir", help="default: review.dir (survey test set)")
    parser.add_argument("--gen-dir", help="default: generations.dir")
    parser.add_argument("--out", default="results/survey")
    parser.add_argument("--boot", type=int, default=500, help="bootstrap resamples (0 = off)")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--dry-run", action="store_true",
                        help="describe the test set and the models, cut / score / write nothing")
    args = parser.parse_args()
    cfg = load(args.config)
    gen_cfg = cfg["generations"]
    gens_dir = Path(args.gen_dir or gen_cfg["dir"])
    out_dir = Path(args.out)
    thresholds = sorted(set(gen_cfg.get("eval_thresholds", [0.35, 0.5, 0.7])
                            + [cfg["detection"]["min_conf"]]))

    test = {"xc": load_xc, "audit": load_audit, "survey": load_survey}[args.testset](cfg, args)
    models = {gen: gens_dir / f"gen_{gen}" / "model.tflite" for gen in args.gens}
    for gen, model_path in models.items():
        if not model_path.exists():
            raise SystemExit(f"{model_path} not found (make_generation.py --gen {gen}; gen_0 is linked "
                             f"by any make_generation.py run)")
    if args.dry_run:
        print("models:", {gen: str(path) for gen, path in models.items()})
        print("dry run: nothing cut, scored or written")
        return
    if args.testset == "audit":
        run_audit(cfg, args, test, models, gens_dir, out_dir)
        return
    labels = [line.strip() for line in models[args.gens[0]].with_name("model_Labels.txt")
              .read_text(encoding="utf-8").splitlines() if line.strip()]
    scores = {gen: score(model_path, test["wavs"], model_path.parent, test["cache"], test["legacy"],
                         args.threads)
              for gen, model_path in models.items()}

    ids, positives, ambiguous = test["ids"], test["positives"], test["ambiguous"]
    # per target: truth, kept windows (0 = ambiguous for that class), scores per generation
    targets = {class_name: (np.array([class_name in positive for positive in positives], float),
                            np.array([class_name not in partial for partial in ambiguous], float),
                            {gen: np.array([scores[gen][window_id].get(class_name, 0.0)
                                            for window_id in ids])
                             for gen in args.gens})
               for class_name in labels}
    targets[ANY] = (np.array([bool(positive) for positive in positives], float),
                    np.array([bool(positive) or not partial
                              for positive, partial in zip(positives, ambiguous, strict=True)],
                             float),
                    {gen: np.array([max(scores[gen][window_id].values(), default=0.0)
                                    for window_id in ids]) for gen in args.gens})

    rng = np.random.default_rng(0)
    boots = [test["resample"](rng) for _ in range(args.boot)]

    hours = test["hours"]

    def metrics(y, scores, weights):
        """All metrics of one resample, flattened in the order of `names`.

        Args:
            y (numpy.ndarray): Positives.
            scores (numpy.ndarray): Scores.
            weights (numpy.ndarray): Weights.

        Returns:
            list[float]: [AP, precision@t1, recall@t1, precision@t2, ...], then (survey
                test set only) false alarms per hour of heldout audio at each threshold.
        """
        ap, pr_points = ap_pr(y, scores, weights, thresholds)
        values = [ap] + [value for point in pr_points for value in point]
        if hours:
            values += [(weights * (1 - y) * (scores >= threshold)).sum() / hours
                       for threshold in thresholds]
        return values

    names = ["AP"] + [f"{metric}@{threshold:g}"
                      for threshold in thresholds for metric in ("precision", "recall")]
    if hours:
        names += [f"false_alarms_per_h@{threshold:g}" for threshold in thresholds]
    rows = []
    for class_name, (y, kept, scores_by_gen) in targets.items():
        weights = test["weights"] * kept
        point = {gen: metrics(y, scores_by_gen[gen], weights) for gen in args.gens}
        boot_metrics = {gen: np.array([metrics(y[sample], scores_by_gen[gen][sample], weights[sample])
                                       for sample in boots])
                        for gen in args.gens}
        ref_gen = args.gens[0]
        for gen in args.gens:
            for j, name in enumerate(names):
                row = {"gen": gen, "class": class_name, "metric": name,
                       "n_pos": int(y.sum()), "value": point[gen][j]}
                if args.boot:
                    row["lo"], row["hi"] = np.nanpercentile(boot_metrics[gen][:, j], [2.5, 97.5])
                    if gen != ref_gen:
                        delta = boot_metrics[gen][:, j] - boot_metrics[ref_gen][:, j]
                        row[f"diff_vs_gen{ref_gen}"] = point[gen][j] - point[ref_gen][j]
                        row["diff_lo"], row["diff_hi"] = np.nanpercentile(delta, [2.5, 97.5])
                rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_path = out_dir / f"compare{test['suffix']}.csv"
    header = ["gen", "class", "metric", "n_pos", "value", "lo", "hi",
              f"diff_vs_gen{args.gens[0]}", "diff_lo", "diff_hi"]
    with open(compare_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: (f"{value:.4f}" if isinstance(value, float) else value)
                             for col, value in row.items()})

    print(f"\n{'class':9s} {'n_pos':>5s}  "
          + "  ".join(f"gen_{gen} AP [95% CI]".rjust(24) for gen in args.gens))
    for class_name in targets:
        cells = []
        for gen in args.gens:
            row = next(row for row in rows
                       if row["gen"] == gen and row["class"] == class_name and row["metric"] == "AP")
            cell = f"{row['value']:.3f} [{row.get('lo', np.nan):.3f}, {row.get('hi', np.nan):.3f}]"
            cells.append(cell.rjust(24))
        print(f"{class_name:9s} {int(targets[class_name][0].sum()):5d}  " + "  ".join(cells))
    for row in rows:
        # skip classes without positives (their difference interval is NaN)
        if row["metric"] == "AP" and "diff_lo" in row and not np.isnan(row["diff_lo"]):
            sig = "" if row["diff_lo"] <= 0 <= row["diff_hi"] else "  (interval excludes 0)"
            diff = row[f"diff_vs_gen{args.gens[0]}"]
            print(f"  {row['class']}: gen_{row['gen']} - gen_{args.gens[0]} AP = {diff:+.3f} "
                  f"[{row['diff_lo']:+.3f}, {row['diff_hi']:+.3f}]{sig}")

    if hours:
        print(f"\nfalse alarms per hour of heldout audio ({hours:.1f} h), windows not of the class "
              f"scoring >= t [95% CI]")
        for threshold in thresholds:
            name = f"false_alarms_per_h@{threshold:g}"
            for class_name in targets:
                cells = []
                for gen in args.gens:
                    row = next(row for row in rows if row["gen"] == gen
                               and row["class"] == class_name and row["metric"] == name)
                    cell = f"{row['value']:.2f} [{row.get('lo', np.nan):.2f}, {row.get('hi', np.nan):.2f}]"
                    cells.append(f"gen_{gen} {cell}")
                print(f"  t={threshold:g} {class_name:9s} " + "   ".join(cells))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for class_name, (y, kept, scores_by_gen) in targets.items():
        if not y.sum():
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for color_index, gen in enumerate(args.gens):
            recall, precision = pr_curve(y, scores_by_gen[gen], test["weights"] * kept)
            ap_value = next(row["value"] for row in rows
                            if row["gen"] == gen and row["class"] == class_name
                            and row["metric"] == "AP")
            ax.plot(recall, precision, color=COLORS[color_index % len(COLORS)], lw=2,
                    drawstyle="steps-post", label=f"gen_{gen}  AP {ap_value:.2f}")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(color="#e4e3df", lw=0.8)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        weighted = "" if args.testset == "xc" else " (weighted)"
        ax.set_xlabel(f"recall{weighted}")
        ax.set_ylabel(f"precision{weighted}")
        ax.set_title(f"{class_name}  ({int(y.sum())} positive {args.testset} test windows)",
                     loc="left", fontsize=11)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / f"pr{test['suffix']}_{class_name}.png", dpi=120)
        plt.close(fig)
    print(f"\nwrote {compare_path} and pr{test['suffix']}_<class>.png")


if __name__ == "__main__":
    main()
