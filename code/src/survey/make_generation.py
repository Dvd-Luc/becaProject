"""Build and train the next model generation: XC clips + reviewed survey clips.

    python src/survey/make_generation.py --gen 1 --dry-run   # counts only, write nothing
    python src/survey/make_generation.py --gen 1             # cut clips, link, train
    python src/survey/make_generation.py --gen 1 --no-train  # build the training folder only
    python src/survey/make_generation.py --gen 2 --dry-run   # + verdicts on gen 1's detections
    python src/survey/make_generation.py --gen 3 --dry-run   # + full-survey audit verdicts (train sites)

gen_0 is the existing model (detection.model): it is only linked, never retrained.
gen_K (K >= 1) uses ALL current verdicts (last verdict per item) of the train queues
judged before it: <review.dir>/train_judgments.csv (v0's detections) and
train_gen<J>_judgments.csv for J < K (gen J's detections, review.py --queue train
--gen J). So gen 1 = train queue only; gen 2 = train + train_gen1; ...
Gens after generations.audit_gen (2) also use the full-survey audit and presence
verdicts (<review.dir>/{audit,presence}_<drive>_judgments.csv, every drive), whose
events are gen_<audit_gen>'s: their windows are chosen from generations.audit_raw_dir
on the recording's 3 s grid and cut from the audit snippets. These queues cover train
and heldout sites by design: their heldout rows are left out (and counted), never
trained on. Every other queue must hold train-split sites only (else an error). The XC
clips (--base-clips) must not come from a recordist of the XC test sample
(<xc_test.dir>/sample.csv), or the run stops.
Survey verdicts are used as follows:
  * event judged as a class or Noise -> its 3 s grid windows, at most
    generations.max_windows_per_item of them, the ones the detecting model (v0, or
    gen J for train_gen<J>) scored highest for the predicted class (skips the gap
    windows bridged by merge_gap_s)
  * background judged Noise (or a class, if a call was hiding there) -> its window
  * Unsure -> nothing

Writes, under generations.dir/gen_K/:
    survey_clips/<class>/<clip_id>_<start>s.wav, survey_clips/metadata.csv
    train/<class>/   hard links: outputs/clips/<class> (XC) + survey_clips/<class>
                     (removed after training unless --keep-train; kept with --no-train)
    model.tflite (+ BirdNET's model_Labels.txt, model_Params.csv), train.log, info.csv

Training uses the hyperparameters in <detection.model>_Params.csv (as v0) and
generations.train_seed as BirdNET's RANDOM_SEED. An existing model.tflite is kept
(use --force to retrain; survey_clips/ and train/ are always rebuilt).

Run from code/.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import load

WIN = 3.0
NOISE, UNSURE = "Noise", "Unsure"
EVENT_KINDS = ("event", "audit", "presence")    # queue kinds that are detected events
META_HEADER = ["file", "class", "item_id", "kind", "predicted", "clip_id", "site",
               "start_s"]
# BirdNET _Params.csv column -> train CLI option (flags: True -> option present)
PARAM_ARGS = {"Hidden units": "--hidden_units", "Dropout": "--dropout",
              "Batchsize": "--batch_size", "Learning rate": "--learning_rate",
              "Crop mode": "--crop_mode", "Crop overlap": "--overlap",
              "Audio speed": "--audio_speed", "Upsamling mode": "--upsampling_mode",
              "Upsamling ratio": "--upsampling_ratio", "focal loss alpha": "--focal-loss-alpha",
              "focal loss gamma": "--focal-loss-gamma"}
PARAM_FLAGS = {"use mixup": "--mixup", "use label smoothing": "--label_smoothing",
               "use focal loss": "--focal-loss"}
# BirdNET has no seed option: set config.RANDOM_SEED before its modules are imported
TRAIN_WRAPPER = ("import sys, birdnet_analyzer.config as c; c.RANDOM_SEED = int(sys.argv[1]); "
                 "sys.argv = ['birdnet_analyzer.train'] + sys.argv[2:]; "
                 "from birdnet_analyzer.train.cli import main; main()")


def read_dicts(path):
    """Rows of a CSV file as dicts.

    Args:
        path (str | Path): CSV file with a header row.

    Returns:
        list[dict[str, str]]: One dict per row, keyed by column name.
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_dicts(path, header, rows):
    """Write dicts to a CSV file with the given header, creating parent folders.

    Args:
        path (Path): Output CSV file.
        header (list[str]): Column names, in order.
        rows (list[dict]): Rows keyed by column name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def model_files(model):
    """A BirdNET model file and its side files.

    Args:
        model (str | Path): The .tflite file.

    Returns:
        list[Path]: [model, <stem>_Labels.txt, <stem>_Params.csv,
            <name>_sample_counts.csv], whether they exist or not.
    """
    model_path = Path(model)
    return [model_path,
            model_path.with_name(model_path.stem + "_Labels.txt"),
            model_path.with_name(model_path.stem + "_Params.csv"),
            model_path.with_name(model_path.name + "_sample_counts.csv")]


def link_gen0(model, gens_dir):
    """Make gen_0/ point to the existing v0 files with relative symlinks.

    Args:
        model (str | Path): detection.model (the v0 .tflite).
        gens_dir (Path): generations.dir.

    Returns:
        Path: gdir/gen_0/model.tflite.
    """
    gen0_dir = gens_dir / "gen_0"
    gen0_dir.mkdir(parents=True, exist_ok=True)
    for src, name in zip(model_files(model), ["model.tflite", "model_Labels.txt",
                                              "model_Params.csv", "model.tflite_sample_counts.csv"]):
        dst = gen0_dir / name
        if src.exists() and not dst.exists():
            dst.symlink_to(os.path.relpath(src.resolve(), gen0_dir.resolve()))
    return gen0_dir / "model.tflite"


def train_args(model):
    """birdnet_analyzer.train options that reproduce a model's training settings.

    Args:
        model (str | Path): The .tflite whose _Params.csv is read.

    Returns:
        list[str]: CLI options (empty, with a note, if _Params.csv is missing).
    """
    params_path = model_files(model)[2]
    if not params_path.exists():
        print(f"note: {params_path} not found, BirdNET defaults used")
        return []
    row = read_dicts(params_path)[0]
    options = []
    for column, option in PARAM_ARGS.items():
        if row.get(column, "") != "":
            options += [option, row[column]]
    for column, option in PARAM_FLAGS.items():
        if row.get(column, "").strip().lower() == "true":
            options.append(option)
    return options


def raw_scores(raw_dir, clip_id, label):
    """Scores of one label in detect.py's raw results for one clip.

    Args:
        raw_dir (Path): detection.dir/raw.
        clip_id (str): Manifest clip id.
        label (str): Class name.

    Returns:
        dict[int, float]: 3 s window index -> confidence (windows below raw_min_conf absent).
    """
    result_path = raw_dir / f"{clip_id}.BirdNET.results.csv"
    scores = {}
    if result_path.exists():
        for window in read_dicts(result_path):
            if (window["Common name"] or window["Scientific name"]) == label:
                scores[round(float(window["Start (s)"]) / WIN)] = float(window["Confidence"])
    return scores


def windows_of(item, raw_dir, max_n):
    """Start times of the 3 s grid windows to cut for one judged item.

    Events longer than max_n windows keep the max_n windows the detecting model scored
    highest for the predicted class, so the gap windows bridged by merge_gap_s are
    dropped. Audit and presence items are snippets of a full-survey file: their times
    are snippet-relative, so the grid is the file's (snippet time + offset_s).

    Args:
        item (dict): Queue row (kind, clip_id, start_s, end_s, predicted; offset_s for
            audit / presence items).
        raw_dir (Path): Raw BirdNET results of the detecting run.
        max_n (int): Maximum windows kept from one event.

    Returns:
        list[float]: Window start times in seconds of the clip / file, sorted.
    """
    offset = float(item.get("offset_s") or 0.0)
    start, end = float(item["start_s"]) + offset, float(item["end_s"]) + offset
    window_indices = list(range(round(start / WIN), max(round(start / WIN) + 1, round(end / WIN))))
    if item["kind"] in EVENT_KINDS and len(window_indices) > max_n:
        scores = raw_scores(raw_dir, item["clip_id"], item["predicted"])
        window_indices = sorted(sorted(window_indices, key=lambda index: -scores.get(index, 0.0))[:max_n])
    return [index * WIN for index in window_indices]


def audit_sources(review_dir, raw_dir):
    """Judged audit and presence queues of the full survey (review.dir/{audit,presence}_<drive>).

    Args:
        review_dir (Path): review.dir.
        raw_dir (Path): Raw results of the full survey the queues were drawn from.

    Returns:
        list[tuple[str, Path]]: (queue name, raw_dir) for each queue with judgments.
    """
    names = sorted(path.name[:-len("_judgments.csv")] for kind in ("audit", "presence")
                   for path in review_dir.glob(f"{kind}_D*_judgments.csv"))
    return [(name, raw_dir) for name in names]


def link(src, dst):
    """Hard-link src to dst, or copy it when linking is impossible (other filesystem).

    Args:
        src (Path): Existing file.
        dst (Path): New path.
    """
    try:
        os.link(src, dst)
    except OSError:                                    # other filesystem
        shutil.copy2(src, dst)


def check_xc_test_leak(base_clips, cfg):
    """Refuse XC training clips from a recordist of the XC test sample.

    Args:
        base_clips (str | Path): XC training clips folder (with metadata.csv).
        cfg (dict): Whole config (xc_test, annotator sections).

    Returns:
        int: Number of test recordists checked (0 if no sample has been drawn).

    Raises:
        SystemExit: If a training clip comes from a test recording or test recordist.
    """
    sample_path = Path(cfg.get("xc_test", {}).get("dir", "data/xc_test")) / "sample.csv"
    clips_meta_path = Path(base_clips) / "metadata.csv"
    if not sample_path.exists():
        return 0
    if not clips_meta_path.exists():
        raise SystemExit(f"{clips_meta_path} not found: cannot check the XC test recordists")
    annot_cfg = cfg["annotator"]
    file_col, group_col = annot_cfg.get("file_col", "file-name"), annot_cfg["group_cols"][0]
    recordist_of = {row[file_col].strip(): row[group_col].strip()
                    for row in read_dicts(annot_cfg["metadata_csv"])}
    sample = read_dicts(sample_path)
    test_recordings = {row[file_col].strip() for row in sample}
    test_recordists = {row[group_col].strip() for row in sample}
    clip_recordings = {row["recording"] for row in read_dicts(clips_meta_path)}
    leak = sorted(recording for recording in clip_recordings
                  if recording in test_recordings or recordist_of.get(recording) in test_recordists)
    if leak:
        raise SystemExit(f"{len(leak)} training recordings belong to the XC test sample "
                         f"({sample_path}), refusing: {leak[:3]}")
    return len(test_recordists)


def main():
    """CLI entry point: check the verdicts, cut survey clips, build train/ and train gen_K."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--gen", type=int, required=True, help="generation number, >= 1")
    parser.add_argument("--review-dir", help="default: review.dir")
    parser.add_argument("--gen-dir", help="default: generations.dir")
    parser.add_argument("--base-clips", default="outputs/clips", help="XC training clips")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--force", action="store_true", help="retrain if model.tflite exists")
    parser.add_argument("--keep-train", action="store_true",
                        help="keep train/ after training (it is removed by default; --no-train keeps it)")
    parser.add_argument("--train-extra", nargs=argparse.REMAINDER, default=[],
                        help="extra birdnet_analyzer.train options, e.g. --epochs 5 (last option)")
    args = parser.parse_args()
    if args.gen < 1:
        raise SystemExit("--gen must be >= 1 (gen_0 is the existing model)")
    cfg = load(args.config)
    gen_cfg, detection_cfg = cfg["generations"], cfg["detection"]
    gens_dir = Path(args.gen_dir or gen_cfg["dir"])
    review_dir = Path(args.review_dir or cfg["review"]["dir"])
    max_n = gen_cfg.get("max_windows_per_item", 3)

    # train queues judged before gen K: v0's detections, then gen 1 .. gen K-1's own
    sources = [("train", Path(detection_cfg["dir"]) / "raw")]
    sources += [(f"train_gen{k}", gens_dir / f"gen_{k}" / "detections" / "raw")
                for k in range(1, args.gen)]
    # full-survey audit / presence verdicts (on gen_<audit_gen>'s events): train sites only
    audit_gen = gen_cfg.get("audit_gen")
    if audit_gen is not None and args.gen > audit_gen:
        sources += audit_sources(review_dir, Path(gen_cfg["audit_raw_dir"]))
    queue, raw_dir_of, verdicts, used_queues, heldout_skipped = {}, {}, {}, [], Counter()
    for queue_name, raw_dir in sources:
        judgments_path = review_dir / f"{queue_name}_judgments.csv"
        if not judgments_path.exists():
            if queue_name == "train":
                raise SystemExit(f"no judgments yet: {judgments_path}")
            print(f"warning: no {judgments_path.name} yet: gen {args.gen} gets nothing from it")
            continue
        queue_rows = {row["item_id"]: row for row in read_dicts(review_dir / f"{queue_name}_queue.csv")}
        queue_verdicts = {row["item_id"]: row["verdict"] for row in read_dicts(judgments_path)}
        missing = [item_id for item_id in queue_verdicts if item_id not in queue_rows]
        if missing:
            raise SystemExit(f"{len(missing)} judged items are not in {queue_name}_queue.csv "
                             f"(queue rebuilt?), e.g. {missing[:2]}")
        if queue_name.startswith(("audit_", "presence_")):   # these mix splits by design
            heldout = [item_id for item_id in queue_verdicts if queue_rows[item_id]["split"] != "train"]
            heldout_skipped[queue_name] = len(heldout)
            queue_verdicts = {item_id: verdict for item_id, verdict in queue_verdicts.items() if item_id not in heldout}
            absent = [item_id for item_id in queue_verdicts if not Path(queue_rows[item_id]["clip_path"]).exists()]
            if absent:
                raise SystemExit(f"{len(absent)} {queue_name} snippets are missing, e.g. "
                                 f"{queue_rows[absent[0]]['clip_path']}: run audit_events.py cut first")
        queue.update(queue_rows)
        verdicts.update(queue_verdicts)
        raw_dir_of.update({item_id: raw_dir for item_id in queue_rows})
        used_queues.append(f"{queue_name} ({len(queue_verdicts)})")
    print("train queues used:", ", ".join(used_queues))
    if heldout_skipped:
        print(f"heldout-site verdicts left out (never trained on): {sum(heldout_skipped.values())} "
              f"({', '.join(f'{name} {n}' for name, n in sorted(heldout_skipped.items()) if n)})")
    leak = sorted({queue[item_id]["site"] for item_id in verdicts if queue[item_id]["split"] != "train"})
    if leak:
        raise SystemExit(f"judgments from non-train sites, refusing: {leak[:5]}")

    plan, skipped = [], Counter()
    for item_id, verdict in verdicts.items():
        item = queue[item_id]
        if verdict == UNSURE:
            skipped["unsure"] += 1
            continue
        offset = float(item.get("offset_s") or 0.0)            # snippet start in the file
        for start in windows_of(item, raw_dir_of[item_id], max_n):
            name = f"{item['clip_id']}_{start:.0f}s.wav"
            plan.append({"file": f"{verdict}/{name}", "class": verdict, "item_id": item_id,
                         "kind": item["kind"], "predicted": item["predicted"],
                         "clip_id": item["clip_id"], "site": item["site"],
                         "start_s": f"{start:.1f}", "_path": item["clip_path"],
                         "_read_s": max(0.0, start - offset)})
    dup_counts = Counter(entry["file"] for entry in plan)               # same window judged twice
    plan = list({entry["file"]: entry for entry in plan}.values())

    n_test_recordists = check_xc_test_leak(args.base_clips, cfg)
    if n_test_recordists:
        print(f"XC test sample: none of its {n_test_recordists} recordists in {args.base_clips}")
    xc_clips = {class_dir.name: sorted(class_dir.glob("*.wav"))
                for class_dir in sorted(Path(args.base_clips).iterdir())
                if class_dir.is_dir() and any(class_dir.glob("*.wav"))}
    survey_counts = Counter(entry["class"] for entry in plan)
    print(f"{len(verdicts)} judged items ({skipped['unsure']} unsure), "
          f"{len(plan)} survey windows, {sum(n > 1 for n in dup_counts.values())} duplicates merged")
    print(f"{'class':10s} {'XC':>6s} {'survey':>7s}")
    for class_name in sorted(set(xc_clips) | set(survey_counts)):
        print(f"{class_name:10s} {len(xc_clips.get(class_name, [])):6d} {survey_counts[class_name]:7d}")
    print("survey windows by kind x class:",
          dict(Counter(f"{entry['kind']}->{entry['class']}" for entry in plan)))
    if args.dry_run:
        return

    gen0_model = link_gen0(detection_cfg["model"], gens_dir)
    gen_dir = gens_dir / f"gen_{args.gen}"
    clips_dir, train_dir = gen_dir / "survey_clips", gen_dir / "train"
    for folder in (clips_dir, train_dir):
        if folder.exists():
            shutil.rmtree(folder)
    for entry in plan:
        dst = clips_dir / entry["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        info = sf.info(entry["_path"])
        audio, sample_rate = sf.read(entry["_path"], start=int(entry["_read_s"] * info.samplerate),
                                     frames=int(WIN * info.samplerate), dtype="float32")
        sf.write(dst, audio, sample_rate, subtype=info.subtype)
    write_dicts(clips_dir / "metadata.csv", META_HEADER,
                [{col: entry[col] for col in META_HEADER} for entry in plan])
    for class_name, files in xc_clips.items():
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        for xc_file in files:
            link(xc_file, train_dir / class_name / f"xc_{xc_file.name}")
    for entry in plan:
        (train_dir / entry["class"]).mkdir(parents=True, exist_ok=True)
        link(clips_dir / entry["file"], train_dir / entry["class"] / f"survey_{Path(entry['file']).name}")
    counts = {class_dir.name: len(list(class_dir.glob("*.wav")))
              for class_dir in sorted(train_dir.iterdir())}
    print(f"built {train_dir}: {counts}")
    write_dicts(gen_dir / "info.csv", ["key", "value"], [
        {"key": "built", "value": datetime.now().isoformat(timespec="seconds")},
        {"key": "judgments", "value": "; ".join(used_queues)},
        {"key": "n_judged", "value": len(verdicts)},
        {"key": "base_clips", "value": args.base_clips},
        {"key": "params_from", "value": detection_cfg["model"]},
        {"key": "train_seed", "value": gen_cfg["train_seed"]},
        {"key": "max_windows_per_item", "value": max_n},
        {"key": "train_counts", "value": counts}])
    print(f"gen_0 -> {gen0_model.resolve()}")
    if args.no_train:
        return

    model = gen_dir / "model.tflite"
    if model.exists() and not args.force:
        print(f"{model} exists, not retrained (--force to retrain)")
    else:
        cmd = [sys.executable, "-c", TRAIN_WRAPPER, str(gen_cfg["train_seed"]), str(train_dir),
               "-o", str(model)] + train_args(detection_cfg["model"]) + args.train_extra
        print(" ".join(cmd[3:]))
        print(f"training... log: {gen_dir / 'train.log'}")
        with open(gen_dir / "train.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"wrote {model}")
    if not args.keep_train:          # only hard links, rebuilt by any run: no need to keep
        shutil.rmtree(train_dir)
        print(f"removed {train_dir} (--keep-train to keep it)")


if __name__ == "__main__":
    main()
