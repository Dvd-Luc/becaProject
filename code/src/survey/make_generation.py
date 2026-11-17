"""Build and train the next model generation: XC clips + reviewed survey clips.

    python src/survey/make_generation.py --gen 1 --dry-run   # counts only, write nothing
    python src/survey/make_generation.py --gen 1             # cut clips, link, train
    python src/survey/make_generation.py --gen 1 --no-train  # build the training folder only

gen_0 is the existing model (detection.model): it is only linked, never retrained.
gen_K (K >= 1) uses ALL current verdicts of <review.dir>/train_judgments.csv (last
verdict per item), from train-split sites only (anything else is an error):
  * event judged as a class or Noise -> its 3 s grid windows, at most
    generations.max_windows_per_item of them, the ones v0 scored highest for the
    predicted class (skips the gap windows bridged by merge_gap_s)
  * background judged Noise (or a class, if a call was hiding there) -> its window
  * Unsure -> nothing

Writes, under generations.dir/gen_K/:
    survey_clips/<class>/<clip_id>_<start>s.wav, survey_clips/metadata.csv
    train/<class>/   hard links: outputs/clips/<class> (XC) + survey_clips/<class>
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
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import load

WIN = 3.0
NOISE, UNSURE = "Noise", "Unsure"
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
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def model_files(model):
    """A BirdNET model file and its side files.

    Args:
        model (str | Path): The .tflite file.

    Returns:
        list[Path]: [model, <stem>_Labels.txt, <stem>_Params.csv,
            <name>_sample_counts.csv], whether they exist or not.
    """
    m = Path(model)
    return [m, m.with_name(m.stem + "_Labels.txt"), m.with_name(m.stem + "_Params.csv"),
            m.with_name(m.name + "_sample_counts.csv")]


def link_gen0(model, gdir):
    """Make gen_0/ point to the existing v0 files with relative symlinks.

    Args:
        model (str | Path): detection.model (the v0 .tflite).
        gdir (Path): generations.dir.

    Returns:
        Path: gdir/gen_0/model.tflite.
    """
    g0 = gdir / "gen_0"
    g0.mkdir(parents=True, exist_ok=True)
    for src, name in zip(model_files(model), ["model.tflite", "model_Labels.txt",
                                              "model_Params.csv", "model.tflite_sample_counts.csv"]):
        dst = g0 / name
        if src.exists() and not dst.exists():
            dst.symlink_to(os.path.relpath(src.resolve(), g0.resolve()))
    return g0 / "model.tflite"


def train_args(model):
    """birdnet_analyzer.train options that reproduce a model's training settings.

    Args:
        model (str | Path): The .tflite whose _Params.csv is read.

    Returns:
        list[str]: CLI options (empty, with a note, if _Params.csv is missing).
    """
    p = model_files(model)[2]
    if not p.exists():
        print(f"note: {p} not found, BirdNET defaults used")
        return []
    row = read_dicts(p)[0]
    out = []
    for k, opt in PARAM_ARGS.items():
        if row.get(k, "") != "":
            out += [opt, row[k]]
    for k, opt in PARAM_FLAGS.items():
        if row.get(k, "").strip().lower() == "true":
            out.append(opt)
    return out


def raw_scores(raw_dir, clip_id, label):
    """Scores of one label in detect.py's raw results for one clip.

    Args:
        raw_dir (Path): detection.dir/raw.
        clip_id (str): Manifest clip id.
        label (str): Class name.

    Returns:
        dict[int, float]: 3 s window index -> confidence (windows below raw_min_conf absent).
    """
    p = raw_dir / f"{clip_id}.BirdNET.results.csv"
    out = {}
    if p.exists():
        for w in read_dicts(p):
            if (w["Common name"] or w["Scientific name"]) == label:
                out[round(float(w["Start (s)"]) / WIN)] = float(w["Confidence"])
    return out


def windows_of(item, raw_dir, max_n):
    """Start times of the 3 s grid windows to cut for one judged item.

    Events longer than max_n windows keep the max_n windows v0 scored highest for the
    predicted class, so the gap windows bridged by merge_gap_s are dropped.

    Args:
        item (dict): Train queue row (kind, clip_id, start_s, end_s, predicted).
        raw_dir (Path): detection.dir/raw.
        max_n (int): Maximum windows kept from one event.

    Returns:
        list[float]: Window start times in seconds, sorted.
    """
    s, e = float(item["start_s"]), float(item["end_s"])
    ks = list(range(round(s / WIN), max(round(s / WIN) + 1, round(e / WIN))))
    if item["kind"] == "event" and len(ks) > max_n:
        sc = raw_scores(raw_dir, item["clip_id"], item["predicted"])
        ks = sorted(sorted(ks, key=lambda k: -sc.get(k, 0.0))[:max_n])
    return [k * WIN for k in ks]


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


def main():
    """CLI entry point: check the verdicts, cut survey clips, build train/ and train gen_K."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="config/project.yaml")
    ap.add_argument("--gen", type=int, required=True, help="generation number, >= 1")
    ap.add_argument("--review-dir", help="default: review.dir")
    ap.add_argument("--gen-dir", help="default: generations.dir")
    ap.add_argument("--base-clips", default="outputs/clips", help="XC training clips")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain if model.tflite exists")
    ap.add_argument("--train-extra", nargs=argparse.REMAINDER, default=[],
                    help="extra birdnet_analyzer.train options, e.g. --epochs 5 (last option)")
    a = ap.parse_args()
    if a.gen < 1:
        raise SystemExit("--gen must be >= 1 (gen_0 is the existing model)")
    cfg = load(a.config)
    g, d = cfg["generations"], cfg["detection"]
    gdir = Path(a.gen_dir or g["dir"])
    rdir = Path(a.review_dir or cfg["review"]["dir"])
    raw_dir = Path(d["dir"]) / "raw"
    max_n = g.get("max_windows_per_item", 3)

    queue = {r["item_id"]: r for r in read_dicts(rdir / "train_queue.csv")}
    jpath = rdir / "train_judgments.csv"
    if not jpath.exists():
        raise SystemExit(f"no judgments yet: {jpath}")
    verdicts = {r["item_id"]: r["verdict"] for r in read_dicts(jpath)}
    missing = [i for i in verdicts if i not in queue]
    if missing:
        raise SystemExit(f"{len(missing)} judged items are not in {rdir / 'train_queue.csv'} "
                         f"(queue rebuilt?), e.g. {missing[:2]}")
    leak = sorted({queue[i]["site"] for i in verdicts if queue[i]["split"] != "train"})
    if leak:
        raise SystemExit(f"judgments from non-train sites, refusing: {leak[:5]}")

    plan, skipped = [], Counter()
    for item_id, verdict in verdicts.items():
        it = queue[item_id]
        if verdict == UNSURE:
            skipped["unsure"] += 1
            continue
        for start in windows_of(it, raw_dir, max_n):
            name = f"{it['clip_id']}_{start:.0f}s.wav"
            plan.append({"file": f"{verdict}/{name}", "class": verdict, "item_id": item_id,
                         "kind": it["kind"], "predicted": it["predicted"],
                         "clip_id": it["clip_id"], "site": it["site"],
                         "start_s": f"{start:.1f}", "_path": it["clip_path"]})
    dup = Counter(p["file"] for p in plan)               # same window judged twice
    plan = list({p["file"]: p for p in plan}.values())

    base = {c.name: sorted(c.glob("*.wav")) for c in sorted(Path(a.base_clips).iterdir())
            if c.is_dir() and any(c.glob("*.wav"))}
    surv = Counter(p["class"] for p in plan)
    print(f"{len(verdicts)} judged items ({skipped['unsure']} unsure), "
          f"{len(plan)} survey windows, {sum(n > 1 for n in dup.values())} duplicates merged")
    print(f"{'class':10s} {'XC':>6s} {'survey':>7s}")
    for c in sorted(set(base) | set(surv)):
        print(f"{c:10s} {len(base.get(c, [])):6d} {surv[c]:7d}")
    print("survey windows by kind x class:",
          dict(Counter(f"{p['kind']}->{p['class']}" for p in plan)))
    if a.dry_run:
        return

    g0 = link_gen0(d["model"], gdir)
    out = gdir / f"gen_{a.gen}"
    sc_dir, tr_dir = out / "survey_clips", out / "train"
    for p in (sc_dir, tr_dir):
        if p.exists():
            shutil.rmtree(p)
    for p in plan:
        dst = sc_dir / p["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        info = sf.info(p["_path"])
        x, sr = sf.read(p["_path"], start=int(float(p["start_s"]) * info.samplerate),
                        frames=int(WIN * info.samplerate), dtype="float32")
        sf.write(dst, x, sr, subtype=info.subtype)
    write_dicts(sc_dir / "metadata.csv", META_HEADER,
                [{k: p[k] for k in META_HEADER} for p in plan])
    for c, files in base.items():
        (tr_dir / c).mkdir(parents=True, exist_ok=True)
        for f in files:
            link(f, tr_dir / c / f"xc_{f.name}")
    for p in plan:
        (tr_dir / p["class"]).mkdir(parents=True, exist_ok=True)
        link(sc_dir / p["file"], tr_dir / p["class"] / f"survey_{Path(p['file']).name}")
    counts = {c.name: len(list(c.glob("*.wav"))) for c in sorted(tr_dir.iterdir())}
    print(f"built {tr_dir}: {counts}")
    write_dicts(out / "info.csv", ["key", "value"], [
        {"key": "built", "value": datetime.now().isoformat(timespec="seconds")},
        {"key": "judgments", "value": str(jpath)},
        {"key": "n_judged", "value": len(verdicts)},
        {"key": "base_clips", "value": a.base_clips},
        {"key": "params_from", "value": d["model"]},
        {"key": "train_seed", "value": g["train_seed"]},
        {"key": "max_windows_per_item", "value": max_n},
        {"key": "train_counts", "value": counts}])
    print(f"gen_0 -> {g0.resolve()}")
    if a.no_train:
        return

    model = out / "model.tflite"
    if model.exists() and not a.force:
        print(f"{model} exists, not retrained (--force to retrain)")
        return
    cmd = [sys.executable, "-c", TRAIN_WRAPPER, str(g["train_seed"]), str(tr_dir),
           "-o", str(model)] + train_args(d["model"]) + a.train_extra
    print(" ".join(cmd[3:]))
    print(f"training... log: {out / 'train.log'}")
    with open(out / "train.log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"wrote {model}")


if __name__ == "__main__":
    main()
