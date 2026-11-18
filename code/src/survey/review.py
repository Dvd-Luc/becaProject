"""Review survey detections: say what each item really is (a class, Noise or unsure).

    python src/survey/review.py --queue train --list     # build/show the queue, no GUI
    python src/survey/review.py --queue train            # judge the items still to do
    python src/survey/review.py --queue train --n 20     # only the next 20 this session
    python src/survey/review.py --queue test             # blind heldout test set

Two queues, each frozen on first use to <review.dir>/<queue>_queue.csv (delete the file to
re-sample it; existing judgments keep their item_id but may no longer be in the queue):

  train  train-split sites only; feeds the next model (make_generation.py)
         * events from detection.dir/events.csv: review.events_per_class per predicted
           class, spread evenly over review.conf_bins (max_conf)
         * background: review.background_per_clip random 3 s windows per analysed clip,
           at least review.context_s away from any event (expected Noise)
  test   heldout-split sites only; scores every model generation (evaluate.py)
         * 3 s windows on BirdNET's grid (0, 3, 6 s ...), stratified by the best v0
           score in detection.dir/raw over review.test_bins, review.test_per_bin per bin.
           <review.dir>/test_strata.csv keeps each bin's population for re-weighting.
         * BLIND: the prediction is never shown.

Each judgment is appended to <review.dir>/<queue>_judgments.csv at once; the last one per
item wins, so `b` + another key corrects it. Items with a judgment are skipped next time.

Keys
    1 2 3 ...   the item is this class (order of annotator.categories, as in annotator.py)
    0 / n       Noise
    y / Enter   the prediction is right (train queue only; background -> Noise)
    u           unsure (left out of training and evaluation)
    Space       play the item          c   play item with its context
    b           back to the previous item          q   quit (everything is saved)

Run from code/.
"""
import argparse
import csv
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import load
from tools.annotator import compute_spectrogram

WIN = 3.0
QUEUE_HEADER = ["item_id", "kind", "clip_id", "site", "split", "period", "clip_path",
                "start_s", "end_s", "predicted", "max_conf", "stratum"]
JUDGE_HEADER = ["item_id", "clip_id", "start_s", "end_s", "kind", "predicted",
                "max_conf", "verdict", "time"]
NOISE, UNSURE = "Noise", "Unsure"


# ---------------------------------------------------------------- IO
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


def append_judgment(path, row):
    """Append one judgment to the judgments CSV (header written on first use).

    Args:
        path (Path): <review.dir>/<queue>_judgments.csv.
        row (dict): Judgment keyed by JUDGE_HEADER.
    """
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JUDGE_HEADER)
        if new:
            writer.writeheader()
        writer.writerow(row)


def load_judgments(path):
    """Read the verdicts given so far.

    Args:
        path (Path): Judgments CSV (may not exist yet).

    Returns:
        dict[str, str]: item_id -> verdict; the last judgment of an item wins.
    """
    return {row["item_id"]: row["verdict"] for row in read_dicts(path)} if path.exists() else {}


def classes_from(cfg):
    """Classes in annotator key order, checked against the model's labels.

    Args:
        cfg (dict): The whole project config.

    Returns:
        list[str]: Class names; key 1 is the first. Falls back to the model's label
            order when annotator.categories does not cover the model labels.
    """
    model = Path(cfg["detection"]["model"])
    labels = [line.strip() for line in model.with_name(model.stem + "_Labels.txt")
              .read_text(encoding="utf-8").splitlines() if line.strip()]
    classes = [category for category in cfg.get("annotator", {}).get("categories", []) if category in labels]
    if sorted(classes) != sorted(labels):
        print(f"note: annotator.categories do not cover the model labels {labels}; "
              f"using the model order")
        classes = labels
    return classes


# ---------------------------------------------------------------- queues
def window_row(kind, clip, start, stratum, predicted="", max_conf=""):
    """Queue row for one 3 s window (background or test item) of a manifest clip.

    Args:
        kind (str): "background" or "test".
        clip (dict): Manifest row of the clip.
        start (float): Window start in seconds.
        stratum (str): Stratum name stored in the queue.
        predicted (str): Expected class ("Noise" for background, "" for test).
        max_conf (str): v0 best score, formatted ("" if not relevant).

    Returns:
        dict[str, str]: Row keyed by QUEUE_HEADER.
    """
    return {"item_id": f"{kind}_{clip['clip_id']}_{start:.0f}", "kind": kind,
            "clip_id": clip["clip_id"], "site": clip["site"], "split": clip["split"],
            "period": clip["period"], "clip_path": clip["clip_path"],
            "start_s": f"{start:.1f}", "end_s": f"{start + WIN:.1f}",
            "predicted": predicted, "max_conf": max_conf, "stratum": stratum}


def analysed_clips(manifest, raw_dir, split):
    """Manifest clips of one split that have a raw BirdNET result.

    Args:
        manifest (list[dict]): Manifest rows.
        raw_dir (Path): detection.dir/raw.
        split (str): "train" or "heldout".

    Returns:
        list[dict]: The matching manifest rows.
    """
    return [clip for clip in manifest if clip["split"] == split
            and (raw_dir / f"{clip['clip_id']}.BirdNET.results.csv").exists()]


def build_train(cfg, manifest, rng):
    """Build the train queue: stratified events plus background windows.

    Events (train sites) are drawn per predicted class, evenly over review.conf_bins,
    topped up from other bins when one is short. Background = review.background_per_clip
    random grid windows per analysed clip, at least context_s away from any event.

    Args:
        cfg (dict): The whole project config.
        manifest (list[dict]): Manifest rows.
        rng (random.Random): Seeded generator.

    Returns:
        tuple[list[dict], None]: (queue rows keyed by QUEUE_HEADER, no strata).
    """
    review_cfg, detection_cfg = cfg["review"], cfg["detection"]
    by_clip = {clip["clip_id"]: clip for clip in manifest}
    events = [event for event in read_dicts(Path(detection_cfg["dir"]) / "events.csv")
              if event["split"] == "train"]
    bins = review_cfg["conf_bins"]
    n_bins = len(bins) - 1
    def bin_of(conf):
        """Index of the conf_bins bin containing c (the last bin includes its upper edge).

        Args:
            conf (float): An event's max_conf.

        Returns:
            int | None: Bin index, None if c is outside the bins.
        """
        for i in range(n_bins):
            if bins[i] <= conf < bins[i + 1] or (i == n_bins - 1 and conf == bins[-1]):
                return i
        return None
    cells = defaultdict(list)
    for event in events:
        bin_idx = bin_of(float(event["max_conf"]))
        if bin_idx is not None:
            cells[(event["label"], bin_idx)].append(event)
    rows = []
    for label in sorted({event["label"] for event in events}):
        want = review_cfg["events_per_class"]
        pools = [rng.sample(cells[(label, bin_idx)], len(cells[(label, bin_idx)]))
                 for bin_idx in range(n_bins)]
        take = [0] * n_bins
        # spread evenly over the bins, then top up from the bins that still have events
        while want and any(take[bin_idx] < len(pools[bin_idx]) for bin_idx in range(n_bins)):
            for bin_idx in range(n_bins):
                if want and take[bin_idx] < len(pools[bin_idx]):
                    take[bin_idx] += 1
                    want -= 1
        for bin_idx in range(n_bins):
            for event in pools[bin_idx][:take[bin_idx]]:
                clip = by_clip[event["clip_id"]]
                item_id = f"event_{event['clip_id']}_{event['label']}_{float(event['start_s']):.0f}"
                rows.append({"item_id": item_id,
                             "kind": "event", "clip_id": event["clip_id"], "site": event["site"],
                             "split": event["split"], "period": event["period"],
                             "clip_path": clip["clip_path"], "start_s": event["start_s"],
                             "end_s": event["end_s"], "predicted": event["label"],
                             "max_conf": event["max_conf"],
                             "stratum": f"{label}|{bins[bin_idx]}-{bins[bin_idx + 1]}"})
    event_spans = defaultdict(list)
    for event in events:
        event_spans[event["clip_id"]].append((float(event["start_s"]), float(event["end_s"])))
    context_s = review_cfg["context_s"]
    for clip in analysed_clips(manifest, Path(detection_cfg["dir"]) / "raw", "train"):
        n_win = int(float(clip["duration_s"]) // WIN)
        free = [k * WIN for k in range(n_win)
                if all(k * WIN + WIN <= ev_start - context_s or k * WIN >= ev_end + context_s
                       for ev_start, ev_end in event_spans[clip["clip_id"]])]
        for start in sorted(rng.sample(free, min(review_cfg["background_per_clip"], len(free)))):
            rows.append(window_row("background", clip, start, "background", predicted=NOISE))
    return rows, None


def build_test(cfg, manifest, rng):
    """Build the blind test queue from heldout clips.

    Every 3 s grid window gets v0's best score over all classes (0 if absent from raw/),
    is put in a review.test_bins stratum, and review.test_per_bin windows are drawn per
    stratum.

    Args:
        cfg (dict): The whole project config.
        manifest (list[dict]): Manifest rows.
        rng (random.Random): Seeded generator.

    Returns:
        tuple[list[dict], list[dict]]: (queue rows keyed by QUEUE_HEADER,
            strata rows with stratum, population, sampled).
    """
    review_cfg, detection_cfg = cfg["review"], cfg["detection"]
    raw_dir = Path(detection_cfg["dir"]) / "raw"
    bins = review_cfg["test_bins"]
    n_bins = len(bins) - 1
    population = defaultdict(list)
    for clip in analysed_clips(manifest, raw_dir, "heldout"):
        best = defaultdict(float)
        for window in read_dicts(raw_dir / f"{clip['clip_id']}.BirdNET.results.csv"):
            k = round(float(window["Start (s)"]) / WIN)
            best[k] = max(best[k], float(window["Confidence"]))
        for k in range(int(float(clip["duration_s"]) // WIN)):
            conf = best.get(k, 0.0)
            bin_idx = next(i for i in range(n_bins) if conf < bins[i + 1] or i == n_bins - 1)
            population[bin_idx].append((clip, k * WIN, conf))
    rows, strata = [], []
    for bin_idx in range(n_bins):
        picked = rng.sample(population[bin_idx], min(review_cfg["test_per_bin"], len(population[bin_idx])))
        name = f"{bins[bin_idx]}-{bins[bin_idx + 1]}"
        strata.append({"stratum": name, "population": len(population[bin_idx]), "sampled": len(picked)})
        for clip, start, conf in sorted(picked, key=lambda entry: (entry[0]["clip_id"], entry[1])):
            rows.append(window_row("test", clip, start, name, max_conf=f"{conf:.4f}"))
    return rows, strata


def get_queue(cfg, queue, review_dir):
    """Return the frozen queue, building and saving it (shuffled) on first use.

    Args:
        cfg (dict): The whole project config.
        queue (str): "train" or "test".
        review_dir (Path): Review folder.

    Returns:
        list[dict]: Queue rows keyed by QUEUE_HEADER.
    """
    path = review_dir / f"{queue}_queue.csv"
    if path.exists():
        return read_dicts(path)
    manifest = read_dicts(Path(cfg["survey"]["dir"]) / "manifest.csv")
    rng = random.Random(f"{cfg['review']['seed']}-{queue}")
    rows, strata = (build_train if queue == "train" else build_test)(cfg, manifest, rng)
    rng.shuffle(rows)                       # any session of --n N gets a mix
    write_dicts(path, QUEUE_HEADER, rows)
    if strata:
        write_dicts(review_dir / "test_strata.csv", ["stratum", "population", "sampled"], strata)
    print(f"built {path}: {len(rows)} items")
    return rows


def describe(queue, rows, judged):
    """Print the queue size, items per stratum and the verdicts so far.

    Args:
        queue (str): Queue name.
        rows (list[dict]): Queue rows.
        judged (dict[str, str]): item_id -> verdict.
    """
    print(f"{queue} queue: {len(rows)} items, {sum(row['item_id'] in judged for row in rows)} judged")
    for stratum, count in sorted(Counter(row["stratum"] for row in rows).items()):
        print(f"    {stratum:28s} {count}")
    if judged:
        print("verdicts:", dict(Counter(judged.values())))


# ---------------------------------------------------------------- GUI
class Reviewer:
    """Matplotlib window to judge queue items one by one (keys in the module docstring).

    Attributes:
        items (list[dict]): Queue rows of this session.
        index (int): Index of the item shown.
        judged (dict[str, str]): item_id -> verdict, updated at each key press.
        classes (list[str]): Classes bound to keys 1, 2, 3, ...
        blind (bool): True for the test queue (prediction hidden, y/Enter disabled).
        audio (numpy.ndarray): Mono samples of the item with its context.
        sample_rate (int): Sample rate.
        load_start (float): Clip time of the first sample of audio, in seconds.
        span (tuple[float, float]): Item start and end, in clip seconds.
    """
    def __init__(self, cfg, queue, items, judged, judgments_path, classes):
        """Set up the figure and key bindings, then show the first item.

        Args:
            cfg (dict): The whole project config (viewer and review sections are used).
            queue (str): "train" or "test".
            items (list[dict]): Queue rows to judge this session.
            judged (dict[str, str]): Verdicts so far (item_id -> verdict).
            judgments_path (Path): Judgments CSV to append to.
            classes (list[str]): Classes bound to keys 1, 2, 3, ...
        """
        import matplotlib.pyplot as plt
        self.plt = plt
        for key in [key for key in plt.rcParams if key.startswith("keymap.")]:
            plt.rcParams[key] = []
        self.viewer_cfg = cfg.get("viewer", {})
        self.context_s = cfg["review"]["context_s"]
        self.queue, self.items, self.judged, self.judgments_path = queue, items, judged, judgments_path
        self.classes, self.blind = classes, queue == "test"
        self.index, self._playing, self._timer = 0, False, None
        self.fig = plt.figure(figsize=(13, 6))
        self.ax = self.fig.add_axes([0.06, 0.12, 0.9, 0.76])
        keys = "   ".join(f"[{key + 1}] {cls}" for key, cls in enumerate(classes))
        ok = "" if self.blind else "[y/Enter] prediction right   "
        self.fig.text(0.06, 0.97, f"{keys}   [0/n] Noise   {ok}[u] unsure   "
                      "[Space] play   [c] play+context   [b] back   [q] quit",
                      fontsize=9, va="top")
        self.msg = self.fig.text(0.06, 0.02, "", fontsize=10, color="darkred")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.show(0)

    def item(self):
        """The queue row currently shown.

        Returns:
            dict: Queue row keyed by QUEUE_HEADER.
        """
        return self.items[self.index]

    def show(self, i):
        """Load item i with review.context_s around it and draw its spectrogram.

        Only that segment is read from the clip. The prediction is hidden for the test queue.

        Args:
            i (int): Index in self.items.
        """
        self._stop()
        self.index = i
        item = self.item()
        start, end = float(item["start_s"]), float(item["end_s"])
        info = sf.info(item["clip_path"])
        sample_rate = info.samplerate
        self.load_start = max(0.0, start - self.context_s)
        load_end = min(info.frames / sample_rate, end + self.context_s)
        audio, _ = sf.read(item["clip_path"], start=int(self.load_start * sample_rate),
                           frames=int((load_end - self.load_start) * sample_rate),
                           dtype="float32", always_2d=True)
        self.audio, self.sample_rate, self.span = audio.mean(axis=1), sample_rate, (start, end)
        viewer_cfg = self.viewer_cfg
        spec, f_low, f_high = compute_spectrogram(
            self.audio, sample_rate, viewer_cfg.get("n_fft", 1024), viewer_cfg.get("hop_length", 256),
            viewer_cfg.get("f_min", 0), min(viewer_cfg.get("f_max", 10000), sample_rate / 2))
        vmax = float(np.percentile(spec, 99.5))
        ax = self.ax
        ax.clear()
        ax.imshow(spec, origin="lower", aspect="auto", cmap=viewer_cfg.get("cmap", "magma"),
                  extent=[self.load_start, self.load_start + len(self.audio) / sample_rate, f_low, f_high],
                  vmin=vmax - viewer_cfg.get("db_range", 70), vmax=vmax, interpolation="nearest")
        ax.axvspan(start, end, ymin=0.97, ymax=1.0, color="cyan", lw=0)   # marker, not over the image
        for edge in (start, end):
            ax.axvline(edge, color="cyan", lw=1.2)
        self.playhead = ax.axvline(start, color="white", lw=1, visible=False)
        ax.set_xlabel("time in clip (s)")
        ax.set_ylabel("frequency (Hz)")
        what = (f"window {start:.0f}-{end:.0f} s" if self.blind else
                f"{item['kind']}  predicted {item['predicted']}"
                + (f" ({float(item['max_conf']):.2f})" if item["max_conf"] else ""))
        done = self.judged.get(item["item_id"])
        ax.set_title(f"[{i + 1}/{len(self.items)}] {item['site']}   {what}"
                     + (f"   -> judged: {done}" if done else ""), loc="left", fontsize=10)
        self._msg("")
        self.fig.canvas.draw_idle()

    def judge(self, verdict):
        """Save a verdict for the current item and move to the next one.

        Args:
            verdict (str): A class name, "Noise" or "Unsure".
        """
        item = self.item()
        append_judgment(self.judgments_path, {
            "item_id": item["item_id"], "clip_id": item["clip_id"], "start_s": item["start_s"],
            "end_s": item["end_s"], "kind": item["kind"], "predicted": item["predicted"],
            "max_conf": item["max_conf"], "verdict": verdict,
            "time": datetime.now().isoformat(timespec="seconds")})
        self.judged[item["item_id"]] = verdict
        if self.index + 1 < len(self.items):
            self.show(self.index + 1)
            self._msg(f"saved: {verdict}")
        else:
            self._msg(f"saved: {verdict}. Session done ({len(self.items)} items), q to quit")
            self.show(self.index)
            self._msg(f"saved: {verdict}. Session done ({len(self.items)} items), q to quit")

    def _msg(self, text):
        """Show a message under the spectrogram.

        Args:
            text (str): Message text ("" clears it).
        """
        self.msg.set_text(text)
        self.fig.canvas.draw_idle()

    def play(self, context):
        """Play the item, or the item with its context, with a moving playhead; again to stop.

        Args:
            context (bool): Play the whole loaded segment instead of the item only.
        """
        if self._playing:
            return self._stop()
        if context:
            start, end = self.load_start, self.load_start + len(self.audio) / self.sample_rate
        else:
            start, end = self.span
        first = int((start - self.load_start) * self.sample_rate)
        last = int((end - self.load_start) * self.sample_rate)
        segment = self.audio[first:last].copy()
        if not len(segment):
            return
        if self.viewer_cfg.get("normalize_playback", True) and np.abs(segment).max() > 0:
            segment *= 0.8 / np.abs(segment).max()
        try:
            import sounddevice as sd
            sd.stop()
            sd.play(segment, self.sample_rate)
        except Exception as err:
            return self._msg(f"cannot play audio: {err}")
        self._playing, self._play_state = True, (start, end - start, time.time())
        self._timer = self.fig.canvas.new_timer(interval=60)
        self._timer.add_callback(self._tick)
        self._timer.start()

    def _tick(self):
        """Timer callback: move the playhead, stop at the end of the segment."""
        start, length, t_started = self._play_state
        elapsed = time.time() - t_started
        if elapsed >= length:
            return self._stop()
        self.playhead.set_xdata([start + elapsed] * 2)
        self.playhead.set_visible(True)
        self.fig.canvas.draw_idle()

    def _stop(self):
        """Stop playback and hide the playhead."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self._playing:
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass
        self._playing = False
        if hasattr(self, "playhead"):
            self.playhead.set_visible(False)

    def _on_key(self, event):
        """Dispatch a key press: verdicts, playback, back, quit.

        Args:
            event (matplotlib.backend_bases.KeyEvent): The key event.
        """
        key = event.key
        if key is None:
            return
        if key.isdigit() and 1 <= int(key) <= len(self.classes):
            self.judge(self.classes[int(key) - 1])
        elif key in ("0", "n"):
            self.judge(NOISE)
        elif key in ("y", "enter"):
            if self.blind:
                self._msg("test queue is blind: choose the class (1-3) or Noise (0)")
            else:
                self.judge(self.item()["predicted"])
        elif key == "u":
            self.judge(UNSURE)
        elif key == " ":
            self.play(context=False)
        elif key == "c":
            self.play(context=True)
        elif key == "b" and self.index > 0:
            self.show(self.index - 1)
        elif key == "q":
            self._stop()
            self.plt.close(self.fig)

    def run(self):
        """Open the window (blocks until it is closed)."""
        self.plt.show()


def main():
    """CLI entry point: load or build the queue, then judge the next items in the GUI."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--queue", choices=["train", "test"], required=True)
    parser.add_argument("--n", type=int, help="only the next N unjudged items this session")
    parser.add_argument("--dir", help="output folder (default: review.dir)")
    parser.add_argument("--list", action="store_true", help="build/show the queue, no GUI")
    args = parser.parse_args()
    cfg = load(args.config)
    review_dir = Path(args.dir or cfg["review"]["dir"])
    classes = classes_from(cfg)
    rows = get_queue(cfg, args.queue, review_dir)
    judgments_path = review_dir / f"{args.queue}_judgments.csv"
    judged = load_judgments(judgments_path)
    describe(args.queue, rows, judged)
    todo = [row for row in rows if row["item_id"] not in judged][:args.n]
    if args.list:
        return
    if not todo:
        print("nothing left to judge in this queue")
        return
    import matplotlib
    if cfg.get("viewer", {}).get("backend"):
        matplotlib.use(cfg["viewer"]["backend"])
    print(f"{len(todo)} items this session")
    Reviewer(cfg, args.queue, todo, judged, judgments_path, classes).run()


if __name__ == "__main__":
    main()
