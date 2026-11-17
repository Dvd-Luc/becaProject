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
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_dicts(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def append_judgment(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JUDGE_HEADER)
        if new:
            w.writeheader()
        w.writerow(row)


def load_judgments(path):
    """{item_id: verdict}, last judgment wins."""
    return {r["item_id"]: r["verdict"] for r in read_dicts(path)} if path.exists() else {}


def classes_from(cfg):
    """Classes in annotator key order, checked against the model's labels."""
    model = Path(cfg["detection"]["model"])
    labels = [l.strip() for l in model.with_name(model.stem + "_Labels.txt")
              .read_text(encoding="utf-8").splitlines() if l.strip()]
    cats = [c for c in cfg.get("annotator", {}).get("categories", []) if c in labels]
    if sorted(cats) != sorted(labels):
        print(f"note: annotator.categories do not cover the model labels {labels}; "
              f"using the model order")
        cats = labels
    return cats


# ---------------------------------------------------------------- queues
def window_row(kind, m, start, stratum, predicted="", max_conf=""):
    return {"item_id": f"{kind}_{m['clip_id']}_{start:.0f}", "kind": kind,
            "clip_id": m["clip_id"], "site": m["site"], "split": m["split"],
            "period": m["period"], "clip_path": m["clip_path"],
            "start_s": f"{start:.1f}", "end_s": f"{start + WIN:.1f}",
            "predicted": predicted, "max_conf": max_conf, "stratum": stratum}


def analysed_clips(manifest, raw_dir, split):
    return [m for m in manifest if m["split"] == split
            and (raw_dir / f"{m['clip_id']}.BirdNET.results.csv").exists()]


def build_train(cfg, manifest, rng):
    r, d = cfg["review"], cfg["detection"]
    by_clip = {m["clip_id"]: m for m in manifest}
    events = [e for e in read_dicts(Path(d["dir"]) / "events.csv") if e["split"] == "train"]
    bins = r["conf_bins"]
    nb = len(bins) - 1
    def bin_of(c):
        for i in range(nb):
            if bins[i] <= c < bins[i + 1] or (i == nb - 1 and c == bins[-1]):
                return i
        return None
    cells = defaultdict(list)
    for e in events:
        b = bin_of(float(e["max_conf"]))
        if b is not None:
            cells[(e["label"], b)].append(e)
    rows = []
    for label in sorted({e["label"] for e in events}):
        want = r["events_per_class"]
        pools = [rng.sample(cells[(label, b)], len(cells[(label, b)])) for b in range(nb)]
        take = [0] * nb
        while want and any(take[b] < len(pools[b]) for b in range(nb)):   # even, then top up
            for b in range(nb):
                if want and take[b] < len(pools[b]):
                    take[b] += 1
                    want -= 1
        for b in range(nb):
            for e in pools[b][:take[b]]:
                m = by_clip[e["clip_id"]]
                rows.append({"item_id": f"event_{e['clip_id']}_{e['label']}_{float(e['start_s']):.0f}",
                             "kind": "event", "clip_id": e["clip_id"], "site": e["site"],
                             "split": e["split"], "period": e["period"],
                             "clip_path": m["clip_path"], "start_s": e["start_s"],
                             "end_s": e["end_s"], "predicted": e["label"],
                             "max_conf": e["max_conf"], "stratum": f"{label}|{bins[b]}-{bins[b + 1]}"})
    ev_by_clip = defaultdict(list)
    for e in events:
        ev_by_clip[e["clip_id"]].append((float(e["start_s"]), float(e["end_s"])))
    ctx = r["context_s"]
    for m in analysed_clips(manifest, Path(d["dir"]) / "raw", "train"):
        n_win = int(float(m["duration_s"]) // WIN)
        free = [k * WIN for k in range(n_win)
                if all(k * WIN + WIN <= s - ctx or k * WIN >= e + ctx
                       for s, e in ev_by_clip[m["clip_id"]])]
        for start in sorted(rng.sample(free, min(r["background_per_clip"], len(free)))):
            rows.append(window_row("background", m, start, "background", predicted=NOISE))
    return rows, None


def build_test(cfg, manifest, rng):
    r, d = cfg["review"], cfg["detection"]
    raw_dir = Path(d["dir"]) / "raw"
    bins = r["test_bins"]
    nb = len(bins) - 1
    pop = defaultdict(list)
    for m in analysed_clips(manifest, raw_dir, "heldout"):
        best = defaultdict(float)
        for w in read_dicts(raw_dir / f"{m['clip_id']}.BirdNET.results.csv"):
            k = round(float(w["Start (s)"]) / WIN)
            best[k] = max(best[k], float(w["Confidence"]))
        for k in range(int(float(m["duration_s"]) // WIN)):
            c = best.get(k, 0.0)
            b = next(i for i in range(nb) if c < bins[i + 1] or i == nb - 1)
            pop[b].append((m, k * WIN, c))
    rows, strata = [], []
    for b in range(nb):
        pick = rng.sample(pop[b], min(r["test_per_bin"], len(pop[b])))
        name = f"{bins[b]}-{bins[b + 1]}"
        strata.append({"stratum": name, "population": len(pop[b]), "sampled": len(pick)})
        for m, start, c in sorted(pick, key=lambda x: (x[0]["clip_id"], x[1])):
            rows.append(window_row("test", m, start, name, max_conf=f"{c:.4f}"))
    return rows, strata


def get_queue(cfg, queue, out):
    path = out / f"{queue}_queue.csv"
    if path.exists():
        return read_dicts(path)
    manifest = read_dicts(Path(cfg["survey"]["dir"]) / "manifest.csv")
    rng = random.Random(f"{cfg['review']['seed']}-{queue}")
    rows, strata = (build_train if queue == "train" else build_test)(cfg, manifest, rng)
    rng.shuffle(rows)                       # any session of --n N gets a mix
    write_dicts(path, QUEUE_HEADER, rows)
    if strata:
        write_dicts(out / "test_strata.csv", ["stratum", "population", "sampled"], strata)
    print(f"built {path}: {len(rows)} items")
    return rows


def describe(queue, rows, judged):
    print(f"{queue} queue: {len(rows)} items, {sum(r['item_id'] in judged for r in rows)} judged")
    for s, n in sorted(Counter(r["stratum"] for r in rows).items()):
        print(f"    {s:28s} {n}")
    if judged:
        print("verdicts:", dict(Counter(judged.values())))


# ---------------------------------------------------------------- GUI
class Reviewer:
    def __init__(self, cfg, queue, items, judged, jpath, classes):
        import matplotlib.pyplot as plt
        self.plt = plt
        for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[k] = []
        self.v = cfg.get("viewer", {})
        self.ctx = cfg["review"]["context_s"]
        self.queue, self.items, self.judged, self.jpath = queue, items, judged, jpath
        self.classes, self.blind = classes, queue == "test"
        self.i, self._playing, self._timer = 0, False, None
        self.fig = plt.figure(figsize=(13, 6))
        self.ax = self.fig.add_axes([0.06, 0.12, 0.9, 0.76])
        keys = "   ".join(f"[{k + 1}] {c}" for k, c in enumerate(classes))
        ok = "" if self.blind else "[y/Enter] prediction right   "
        self.fig.text(0.06, 0.97, f"{keys}   [0/n] Noise   {ok}[u] unsure   "
                      "[Space] play   [c] play+context   [b] back   [q] quit",
                      fontsize=9, va="top")
        self.msg = self.fig.text(0.06, 0.02, "", fontsize=10, color="darkred")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.show(0)

    def item(self):
        return self.items[self.i]

    def show(self, i):
        self._stop()
        self.i = i
        it = self.item()
        s, e = float(it["start_s"]), float(it["end_s"])
        info = sf.info(it["clip_path"])
        sr = info.samplerate
        self.t0 = max(0.0, s - self.ctx)
        t1 = min(info.frames / sr, e + self.ctx)
        x, _ = sf.read(it["clip_path"], start=int(self.t0 * sr),
                       frames=int((t1 - self.t0) * sr), dtype="float32", always_2d=True)
        self.x, self.sr, self.span = x.mean(axis=1), sr, (s, e)
        v = self.v
        S, flo, fhi = compute_spectrogram(self.x, sr, v.get("n_fft", 1024),
                                          v.get("hop_length", 256), v.get("f_min", 0),
                                          min(v.get("f_max", 10000), sr / 2))
        vmax = float(np.percentile(S, 99.5))
        ax = self.ax
        ax.clear()
        ax.imshow(S, origin="lower", aspect="auto", cmap=v.get("cmap", "magma"),
                  extent=[self.t0, self.t0 + len(self.x) / sr, flo, fhi],
                  vmin=vmax - v.get("db_range", 70), vmax=vmax, interpolation="nearest")
        ax.axvspan(s, e, ymin=0.97, ymax=1.0, color="cyan", lw=0)   # marker, not over the image
        for t in (s, e):
            ax.axvline(t, color="cyan", lw=1.2)
        self.playhead = ax.axvline(s, color="white", lw=1, visible=False)
        ax.set_xlabel("time in clip (s)")
        ax.set_ylabel("frequency (Hz)")
        what = (f"window {s:.0f}-{e:.0f} s" if self.blind else
                f"{it['kind']}  predicted {it['predicted']}"
                + (f" ({float(it['max_conf']):.2f})" if it["max_conf"] else ""))
        done = self.judged.get(it["item_id"])
        ax.set_title(f"[{i + 1}/{len(self.items)}] {it['site']}   {what}"
                     + (f"   -> judged: {done}" if done else ""), loc="left", fontsize=10)
        self._msg("")
        self.fig.canvas.draw_idle()

    def judge(self, verdict):
        it = self.item()
        append_judgment(self.jpath, {
            "item_id": it["item_id"], "clip_id": it["clip_id"], "start_s": it["start_s"],
            "end_s": it["end_s"], "kind": it["kind"], "predicted": it["predicted"],
            "max_conf": it["max_conf"], "verdict": verdict,
            "time": datetime.now().isoformat(timespec="seconds")})
        self.judged[it["item_id"]] = verdict
        if self.i + 1 < len(self.items):
            self.show(self.i + 1)
            self._msg(f"saved: {verdict}")
        else:
            self._msg(f"saved: {verdict}. Session done ({len(self.items)} items), q to quit")
            self.show(self.i)
            self._msg(f"saved: {verdict}. Session done ({len(self.items)} items), q to quit")

    def _msg(self, t):
        self.msg.set_text(t)
        self.fig.canvas.draw_idle()

    def play(self, context):
        if self._playing:
            return self._stop()
        s, e = (self.t0, self.t0 + len(self.x) / self.sr) if context else self.span
        seg = self.x[int((s - self.t0) * self.sr):int((e - self.t0) * self.sr)].copy()
        if not len(seg):
            return
        if self.v.get("normalize_playback", True) and np.abs(seg).max() > 0:
            seg *= 0.8 / np.abs(seg).max()
        try:
            import sounddevice as sd
            sd.stop()
            sd.play(seg, self.sr)
        except Exception as ex:
            return self._msg(f"cannot play audio: {ex}")
        self._playing, self._p = True, (s, e - s, time.time())
        self._timer = self.fig.canvas.new_timer(interval=60)
        self._timer.add_callback(self._tick)
        self._timer.start()

    def _tick(self):
        s, length, t = self._p
        el = time.time() - t
        if el >= length:
            return self._stop()
        self.playhead.set_xdata([s + el] * 2)
        self.playhead.set_visible(True)
        self.fig.canvas.draw_idle()

    def _stop(self):
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

    def _on_key(self, ev):
        k = ev.key
        if k is None:
            return
        if k.isdigit() and 1 <= int(k) <= len(self.classes):
            self.judge(self.classes[int(k) - 1])
        elif k in ("0", "n"):
            self.judge(NOISE)
        elif k in ("y", "enter"):
            if self.blind:
                self._msg("test queue is blind: choose the class (1-3) or Noise (0)")
            else:
                self.judge(self.item()["predicted"])
        elif k == "u":
            self.judge(UNSURE)
        elif k == " ":
            self.play(context=False)
        elif k == "c":
            self.play(context=True)
        elif k == "b" and self.i > 0:
            self.show(self.i - 1)
        elif k == "q":
            self._stop()
            self.plt.close(self.fig)

    def run(self):
        self.plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="config/project.yaml")
    ap.add_argument("--queue", choices=["train", "test"], required=True)
    ap.add_argument("--n", type=int, help="only the next N unjudged items this session")
    ap.add_argument("--dir", help="output folder (default: review.dir)")
    ap.add_argument("--list", action="store_true", help="build/show the queue, no GUI")
    a = ap.parse_args()
    cfg = load(a.config)
    out = Path(a.dir or cfg["review"]["dir"])
    classes = classes_from(cfg)
    rows = get_queue(cfg, a.queue, out)
    jpath = out / f"{a.queue}_judgments.csv"
    judged = load_judgments(jpath)
    describe(a.queue, rows, judged)
    todo = [r for r in rows if r["item_id"] not in judged][:a.n]
    if a.list:
        return
    if not todo:
        print("nothing left to judge in this queue")
        return
    import matplotlib
    if cfg.get("viewer", {}).get("backend"):
        matplotlib.use(cfg["viewer"]["backend"])
    print(f"{len(todo)} items this session")
    Reviewer(cfg, a.queue, todo, judged, jpath, classes).run()


if __name__ == "__main__":
    main()
