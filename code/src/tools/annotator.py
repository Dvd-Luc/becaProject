"""Spectrogram annotator for long recordings (up to ~5 min and more).

Label time regions with categories from a config file. Labels are saved in the
Audacity label format (start <TAB> end <TAB> label), one file per recording:
    <labels_dir>/<recording stem>.txt
so they work directly with annotation_tools.py and make_training_clips.py.

Usage
    python annotator.py --config annotator_config.yaml                 # files still to do
    python annotator.py --config annotator_config.yaml --type drum --n 25
    python annotator.py --config annotator_config.yaml --list          # preview the queue
    python annotator.py --config annotator_config.yaml --status        # progress summary
    python annotator.py --config annotator_config.yaml --all     # also labelled ones

A recording is "to do" when labels_dir has no <stem>.txt for it. Work in progress is
autosaved to <stem>.partial (ignored by the status tool) and reloaded next time.

Controls
    drag on spectrogram     create a selection (drag its edges, or inside it, to adjust)
    1 .. 9, 0               give the selection the category with that number
    right-click a label     take it back into the selection to adjust / relabel
    Enter                   re-commit an adjusted label with its previous category
    Backspace / Delete      delete the selection (or the label being edited)
    Esc                     cancel (restores a label being edited)
    Space                   play / stop the selection (the visible part if none)
    mouse wheel             zoom in time around the cursor
    shift + wheel, <- ->    scroll in time (the slider at the bottom works too)
    Up / Down, + / -        zoom in / out around the centre of the view
    [  ]                    lower / raise spectrogram contrast
    s                       save labels and go to the next recording
    x                       mark recording as "no calls" (empty file) and go to the next
    b                       go back to the previous recording
"""
import argparse
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import matplotlib
import numpy as np
import soundfile as sf
import yaml
import csv
from collections import Counter
from matplotlib.patches import Patch
from matplotlib.widgets import Slider, SpanSelector

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}
CAT_KEYS = "1234567890"
MIN_SEL = 0.02      # s, smaller selections are ignored
MIN_VIEW = 0.5      # s, maximum zoom
MAX_COLS = 2500     # spectrogram columns drawn at once (block max-pooling above that)
HELP = ("drag: select | 1-9: label | right-click label: edit | Space: play | "
        "Backspace: delete | wheel: zoom | shift+wheel / arrows: scroll | "
        "s: save+next | x: no calls | b: back")


# ---------------------------------------------------------------- config / IO
@dataclass
class Config:
    audio_dir: str
    labels_dir: str
    categories: list
    n_fft: int = 1024
    hop_length: int = 256
    f_min: float = 0.0
    f_max: float = 10000.0
    view_seconds: float = 20.0
    db_range: float = 70.0
    cmap: str = "magma"
    normalize_playback: bool = True
    backend: str = ""          # "TkAgg", "QtAgg"... empty = matplotlib default
    metadata_csv: str = ""     # optional CSV: ordering, --type filter, status by quality
    file_col: str = "file-name"
    q_col: str = "q"
    type_col: str = "type"
    status_file: str = "annotation_status.csv"
    group_cols: list = field(default_factory=list) #CSV columns to balance across when selecting files (e.g. [rec, loc]). Empty = off

def load_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise SystemExit(f"unknown keys in {path}: {sorted(unknown)}")
    cfg = Config(**raw)
    if not 1 <= len(cfg.categories) <= len(CAT_KEYS):
        raise SystemExit(f"'categories' needs between 1 and {len(CAT_KEYS)} entries")
    return cfg


def read_labels(path):
    labels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("\\"):      # skip frequency-range lines
            continue
        p = line.split("\t")
        if len(p) >= 3:
            labels.append({"start": float(p[0].replace(",", ".")),
                           "end": float(p[1].replace(",", ".")),
                           "cat": p[2].strip()})
    return sorted(labels, key=lambda l: l["start"])


def write_labels(path, labels):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for l in sorted(labels, key=lambda l: l["start"]):
            f.write(f"{l['start']:.6f}\t{l['end']:.6f}\t{l['cat']}\n")


def load_metadata(cfg):
    """{file name: {'q': ..., 'type': ...}} from the CSV; {} if none is configured."""
    if not cfg.metadata_csv:
        return {}
    with open(cfg.metadata_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and cfg.file_col not in rows[0]:
        raise SystemExit(f"column '{cfg.file_col}' not found in {cfg.metadata_csv} "
                         f"(columns: {list(rows[0])})")
    for c in cfg.group_cols:
        if rows and c not in rows[0]:
            raise SystemExit(f"group column '{c}' not found in {cfg.metadata_csv} "
                             f"(columns: {list(rows[0])})")
    group = lambda r: "|".join((r.get(c) or "").strip() for c in cfg.group_cols)
    return {r[cfg.file_col].strip(): {"q": (r.get(cfg.q_col) or "").strip().upper(),
                                      "type": r.get(cfg.type_col) or "",
                                      "group": group(r)}
            for r in rows}


def label_status(cfg, name):
    """('todo' | 'done' | 'no_calls', list of label names) for a recording name."""
    lab = Path(cfg.labels_dir) / f"{Path(name).stem}.txt"
    if not lab.exists():
        return "todo", []
    names = [l["cat"] for l in read_labels(lab)]
    return ("done" if names else "no_calls"), names


def balance_groups(files, info, done, max_per_group=None):
    """Interleave files across groups: always take the next file from the group
    with the fewest recordings labelled so far. Order inside a group is kept
    (quality first). max_per_group caps labelled + queued recordings per group."""
    queues = {}
    for p in files:
        queues.setdefault(info(p)["group"], []).append(p)
    count = Counter(done)
    if max_per_group:
        queues = {g: v[:max(0, max_per_group - count[g])] for g, v in queues.items()}
    queues = {g: v for g, v in queues.items() if v}
    out = []
    while queues:
        g = min(queues, key=lambda g: (count[g], g)) # Example : (2, "B") vs (5, "A")
        out.append(queues[g].pop(0))
        count[g] += 1
        if not queues[g]:
            del queues[g]
    return out


def list_files(cfg, include_done=False, type_filter=None, limit=None, max_per_group=None):
    meta = load_metadata(cfg)
    files = [p for p in Path(cfg.audio_dir).iterdir() if p.suffix.lower() in AUDIO_EXT]
    info = lambda p: meta.get(p.name, {"q": "", "type": "", "group": ""})
    if meta:
        unlisted = [p.name for p in files if p.name not in meta]
        if unlisted:
            print(f"note: {len(unlisted)} audio files are not in the CSV "
                  f"(placed last), e.g. {unlisted[:3]}")
        absent = [n for n in meta if not (Path(cfg.audio_dir) / n).exists()]
        if absent:
            print(f"note: {len(absent)} CSV recordings not found in audio_dir, "
                  f"e.g. {absent[:3]}")
    files.sort(key=lambda p: (info(p)["q"] == "", info(p)["q"], p.name))   # A before B

    status = {p: label_status(cfg, p.name)[0] for p in files}
    done = Counter(info(p)["group"] for p in files if status[p] != "todo")

    if type_filter:
        if not meta:
            raise SystemExit("--type needs metadata_csv in the config")
        files = [p for p in files if type_filter.lower() in info(p)["type"].lower()]
    if not include_done:
        files = [p for p in files if status[p] == "todo"]
        if cfg.group_cols:
            files = balance_groups(files, info, done, max_per_group)
    return files[:limit] if limit else files


def build_status(cfg):
    meta = load_metadata(cfg)
    names = list(meta) + sorted(p.name for p in Path(cfg.audio_dir).iterdir()
                                if p.suffix.lower() in AUDIO_EXT and p.name not in meta)
    rows = []
    for name in names:
        status, labs = label_status(cfg, name)
        m = meta.get(name, {"q": "", "type": "", "group": ""})
        rows.append({"recording": name, "q": m["q"], "type": m["type"],
                     "group": m["group"],
                     "status": status, "n_labels": len(labs),
                     "label_names": ";".join(sorted(set(labs)))})
    return rows


def write_status(cfg):
    rows = build_status(cfg)
    if rows:
        with open(cfg.status_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def print_status(cfg):
    rows = write_status(cfg)
    print(f"{len(rows)} recordings")
    print("by status:", dict(Counter(r["status"] for r in rows)))
    if any(r["q"] for r in rows):
        print("by quality:", dict(Counter(r["q"] for r in rows)))
        by = Counter((r["q"], r["status"]) for r in rows)
        print("quality x status:", {f"{q}/{s}": n for (q, s), n in sorted(by.items())})
    total = Counter()
    for lab in Path(cfg.labels_dir).glob("*.txt"):
        total.update(l["cat"] for l in read_labels(lab))
    if cfg.group_cols:
        g = Counter(r["group"] for r in rows if r["status"] != "todo")
        print(f"labelled recordings per group ({len(g)} groups):", dict(g.most_common()))
    print("labels per class:", dict(total))
    bad = {n: c for n, c in total.items() if n not in cfg.categories}
    print("UNKNOWN label names (typos?):", bad or "none")
    print(f"saved {cfg.status_file}")


def compute_spectrogram(x, sr, n_fft, hop, f_min, f_max):
    """Magnitude spectrogram in dB, only the bins between f_min and f_max."""
    df = sr / n_fft
    b0 = int(f_min // df)
    b1 = min(int(f_max // df) + 1, n_fft // 2 + 1)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(x, n_fft)
    S = np.empty((b1 - b0, n_frames), dtype=np.float32)
    for i0 in range(0, n_frames, 1500):                    # chunks keep memory low
        i1 = min(i0 + 1500, n_frames)
        frames = windows[i0 * hop:(i1 - 1) * hop + 1:hop] * win
        mag = np.abs(np.fft.rfft(frames, axis=1))[:, b0:b1]
        S[:, i0:i1] = 20 * np.log10(mag.T + 1e-10)
    return S, b0 * df, b1 * df


# ---------------------------------------------------------------- application
class Annotator:
    def __init__(self, cfg, files):
        import matplotlib.pyplot as plt
        self.plt = plt
        for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[k] = []                           # free the keys for our shortcuts

        self.cfg, self.files, self.idx = cfg, files, 0
        self.meta = load_metadata(cfg)
        cmap = matplotlib.colormaps["tab10"]
        self.colors = {c: cmap(i % 10) for i, c in enumerate(cfg.categories)}
        self.edit_orig = None
        self._label_artists = []
        self._playing = False
        self._timer = None

        self.fig = plt.figure(figsize=(14, 7.5))
        self.ax = self.fig.add_axes([0.06, 0.17, 0.74, 0.73])
        ax_sl = self.fig.add_axes([0.06, 0.08, 0.74, 0.03])
        self.slider = Slider(ax_sl, "", 0.0, 1.0, valinit=0.0, valfmt="%.1f s")
        self.slider.on_changed(lambda v: self.set_view(v, self.view[1] - self.view[0]))
        self.im = self.ax.imshow(np.zeros((2, 2)), origin="lower", aspect="auto",
                                 cmap=cfg.cmap, interpolation="nearest")
        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel("frequency (Hz)")
        self.playhead = self.ax.axvline(0, color="cyan", lw=1.5, visible=False)

        self.span = SpanSelector(
            self.ax, self._on_select, "horizontal", minspan=MIN_SEL, useblit=True,
            interactive=True, drag_from_anywhere=True, button=1,
            props=dict(facecolor="red", alpha=0.3),
            handle_props=dict(color="red", linewidth=2))

        handles = [Patch(facecolor=self.colors[c], alpha=0.6, label=f"[{CAT_KEYS[i]}]  {c}")
                   for i, c in enumerate(cfg.categories)]
        self.ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       frameon=False, title="Categories")
        self.fig.text(0.06, 0.975, HELP, fontsize=8, va="top")
        self.msg = self.fig.text(0.06, 0.02, "", fontsize=10, color="darkred")

        c = self.fig.canvas
        c.mpl_connect("key_press_event", self._on_key)
        c.mpl_connect("scroll_event", self._on_scroll)
        c.mpl_connect("button_press_event", self._on_click)
        self.load_file(0)

    # ---- loading / saving
    def _txt(self):
        return Path(self.cfg.labels_dir) / f"{self.files[self.idx].stem}.txt"

    def _partial(self):
        return self._txt().with_suffix(".partial")

    def load_file(self, i):
        self._stop_play()
        self.idx = i
        path = self.files[i]
        self.span.clear()
        self.edit_orig = None
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        self.x, self.sr = x.mean(axis=1), sr
        self.duration = len(self.x) / sr
        c = self.cfg
        self.S, self.f_lo, self.f_hi = compute_spectrogram(
            self.x, sr, c.n_fft, c.hop_length, c.f_min, min(c.f_max, sr / 2))
        self.vmax = float(np.percentile(self.S[:, ::max(1, self.S.shape[1] // 2000)], 99.5))
        self.db_range = c.db_range
        if self._partial().exists():
            self.labels, note = read_labels(self._partial()), "restored unsaved work"
        elif self._txt().exists():
            self.labels, note = read_labels(self._txt()), "loaded saved labels"
        else:
            self.labels, note = [], ""
        self.view = (0.0, min(c.view_seconds, self.duration))
        self.refresh()
        self._msg(note)
        try:
            self.fig.canvas.manager.set_window_title(f"{path.name} [{i + 1}/{len(self.files)}]")
        except Exception:
            pass

    def _autosave(self):
        write_labels(self._partial(), self.labels)

    def save_and_next(self):
        if self.get_selection() is not None:
            return self._msg("Uncommitted selection: press a category key or Backspace first")
        if not self.labels:
            return self._msg("No labels: press x to confirm that this recording has no calls")
        self._finish()

    def mark_empty(self):
        if self.labels or self.get_selection() is not None:
            return self._msg("This recording has labels: delete them first if it has no calls")
        self._finish()

    def _finish(self):
        write_labels(self._txt(), self.labels)
        self._partial().unlink(missing_ok=True)
        try:
            write_status(self.cfg)
        except Exception as e:
            print(f"could not update {self.cfg.status_file}: {e}")
        done = self.files[self.idx].name
        if self.idx + 1 < len(self.files):
            self.load_file(self.idx + 1)
            self._msg(f"saved {done}")
        else:
            self._msg(f"saved {done}. That was the last recording: all done")

    def go_back(self):
        if self.idx > 0:
            self.load_file(self.idx - 1)

    # ---- selection and labels
    def _on_select(self, vmin, vmax):
        self._msg(f"selection {vmin:.2f}-{vmax:.2f} s ({vmax - vmin:.2f} s): "
                  "press a category key, Space to listen")

    def get_selection(self):
        if not self.span.get_visible():
            return None
        a, b = self.span.extents
        a, b = max(0.0, a), min(self.duration, b)
        return (a, b) if b - a >= MIN_SEL else None

    def commit(self, cat_idx):
        sel = self.get_selection()
        if sel is None:
            return self._msg("No selection: drag on the spectrogram first")
        if cat_idx is None:
            if self.edit_orig is None:
                return self._msg("Press a category key (1-9) to label the selection")
            cat = self.edit_orig["cat"]
        else:
            cat = self.cfg.categories[cat_idx]
        self.labels.append({"start": sel[0], "end": sel[1], "cat": cat})
        self.labels.sort(key=lambda l: l["start"])
        self.span.clear()
        self.edit_orig = None
        self._autosave()
        self.refresh()
        self._msg(f"labelled {sel[0]:.2f}-{sel[1]:.2f} s as {cat}")

    def delete_selection(self):
        if self.get_selection() is None:
            return self._msg("Nothing selected: right-click a label to select it first")
        was_edit = self.edit_orig is not None
        self.span.clear()
        self.edit_orig = None
        if was_edit:
            self._autosave()
        self.refresh()
        self._msg("label deleted" if was_edit else "selection cleared")

    def cancel(self):
        if self.edit_orig is not None:
            self.labels.append(self.edit_orig)
            self.labels.sort(key=lambda l: l["start"])
            self.edit_orig = None
        self.span.clear()
        self.refresh()

    def _on_click(self, event):
        if event.button != 3 or event.inaxes is not self.ax or event.xdata is None:
            return
        if self.get_selection() is not None:
            return self._msg("Commit or delete the current selection first")
        hits = [l for l in self.labels if l["start"] <= event.xdata <= l["end"]]
        if not hits:
            return
        l = min(hits, key=lambda l: l["end"] - l["start"])
        self.labels.remove(l)
        self.edit_orig = dict(l)
        self.span.extents = (l["start"], l["end"])
        self.span.set_visible(True)
        self.refresh()
        self._msg(f"editing '{l['cat']}': adjust the edges, then press a category key "
                  "or Enter (Backspace deletes, Esc cancels)")

    # ---- view
    def set_view(self, t0, length):
        length = min(max(length, MIN_VIEW), self.duration)
        t0 = min(max(t0, 0.0), self.duration - length)
        self.view = (t0, t0 + length)
        self.refresh()

    def zoom(self, factor, center=None):
        t0, t1 = self.view
        length = t1 - t0
        c = (t0 + t1) / 2 if center is None else center
        new = min(max(length * factor, MIN_VIEW), self.duration)
        self.set_view(c - (c - t0) / length * new, new)

    def pan(self, frac):
        t0, t1 = self.view
        self.set_view(t0 + frac * (t1 - t0), t1 - t0)

    def _on_scroll(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if event.key == "shift":
            self.pan(0.1 if event.button == "up" else -0.1)
        else:
            self.zoom(0.8 if event.button == "up" else 1.25, center=event.xdata)

    # ---- drawing
    def refresh(self):
        self._draw_image()
        self._draw_labels()
        t0, t1 = self.view
        m = self.meta.get(self.files[self.idx].name)
        tag = f"   q={m['q']} type={m['type']}" if m else ""
        self.ax.set_title(
            f"[{self.idx + 1}/{len(self.files)}] {self.files[self.idx].name}{tag}   "
            f"{self.duration:.1f} s   {len(self.labels)} labels   view {t0:.1f}-{t1:.1f} s",
            loc="left", fontsize=10)
        vmax = max(self.duration - (t1 - t0), 1e-3)
        self.slider.eventson = False
        self.slider.valmax = vmax
        self.slider.ax.set_xlim(0, vmax)
        self.slider.set_val(t0)
        self.slider.eventson = True
        self.fig.canvas.draw_idle()

    def _draw_image(self):
        t0, t1 = self.view
        sr, hop, n_fft = self.sr, self.cfg.hop_length, self.cfg.n_fft
        n_frames = self.S.shape[1]
        c0 = max(0, int((t0 * sr - n_fft / 2) // hop))
        c1 = min(n_frames, int(np.ceil((t1 * sr - n_fft / 2) / hop)) + 1)
        sl = self.S[:, c0:c1]
        step = max(1, sl.shape[1] // MAX_COLS)
        if step > 1:                                        # max-pool: keep short calls visible
            k = sl.shape[1] // step * step
            sl = sl[:, :k].reshape(sl.shape[0], -1, step).max(axis=2)
        xa = (c0 * hop + n_fft / 2) / sr - hop / (2 * sr)
        xb = xa + sl.shape[1] * step * hop / sr
        self.im.set_data(sl)
        self.im.set_extent([xa, xb, self.f_lo, self.f_hi])
        self.im.set_clim(self.vmax - self.db_range, self.vmax)
        self.ax.set_xlim(t0, t1)
        self.ax.set_ylim(self.f_lo, self.f_hi)

    def _draw_labels(self):
        for a in self._label_artists:
            a.remove()
        self._label_artists = []
        t0, t1 = self.view
        for l in self.labels:
            if l["end"] < t0 or l["start"] > t1:
                continue
            col = self.colors.get(l["cat"], (0.5, 0.5, 0.5, 1.0))
            self._label_artists.append(
                self.ax.axvspan(l["start"], l["end"], color=col, alpha=0.3, lw=0))
            self._label_artists.append(self.ax.text(
                max(l["start"], t0), 0.98, l["cat"], transform=self.ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=9, color="white", clip_on=True,
                bbox=dict(facecolor=col, alpha=0.8, pad=1.5, lw=0)))
        self.ax.set_xlim(t0, t1)

    def _msg(self, text):
        self.msg.set_text(text)
        self.fig.canvas.draw_idle()

    # ---- playback
    def toggle_play(self):
        if self._playing:
            return self._stop_play()
        sel = self.get_selection()
        t0, t1 = sel if sel else self.view
        seg = self.x[int(t0 * self.sr):int(t1 * self.sr)].copy()
        if len(seg) == 0:
            return
        if self.cfg.normalize_playback and np.abs(seg).max() > 0:
            seg *= 0.8 / np.abs(seg).max()
        try:
            import sounddevice as sd
            sd.stop()
            sd.play(seg, self.sr)
        except Exception as e:                              # no audio device / sounddevice
            return self._msg(f"cannot play audio: {e}")
        self._playing = True
        self._p_t0, self._p_len, self._p_start = t0, t1 - t0, time.time()
        self._timer = self.fig.canvas.new_timer(interval=60)
        self._timer.add_callback(self._tick)
        self._timer.start()

    def _tick(self):
        el = time.time() - self._p_start
        if el >= self._p_len:
            return self._stop_play()
        self.playhead.set_xdata([self._p_t0 + el] * 2)
        self.playhead.set_visible(True)
        self.fig.canvas.draw_idle()

    def _stop_play(self):
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
        self.playhead.set_visible(False)
        self.fig.canvas.draw_idle()

    # ---- keyboard
    def _on_key(self, event):
        k, n = event.key, len(self.cfg.categories)
        if k is None:
            return
        if len(k) == 1 and k in CAT_KEYS[:n]:
            self.commit(CAT_KEYS.index(k))
        elif k == "enter":
            self.commit(None)
        elif k == " ":
            self.toggle_play()
        elif k in ("backspace", "delete"):
            self.delete_selection()
        elif k == "escape":
            self.cancel()
        elif k == "left":
            self.pan(-0.25)
        elif k == "right":
            self.pan(0.25)
        elif k in ("up", "+", "="):
            self.zoom(0.7)
        elif k in ("down", "-"):
            self.zoom(1 / 0.7)
        elif k == "[":
            self.db_range = max(10.0, self.db_range - 5)
            self.refresh()
        elif k == "]":
            self.db_range += 5
            self.refresh()
        elif k == "s":
            self.save_and_next()
        elif k == "x":
            self.mark_empty()
        elif k == "b":
            self.go_back()

    def run(self):
        self.plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="annotator_config.yaml")
    ap.add_argument("--all", action="store_true", help="include already labelled recordings")
    ap.add_argument("--type", help="only recordings whose CSV 'type' contains this text")
    ap.add_argument("--n", type=int, help="only the next N recordings to annotate")
    ap.add_argument("--status", action="store_true",
                    help="print the progress summary, rewrite the status CSV and exit")
    ap.add_argument("--list", action="store_true",
                    help="list the recordings that would be annotated and exit")
    ap.add_argument("--max-per-group", type=int,
                help="cap on labelled + queued recordings per group (e.g. per recordist)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.status:
        return print_status(cfg)
    files = list_files(cfg, include_done=args.all, type_filter=args.type,
            limit=args.n, max_per_group=args.max_per_group)
    if not files:
        print("Nothing to annotate with these options.")
        return
    if args.list:
        meta = load_metadata(cfg)
        for p in files:
            m = meta.get(p.name, {})
            print(f"  {p.name}  q={m.get('q', '')}  type={m.get('type', '')}, group={m.get('group', '')}")
        return
    if cfg.backend:
        matplotlib.use(cfg.backend)
    print(f"{len(files)} recordings to annotate")
    Annotator(cfg, files).run()

if __name__ == "__main__":
    main()
