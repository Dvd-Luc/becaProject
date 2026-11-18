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
    """Annotator settings: annotator_config.yaml keys, or viewer + annotator in project.yaml.

    Attributes:
        audio_dir (str): Folder of the recordings.
        labels_dir (str): Folder of the Audacity label files.
        categories (list[str]): Label names, bound to keys 1..9, 0.
        n_fft (int): FFT size of the spectrogram.
        hop_length (int): Hop between spectrogram frames, in samples.
        f_min (float): Lowest frequency shown, Hz.
        f_max (float): Highest frequency shown, Hz.
        view_seconds (float): Initial view width, seconds.
        db_range (float): Colour range below the maximum, dB.
        cmap (str): Matplotlib colormap.
        normalize_playback (bool): Normalise the played segment to 0.8 full scale.
        backend (str): Matplotlib backend ("" = default).
        metadata_csv (str): Optional XC metadata CSV (ordering, --type, status).
        file_col (str): Its file-name column.
        q_col (str): Its quality column.
        type_col (str): Its sound-type column.
        status_file (str): Status CSV rewritten after each save.
        group_cols (list[str]): CSV columns to balance the queue across (e.g. [rec]).
    """
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
    # CSV columns to balance across when selecting files (e.g. [rec, loc]); empty = off
    group_cols: list = field(default_factory=list)

def load_config(path):
    """Load the annotator settings from the standalone file or the unified project.yaml.

    Args:
        path (str | Path): annotator_config.yaml (flat keys) or project.yaml (viewer +
            annotator sections merged).

    Returns:
        Config: The settings.

    Raises:
        SystemExit: On unknown keys (a new key must also be a Config field) or a wrong
            number of categories.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "annotator" in raw:                        # unified project.yaml
        raw = {**raw.get("viewer", {}), **raw["annotator"]}
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise SystemExit(f"unknown keys in {path}: {sorted(unknown)}")
    cfg = Config(**raw)
    if not 1 <= len(cfg.categories) <= len(CAT_KEYS):
        raise SystemExit(f"'categories' needs between 1 and {len(CAT_KEYS)} entries")
    return cfg


def read_labels(path):
    """Read an Audacity label file, sorted by start.

    Frequency-range lines (starting with a backslash) are skipped and decimal commas
    accepted.

    Args:
        path (str | Path): Label file.

    Returns:
        list[dict]: Labels with start (float s), end (float s) and cat (str).
    """
    labels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("\\"):      # skip frequency-range lines
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            labels.append({"start": float(parts[0].replace(",", ".")),
                           "end": float(parts[1].replace(",", ".")),
                           "cat": parts[2].strip()})
    return sorted(labels, key=lambda label: label["start"])


def write_labels(path, labels):
    """Write labels in Audacity format (start TAB end TAB label), sorted by start.

    Args:
        path (str | Path): Label file (parent folders are created).
        labels (list[dict]): Labels with start, end and cat.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for label in sorted(labels, key=lambda label: label["start"]):
            f.write(f"{label['start']:.6f}\t{label['end']:.6f}\t{label['cat']}\n")


def load_metadata(cfg):
    """Recording info from the metadata CSV.

    Args:
        cfg (Config): Settings (metadata_csv, file_col, q_col, type_col, group_cols).

    Returns:
        dict[str, dict[str, str]]: File name -> {q, type, group}; {} if no CSV is configured.

    Raises:
        SystemExit: If a configured column is missing from the CSV.
    """
    if not cfg.metadata_csv:
        return {}
    with open(cfg.metadata_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and cfg.file_col not in rows[0]:
        raise SystemExit(f"column '{cfg.file_col}' not found in {cfg.metadata_csv} "
                         f"(columns: {list(rows[0])})")
    for col in cfg.group_cols:
        if rows and col not in rows[0]:
            raise SystemExit(f"group column '{col}' not found in {cfg.metadata_csv} "
                             f"(columns: {list(rows[0])})")
    group = lambda row: "|".join((row.get(col) or "").strip() for col in cfg.group_cols)
    return {row[cfg.file_col].strip(): {"q": (row.get(cfg.q_col) or "").strip().upper(),
                                        "type": row.get(cfg.type_col) or "",
                                        "group": group(row)}
            for row in rows}


def label_status(cfg, name):
    """Annotation status of one recording.

    Args:
        cfg (Config): Settings (labels_dir).
        name (str): Recording file name.

    Returns:
        tuple[str, list[str]]: ("todo" | "done" | "no_calls", label names in its file).
    """
    label_path = Path(cfg.labels_dir) / f"{Path(name).stem}.txt"
    if not label_path.exists():
        return "todo", []
    names = [label["cat"] for label in read_labels(label_path)]
    return ("done" if names else "no_calls"), names


def balance_groups(files, info, done, max_per_group=None):
    """Interleave files across groups, always taking from the least-labelled group.

    Order inside a group is kept (quality first).

    Args:
        files (list[Path]): Recordings to queue.
        info (Callable[[Path], dict]): File -> {q, type, group}.
        done (Counter): Recordings already labelled per group.
        max_per_group (int | None): Cap on labelled + queued recordings per group.

    Returns:
        list[Path]: The queue order.
    """
    queues = {}
    for path in files:
        queues.setdefault(info(path)["group"], []).append(path)
    count = Counter(done)
    if max_per_group:
        queues = {group: group_files[:max(0, max_per_group - count[group])]
                  for group, group_files in queues.items()}
    queues = {group: group_files for group, group_files in queues.items() if group_files}
    order = []
    while queues:
        group = min(queues, key=lambda group: (count[group], group)) # Example : (2, "B") vs (5, "A")
        order.append(queues[group].pop(0))
        count[group] += 1
        if not queues[group]:
            del queues[group]
    return order


def list_files(cfg, include_done=False, type_filter=None, limit=None, max_per_group=None):
    """Recordings to annotate, best quality first.

    Args:
        cfg (Config): Settings.
        include_done (bool): Also list recordings that already have a label file.
        type_filter (str | None): Keep recordings whose CSV type contains this text.
        limit (int | None): Return at most this many files.
        max_per_group (int | None): Cap per group when balancing across group_cols.

    Returns:
        list[Path]: Recordings in annotation order.

    Raises:
        SystemExit: If type_filter is used without metadata_csv.
    """
    meta = load_metadata(cfg)
    files = [path for path in Path(cfg.audio_dir).iterdir() if path.suffix.lower() in AUDIO_EXT]
    info = lambda path: meta.get(path.name, {"q": "", "type": "", "group": ""})
    if meta:
        unlisted = [path.name for path in files if path.name not in meta]
        if unlisted:
            print(f"note: {len(unlisted)} audio files are not in the CSV "
                  f"(placed last), e.g. {unlisted[:3]}")
        absent = [name for name in meta if not (Path(cfg.audio_dir) / name).exists()]
        if absent:
            print(f"note: {len(absent)} CSV recordings not found in audio_dir, "
                  f"e.g. {absent[:3]}")
    files.sort(key=lambda path: (info(path)["q"] == "", info(path)["q"], path.name))   # A before B

    status = {path: label_status(cfg, path.name)[0] for path in files}
    done = Counter(info(path)["group"] for path in files if status[path] != "todo")

    if type_filter:
        if not meta:
            raise SystemExit("--type needs metadata_csv in the config")
        files = [path for path in files if type_filter.lower() in info(path)["type"].lower()]
    if not include_done:
        files = [path for path in files if status[path] == "todo"]
        if cfg.group_cols:
            files = balance_groups(files, info, done, max_per_group)
    return files[:limit] if limit else files


def build_status(cfg):
    """One status row per recording (CSV recordings first, then other audio files).

    Args:
        cfg (Config): Settings.

    Returns:
        list[dict]: Rows with recording, q, type, group, status, n_labels, label_names.
    """
    meta = load_metadata(cfg)
    names = list(meta) + sorted(path.name for path in Path(cfg.audio_dir).iterdir()
                                if path.suffix.lower() in AUDIO_EXT and path.name not in meta)
    rows = []
    for name in names:
        status, label_names = label_status(cfg, name)
        rec_meta = meta.get(name, {"q": "", "type": "", "group": ""})
        rows.append({"recording": name, "q": rec_meta["q"], "type": rec_meta["type"],
                     "group": rec_meta["group"],
                     "status": status, "n_labels": len(label_names),
                     "label_names": ";".join(sorted(set(label_names)))})
    return rows


def write_status(cfg):
    """Rewrite the status CSV.

    Args:
        cfg (Config): Settings (status_file).

    Returns:
        list[dict]: The rows written (see build_status).
    """
    rows = build_status(cfg)
    if rows:
        with open(cfg.status_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def print_status(cfg):
    """Rewrite the status CSV and print progress, label counts and unknown label names.

    Args:
        cfg (Config): Settings.
    """
    rows = write_status(cfg)
    print(f"{len(rows)} recordings")
    print("by status:", dict(Counter(row["status"] for row in rows)))
    if any(row["q"] for row in rows):
        print("by quality:", dict(Counter(row["q"] for row in rows)))
        by_q_status = Counter((row["q"], row["status"]) for row in rows)
        print("quality x status:", {f"{quality}/{status}": count
                                    for (quality, status), count in sorted(by_q_status.items())})
    total = Counter()
    for label_path in Path(cfg.labels_dir).glob("*.txt"):
        total.update(label["cat"] for label in read_labels(label_path))
    if cfg.group_cols:
        group_counts = Counter(row["group"] for row in rows if row["status"] != "todo")
        print(f"labelled recordings per group ({len(group_counts)} groups):",
              dict(group_counts.most_common()))
    print("labels per class:", dict(total))
    unknown_names = {name: count for name, count in total.items() if name not in cfg.categories}
    print("UNKNOWN label names (typos?):", unknown_names or "none")
    print(f"saved {cfg.status_file}")


def compute_spectrogram(audio, sample_rate, n_fft, hop, f_min, f_max):
    """Magnitude spectrogram in dB, only the bins between f_min and f_max.

    Args:
        audio (numpy.ndarray): Mono samples.
        sample_rate (int): Sample rate.
        n_fft (int): FFT size.
        hop (int): Hop between frames, in samples.
        f_min (float): Lowest frequency kept, Hz.
        f_max (float): Highest frequency kept, Hz.

    Returns:
        tuple[numpy.ndarray, float, float]: (dB array of shape (bins, frames),
            lowest and highest frequency of the kept bins in Hz).
    """
    bin_hz = sample_rate / n_fft
    bin_lo = int(f_min // bin_hz)
    bin_hi = min(int(f_max // bin_hz) + 1, n_fft // 2 + 1)
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    n_frames = 1 + (len(audio) - n_fft) // hop
    window_fn = np.hanning(n_fft).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(audio, n_fft)
    spec = np.empty((bin_hi - bin_lo, n_frames), dtype=np.float32)
    for chunk_start in range(0, n_frames, 1500):                    # chunks keep memory low
        chunk_end = min(chunk_start + 1500, n_frames)
        frames = windows[chunk_start * hop:(chunk_end - 1) * hop + 1:hop] * window_fn
        mag = np.abs(np.fft.rfft(frames, axis=1))[:, bin_lo:bin_hi]
        spec[:, chunk_start:chunk_end] = 20 * np.log10(mag.T + 1e-10)
    return spec, bin_lo * bin_hz, bin_hi * bin_hz


# ---------------------------------------------------------------- application
class Annotator:
    """Matplotlib spectrogram window to label time regions, one recording at a time.

    Attributes:
        cfg (Config): Settings.
        files (list[Path]): Recordings of this session.
        idx (int): Index of the recording shown.
        labels (list[dict]): Committed labels of that recording (start, end, cat).
        edit_orig (dict | None): Label taken back for editing, restored by Esc.
        audio (numpy.ndarray): Mono samples of the recording.
        sample_rate (int): Sample rate.
        duration (float): Recording length, seconds.
        spec (numpy.ndarray): Its dB spectrogram.
        view (tuple[float, float]): Visible time range, seconds.
    """
    def __init__(self, cfg, files):
        """Build the figure, widgets and key bindings, then load the first file.

        Args:
            cfg (Config): Settings.
            files (list[Path]): Recordings to annotate, in order.
        """
        import matplotlib.pyplot as plt
        self.plt = plt
        for key in [key for key in plt.rcParams if key.startswith("keymap.")]:
            plt.rcParams[key] = []                           # free the keys for our shortcuts

        self.cfg, self.files, self.idx = cfg, files, 0
        self.meta = load_metadata(cfg)
        cmap = matplotlib.colormaps["tab10"]
        self.colors = {cat: cmap(i % 10) for i, cat in enumerate(cfg.categories)}
        self.edit_orig = None
        self._label_artists = []
        self._playing = False
        self._timer = None

        self.fig = plt.figure(figsize=(14, 7.5))
        self.ax = self.fig.add_axes([0.06, 0.17, 0.74, 0.73])
        slider_ax = self.fig.add_axes([0.06, 0.08, 0.74, 0.03])
        self.slider = Slider(slider_ax, "", 0.0, 1.0, valinit=0.0, valfmt="%.1f s")
        self.slider.on_changed(lambda value: self.set_view(value, self.view[1] - self.view[0]))
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

        handles = [Patch(facecolor=self.colors[cat], alpha=0.6, label=f"[{CAT_KEYS[i]}]  {cat}")
                   for i, cat in enumerate(cfg.categories)]
        self.ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       frameon=False, title="Categories")
        self.fig.text(0.06, 0.975, HELP, fontsize=8, va="top")
        self.msg = self.fig.text(0.06, 0.02, "", fontsize=10, color="darkred")

        canvas = self.fig.canvas
        canvas.mpl_connect("key_press_event", self._on_key)
        canvas.mpl_connect("scroll_event", self._on_scroll)
        canvas.mpl_connect("button_press_event", self._on_click)
        self.load_file(0)

    # ---- loading / saving
    def _txt(self):
        """Label file of the current recording.

        Returns:
            Path: <labels_dir>/<stem>.txt.
        """
        return Path(self.cfg.labels_dir) / f"{self.files[self.idx].stem}.txt"

    def _partial(self):
        """Autosave file of the current recording.

        Returns:
            Path: <labels_dir>/<stem>.partial.
        """
        return self._txt().with_suffix(".partial")

    def load_file(self, i):
        """Load recording i, compute its spectrogram and restore its labels.

        Unsaved work (.partial) takes precedence over the saved .txt.

        Args:
            i (int): Index in self.files.
        """
        self._stop_play()
        self.idx = i
        path = self.files[i]
        self.span.clear()
        self.edit_orig = None
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        self.audio, self.sample_rate = audio.mean(axis=1), sample_rate
        self.duration = len(self.audio) / sample_rate
        cfg = self.cfg
        self.spec, self.f_lo, self.f_hi = compute_spectrogram(
            self.audio, sample_rate, cfg.n_fft, cfg.hop_length, cfg.f_min, min(cfg.f_max, sample_rate / 2))
        self.vmax = float(np.percentile(self.spec[:, ::max(1, self.spec.shape[1] // 2000)], 99.5))
        self.db_range = cfg.db_range
        if self._partial().exists():
            self.labels, note = read_labels(self._partial()), "restored unsaved work"
        elif self._txt().exists():
            self.labels, note = read_labels(self._txt()), "loaded saved labels"
        else:
            self.labels, note = [], ""
        self.view = (0.0, min(cfg.view_seconds, self.duration))
        self.refresh()
        self._msg(note)
        try:
            self.fig.canvas.manager.set_window_title(f"{path.name} [{i + 1}/{len(self.files)}]")
        except Exception:
            pass

    def _autosave(self):
        """Save the current labels to the .partial file."""
        write_labels(self._partial(), self.labels)

    def save_and_next(self):
        """Save the labels and go to the next recording (refuses an uncommitted selection)."""
        if self.get_selection() is not None:
            return self._msg("Uncommitted selection: press a category key or Backspace first")
        if not self.labels:
            return self._msg("No labels: press x to confirm that this recording has no calls")
        self._finish()

    def mark_empty(self):
        """Save an empty label file ("listened, no calls") and go to the next recording."""
        if self.labels or self.get_selection() is not None:
            return self._msg("This recording has labels: delete them first if it has no calls")
        self._finish()

    def _finish(self):
        """Write the .txt, drop the .partial, update the status CSV and load the next file."""
        write_labels(self._txt(), self.labels)
        self._partial().unlink(missing_ok=True)
        try:
            write_status(self.cfg)
        except Exception as err:
            print(f"could not update {self.cfg.status_file}: {err}")
        done = self.files[self.idx].name
        if self.idx + 1 < len(self.files):
            self.load_file(self.idx + 1)
            self._msg(f"saved {done}")
        else:
            self._msg(f"saved {done}. That was the last recording: all done")

    def go_back(self):
        """Load the previous recording."""
        if self.idx > 0:
            self.load_file(self.idx - 1)

    # ---- selection and labels
    def _on_select(self, vmin, vmax):
        """SpanSelector callback: report the new selection.

        Args:
            vmin (float): Selection start, seconds.
            vmax (float): Selection end, seconds.
        """
        self._msg(f"selection {vmin:.2f}-{vmax:.2f} s ({vmax - vmin:.2f} s): "
                  "press a category key, Space to listen")

    def get_selection(self):
        """Current selection, clipped to the recording.

        Returns:
            tuple[float, float] | None: (start, end) in seconds, None if nothing (or less
                than MIN_SEL) is selected.
        """
        if not self.span.get_visible():
            return None
        start, end = self.span.extents
        start, end = max(0.0, start), min(self.duration, end)
        return (start, end) if end - start >= MIN_SEL else None

    def commit(self, cat_idx):
        """Turn the selection into a label.

        Args:
            cat_idx (int | None): Index in cfg.categories; None keeps the category of the
                label being edited.
        """
        selection = self.get_selection()
        if selection is None:
            return self._msg("No selection: drag on the spectrogram first")
        if cat_idx is None:
            if self.edit_orig is None:
                return self._msg("Press a category key (1-9) to label the selection")
            cat = self.edit_orig["cat"]
        else:
            cat = self.cfg.categories[cat_idx]
        self.labels.append({"start": selection[0], "end": selection[1], "cat": cat})
        self.labels.sort(key=lambda label: label["start"])
        self.span.clear()
        self.edit_orig = None
        self._autosave()
        self.refresh()
        self._msg(f"labelled {selection[0]:.2f}-{selection[1]:.2f} s as {cat}")

    def delete_selection(self):
        """Clear the selection; if it was a label being edited, that label is deleted."""
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
        """Cancel the selection, restoring a label being edited."""
        if self.edit_orig is not None:
            self.labels.append(self.edit_orig)
            self.labels.sort(key=lambda label: label["start"])
            self.edit_orig = None
        self.span.clear()
        self.refresh()

    def _on_click(self, event):
        """Right-click on a label: take it back into the selection to adjust or relabel it.

        Args:
            event (matplotlib.backend_bases.MouseEvent): The click.
        """
        if event.button != 3 or event.inaxes is not self.ax or event.xdata is None:
            return
        if self.get_selection() is not None:
            return self._msg("Commit or delete the current selection first")
        hits = [label for label in self.labels if label["start"] <= event.xdata <= label["end"]]
        if not hits:
            return
        label = min(hits, key=lambda label: label["end"] - label["start"])
        self.labels.remove(label)
        self.edit_orig = dict(label)
        self.span.extents = (label["start"], label["end"])
        self.span.set_visible(True)
        self.refresh()
        self._msg(f"editing '{label['cat']}': adjust the edges, then press a category key "
                  "or Enter (Backspace deletes, Esc cancels)")

    # ---- view
    def set_view(self, start, length):
        """Show a time range, kept inside the recording and the zoom limits.

        Args:
            start (float): Start, seconds.
            length (float): Width, seconds.
        """
        length = min(max(length, MIN_VIEW), self.duration)
        start = min(max(start, 0.0), self.duration - length)
        self.view = (start, start + length)
        self.refresh()

    def zoom(self, factor, center=None):
        """Zoom the time axis.

        Args:
            factor (float): New width / old width (< 1 zooms in).
            center (float | None): Time kept fixed, seconds (None: centre of the view).
        """
        start, end = self.view
        length = end - start
        pivot = (start + end) / 2 if center is None else center
        new_length = min(max(length * factor, MIN_VIEW), self.duration)
        self.set_view(pivot - (pivot - start) / length * new_length, new_length)

    def pan(self, frac):
        """Scroll the view.

        Args:
            frac (float): Shift as a fraction of the view width (negative = left).
        """
        start, end = self.view
        self.set_view(start + frac * (end - start), end - start)

    def _on_scroll(self, event):
        """Mouse wheel: zoom around the cursor; shift + wheel: scroll.

        Args:
            event (matplotlib.backend_bases.MouseEvent): The scroll event.
        """
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if event.key == "shift":
            self.pan(0.1 if event.button == "up" else -0.1)
        else:
            self.zoom(0.8 if event.button == "up" else 1.25, center=event.xdata)

    # ---- drawing
    def refresh(self):
        """Redraw the spectrogram, labels, title and slider for the current view."""
        self._draw_image()
        self._draw_labels()
        start, end = self.view
        rec_meta = self.meta.get(self.files[self.idx].name)
        tag = f"   q={rec_meta['q']} type={rec_meta['type']}" if rec_meta else ""
        self.ax.set_title(
            f"[{self.idx + 1}/{len(self.files)}] {self.files[self.idx].name}{tag}   "
            f"{self.duration:.1f} s   {len(self.labels)} labels   view {start:.1f}-{end:.1f} s",
            loc="left", fontsize=10)
        slider_max = max(self.duration - (end - start), 1e-3)
        self.slider.eventson = False
        self.slider.valmax = slider_max
        self.slider.ax.set_xlim(0, slider_max)
        self.slider.set_val(start)
        self.slider.eventson = True
        self.fig.canvas.draw_idle()

    def _draw_image(self):
        """Draw the visible part of the spectrogram, max-pooled to MAX_COLS columns."""
        start, end = self.view
        sample_rate, hop, n_fft = self.sample_rate, self.cfg.hop_length, self.cfg.n_fft
        n_frames = self.spec.shape[1]
        col_start = max(0, int((start * sample_rate - n_fft / 2) // hop))
        col_end = min(n_frames, int(np.ceil((end * sample_rate - n_fft / 2) / hop)) + 1)
        visible = self.spec[:, col_start:col_end]
        step = max(1, visible.shape[1] // MAX_COLS)
        if step > 1:                                        # max-pool: keep short calls visible
            n_kept = visible.shape[1] // step * step
            visible = visible[:, :n_kept].reshape(visible.shape[0], -1, step).max(axis=2)
        x_left = (col_start * hop + n_fft / 2) / sample_rate - hop / (2 * sample_rate)
        x_right = x_left + visible.shape[1] * step * hop / sample_rate
        self.im.set_data(visible)
        self.im.set_extent([x_left, x_right, self.f_lo, self.f_hi])
        self.im.set_clim(self.vmax - self.db_range, self.vmax)
        self.ax.set_xlim(start, end)
        self.ax.set_ylim(self.f_lo, self.f_hi)

    def _draw_labels(self):
        """Draw the labels that overlap the view as coloured spans with their name."""
        for artist in self._label_artists:
            artist.remove()
        self._label_artists = []
        start, end = self.view
        for label in self.labels:
            if label["end"] < start or label["start"] > end:
                continue
            color = self.colors.get(label["cat"], (0.5, 0.5, 0.5, 1.0))
            self._label_artists.append(
                self.ax.axvspan(label["start"], label["end"], color=color, alpha=0.3, lw=0))
            self._label_artists.append(self.ax.text(
                max(label["start"], start), 0.98, label["cat"], transform=self.ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=9, color="white", clip_on=True,
                bbox=dict(facecolor=color, alpha=0.8, pad=1.5, lw=0)))
        self.ax.set_xlim(start, end)

    def _msg(self, text):
        """Show a message under the spectrogram.

        Args:
            text (str): Message text ("" clears it).
        """
        self.msg.set_text(text)
        self.fig.canvas.draw_idle()

    # ---- playback
    def toggle_play(self):
        """Play the selection (or the visible part) with a moving playhead; again to stop."""
        if self._playing:
            return self._stop_play()
        selection = self.get_selection()
        start, end = selection if selection else self.view
        segment = self.audio[int(start * self.sample_rate):int(end * self.sample_rate)].copy()
        if len(segment) == 0:
            return
        if self.cfg.normalize_playback and np.abs(segment).max() > 0:
            segment *= 0.8 / np.abs(segment).max()
        try:
            import sounddevice as sd
            sd.stop()
            sd.play(segment, self.sample_rate)
        except Exception as err:                              # no audio device / sounddevice
            return self._msg(f"cannot play audio: {err}")
        self._playing = True
        self._play_from, self._play_len, self._play_clock = start, end - start, time.time()
        self._timer = self.fig.canvas.new_timer(interval=60)
        self._timer.add_callback(self._tick)
        self._timer.start()

    def _tick(self):
        """Timer callback: move the playhead, stop at the end of the segment."""
        elapsed = time.time() - self._play_clock
        if elapsed >= self._play_len:
            return self._stop_play()
        self.playhead.set_xdata([self._play_from + elapsed] * 2)
        self.playhead.set_visible(True)
        self.fig.canvas.draw_idle()

    def _stop_play(self):
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
        self.playhead.set_visible(False)
        self.fig.canvas.draw_idle()

    # ---- keyboard
    def _on_key(self, event):
        """Dispatch a key press to the matching action (see Controls in the module docstring).

        Args:
            event (matplotlib.backend_bases.KeyEvent): The key event.
        """
        key, n_cats = event.key, len(self.cfg.categories)
        if key is None:
            return
        if len(key) == 1 and key in CAT_KEYS[:n_cats]:
            self.commit(CAT_KEYS.index(key))
        elif key == "enter":
            self.commit(None)
        elif key == " ":
            self.toggle_play()
        elif key in ("backspace", "delete"):
            self.delete_selection()
        elif key == "escape":
            self.cancel()
        elif key == "left":
            self.pan(-0.25)
        elif key == "right":
            self.pan(0.25)
        elif key in ("up", "+", "="):
            self.zoom(0.7)
        elif key in ("down", "-"):
            self.zoom(1 / 0.7)
        elif key == "[":
            self.db_range = max(10.0, self.db_range - 5)
            self.refresh()
        elif key == "]":
            self.db_range += 5
            self.refresh()
        elif key == "s":
            self.save_and_next()
        elif key == "x":
            self.mark_empty()
        elif key == "b":
            self.go_back()

    def run(self):
        """Open the window (blocks until it is closed)."""
        self.plt.show()


def main():
    """CLI entry point: --status / --list, or open the annotator on the selected recordings."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--all", action="store_true", help="include already labelled recordings")
    parser.add_argument("--type", help="only recordings whose CSV 'type' contains this text")
    parser.add_argument("--n", type=int, help="only the next N recordings to annotate")
    parser.add_argument("--status", action="store_true",
                        help="print the progress summary, rewrite the status CSV and exit")
    parser.add_argument("--list", action="store_true",
                        help="list the recordings that would be annotated and exit")
    parser.add_argument("--max-per-group", type=int,
                        help="cap on labelled + queued recordings per group (e.g. per recordist)")
    args = parser.parse_args()
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
        for path in files:
            rec_meta = meta.get(path.name, {})
            print(f"  {path.name}  q={rec_meta.get('q', '')}  type={rec_meta.get('type', '')}, "
                  f"group={rec_meta.get('group', '')}")
        return
    if cfg.backend:
        matplotlib.use(cfg.backend)
    print(f"{len(files)} recordings to annotate")
    Annotator(cfg, files).run()

if __name__ == "__main__":
    main()
