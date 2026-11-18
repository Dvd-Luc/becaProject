"""Cut 3 s BirdNET training windows from long recordings using Audacity labels.

Inputs
  --audio-dir   recordings (wav/flac/ogg/mp3)
  --labels-dir  one Audacity label export per recording, same stem:
                rec01.wav  <->  rec01.txt   (start <TAB> end <TAB> label)

Outputs
  out-dir/<label>/<recording>_<start>s.wav   3 s clips, one folder per call type
  out-dir/Noise/...                          3 s clips with no labelled call
  out-dir/metadata.csv                       source file, window, source call, kind

Rules
  * call < 3 s : window of 3 s containing the whole call, random position
  * call > 3 s : windows of 3 s that overlap the call by at least TAU seconds,
                 chained from a random end of the call, consecutive windows
                 overlapping by 0..MAX_OVERLAP seconds
  * windows are shifted to stay inside the recording
  * windows containing a call of a different type are rejected
  * --runs N draws N different random windows per call (more near-duplicate clips of
    the same call, not new information); --class-runs Song=3 does it for one class only
  * noise windows are random 3 s windows at least --noise-margin s away from every
    label, not overlapping each other. --noise-per-file is a MAXIMUM: short files, or
    files crowded with calls, simply give fewer. --noise-total N then keeps a random
    subset of N over all recordings
"""
import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path

import soundfile as sf

WIN = 3.0
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}


# ---------- helpers ----------
def read_labels(path):
    """Parse an Audacity label export.

    Args:
        path (str | Path): Label file.

    Returns:
        list[tuple[float, float, str]]: (start s, end s, label).
    """
    labels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("\\"):  # skip freq-range lines
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        start = float(parts[0].replace(",", "."))
        end = float(parts[1].replace(",", "."))
        labels.append((start, end, parts[2].strip()))
    return labels


def overlap(a_start, a_end, b_start, b_end):
    """Length of the overlap of two intervals.

    Args:
        a_start (float): Start of the first interval.
        a_end (float): End of the first interval.
        b_start (float): Start of the second interval.
        b_end (float): End of the second interval.

    Returns:
        float: Overlap in seconds, 0 if disjoint.
    """
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def clamp_start(start, rec_len):
    """Shift a window so it lies inside the recording.

    Args:
        start (float): Wanted window start, seconds.
        rec_len (float): Recording length, seconds.

    Returns:
        float | None: Clamped start, None if the recording is shorter than 3 s.
    """
    if rec_len < WIN:
        return None
    return min(max(start, 0.0), rec_len - WIN)


def has_conflict(win_start, idx, labels, tol):
    """True if the window contains a call of a different type.

    Args:
        win_start (float): Window start, seconds.
        idx (int): Index of the call the window is made for.
        labels (list[tuple[float, float, str]]): All labels of the recording.
        tol (float): Overlap tolerated with another call type, seconds.

    Returns:
        bool: Whether the window must be rejected.
    """
    _, _, label = labels[idx]
    for j, (other_start, other_end, other_label) in enumerate(labels):
        shared = overlap(win_start, win_start + WIN, other_start, other_end)
        if j != idx and other_label != label and shared > tol:
            return True
    return False


def safe(name):
    """Make a label usable as a folder name.

    Args:
        name (str): Label text.

    Returns:
        str: The name with unsafe characters replaced by "_".
    """
    return re.sub(r"[^\w\-. ,]", "_", name).strip()


# ---------- window generators ----------
def short_window(idx, labels, rec_len, rng, tol, tries=20):
    """Window for a call <= 3 s: random position, whole call inside.

    Args:
        idx (int): Index of the call in labels.
        labels (list[tuple[float, float, str]]): All labels of the recording.
        rec_len (float): Recording length, seconds.
        rng (random.Random): Generator.
        tol (float): Overlap tolerated with another call type, seconds.
        tries (int): Random positions tried before giving up.

    Returns:
        list[float]: [window start], or [] if no valid position was found.
    """
    start, end, _ = labels[idx]
    for _ in range(tries):
        delta = rng.uniform(0, WIN - (end - start))      # padding placed BEFORE the call
        win_start = clamp_start(start - delta, rec_len)
        if win_start is None:
            return []
        if win_start > start or win_start + WIN < end:                 # safety: call must be inside
            continue
        if has_conflict(win_start, idx, labels, tol):
            continue
        return [win_start]
    return []


def long_windows(idx, labels, rec_len, rng, tau, max_overlap, tol):
    """Windows for a call > 3 s: a chain, each overlapping the call by >= tau.

    Args:
        idx (int): Index of the call in labels.
        labels (list[tuple[float, float, str]]): All labels of the recording.
        rec_len (float): Recording length, seconds.
        rng (random.Random): Generator.
        tau (float): Minimum part of the call in each window, seconds.
        max_overlap (float): Maximum overlap between consecutive windows, seconds.
        tol (float): Overlap tolerated with another call type, seconds.

    Returns:
        list[float]: Window starts.
    """
    start, end, _ = labels[idx]
    delta = rng.uniform(tau, WIN)                  # overlap of the first window
    if rng.random() < 0.5:                         # start from the beginning
        win_start, direction = start - (WIN - delta), +1
    else:                                          # start from the end
        win_start, direction = end - delta, -1

    starts = []
    while overlap(win_start, win_start + WIN, start, end) >= tau:      # unclamped test -> loop ends
        clamped = clamp_start(win_start, rec_len)
        if clamped is None:
            break
        if (overlap(clamped, clamped + WIN, start, end) >= tau
            and not has_conflict(clamped, idx, labels, tol)
            and all(abs(clamped - prev) > 1e-3 for prev in starts)):
            starts.append(clamped)
        win_start += direction * rng.uniform(WIN - max_overlap, WIN)
    return starts


def noise_windows(labels, rec_len, max_n, rng, margin, max_tries=200):
    """Up to n random windows away from every label and not overlapping each other.

    Returns fewer (possibly none) when the recording has too little free time, e.g. a
    short file or one full of calls.

    Args:
        labels (list[tuple[float, float, str]]): All labels of the recording.
        rec_len (float): Recording length, seconds.
        max_n (int): Maximum number of windows.
        rng (random.Random): Generator.
        margin (float): Minimum distance to any label, seconds.
        max_tries (int): Random draws per wanted window.

    Returns:
        list[float]: Window starts.
    """
    starts = []
    if rec_len < WIN:
        return starts
    for _ in range(max_tries * max(max_n, 1)):
        if len(starts) >= max_n:
            break
        win_start = rng.uniform(0, rec_len - WIN)
        if (all(overlap(win_start, win_start + WIN, start - margin, end + margin) == 0
                for start, end, _ in labels)
            and all(abs(win_start - prev) >= WIN for prev in starts)):
            starts.append(win_start)
    return starts


# ---------- I/O ----------
def save_clip(audio_path, sample_rate, win_start, folder):
    """Write the 3 s window starting at w0 to folder/<recording>_<start>s.wav.

    Args:
        audio_path (Path): Source recording.
        sample_rate (int): Its sample rate.
        win_start (float): Window start, seconds.
        folder (Path): Class folder (created if needed).

    Returns:
        Path: The clip written.
    """
    data, _ = sf.read(audio_path, start=int(round(win_start * sample_rate)),
                      frames=int(round(WIN * sample_rate)))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{audio_path.stem}_{win_start:07.2f}s.wav"
    sf.write(path, data, sample_rate)
    return path


def process(audio_path, label_path, args, rng, writer):
    """Cut the call windows of one recording and draw its noise windows.

    Call clips are written at once; noise windows are only returned so that
    --noise-total can subsample them over all recordings.

    Args:
        audio_path (Path): Recording.
        label_path (Path): Its Audacity label file.
        args (argparse.Namespace): CLI options (out_dir, runs, class_runs, tau, ...).
        rng (random.Random): Generator.
        writer (csv.writer): metadata.csv writer.

    Returns:
        list[tuple[Path, float]]: (recording, window start) of the noise windows.
    """
    info = sf.info(audio_path)
    sample_rate, rec_len = info.samplerate, info.frames / info.samplerate
    labels = read_labels(label_path)

    def emit(win_start, label, call_start, call_end, kind):
        """Save one call clip and write its metadata row.

        Args:
            win_start (float): Window start, seconds.
            label (str): Class name.
            call_start (float): Call start, seconds.
            call_end (float): Call end, seconds.
            kind (str): "short" or "long".
        """
        folder = Path(args.out_dir) / safe(label)
        clip = save_clip(audio_path, sample_rate, win_start, folder)
        writer.writerow([audio_path.name, label, f"{win_start:.3f}", f"{win_start + WIN:.3f}",
                         f"{call_start:.3f}", f"{call_end:.3f}", kind, clip])

    for idx, (start, end, label) in enumerate(labels):
        if end <= start:
            print(f"  skip point label at {start:.2f}s in {audio_path.name} "
                  "(use a region label)")
            continue
        for _ in range(args.class_runs.get(label, args.runs)):
            if end - start <= WIN:
                starts, kind = short_window(idx, labels, rec_len, rng,
                                            args.foreign_tol), "short"
            else:
                starts, kind = long_windows(idx, labels, rec_len, rng, args.tau,
                                            args.max_overlap,
                                            args.foreign_tol), "long"
            for win_start in starts:
                emit(win_start, label, start, end, kind)

    return [(audio_path, win_start) for win_start in
            noise_windows(labels, rec_len, args.noise_per_file, rng, args.noise_margin)]


def main():
    """CLI entry point: process every labelled recording, then write the noise clips and a summary."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tau", type=float, default=1.0,
                        help="min seconds of a long call kept in each window")
    parser.add_argument("--max-overlap", type=float, default=1.0,
                        help="max overlap (s) between consecutive windows of a long call")
    parser.add_argument("--runs", type=int, default=1,
                        help="random windows generated per call (all from the same file!)")
    parser.add_argument("--class-runs", nargs="*", default=[], metavar="CLASS=N",
                        help="runs for specific classes, e.g. --class-runs Song=3")
    parser.add_argument("--foreign-tol", type=float, default=0.1,
                        help="reject window if another call type overlaps more than this (s)")
    parser.add_argument("--noise-per-file", type=int, default=3,
                        help="MAXIMUM noise windows per recording (fewer if there is no room)")
    parser.add_argument("--noise-total", type=int, default=None,
                        help="keep at most this many noise windows overall (random subset)")
    parser.add_argument("--noise-margin", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.class_runs = {cls: int(runs) for cls, runs in (spec.split("=", 1) for spec in args.class_runs)}

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = n_unlabelled = 0
    noise = []
    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["recording", "label", "win_start", "win_end",
                         "call_start", "call_end", "kind", "clip"])
        for audio in sorted(Path(args.audio_dir).iterdir()):
            if audio.suffix.lower() not in AUDIO_EXT:
                continue
            label_path = Path(args.labels_dir) / f"{audio.stem}.txt"
            if not label_path.exists():
                n_unlabelled += 1
                continue
            n_files += 1
            print(audio.name)
            noise += process(audio, label_path, args, rng, writer)

        n_found = len(noise)
        if args.noise_total is not None and n_found > args.noise_total:
            noise = rng.sample(noise, args.noise_total)
        for audio, win_start in sorted(noise, key=lambda item: (item[0].name, item[1])):
            sample_rate = sf.info(audio).samplerate
            clip = save_clip(audio, sample_rate, win_start, out_dir / "Noise")
            writer.writerow([audio.name, "Noise", f"{win_start:.3f}", f"{win_start + WIN:.3f}",
                             "", "", "noise", clip])

    with open(out_dir / "metadata.csv", newline="", encoding="utf-8") as f:
        counts = Counter(row["label"] for row in csv.DictReader(f))
    print(f"\n{n_files} labelled recordings processed, "
          f"{n_unlabelled} recordings without a label file skipped")
    for label, count in sorted(counts.items()):
        print(f"  {label:<30} {count:>6} clips")
    print(f"noise windows: {n_found} possible with --noise-per-file {args.noise_per_file}, "
          f"{len(noise)} kept")


if __name__ == "__main__":
    main()
