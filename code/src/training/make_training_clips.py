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
    """Parse an Audacity label export -> list of (start, end, label)."""
    labels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("\\"):  # skip freq-range lines
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        s = float(parts[0].replace(",", "."))
        e = float(parts[1].replace(",", "."))
        labels.append((s, e, parts[2].strip()))
    return labels


def overlap(a0, a1, b0, b1):
    """Length of the overlap between [a0, a1] and [b0, b1] (0 if disjoint)."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def clamp_start(start, rec_len):
    """Shift a window so it lies inside [0, rec_len]. None if file < 3 s."""
    if rec_len < WIN:
        return None
    return min(max(start, 0.0), rec_len - WIN)


def has_conflict(w0, idx, labels, tol):
    """True if the window contains a call of a different type (> tol seconds)."""
    _, _, lab = labels[idx]
    for j, (o0, o1, olab) in enumerate(labels):
        if j != idx and olab != lab and overlap(w0, w0 + WIN, o0, o1) > tol:
            return True
    return False


def safe(name):
    """Make a label usable as a folder name."""
    return re.sub(r"[^\w\-. ,]", "_", name).strip()


# ---------- window generators ----------
def short_window(idx, labels, rec_len, rng, tol, tries=20):
    """Call <= 3 s: random position, whole call inside the window."""
    s, e, _ = labels[idx]
    for _ in range(tries):
        delta = rng.uniform(0, WIN - (e - s))      # padding placed BEFORE the call
        w0 = clamp_start(s - delta, rec_len)
        if w0 is None:
            return []
        if w0 > s or w0 + WIN < e:                 # safety: call must be inside
            continue
        if has_conflict(w0, idx, labels, tol):
            continue
        return [w0]
    return []


def long_windows(idx, labels, rec_len, rng, tau, max_overlap, tol):
    """Call > 3 s: chain of windows, each overlapping the call by >= tau."""
    s, e, _ = labels[idx]
    delta = rng.uniform(tau, WIN)                  # overlap of the first window
    if rng.random() < 0.5:                         # start from the beginning
        w0, direction = s - (WIN - delta), +1
    else:                                          # start from the end
        w0, direction = e - delta, -1

    out = []
    while overlap(w0, w0 + WIN, s, e) >= tau:      # unclamped test -> loop ends
        c = clamp_start(w0, rec_len)
        if c is None:
            break
        if (overlap(c, c + WIN, s, e) >= tau
                and not has_conflict(c, idx, labels, tol)
                and all(abs(c - o) > 1e-3 for o in out)):
            out.append(c)
        w0 += direction * rng.uniform(WIN - max_overlap, WIN)
    return out


def noise_windows(labels, rec_len, n, rng, margin, max_tries=200):
    """Up to n random windows, `margin` seconds away from every label and not
    overlapping each other. Returns fewer (possibly none) when the recording has
    too little free time, e.g. a short file or one full of calls."""
    out = []
    if rec_len < WIN:
        return out
    for _ in range(max_tries * max(n, 1)):
        if len(out) >= n:
            break
        w0 = rng.uniform(0, rec_len - WIN)
        if (all(overlap(w0, w0 + WIN, s - margin, e + margin) == 0
                for s, e, _ in labels)
                and all(abs(w0 - o) >= WIN for o in out)):
            out.append(w0)
    return out


# ---------- I/O ----------
def save_clip(audio_path, sr, w0, folder):
    """Write the 3 s window starting at w0 to folder/<recording>_<start>s.wav."""
    data, _ = sf.read(audio_path, start=int(round(w0 * sr)),
                      frames=int(round(WIN * sr)))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{audio_path.stem}_{w0:07.2f}s.wav"
    sf.write(path, data, sr)
    return path


def process(audio_path, label_path, args, rng, writer):
    """Cut the call windows of one recording and draw its noise windows.

    Call clips are written at once; noise windows are returned as [(audio_path, start)]
    so --noise-total can subsample them over all recordings.
    """
    info = sf.info(audio_path)
    sr, rec_len = info.samplerate, info.frames / info.samplerate
    labels = read_labels(label_path)

    def emit(w0, label, src0, src1, kind):
        folder = Path(args.out_dir) / safe(label)
        clip = save_clip(audio_path, sr, w0, folder)
        writer.writerow([audio_path.name, label, f"{w0:.3f}", f"{w0 + WIN:.3f}",
                         f"{src0:.3f}", f"{src1:.3f}", kind, clip])

    for idx, (s, e, label) in enumerate(labels):
        if e <= s:
            print(f"  skip point label at {s:.2f}s in {audio_path.name} "
                  "(use a region label)")
            continue
        for _ in range(args.class_runs.get(label, args.runs)):
            if e - s <= WIN:
                wins, kind = short_window(idx, labels, rec_len, rng,
                                          args.foreign_tol), "short"
            else:
                wins, kind = long_windows(idx, labels, rec_len, rng, args.tau,
                                          args.max_overlap,
                                          args.foreign_tol), "long"
            for w0 in wins:
                emit(w0, label, s, e, kind)

    return [(audio_path, w0) for w0 in
            noise_windows(labels, rec_len, args.noise_per_file, rng, args.noise_margin)]


def main():
    """CLI entry point: process every labelled recording, then write the noise clips and a summary."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tau", type=float, default=1.0,
                   help="min seconds of a long call kept in each window")
    p.add_argument("--max-overlap", type=float, default=1.0,
                   help="max overlap (s) between consecutive windows of a long call")
    p.add_argument("--runs", type=int, default=1,
                   help="random windows generated per call (all from the same file!)")
    p.add_argument("--class-runs", nargs="*", default=[], metavar="CLASS=N",
                   help="runs for specific classes, e.g. --class-runs Song=3")
    p.add_argument("--foreign-tol", type=float, default=0.1,
                   help="reject window if another call type overlaps more than this (s)")
    p.add_argument("--noise-per-file", type=int, default=3,
                   help="MAXIMUM noise windows per recording (fewer if there is no room)")
    p.add_argument("--noise-total", type=int, default=None,
                   help="keep at most this many noise windows overall (random subset)")
    p.add_argument("--noise-margin", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.class_runs = {k: int(v) for k, v in (x.split("=", 1) for x in args.class_runs)}

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_files = n_unlabelled = 0
    noise = []
    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["recording", "label", "win_start", "win_end",
                    "call_start", "call_end", "kind", "clip"])
        for audio in sorted(Path(args.audio_dir).iterdir()):
            if audio.suffix.lower() not in AUDIO_EXT:
                continue
            lab = Path(args.labels_dir) / f"{audio.stem}.txt"
            if not lab.exists():
                n_unlabelled += 1
                continue
            n_files += 1
            print(audio.name)
            noise += process(audio, lab, args, rng, w)

        n_found = len(noise)
        if args.noise_total is not None and n_found > args.noise_total:
            noise = rng.sample(noise, args.noise_total)
        for audio, w0 in sorted(noise, key=lambda x: (x[0].name, x[1])):
            sr = sf.info(audio).samplerate
            clip = save_clip(audio, sr, w0, out / "Noise")
            w.writerow([audio.name, "Noise", f"{w0:.3f}", f"{w0 + WIN:.3f}",
                        "", "", "noise", clip])

    with open(out / "metadata.csv", newline="", encoding="utf-8") as f:
        counts = Counter(r["label"] for r in csv.DictReader(f))
    print(f"\n{n_files} labelled recordings processed, "
          f"{n_unlabelled} recordings without a label file skipped")
    for lab, n in sorted(counts.items()):
        print(f"  {lab:<30} {n:>6} clips")
    print(f"noise windows: {n_found} possible with --noise-per-file {args.noise_per_file}, "
          f"{len(noise)} kept")


if __name__ == "__main__":
    main()