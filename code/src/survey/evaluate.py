"""Compare model generations on the blind heldout test set (review.py --queue test).

    python src/survey/evaluate.py --gens 0 1
    python src/survey/evaluate.py --gens 0 1 2 --boot 1000

Every generation scores exactly the same 3 s windows, so the numbers are comparable:
  1. the judged test windows are cut once to <review.dir>/test_windows/<item_id>.wav
  2. each generations.dir/gen_K/model.tflite scores them (BirdNET, min_conf 0.01) into
     gen_K/test_scores/ (kept; delete it to re-score)
  3. per class, plus "any" (any snipe class vs Noise): average precision (AP) and
     precision / recall at generations.eval_thresholds, with 95% bootstrap intervals
     (items resampled within strata) and the paired difference to the first generation.

Weights: the test windows were sampled per v0-score stratum, so each judged window
stands for population / n_judged windows of its stratum (test_strata.csv). Without
weights, rare high-score windows would be over-represented. Unsure verdicts are left out.
Windows judged Noise are negatives for every class.

Writes results/survey/compare.csv, results/survey/pr_<class>.png.

Run from code/.
"""
import argparse
import csv
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import load

WIN = 3.0
warnings.filterwarnings("ignore", "All-NaN slice")   # precision is NaN when nothing passes
NOISE, UNSURE = "Noise", "Unsure"
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


def cut_windows(items, wdir):
    """Cut each judged test window once to wdir/<item_id>.wav (existing files are kept).

    Args:
        items (list[dict]): Test queue rows.
        wdir (Path): <review.dir>/test_windows.
    """
    wdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for it in items:
        dst = wdir / f"{it['item_id']}.wav"
        if dst.exists():
            continue
        info = sf.info(it["clip_path"])
        x, sr = sf.read(it["clip_path"], start=int(float(it["start_s"]) * info.samplerate),
                        frames=int(WIN * info.samplerate), dtype="float32")
        sf.write(dst, x, sr, subtype=info.subtype)
        n += 1
    if n:
        print(f"cut {n} test windows into {wdir}")


def score(model, wdir, sdir, items, threads):
    """Scores of one model on the test windows; runs BirdNET if results are missing.

    Args:
        model (Path): The generation's model.tflite.
        wdir (Path): Folder of the cut test windows.
        sdir (Path): The generation's test_scores/ folder.
        items (list[dict]): Test queue rows to score.
        threads (int | None): BirdNET CPU threads (None: BirdNET's default).

    Returns:
        dict[str, dict[str, float]]: item_id -> {label: score}; labels below 0.01 absent.

    Raises:
        SystemExit: If BirdNET produced no result for an item.
    """
    todo = [it for it in items if not (sdir / f"{it['item_id']}.BirdNET.results.csv").exists()]
    if todo:
        cmd = [sys.executable, "-m", "birdnet_analyzer.analyze", str(wdir), "-o", str(sdir),
               "-c", str(model), "--min_conf", "0.01", "--overlap", "0", "--rtype", "csv"]
        if threads:
            cmd += ["-t", str(threads)]
        print(f"scoring {len(items)} windows with {model}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    out = {}
    for it in items:
        p = sdir / f"{it['item_id']}.BirdNET.results.csv"
        if not p.exists():
            raise SystemExit(f"no BirdNET result for {it['item_id']} in {sdir}")
        s = defaultdict(float)
        for w in read_dicts(p):
            if float(w["Start (s)"]) < 0.5:              # the one 3 s segment
                lab = w["Common name"] or w["Scientific name"]
                s[lab] = max(s[lab], float(w["Confidence"]))
        out[it["item_id"]] = s
    return out


def ap_pr(y, s, w, thresholds):
    """Weighted average precision, and precision / recall per threshold.

    Args:
        y (numpy.ndarray): 1.0 for positive windows, 0.0 otherwise.
        s (numpy.ndarray): Model scores.
        w (numpy.ndarray): Window weights (stratum population / judged windows).
        thresholds (list[float]): Score thresholds.

    Returns:
        tuple[float, list[tuple[float, float]]]: (AP, [(precision, recall)] per
            threshold); NaN where undefined (no positives, nothing above a threshold).
    """
    pos = (w * y).sum()
    if pos == 0:
        return np.nan, [(np.nan, np.nan)] * len(thresholds)
    o = np.argsort(-s, kind="stable")
    tp, fp = np.cumsum((w * y)[o]), np.cumsum((w * (1 - y))[o])
    prec = tp / np.maximum(tp + fp, 1e-12)
    ap = float((prec * (w * y)[o]).sum() / pos)
    pr = []
    for t in thresholds:
        m = s >= t
        sel = (w * m).sum()
        pr.append(((w * y * m).sum() / sel if sel else np.nan, (w * y * m).sum() / pos))
    return ap, pr


def pr_curve(y, s, w):
    """Weighted recall and precision along the ranking by score, for plotting.

    Args:
        y (numpy.ndarray): 1.0 for positive windows, 0.0 otherwise.
        s (numpy.ndarray): Model scores.
        w (numpy.ndarray): Window weights.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: (recall, precision), one value per window.
    """
    o = np.argsort(-s, kind="stable")
    tp, fp = np.cumsum((w * y)[o]), np.cumsum((w * (1 - y))[o])
    return tp / max((w * y).sum(), 1e-12), tp / np.maximum(tp + fp, 1e-12)


def main():
    """CLI entry point: score the test windows with each generation, compute metrics, plot."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="config/project.yaml")
    ap.add_argument("--gens", type=int, nargs="+", required=True)
    ap.add_argument("--review-dir", help="default: review.dir")
    ap.add_argument("--gen-dir", help="default: generations.dir")
    ap.add_argument("--out", default="results/survey")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples (0 = off)")
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    cfg = load(a.config)
    g = cfg["generations"]
    rdir, gdir, out = Path(a.review_dir or cfg["review"]["dir"]), Path(a.gen_dir or g["dir"]), Path(a.out)
    thresholds = sorted(set(g.get("eval_thresholds", [0.35, 0.5, 0.7])
                            + [cfg["detection"]["min_conf"]]))

    queue = {r["item_id"]: r for r in read_dicts(rdir / "test_queue.csv")}
    jp = rdir / "test_judgments.csv"
    if not jp.exists():
        raise SystemExit(f"no test judgments yet: {jp}")
    verdict = {r["item_id"]: r["verdict"] for r in read_dicts(jp)}
    items = [queue[i] for i in verdict if i in queue and verdict[i] != UNSURE]
    pop = {r["stratum"]: int(r["population"]) for r in read_dicts(rdir / "test_strata.csv")}
    n_j = Counter(it["stratum"] for it in items)
    w = np.array([pop[it["stratum"]] / n_j[it["stratum"]] for it in items])
    strata = np.array([it["stratum"] for it in items])
    v = [verdict[it["item_id"]] for it in items]
    print(f"{len(items)} judged test windows (unsure left out): {dict(Counter(v))}")
    print("judged per stratum:", {s: f"{n_j[s]}/{pop[s]}" for s in pop})

    wdir = rdir / "test_windows"
    cut_windows(items, wdir)
    models = {k: gdir / f"gen_{k}" / "model.tflite" for k in a.gens}
    for k, m in models.items():
        if not m.exists():
            raise SystemExit(f"{m} not found (make_generation.py --gen {k}; gen_0 is linked "
                             f"by any make_generation.py run)")
    labels = [l.strip() for l in models[a.gens[0]].with_name("model_Labels.txt")
              .read_text(encoding="utf-8").splitlines() if l.strip()]
    scores = {k: score(m, wdir, gdir / f"gen_{k}" / "test_scores", items, a.threads)
              for k, m in models.items()}

    targets = {c: (np.array([x == c for x in v], float),
                   {k: np.array([scores[k][it["item_id"]].get(c, 0.0) for it in items])
                    for k in a.gens})
               for c in labels}
    targets["any"] = (np.array([x != NOISE for x in v], float),
                      {k: np.array([max(scores[k][it["item_id"]].values(), default=0.0)
                                    for it in items]) for k in a.gens})

    rng = np.random.default_rng(0)
    groups = {s: np.where(strata == s)[0] for s in np.unique(strata)}
    boots = [np.concatenate([rng.choice(ix, len(ix)) for ix in groups.values()])
             for _ in range(a.boot)]

    def metrics(y, s, ww):
        """All metrics of one resample, flattened in the order of `names`.

        Args:
            y (numpy.ndarray): Positives.
            s (numpy.ndarray): Scores.
            ww (numpy.ndarray): Weights.

        Returns:
            list[float]: [AP, precision@t1, recall@t1, precision@t2, ...].
        """
        apv, pr = ap_pr(y, s, ww, thresholds)
        return [apv] + [x for p in pr for x in p]

    names = ["AP"] + [f"{m}@{t:g}" for t in thresholds for m in ("precision", "recall")]
    rows = []
    for c, (y, s_by) in targets.items():
        point = {k: metrics(y, s_by[k], w) for k in a.gens}
        bs = {k: np.array([metrics(y[b], s_by[k][b], w[b]) for b in boots]) for k in a.gens}
        ref = a.gens[0]
        for k in a.gens:
            for j, name in enumerate(names):
                row = {"gen": k, "class": c, "metric": name,
                       "n_pos": int(y.sum()), "value": point[k][j]}
                if a.boot:
                    row["lo"], row["hi"] = np.nanpercentile(bs[k][:, j], [2.5, 97.5])
                    if k != ref:
                        dlt = bs[k][:, j] - bs[ref][:, j]
                        row[f"diff_vs_gen{ref}"] = point[k][j] - point[ref][j]
                        row["diff_lo"], row["diff_hi"] = np.nanpercentile(dlt, [2.5, 97.5])
                rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    header = ["gen", "class", "metric", "n_pos", "value", "lo", "hi",
              f"diff_vs_gen{a.gens[0]}", "diff_lo", "diff_hi"]
    with open(out / "compare.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{x:.4f}" if isinstance(x, float) else x) for k, x in r.items()})

    print(f"\n{'class':9s} {'n_pos':>5s}  " + "  ".join(f"gen_{k} AP [95% CI]".rjust(24) for k in a.gens))
    for c in targets:
        cells = []
        for k in a.gens:
            r = next(r for r in rows if r["gen"] == k and r["class"] == c and r["metric"] == "AP")
            cells.append(f"{r['value']:.3f} [{r.get('lo', np.nan):.3f}, {r.get('hi', np.nan):.3f}]".rjust(24))
        print(f"{c:9s} {int(targets[c][0].sum()):5d}  " + "  ".join(cells))
    for r in rows:
        if r["metric"] == "AP" and "diff_lo" in r:
            sig = "" if r["diff_lo"] <= 0 <= r["diff_hi"] else "  (interval excludes 0)"
            print(f"  {r['class']}: gen_{r['gen']} - gen_{a.gens[0]} AP = {r[f'diff_vs_gen{a.gens[0]}']:+.3f} "
                  f"[{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}]{sig}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for c, (y, s_by) in targets.items():
        if not y.sum():
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for i, k in enumerate(a.gens):
            rec, prec = pr_curve(y, s_by[k], w)
            apv = next(r["value"] for r in rows if r["gen"] == k and r["class"] == c and r["metric"] == "AP")
            ax.plot(rec, prec, color=COLORS[i % len(COLORS)], lw=2, drawstyle="steps-post",
                    label=f"gen_{k}  AP {apv:.2f}")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(color="#e4e3df", lw=0.8)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("recall (weighted)")
        ax.set_ylabel("precision (weighted)")
        ax.set_title(f"{c}  ({int(y.sum())} positive test windows)", loc="left", fontsize=11)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(out / f"pr_{c}.png", dpi=120)
        plt.close(fig)
    print(f"\nwrote {out / 'compare.csv'} and pr_<class>.png")


if __name__ == "__main__":
    main()
