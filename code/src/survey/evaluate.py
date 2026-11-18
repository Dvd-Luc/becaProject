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


def cut_windows(items, windows_dir):
    """Cut each judged test window once to wdir/<item_id>.wav (existing files are kept).

    Args:
        items (list[dict]): Test queue rows.
        windows_dir (Path): <review.dir>/test_windows.
    """
    windows_dir.mkdir(parents=True, exist_ok=True)
    n_cut = 0
    for item in items:
        dst = windows_dir / f"{item['item_id']}.wav"
        if dst.exists():
            continue
        info = sf.info(item["clip_path"])
        audio, sample_rate = sf.read(item["clip_path"], start=int(float(item["start_s"]) * info.samplerate),
                                     frames=int(WIN * info.samplerate), dtype="float32")
        sf.write(dst, audio, sample_rate, subtype=info.subtype)
        n_cut += 1
    if n_cut:
        print(f"cut {n_cut} test windows into {windows_dir}")


def score(model, windows_dir, scores_dir, items, threads):
    """Scores of one model on the test windows; runs BirdNET if results are missing.

    Args:
        model (Path): The generation's model.tflite.
        windows_dir (Path): Folder of the cut test windows.
        scores_dir (Path): The generation's test_scores/ folder.
        items (list[dict]): Test queue rows to score.
        threads (int | None): BirdNET CPU threads (None: BirdNET's default).

    Returns:
        dict[str, dict[str, float]]: item_id -> {label: score}; labels below 0.01 absent.

    Raises:
        SystemExit: If BirdNET produced no result for an item.
    """
    todo = [item for item in items if not (scores_dir / f"{item['item_id']}.BirdNET.results.csv").exists()]
    if todo:
        cmd = [sys.executable, "-m", "birdnet_analyzer.analyze", str(windows_dir), "-o", str(scores_dir),
               "-c", str(model), "--min_conf", "0.01", "--overlap", "0", "--rtype", "csv"]
        if threads:
            cmd += ["-t", str(threads)]
        print(f"scoring {len(items)} windows with {model}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    scores = {}
    for item in items:
        result_path = scores_dir / f"{item['item_id']}.BirdNET.results.csv"
        if not result_path.exists():
            raise SystemExit(f"no BirdNET result for {item['item_id']} in {scores_dir}")
        item_scores = defaultdict(float)
        for window in read_dicts(result_path):
            if float(window["Start (s)"]) < 0.5:              # the one 3 s segment
                label = window["Common name"] or window["Scientific name"]
                item_scores[label] = max(item_scores[label], float(window["Confidence"]))
        scores[item["item_id"]] = item_scores
    return scores


def ap_pr(y, scores, weights, thresholds):
    """Weighted average precision, and precision / recall per threshold.

    Args:
        y (numpy.ndarray): 1.0 for positive windows, 0.0 otherwise.
        scores (numpy.ndarray): Model scores.
        weights (numpy.ndarray): Window weights (stratum population / judged windows).
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
    """CLI entry point: score the test windows with each generation, compute metrics, plot."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--gens", type=int, nargs="+", required=True)
    parser.add_argument("--review-dir", help="default: review.dir")
    parser.add_argument("--gen-dir", help="default: generations.dir")
    parser.add_argument("--out", default="results/survey")
    parser.add_argument("--boot", type=int, default=500, help="bootstrap resamples (0 = off)")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    cfg = load(args.config)
    gen_cfg = cfg["generations"]
    review_dir = Path(args.review_dir or cfg["review"]["dir"])
    gens_dir = Path(args.gen_dir or gen_cfg["dir"])
    out_dir = Path(args.out)
    thresholds = sorted(set(gen_cfg.get("eval_thresholds", [0.35, 0.5, 0.7])
                            + [cfg["detection"]["min_conf"]]))

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
    cut_windows(items, windows_dir)
    models = {gen: gens_dir / f"gen_{gen}" / "model.tflite" for gen in args.gens}
    for gen, model_path in models.items():
        if not model_path.exists():
            raise SystemExit(f"{model_path} not found (make_generation.py --gen {gen}; gen_0 is linked "
                             f"by any make_generation.py run)")
    labels = [line.strip() for line in models[args.gens[0]].with_name("model_Labels.txt")
              .read_text(encoding="utf-8").splitlines() if line.strip()]
    scores = {gen: score(model_path, windows_dir, gens_dir / f"gen_{gen}" / "test_scores", items,
                         args.threads)
              for gen, model_path in models.items()}

    targets = {class_name: (np.array([verdict == class_name for verdict in verdicts], float),
                            {gen: np.array([scores[gen][item["item_id"]].get(class_name, 0.0)
                                            for item in items])
                             for gen in args.gens})
               for class_name in labels}
    targets["any"] = (np.array([verdict != NOISE for verdict in verdicts], float),
                      {gen: np.array([max(scores[gen][item["item_id"]].values(), default=0.0)
                                      for item in items]) for gen in args.gens})

    rng = np.random.default_rng(0)
    groups = {stratum: np.where(strata == stratum)[0] for stratum in np.unique(strata)}
    boots = [np.concatenate([rng.choice(indices, len(indices)) for indices in groups.values()])
             for _ in range(args.boot)]

    def metrics(y, scores, weights):
        """All metrics of one resample, flattened in the order of `names`.

        Args:
            y (numpy.ndarray): Positives.
            scores (numpy.ndarray): Scores.
            weights (numpy.ndarray): Weights.

        Returns:
            list[float]: [AP, precision@t1, recall@t1, precision@t2, ...].
        """
        ap, pr_points = ap_pr(y, scores, weights, thresholds)
        return [ap] + [value for point in pr_points for value in point]

    names = ["AP"] + [f"{metric}@{threshold:g}"
                      for threshold in thresholds for metric in ("precision", "recall")]
    rows = []
    for class_name, (y, scores_by_gen) in targets.items():
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
    header = ["gen", "class", "metric", "n_pos", "value", "lo", "hi",
              f"diff_vs_gen{args.gens[0]}", "diff_lo", "diff_hi"]
    with open(out_dir / "compare.csv", "w", newline="", encoding="utf-8") as f:
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

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for class_name, (y, scores_by_gen) in targets.items():
        if not y.sum():
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for item_id, gen in enumerate(args.gens):
            recall, precision = pr_curve(y, scores_by_gen[gen], weights)
            ap_value = next(row["value"] for row in rows
                            if row["gen"] == gen and row["class"] == class_name
                            and row["metric"] == "AP")
            ax.plot(recall, precision, color=COLORS[item_id % len(COLORS)], lw=2, drawstyle="steps-post",
                    label=f"gen_{gen}  AP {ap_value:.2f}")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(color="#e4e3df", lw=0.8)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("recall (weighted)")
        ax.set_ylabel("precision (weighted)")
        ax.set_title(f"{class_name}  ({int(y.sum())} positive test windows)", loc="left", fontsize=11)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / f"pr_{class_name}.png", dpi=120)
        plt.close(fig)
    print(f"\nwrote {out_dir / 'compare.csv'} and pr_<class>.png")


if __name__ == "__main__":
    main()
