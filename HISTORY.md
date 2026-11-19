# Project history

What each model and dataset is, what it may be used for, and what each experiment
gave. The README explains how the pipeline works; this file records what was actually
done. Paths are relative to `code/`. Update it whenever a model, a dataset or a result
changes (the `/wrap-up` routine does this).

## Models

| Model | Trained on | Model file | Detections on the survey clips |
|---|---|---|---|
| **v0 = gen_0** | XC clips only: 3 s windows cut from the hand-annotated XC recordings (`outputs/clips`: Call 633, Drumming 473, Song 749, Noise 184) | `models/final/gallinago_v0.tflite`; `generations/gen_0/` only holds links to it | `data/survey/detections/` (all 264 clips; the default `detection.dir`) |
| **gen_1** | XC clips + 728 survey windows from 496 train-queue verdicts on v0's detections (597 Noise, 59 Call, 40 Song, 32 Drumming); same hyperparameters as v0, seed 1 | `generations/gen_1/model.tflite` | `generations/gen_1/detections/` (198 train-site clips only) |
| gen_2 | *planned:* gen_1's data + verdicts on gen_1's own detections (queue `train_gen1`) | – | – |

## Datasets

| Dataset | What it is | Size | Use | Path |
|---|---|---|---|---|
| XC recordings | Xeno-canto *Gallinago gallinago*, quality A/B | 1536 recordings | source | `data/xc_gallinago_gallinago/` |
| XC training annotations | hand labels (Audacity format) | 115 recordings, 109 recordists | **training only** | `data/annotation_labels/` |
| Survey clips | 3 clips of 20 min per site (one per third of the season) | 88 sites, 264 clips, ~88 h | source | `data/survey/clips/`, `manifest.csv` |
| Site split | per drive, seeded | **66 train / 22 heldout** sites | heldout sites never train anything | `data/survey/sites.csv` |
| Train queue | v0 detections (events over confidence bins) + 1 random background window per clip, train sites | 496 verdicts | **training** (gen_1, later gens) | `data/survey/review/train_*` |
| Survey test set | blind 3 s windows from heldout sites, stratified by v0 score, prediction hidden | 361 judged (16 Call, 2 Song, 0 Drumming) | **test only**: field AP, false alarms per hour | `data/survey/review/test_*` |
| XC test set | not-yet-annotated XC recordings by 60 recordists absent from all training data, annotated blind (no name or tag shown) | 78 recordings, 727 windows (250 Drumming, 216 Song, 75 Call) | **test only**: recall/AP on unseen recordists | `data/xc_test/` |

Snipe song and drumming are rare at the survey sites: drumming was confirmed at 2 of
66 train sites and song at 4. Field recall for these classes therefore cannot be
measured; the XC test set measures it instead, and the survey test set measures field
false alarms (which need no snipe present).

## Results

**Survey test set, gen_0 vs gen_1** (`results/survey/compare.csv`; 95 % CI from a stratified bootstrap):

| Metric | gen_0 | gen_1 |
|---|---|---|
| AP Call (16 pos.) | 0.954 [0.861, 1.000] | 0.981 [0.925, 1.000] |
| AP any (18 pos.) | 0.790 [0.614, 0.985] | 0.714 [0.482, 0.985] |
| False alarms / h, any class, t = 0.5 | 17.5 [16.8, 18.0] | 0.28 [0.00, 0.73] |
| False alarms / h, Drumming, t = 0.5 | 15.4 [14.1, 16.7] | 0.23 [0.00, 0.68] |
| False alarms / h, any class, t = 0.35 | 102.1 [101.6, 102.6] | 0.78 [0.10, 1.63] |

**XC test set, gen_0 vs gen_1** (`results/survey/compare_xc.csv`; bootstrap over recordists):

| Class | AP gen_0 | AP gen_1 | gen_1 − gen_0 |
|---|---|---|---|
| Call | 0.989 | 0.984 | −0.005 [−0.020, +0.005] |
| Drumming | 0.934 | 0.930 | −0.004 [−0.017, +0.009] |
| Song | 0.903 | 0.942 | +0.039 [+0.004, +0.105] |

**Reading:** training on field verdicts (gen_1) removed almost all field false alarms
(about 60× fewer at 0.5) without losing recognition on unseen XC recordists, and
improved Song. Gen_1 is much more conservative: on the train sites it gives 81 events
at ≥ 0.5 where v0 gave 3709 at ≥ 0.35.

## Timeline

- **2026-11-09 → 11-12**: XC download, annotation tools, first annotations.
- **2026-11-17**: training pipeline (clips, recordist-grouped folds, learning curve) → v0. Survey sampling of 88 sites (66 train / 22 heldout), v0 detections, review GUI; 361 test and 496 train verdicts; gen_1 trained.
- **2026-11-18**: gen_0 vs gen_1 on the survey test set (too few positives to conclude). XC test set drawn, annotated blind and evaluated. Output cleanup (old learning-curve models deleted).
- **2026-11-19**: field false-alarm metric added (gen_1: about 60× fewer). gen_1 run on the train sites; review queue `train_gen1` started for gen_2.
