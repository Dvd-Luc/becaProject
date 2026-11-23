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
| **gen_2** | gen_1's data + 250 verdicts on gen_1's own train-site events at ≥ 0.35 (queue `train_gen1`: 52 events not already judged + 198 background; 221 Noise, 19 Call, 4 Drumming, 6 Unsure) → 989 survey windows; totals Call 727, Drumming 510, Song 789, Noise 1002 (33 %); same hyperparameters, seed 1 | `generations/gen_2/model.tflite` | – |

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
| Full survey, gen_2 | gen_2 on **every** recording, drive by drive (MeSU cluster, commit `7fd9141`, 64 cores, 2 s per audio-hour) | D4: 29 sites, 5 752 files (5 749 h), 4 408 events at ≥ 0.5 (Call 2 605, Drumming 755, Song 1 048); D1–D3 to do | **result** (detections) | `data/survey/full/gen_2/` |
| Audit sample D4 | random sample of D4's full-survey events, 30 per class × `max_conf` bin (0.5 / 0.7 / 0.9 / 1.0), judged blind; train and heldout sites | 270 events | **precision of the full-survey events**; only train-site verdicts may ever train | `data/survey/review/audit_D4_*` |

Snipe song and drumming are rare at the survey sites: drumming was confirmed at 2 of
66 train sites and song at 4. Field recall for these classes therefore cannot be
measured; the XC test set measures it instead, and the survey test set measures field
false alarms (which need no snipe present).

## Results

**Survey test set, gens 0–2** (`results/survey/compare.csv`; 95 % CI from a stratified bootstrap):

| Metric | gen_0 | gen_1 | gen_2 |
|---|---|---|---|
| AP Call (16 pos.) | 0.954 [0.861, 1.000] | 0.981 [0.925, 1.000] | 0.989 [0.957, 1.000] |
| AP any (18 pos.) | 0.790 [0.614, 0.985] | 0.714 [0.482, 0.985] | 0.759 [0.518, 1.000] |
| False alarms / h, any class, t = 0.5 | 17.5 [16.8, 18.0] | 0.28 [0.00, 0.73] | 0.05 [0.00, 0.15] |
| False alarms / h, Drumming, t = 0.5 | 15.4 [14.1, 16.7] | 0.23 [0.00, 0.68] | 0.00 [0.00, 0.00] |
| False alarms / h, any class, t = 0.35 | 102.1 [101.6, 102.6] | 0.78 [0.10, 1.63] | 0.43 [0.05, 0.88] |

(Song AP rests on 2 positive windows and is not interpretable.)

**XC test set, gens 0–2** (`results/survey/compare_xc.csv`; bootstrap over recordists):

| Class | AP gen_0 | AP gen_1 | AP gen_2 | gen_1 − gen_0 | gen_2 − gen_1 |
|---|---|---|---|---|---|
| Call | 0.989 | 0.984 | 0.985 | −0.005 [−0.020, +0.005] | +0.001 [−0.006, +0.006] |
| Drumming | 0.934 | 0.930 | 0.937 | −0.004 [−0.017, +0.009] | +0.007 [−0.006, +0.021] |
| Song | 0.903 | 0.942 | 0.916 | +0.039 [+0.004, +0.105] | −0.026 [−0.088, +0.004] |

**Full-survey audit, D4, gen_2 events at ≥ 0.5** (`results/survey/audit_full_gen2.csv`; 270 blind verdicts, 8 Unsure; precision weighted by stratum population, 95 % CI from a site bootstrap):

| Class | Events | Precision | 0.5–0.7 | 0.7–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|
| Call | 2 605 | 0.750 [0.612, 0.912] | 0.462 | 0.654 | 0.967 |
| Drumming | 755 | 0.951 [0.751, 1.000] | 0.933 | 0.967 | 1.000 |
| Song | 1 048 | 0.987 [0.923, 1.000] | 0.967 | 1.000 | 1.000 |

All errors were events judged Noise (no confusion between snipe classes). 13 of the 24
Call errors came from one recorder (068 Bouverans). 90 % of Drumming and 79 % of Song
events are at Granges-Narboz (recorders 044 and 046 train, 045 heldout), so the sample
mostly measures that place. Outside it, 3 of the 5 Drumming events judged were Noise.
Decisions: Call threshold raised to 0.9; site-by-site presence check of Drumming and
Song outside the hotspot; next audits sampled per site; gen 3 decided after D1–D3.

**Reading:** training on field verdicts (gen_1) removed almost all field false alarms
(about 60× fewer at 0.5) without losing recognition on unseen XC recordists, and
improved Song. Gen_1 is much more conservative: on the train sites it gives 81 events
at ≥ 0.5 where v0 gave 3709 at ≥ 0.35 (124 at ≥ 0.35).

Gen_2 (one more round on gen_1's own events) is a small step: field false alarms fall
further (any class at 0.5: 0.28 → 0.05 / h) and XC recognition holds for Call and
Drumming; Song on the XC test is back between gen_0 and gen_1 (−0.026 vs gen_1, CI
touches 0). The gains are now within the noise of these test sets: the training
signal from field verdicts is nearly exhausted at these sites.

## Timeline

- **2026-11-09 → 11-12**: XC download, annotation tools, first annotations.
- **2026-11-17**: training pipeline (clips, recordist-grouped folds, learning curve) → v0. Survey sampling of 88 sites (66 train / 22 heldout), v0 detections, review GUI; 361 test and 496 train verdicts; gen_1 trained.
- **2026-11-18**: gen_0 vs gen_1 on the survey test set (too few positives to conclude). XC test set drawn, annotated blind and evaluated. Output cleanup (old learning-curve models deleted).
- **2026-11-19**: field false-alarm metric added (gen_1: about 60× fewer). gen_1 run on the train sites; review queue `train_gen1` rebuilt at gen_0's settings (events ≥ 0.35) and judged (250); gen_2 trained and evaluated on both test sets.
- **2026-11-20 → 11-23**: full survey moved to the MeSU cluster (Slurm job per drive). D4 done with gen_2 (5 752 files, 4 408 events at ≥ 0.5). Audit tool added and the D4 audit sample frozen (270 events, snippets cut).
