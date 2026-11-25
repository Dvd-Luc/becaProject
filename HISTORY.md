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
| Full survey, gen_2 | gen_2 on **every** recording, drive by drive (MeSU cluster, commit `7fd9141`, 64 cores, 2 s per audio-hour) | D1 + D2 + D4: 76 sites, 13 896 files (13 881 h), 9 019 kept events (Call ≥ 0.9, others ≥ 0.5). D1: 32 sites, 5 507 files, Call 1 401, Drumming 407, Song 367. D2: 15 sites, 2 637 files, Call 431, Drumming 2 404, Song 980. D3: 12 sites, 2 335 files (2 334 h), Call 753, Drumming 51, Song 129. D4: 29 sites, 5 752 files, Call 1 226, Drumming 755, Song 1 048. **All four drives: 88 sites, 16 231 files (16 215 h), 9 952 kept events** | **result** (detections) | `data/survey/full/gen_2/` |
| Audit sample D4 | random sample of D4's full-survey events, 30 per class × `max_conf` bin (0.5 / 0.7 / 0.9 / 1.0), judged blind; train and heldout sites | 270 events | **precision of the full-survey events**; only train-site verdicts may ever train | `data/survey/review/audit_D4_*` |
| Audit samples D1, D2, D3 | random sample per class × site × `max_conf` bin (3 per stratum for D1, 5 for D2 and D3), judged blind; presence candidates cut at the same time | D1 253 + 4 presence, D2 191, D3 143 (presence queues of D2, D3 empty) | as D4 | `data/survey/review/{audit,presence}_D{1,2,3}_*` |
| Heldout audit events | the audit events of heldout sites that gen_2 keeps (best window ≥ class floor), cut into their 3 s windows | 150 events, 17 sites (Call 50 true / 2 false, Drumming 32 / 20, Song 32 / 14) | **test only**: how a generation filters gen_2's events (`evaluate.py --testset audit`) | `data/survey/review/audit_D*` (heldout rows) |

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

**Presence per site, D4** (`results/survey/presence_full_gen2.csv`; audit verdicts + 72
blind verdicts on the best events of each site not yet confirmed, up to 5 per site × class):

| Class | Sites with events | Present | Not confirmed |
|---|---|---|---|
| Drumming | 18 / 29 | 12 | 6 (047, 053, 058, 062, 065, 067) |
| Song | 24 / 29 | 21 | 3 (053, 056, 062) |

Snipe are far more widespread than the 20-min clips showed (drumming then confirmed at
2 of 66 train sites). Many sites rest on one or two low-score events (best score 0.5–0.65),
which supports keeping 0.5 for Drumming and Song. "Not confirmed" means that the best events
were something else, not that the class is absent. Site-level maps should filter
`events.csv` with this table.

**Full-survey audit, D1 + D2 + D4, kept gen_2 events** (`results/survey/audit_full_gen2.csv`, commit `88a6bde`; 714 blind verdicts, 20 Unsure; weighted by stratum population, site bootstrap):

| Class | Kept events | Precision |
|---|---|---|
| Call (≥ 0.9) | 3 058 | 0.952 [0.910, 0.988] |
| Drumming | 3 566 | 0.896 [0.713, 0.960] |
| Song | 2 395 | 0.946 [0.914, 0.976] |
| All kept | 9 019 | 0.928 [0.860, 0.959] |

Drumming depends on the region. Correct / judged (unweighted), per `max_conf` bin:

| Drumming | 0.5–0.7 | 0.7–0.9 | ≥ 0.9 |
|---|---|---|---|
| D4 (Doubs / Jura) | 28/30 | 29/30 | 30/30 |
| D1 (Massif Central) | 19/56 | 22/27 | 4/4 |
| D2 (Massif Central) | 12/35 | 11/22 | 12/12 |

The pooled Drumming value looks good because D2's two big sites, 011 Egliseneuve-d'Entraigues
(1 252 events) and 039 Saint-Jean-Lachalm (986), are correct and carry most of the weight. The
errors cluster at a few sites: 038 Costaros 0/10, 040 Les Vastres 0/10, 017 Rentières 0/5,
014 Olby 0/4, 037 Landos 1/7, 001 Tanavelle 1/6. By ear they are mostly **horse neighing**. A
higher Drumming threshold would not fix it (D2 0.7–0.9: half wrong) and would lose presence
evidence: 6 of 30 Drumming sites and 19 of 44 Song sites are confirmed only by events < 0.7.
All errors are Noise except 2 Call → Drumming and 2 Song → Drumming. Three small strata got
only Unsure verdicts and are left out of the weights.

Presence per site, D1 + D2 + D4: Drumming present at 30 sites, not confirmed at 25; Song
present at 44, not confirmed at 8.

The survey test set gave gen_2 0.00 Drumming false alarms / h, but its 20-min clips missed
the horse sites: for Drumming in the field, the heldout audit verdicts are now the check.

Gen 3 material (train-site verdicts only): Drumming → Noise 81, Song → Noise 41,
Call → Noise 33. The 19 heldout Drumming errors stay out of training. Plan: decide after
the D3 audit; check a gen 3 on the gen_2 events (heldout audit verdicts) and the XC test,
then re-run the full survey with it.

**Full-survey audit, all four drives** (`audit_full_gen2.csv`; 857 blind verdicts, 25 Unsure):
kept events 0.927 [0.865, 0.957], Call (≥ 0.9) 0.956 [0.919, 0.985], Drumming 0.894
[0.713, 0.954], Song 0.931 [0.901, 0.972]. D3 adds Drumming errors at 078 La Planée (0/6),
072 Bouverans and 075 Frasne, and Song at 074 Bonnevaux (0/10). Presence: Drumming present
at 33 sites, not confirmed at 29; Song present at 52, not confirmed at 9.

**Recording schedule** (checked against computed sunsets at each position): every evening
is two one-hour files from **sunset to sunset + 2 h**; the first file starts at sunset
(median +1 min) in the recorder's own clock. Most clocks stayed on UTC+1 after the
29 March time change, some follow legal time. The page earlier assumed −1 h → +1 h; fixed
(`viz.schedule_start_min: 0`). Median kept event: Call 33, Song 37, Drumming 45 min after
sunset. Data note: 027 Anzat-le-Luguet is named `_15_` but lies in Puy-de-Dôme (63).

**Gen 3 plan** (rules fixed before any gen 3 result). Training: every earlier train queue +
the train-site audit and presence verdicts (694 + 68; 175 heldout left out): Drumming 244
field windows (gen_2: 37), 227 new Noise windows (mostly horses). Adopt gen 3 for a full
re-run only if: (1) XC test AP of no class significantly lower than gen_2; (2) survey test
set false alarms / h not higher; (3) heldout audit events: ≥ 90 % of the true events of
each class kept, and ≥ half of the 20 false Drumming events removed, at the fixed floors
(0.5, Call 0.9). Gen_2 reference on (3): keeps all by construction (rescore = survey
scores), precision of kept Call 0.988, Drumming 0.648 [0.02, 0.98], Song 0.883.

Data note: D1 site 032 Cussac holds two extra folders (`Porzana porzana` 186,
`Zapornia pusilla` 63 WAVs): 16–30 s clips from an earlier BirdNET crake extraction,
cut from the site's own recordings. The survey globs skip them, as they should.

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
- **2026-11-23 → 11-25**: D4 audit judged; Call threshold raised to 0.9; presence check; local visualisation page. D1 and D2 run on the cluster (13 896 files in all), D3 submitted. Audit sampled per class × site × bin from D1 on; D1/D2 judged (444 + 4). Drumming false alarms, mostly horse neighing, at a few Massif Central sites. D3 run and audited (143); recording schedule corrected on the page (sunset → +2 h); page: IGN base maps, départements, zoom. Gen 3 material and the heldout audit test set added; gen 3 training started.
