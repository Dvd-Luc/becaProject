# becaProject: Common Snipe detection with BirdNET

Detecting **Common Snipe** (*Gallinago gallinago*, "bécassine") vocalisations in
passive acoustic recordings with a custom **BirdNET** classifier. The model has three
classes: **Drumming**, **Song** and **Call**.

The pipeline has two halves:

1. **Training on Xeno-canto (XC)**: download recordings → annotate them by hand → cut
   3 s training clips → train and evaluate BirdNET with cross-validation grouped by
   recordist (plus a learning curve).
2. **Survey**: apply the model to 20 min clips sampled from field recordings at ~80
   sites → review its detections → train the next model *generation* on the reviewed
   material → compare generations on a fixed, blind test set.

Everything is plain Python scripts: no package, no build step. Each script's docstring
holds its full usage (`python <script> --help`, or the documentation site below).

---

## Contents

- [Setup](#setup)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Part 1: training on Xeno-canto](#part-1-training-on-xeno-canto)
- [Part 2: the survey](#part-2-the-survey)
- [Script reference](#script-reference)
- [Documentation site](#documentation-site)
- [Good to know](#good-to-know)

---

## Setup

```bash
cd becaProject
source .becavenv/bin/activate      # Python 3.12 virtualenv
cd code                            # every command below is run from code/
```

The virtualenv contains `birdnet_analyzer` 2.4 (called as `python -m
birdnet_analyzer.train` / `.analyze`, with TensorFlow), `numpy`, `soundfile`,
`matplotlib`, `pyyaml`, `requests`, `sounddevice` (audio playback in the GUIs) and
`pdoc` (documentation).

To create it from scratch:

```bash
python3 -m venv .becavenv && source .becavenv/bin/activate
pip install birdnet-analyzer==2.4.* numpy soundfile matplotlib pyyaml requests sounddevice pdoc
```

Downloading from Xeno-canto needs an API key from your xeno-canto account page:

```bash
export XC_API_KEY="your_key"
```

**What is not in git:** audio (`*.wav`, `*.mp3`, `*.flac`), images, models
(`*.tflite`), training logs and `__pycache__`. Only code, configuration, label files
and CSV results are versioned. **Back up the `.tflite` models yourself**, in particular
`models/final/` and `generations/`.

---

## Repository layout

```
code/
├── config/project.yaml          unified configuration, one section per stage
├── src/
│   ├── common/config.py         YAML loader
│   ├── tools/                   fetching, annotation, confusion matrices
│   ├── training/                training clips, folds, learning curve
│   └── survey/                  sampling, detection, review, generations, evaluation
├── data/
│   ├── xc_gallinago_gallinago/  XC audio + metadata.csv
│   ├── annotation_labels/       one Audacity label file per XC recording
│   └── survey/                  sites.csv, manifest.csv, plan/, clips/, detections/, review/
├── outputs/                     clips/, folds/, lc/ (training material)
├── models/                      final/ (v0 model), lc/ (learning-curve models + metrics)
├── generations/                 gen_0 (= v0), gen_1, ... (created by make_generation.py)
└── results/                     lc/ (learning curve), survey/ (generation comparison)
```

---

## Configuration

`code/config/project.yaml` holds one section per stage:

| Section | Used by | Content |
|---|---|---|
| `classes`, `label_map` | reference | the three classes; `Other` is mapped to Noise |
| `xc` | reference | XC data paths and CSV columns |
| `viewer` | `annotator.py`, `review.py` | spectrogram settings (FFT, frequency range, colours) |
| `annotator` | `annotator.py` | folders, metadata CSV, categories (keys 1–9) |
| `learning_curve` | reference | the values used for the learning curve |
| `survey` | `sample_clips.py`, `detect.py` | drives, file patterns, sampling, train/heldout split |
| `detection` | `detect.py`, `review.py`, `make_generation.py`, `evaluate.py` | model, thresholds, event merging |
| `review` | `review.py`, `make_generation.py`, `evaluate.py`, `audit_events.py` | review queues, audit sample |
| `generations` | `make_generation.py`, `evaluate.py`, `detect.py --gen` | training seed, evaluation thresholds |
| `xc_test` | `annotation_tools.py sample-test`, `evaluate.py --testset xc`, `make_generation.py` | XC test sample: seed, strata, caps |
| `annotator_xc_test` | `annotator.py --section annotator_xc_test` | annotating the XC test sample (own labels folder) |
| `viz` | `build_viz.py` | microphone GeoPackages, schedule offset, output page |
| `drive_detection` | `detect_drive.py` | generation, output folder, thresholds, threads, batch size |

The training scripts (Part 1) don't read the YAML: they take everything as
command-line options. Paths in the YAML are relative to `code/`, which is why every
script must be run from there.

> `annotator.py` rejects unknown keys in `viewer` / `annotator` / `annotator_xc_test`:
> a key added there must also be added to its `Config` dataclass.

---

## Part 1: training on Xeno-canto

### 1. Fetch the recordings

```bash
python src/tools/recording_fetch_xc.py
```

Downloads every quality A and B *Gallinago gallinago* recording into
`data/xc_gallinago_gallinago/`, with `metadata.csv` (XC fields: `file-name`, `rec` =
recordist, `q` = quality, `type`, ...). Files already downloaded are skipped.

### 2. Annotate

Label time regions in **Audacity label format**: one file per recording,
`data/annotation_labels/<recording stem>.txt`, one `start<TAB>end<TAB>label` line per
call. Labels must be `Drumming`, `Song` or `Call` (or `Other`): the label text becomes
the class folder name later on. **An empty file means "listened, no calls".**

With the built-in spectrogram annotator:

```bash
python src/tools/annotator.py                  # recordings still to do, best quality first
python src/tools/annotator.py --type drum --n 25
python src/tools/annotator.py --list           # preview the queue
python src/tools/annotator.py --status         # progress summary
python src/tools/annotator.py --max-per-group 5   # cap per recordist (group_cols)
```

Main keys: drag to select, `1`–`9` to label, right-click a label to edit it, `Space`
to play, `s` to save and go to the next recording, `x` for "no calls", `b` to go back.
Unsaved work is autosaved to `<stem>.partial` and restored next time.

Annotating in Audacity instead works too. `annotation_tools.py` tracks progress:

```bash
python src/tools/annotation_tools.py status --csv data/xc_gallinago_gallinago/metadata.csv \
    --labels-dir data/annotation_labels --allowed Drumming Song Call Other
python src/tools/annotation_tools.py next --csv data/xc_gallinago_gallinago/metadata.csv \
    --labels-dir data/annotation_labels --n 10 --type drum
```

To summarise the annotations (labels per class, durations, recordists, concentration
in one file):

```bash
python src/training/label_stats.py --labels-dir data/annotation_labels \
    --xc-csv data/xc_gallinago_gallinago/metadata.csv
```

### 3. Cut 3 s training clips

```bash
python src/training/make_training_clips.py --audio-dir data/xc_gallinago_gallinago \
    --labels-dir data/annotation_labels --out-dir outputs/clips
```

→ `outputs/clips/<label>/*.wav` (3 s windows), `outputs/clips/Noise/` and
`outputs/clips/metadata.csv`.

- A call shorter than 3 s gets one window containing the whole call, at a random position.
- A longer call gets a chain of windows, each covering at least `--tau` s of it.
- Windows containing another call type are rejected.
- Noise windows are drawn away from every label (`--noise-per-file`, `--noise-total`).

Add `--clips-csv outputs/clips/metadata.csv` to `label_stats.py` to see the clips per class.

### 4. Cross-validation (5 folds, grouped by recordist)

```bash
python src/training/make_folds.py --clips-csv outputs/clips/metadata.csv \
    --xc-csv data/xc_gallinago_gallinago/metadata.csv --out-dir outputs/folds \
    --folds 5 --drop-labels Other
bash src/training/birdnet_training.sh       # one model per fold → models/fold_K.tflite
bash src/training/model_analysis.sh         # predictions on each fold's val set → models/preds/
python src/tools/confusion_matrix.py outputs/folds models/preds results/confusion_matrices.csv
```

All clips of one recordist stay on the same side of each split, so validation measures
generalisation to *unseen sources*. `--seed` fixes the fold assignment. The clips are
hard-linked, so no disk space is duplicated.

### 5. Learning curve

```bash
python src/training/run_learning_curve.py --clips-csv outputs/clips/metadata.csv \
    --xc-csv data/xc_gallinago_gallinago/metadata.csv --drop-labels Other
python src/training/plot_learning_curve.py --models-dir models/lc --work-dir outputs/lc \
    --out-dir results/lc
```

Trains one model per (training fraction × subset seed × fold): by default fractions
0.25/0.5/0.75/1.0, subset seeds 0/1/2, 5 folds. Layout:
`outputs/lc/frac_X/seed_S/fold_K/` and `models/lc/frac_X/seed_S/fold_K.tflite`. The
run can be resumed: existing folds and models are skipped. The plot step writes
`learning_curve_long.csv`, `learning_curve_summary.csv` and one PNG per metric, and
prints the paired gap between the two largest fractions.

The final model used for the survey is `models/final/gallinago_v0.tflite` (generation 0).

---

## Part 2: the survey

The survey loop: **sample → detect → review → train the next generation → evaluate →
detect again with the better model.**

### 6. Sample clips (one drive at a time)

The field recordings are on several drives that can't be plugged in together. Plug in
one drive and run it under its name (a key of `survey.drives`):

```bash
python src/survey/sample_clips.py --drive D1 --dry-run   # show this drive's plan
python src/survey/sample_clips.py --drive D1             # plan (once) + cut the clips
python src/survey/sample_clips.py --drive D2 --root /mnt/e   # other mount point
```

For each site, the date range is cut into `survey.sampling.periods` parts, and one 20 min
clip is drawn per part. The sites are split into **train / heldout** within each drive
(`survey.split.fractions`).

**heldout sites are never used for training: they are the test set.**

The plan of each drive is frozen in `data/survey/plan/<drive>_{sites,manifest}.csv` and
reused as is; delete it to resample that drive. After each run, `data/survey/sites.csv`
and `manifest.csv` are rebuilt from all frozen drives, and the clips go to
`data/survey/clips/`.

### 7. Detect

```bash
python src/survey/detect.py --dry-run            # how many clips would be analysed
python src/survey/detect.py                      # analyse new clips + rebuild the tables
python src/survey/detect.py --min-conf 0.35      # other event threshold for this run
python src/survey/detect.py --post-only --min-conf 0.4   # new threshold on the saved scores, model not re-run
python src/survey/detect.py --split heldout      # one split only
```

Runs `detection.model` on every clip of the manifest that has no result yet (the run
can be resumed) → `data/survey/detections/`:

- `raw/<clip_id>.BirdNET.results.csv`: every 3 s window ≥ `raw_min_conf`. Because the
  low-scoring windows are kept, the threshold can be changed later without running the
  model again.
- `events.csv`: windows ≥ `min_conf`, merged per label when they are ≤ `merge_gap_s` apart.
- `clip_summary.csv`: one row per clip × class, zeros included.

About 25 s per 20 min clip on CPU.

### 8. Review the detections

```bash
python src/survey/review.py --queue train --list   # build/show the queue, no GUI
python src/survey/review.py --queue train --n 30   # judge the next 30 items
python src/survey/review.py --queue test           # blind heldout test set
```

There are two main queues, each frozen on first use in `data/survey/review/<queue>_queue.csv`
(delete it to resample). For later generations, `--queue train --gen K` builds a further
train queue (`train_gen<K>`) from generation K's own detections (see step 9).

| Queue | Sites | Content | Purpose |
|---|---|---|---|
| `train` | train | ~100 events per predicted class over confidence bins + background windows | training material for the next generation |
| `test` | heldout | 3 s windows stratified by v0 score, **prediction hidden** | fixed benchmark for every generation |

Keys:

| Key | Verdict |
|---|---|
| `1` / `2` / `3` | Drumming / Song / Call (same order as the annotator) |
| `0` or `n` | Noise |
| `y` or `Enter` | prediction is right (train queue only) |
| `u` | unsure (left out of training and evaluation) |
| `Space` / `c` | play the item / play it with its context |
| `b` / `q` | back (to change a verdict) / quit |

Every verdict is saved at once to `<queue>_judgments.csv`, and the last verdict of an
item wins. A new session resumes where the last one stopped. **Judge the whole test
queue**: a partly judged low-score stratum gives very wide intervals.

### 9. Train the next generation and compare

```bash
python src/survey/make_generation.py --gen 1 --dry-run   # clips per class
python src/survey/make_generation.py --gen 1             # cut, build, train
python src/survey/evaluate.py --gens 0 1                 # compare on the test queue
python src/survey/detect.py --gen 1                      # whole survey with gen_1
```

**`make_generation.py`** builds generation K:

- It turns the train verdicts into 3 s clips (`generations/gen_K/survey_clips/`):
  - false positives and confirmed background → **Noise**;
  - confirmed or relabelled calls → their class;
  - Unsure → skipped.
- It adds the clips to the XC clips (`gen_K/train/`, hard links, removed after training
  unless `--keep-train`).
- It trains with v0's hyperparameters and `generations.train_seed` →
  `gen_K/model.tflite`.
- `gen_0/` holds links to v0.
- Verdicts from heldout sites are refused.

**`evaluate.py`** scores every listed generation on **the same** judged test windows:

- Per class, and for "any snipe vs Noise", it gives average precision (AP) plus
  precision and recall at `generations.eval_thresholds`, with 95 % bootstrap intervals.
- For each newer generation it adds the paired difference to the first one listed.
- Each window is weighted by the population of its score stratum, which corrects for
  the stratified sampling.
- **False alarms per hour** (`false_alarms_per_h@t`): windows judged *not* to be a
  class that score ≥ t for it, per hour of heldout audio (~22 h). This needs no snipe
  at the sites, so it measures field false alarms even for classes too rare in the
  field for recall to be measured (Song and Drumming).
- Output: `results/survey/compare.csv` and `pr_<class>.png`.

**Reading the result:** look at the `diff_vs_gen0` column and its interval. If the
interval excludes 0, the change is probably real. Adding a generation later only means
one more value after `--gens`: no new labelling is needed.

**The loop (each generation learns from the previous one's own errors):**

```bash
python src/survey/detect.py --gen 1 --split train           # gen 1 on the train sites
python src/survey/review.py --queue train --gen 1 --list    # freeze queue train_gen1
python src/survey/review.py --queue train --gen 1           # judge it
python src/survey/make_generation.py --gen 2                # train + train_gen1 verdicts
python src/survey/evaluate.py --gens 0 1 2                  # and --testset xc
```

The `train_gen<K>` queue skips anything already judged in the earlier train queues, and
generation K is trained on every train queue judged before it (gen 2: `train` +
`train_gen1`). See `HISTORY.md` for what each generation was trained on.

**From gen 3: the full-survey audit verdicts** (step 12) are training material too.
Generations after `generations.audit_gen` (2) also read every
`audit_<drive>` / `presence_<drive>` queue. Their **heldout rows are left out**: the
audit covers both splits by design, and its heldout events become a test set instead.
Windows are chosen on the recording's 3 s grid from `generations.audit_raw_dir` (gen 2's
full-survey raw results) and cut from the audit snippets (no drive needed).

```bash
python src/survey/make_generation.py --gen 3 --dry-run      # counts per class and source
python src/survey/make_generation.py --gen 3                # cut, build, train
python src/survey/evaluate.py --gens 2 3 --testset xc       # recall on unseen recordists
python src/survey/evaluate.py --gens 2 3                    # field false alarms (survey test set)
python src/survey/evaluate.py --gens 2 3 --testset audit    # heldout audit events
```

`--testset audit` rescores the **heldout** audit events that gen 2 keeps. They are cut
into their 3 s windows (`review/audit_windows/`). An event is kept if its best window
inside the event reaches the class floor (`drive_detection.min_conf` /
`min_event_conf`). These floors are fixed and never tuned on this set. Per class it
reports:
- the share of true events kept (weighted by audit strata)
- the share of false events removed
- the precision of the kept events

The bootstrap resamples heldout sites. Gen 2 is the reference: it keeps 100 % by
construction, and its rescored scores equal the survey's (checked). This measures how a
generation filters gen 2's events. Recall comes from the XC test.

### 10. XC test set (recordists never used for training)

The survey test set has few Song and Drumming positives. A second benchmark comes from
XC recordings that were never annotated, by recordists that **never appear in the
training annotations**:

```bash
python src/tools/annotation_tools.py sample-test --csv data/xc_gallinago_gallinago/metadata.csv \
    --labels-dir data/annotation_labels --dry-run      # show the draw
python src/tools/annotation_tools.py sample-test --csv data/xc_gallinago_gallinago/metadata.csv \
    --labels-dir data/annotation_labels                # freeze data/xc_test/sample.csv
python src/tools/annotator.py --section annotator_xc_test   # annotate it (blind)
python src/survey/evaluate.py --gens 0 1 --testset xc       # compare generations
```

- **The sample** (`xc_test` section) is stratified by XC `type` tag (`drum`, `song`, plus
  `random` from all eligible recordings). The tags are often wrong, so they only enrich
  the sample: the truth is the annotation. At most `max_per_recordist` recordings per
  recordist, and only recordists with at most `max_recordist_recordings` unannotated
  recordings (each test recordist is lost for training, with all their recordings).
  The sample is frozen with its settings (`sample_settings.json`); a changed setting
  stops the run (`--force` to redraw).
- **Annotation** goes to `data/xc_test/labels/`, never to `data/annotation_labels/`, so
  the test recordings never become training clips. The normal annotator skips every
  test recordist (`annotator.exclude_csv`), and `make_generation.py` refuses XC clips
  from a test recordist.
- **Evaluation:** each annotated recording is cut into a grid of 3 s windows
  (`data/xc_test/windows/`). A window is positive for a class when a label overlaps it
  by at least min(1 s, label length), the rule of `make_training_clips.py`. A smaller
  overlap leaves it out of that class. The metrics are unweighted, and the bootstrap
  resamples recordists. Annotate in any order and re-run at any time: only annotated
  recordings count. Output: `results/survey/compare_xc.csv`, `pr_xc_<class>.png`.
- XC recordings are focal and clean, so this measures generalisation to **new
  recordists**, not precision in the field. The survey test set stays the reference
  for that.

### 11. Detect on every recording, straight from the drives

Steps 6–7 only analyse the sampled 20 min clips. `detect_drive.py` runs a generation
(`drive_detection.gen`, gen 2 by default) on **every** recording of a drive. BirdNET reads
the WAV files on the plugged-in drive, and only CSVs are written to the computer:

```bash
python src/survey/detect_drive.py --drive D1 --dry-run     # files, hours per site, done / to do
python src/survey/detect_drive.py --drive D1 --limit 20    # speed test
python src/survey/detect_drive.py --drive D1               # whole drive, with a progress bar
python src/survey/detect_drive.py --post-only --min-conf 0.35   # new threshold on the saved scores, model not re-run
```

- **One drive per run**, as in step 6. The drive's file list is frozen on the first run
  (`files/<drive>_files.csv`, with each file's duration and split). If the sites found
  belong to another drive in `sites.csv`, the run stops.
- **Resumable file by file:** stop at any time (Ctrl-C, unplug) and run the same
  command again. There are about 16 200 one-hour files in all, so a drive takes several
  nights.
- **Speed:** BirdNET runs the custom model on the CPU only (TFLite, no GPU). Tune
  `threads` and `batch_size` in the config after the `--limit 20` test. WSL gets only
  part of the RAM by default: raise it in `C:\Users\<you>\.wslconfig` (`[wsl2]`
  `memory=12GB`), then run `wsl --shutdown`.
- **Output** (`drive_detection.dir`, `data/survey/full/gen_2/`):
  - `events.csv` gives each event's absolute `event_time`.
  - `file_summary.csv` gives one row per file and class, with zeros and the file
    duration, for detection rates per hour, site or date.
  - Both are rebuilt from every drive done so far.
- This is inference, not an experiment. The `split` column is kept so that any later
  review of these detections for training uses train sites only.

#### Running on the MeSU cluster

On the laptop a drive takes days. The Sorbonne MeSU cluster
([docs](https://sacado.sorbonne-universite.fr/documentation-mcmesu/), Slurm) runs a drive
in a few hours on one 64-core node. The whole survey uses about 1 300 of the 30 000 yearly core-hours.
The cluster is a separate computer, so it needs its own copy of the code (git), the model
(scp), a Python environment (pip) and the audio (rsync). The whole survey is run there
from scratch, so all results come from one machine and one environment.

| What | How | Where on the cluster |
|---|---|---|
| scripts, config, `sites.csv`, model labels | `git clone` | `$HOME/becaProject/` |
| `generations/gen_2/model.tflite` (gitignored) | `scp` from the laptop | same path in the clone |
| Python packages | `pip install -r cluster/requirements.txt` | `$HOME/becavenv/` |
| audio of one drive | `rsync` from the laptop | `$SCRATCH/beca/audio/<drive>/` |

Commands are typed on the login node (setup, copies, submitting only: heavy processes
there are killed). The detection runs as a Slurm job (`cluster/detect_drive.sbatch`).

```bash
# 1. once, on the laptop: SSH key, then log in
ssh-keygen                                              # if you have no key yet
ssh-copy-id <user>@login.mesu.sorbonne-universite.fr
ssh <user>@login.mesu.sorbonne-universite.fr

# 2. once, on the cluster: project + environment
git clone https://github.com/Dvd-Luc/becaProject.git $HOME/becaProject
module avail python                                     # set the exact name in cluster/detect_drive.sbatch
module load python/3.12
python -m venv $HOME/becavenv && source $HOME/becavenv/bin/activate
pip install -r $HOME/becaProject/code/cluster/requirements.txt
mkdir -p $HOME/becaProject/code/logs $SCRATCH/beca/audio
#    and from the laptop (in code/):
scp generations/gen_2/model.tflite <user>@login.mesu.sorbonne-universite.fr:becaProject/code/generations/gen_2/

# 3. per drive, on the laptop with the drive plugged in (resumable: re-run if cut)
rsync -av --partial --prune-empty-dirs --include='*/' --include='*.wav' --exclude='*' \
    /mnt/d/2026_Becacoustic_LPO_BFC_part1/ <user>@login.mesu.sorbonne-universite.fr:/scratch/<user>/beca/audio/D4/

# 4. per drive, on the cluster (in $HOME/becaProject/code)
srun -p std -c 4 --time=1:0:0 --pty bash                # short interactive job: reads every WAV header
python src/survey/detect_drive.py --drive D4 --root $SCRATCH/beca/audio/D4 --dry-run
exit
sbatch --export=ALL,DRIVE=D4,EXTRA="--limit 64" cluster/detect_drive.sbatch   # test job: speed line in logs/
sbatch --export=ALL,DRIVE=D4 cluster/detect_drive.sbatch                      # the whole drive
myqueue; tail -f logs/beca_detect_<jobid>.out           # follow; scancel <jobid> to stop

# 5. on the laptop: fetch the results, rebuild the tables, then commit [DATA]
rsync -av <user>@login.mesu.sorbonne-universite.fr:becaProject/code/data/survey/full/gen_2/ data/survey/full/gen_2/
python src/survey/detect_drive.py --post-only
```

- The source of the rsync is the folder that holds the site folders (the `rel_path`
  column of `files/<drive>_files.csv` is relative to it).
- Files not uploaded yet are skipped with a warning, so a job can start before the upload ends.
  A job that hits the walltime can be submitted again. Both resume where they stopped.
- Code changes: commit and push on the laptop, then `git pull` on the cluster. The job
  log prints the commit it ran.
- `$SCRATCH` is purged after 3 months and is not backed up. Delete
  `$SCRATCH/beca/audio/<drive>` once its results are back; the originals stay on the drives.

### 12. Audit the full-survey events

Step 11 gives thousands of events per drive. The test sets don't say how many of them are
right, because they hold too few field Song and Drumming events. `audit_events.py` draws a random
sample of the events, you judge it blind, and the verdicts give the **precision of all the
events**, with confidence intervals. Do this one drive at a time, as soon as its events exist:

```bash
python src/survey/audit_events.py sample --drive D1 --dry-run   # strata and their populations
python src/survey/audit_events.py sample --drive D1             # draw and freeze the sample
python src/survey/audit_events.py cut --drive D1 --root /mnt/d  # snippets: audit + presence candidates
python src/survey/review.py --queue audit --drive D1            # judge (blind), in several sessions if needed
python src/survey/audit_events.py presence --drive D1           # after the audit: sites still to confirm
python src/survey/review.py --queue presence --drive D1         # judge them (blind), snippets already cut
python src/survey/audit_events.py report                        # precision + presence, every drive so far
```

- **sample**: each event goes in a stratum and a random sample is drawn per stratum,
  seeded by `review.seed` and the drive. `review.audit_strata` picks the strata:
  - `site_bin` (default, D1 on): class × site × `max_conf` bin (`review.audit_bins`),
    `review.audit_per_site_bin` (3) events each. Every site is heard at low and high
    scores, so precision can be compared between sites and between bins, and one hotspot
    can't fill the sample. `--per-stratum N` draws more for one drive (D2: 5).
  - `site`: class × site, `review.audit_per_site` (5) events each, drawn at random, so
    mostly low scores are heard.
  - `bin`: class × `max_conf` bin, `review.audit_per_stratum` (30) each. Used for D4, to
    choose the thresholds.

  The sample, the population of each stratum and the stratum kind are frozen in
  `data/survey/review/audit_<drive>_{queue,strata}.csv`. `--force` redraws, and only
  while nothing is judged yet.
- **One audit per generation.** The queues belong to the generation in
  `drive_detection.gen`: `audit_<drive>` for gen 2, and `audit_gen<K>_<drive>` after it
  (`review.py --queue audit --drive D` picks the right one). A gen 3 event that gen 2's
  audit already judged gets that verdict copied, so you never judge the same audio twice:
  - "Already judged" means same file, same class, start within 1.5 s, and not Unsure.
  - The copy is marked `copied:<queue>@<time>` in the time column.
  - The sample stays random. Copied verdicts never block `--force`.
  - A site × class confirmed by any earlier generation stays **present**, and is never
    queued for the presence check again.
- **presence**: for each class of `review.presence_labels` (Drumming, Song) and each site
  with events of it: if no verdict confirms the class there yet, the best events not yet
  judged (highest `max_conf`) are queued, up to `review.presence_per_site` (5) judged per
  site. The queue is frozen in `presence_<drive>_queue.csv`. Run it after the audit is judged, so
  sites the audit already confirmed are skipped.
- **cut**: writes the audio of every queued event (audit and presence), with 10 s on each
  side, as a snippet in `data/survey/full/gen_2/audit/snippets/` (gitignored WAVs). It
  reads from the plugged-in drive or from the cluster copy
  (`--root $SCRATCH/beca/audio/<drive>`). Before the presence queue exists, it also cuts
  the presence candidates: the best `presence_per_site` events per class × site outside
  the audit queue. The presence queue only ever takes events among them, so one cut right
  after `sample` covers both queues and the drive is needed once.
- **review**: the queue is **blind**, so neither the predicted class nor the score is shown. You
  say what each event really is (class, Noise or Unsure).
- **report** (`--split heldout|train` for one split; this needs site strata): an event
  is correct when the verdict is its predicted class (Unsure left out). For gen 3 and
  later, **heldout precision is the number to report**. These generations trained on
  the train sites' audit verdicts, so their train-site precision is optimistic. Precision is given per class × bin, per class, over the events still kept by
  `min_event_conf` ("kept") and overall, each stratum weighted by its population. The 95 %
  CI comes from a bootstrap that resamples sites, because events cluster by site. It also gives:
  - correct / judged per class × site, with a Wilson CI
  - a confusion table (predicted → verdict, judged and estimated events)
  - each site's presence status per class: present, not confirmed or not judged

  Outputs: `results/survey/audit_full_gen2.csv`, `_sites.csv`, `_confusion.csv`,
  `presence_full_gen2.csv`. Run it any time; strata not judged yet are flagged.
- **Call threshold:** the D4 audit gave Call a precision of 0.46 and 0.65 in the 0.5–0.7 and 0.7–0.9 bins, but
  0.97 at ≥ 0.9. So `drive_detection.min_event_conf: {Call: 0.9}` keeps a Call event
  only if its best window reaches 0.9. Windows are still merged from 0.5, so the kept
  events are exactly the audited ≥ 0.9 stratum. Changing it needs only `detect_drive.py --post-only`.
- The sample covers train and heldout sites, which is fine for measuring the frozen model.
  Only train-site verdicts may ever be used to train a later generation.
- The audit gives precision only. Recall still comes from the XC test set.

### 13. Visualise the full survey (local page)

```bash
python src/survey/build_viz.py --dry-run   # matched positions, counts, status (prints no position)
python src/survey/build_viz.py             # writes results/survey/viz/full_gen3.html (~10 MB)
```

Run `audit_events.py report --split train` and `--split heldout` first, so the page can show precision.

This writes **one self-contained HTML file** (~10 MB). Open it by double-click: no server.
It works on a phone too: filters in a drawer, a sticky section bar, tap for details.
Explanations are folded behind the **i** next to each panel title, and each panel's SVG / PNG / CSV
downloads are in its **Export** menu.
Data, d3, d3-tile, the département outlines and the audio examples are inlined.

- **Thresholds:** the viewer picks the threshold per class (0.5 / 0.7 / 0.9; ● marks the
  survey thresholds). Every view is recomputed, and the audited precision at that
  threshold is shown for heldout sites (train in smaller type).
- **Audio:** one 8 s example per site × class, with a spectrogram, in the site card.
  "Verified" means you judged that event correct. Clips are cut from the audit
  snippets (no drive needed) and cached in `results/survey/viz/audio_gen<K>/`.

Linked views:
- **Map:** bubble area = kept events per hour recorded. Fill = presence status from
  your verdicts. Zoom and pan, one button per département. **Base map** (off by default):
  Plan IGN, aerial photo, relief, land cover (CORINE 2018) or wetlands and water
  (CORINE HR 2015). It is loaded from IGN's Géoplateforme only when chosen, so it needs
  internet; IGN then sees which tiles are shown, never the site positions.
- **Selected site:** events per hour evening by evening; hatched evenings had no
  recording. Scroll to zoom the dates, drag to pan, "Reset zoom".
- **Presence grid:** status per site × class, or detection per month. Sites are grouped
  by **département**: the code in the site name (`_25_` = Doubs), or for names without
  one (the RNNSG sites), the département holding the position. Labels read
  "commune · recorder" (`Sainte Colombe · 041`).
- **Time from sunset:** read from the recorders' schedule, not the clock. Each evening is
  two one-hour files from sunset (`viz.schedule_start_min: 0`): the first file starts at
  sunset in the recorder's own clock (checked against computed sunsets, median +1 min;
  most clocks stayed on UTC+1 after the March time change). Events are placed from the
  evening's first file start. Set-up tests (before noon, shorter than 30 min) are left
  out; a single-file evening is matched to its neighbours as file 1 or file 2. 5-min
  bins, 2 or 1 min when zoomed in.
- **Season overview:** all sites pooled, events per hour each week, and the number of
  sites recording. For each site: the recording period and the first and last
  detection. A hollow dot is within 3 days of the start or end of recording.
- **Weekly heatmap:** events per hour, by site and week.

Each view has **SVG / PNG / CSV** buttons (map exports embed the base-map tiles, © IGN).
Re-run the script after each new drive or audit.

- **Microphone positions are sensitive** (snipe breeding sites) **and never leave this computer.**
  - The GeoPackages (`viz.sources`, copied to `code/data/geo/`) and the page are gitignored.
  - The script never prints a position.
  - Don't upload the page or map exports anywhere. `viz.round_coords` can blur the positions written into the page.
  - Only `data/geo/departements.geojson` (public outlines, france-geojson / Etalab
    Licence Ouverte, rounded to 0.001°) is committed.
- Positions are matched to `sites.csv` by the exact site name (`Station` column), or else
  by the recorder number (`MIC011`, `Mic049`).
- Needs `pyproj` (Lambert-93 → WGS84) in `.becavenv`.

---

## Script reference

All paths relative to `code/`.

| Script | Stage | Reads | Writes |
|---|---|---|---|
| `src/tools/recording_fetch_xc.py` | 1 fetch | XC API (`XC_API_KEY`) | `data/xc_gallinago_gallinago/` |
| `src/tools/annotator.py` | 2, 10 annotate | XC audio + metadata, `project.yaml` | `data/annotation_labels/*.txt` (or `data/xc_test/labels/`), status CSV |
| `src/tools/annotation_tools.py` | 2 annotate, 10 sample | metadata, label files | `annotation_status.csv`, batch folders, `data/xc_test/sample.csv` |
| `src/training/label_stats.py` | 2–3 | label files, metadata, clip metadata | console summary |
| `src/training/make_training_clips.py` | 3 clips | audio + labels | `outputs/clips/` |
| `src/training/make_folds.py` | 4 folds | clip + XC metadata | `outputs/folds/fold_K/`, `fold_stats.csv` |
| `src/training/birdnet_training.sh` | 4 train | `outputs/folds/` | `models/fold_K.tflite` |
| `src/training/model_analysis.sh` | 4 evaluate | folds + models | `models/preds/fold_K/` |
| `src/tools/confusion_matrix.py` | 4 evaluate | folds + predictions | confusion-matrix CSV |
| `src/training/run_learning_curve.py` | 5 learning curve | clip + XC metadata | `outputs/lc/`, `models/lc/` |
| `src/training/plot_learning_curve.py` | 5 learning curve | `models/lc/`, `outputs/lc/` | `results/lc/` |
| `src/survey/sample_clips.py` | 6 sample | a survey drive | `data/survey/{plan,clips}/`, `sites.csv`, `manifest.csv` |
| `src/survey/detect.py` | 7 detect | manifest, clips, model | `data/survey/detections/` (or `generations/gen_K/detections/`) |
| `src/survey/detect_drive.py` | 11 full survey | a survey drive, `sites.csv`, a generation | `data/survey/full/gen_K/` (`files/`, `raw/`, `events.csv`, `file_summary.csv`) |
| `src/survey/audit_events.py` | 12 audit | full-survey `events.csv` + `files/`, a drive, audit judgments | `data/survey/review/audit_<drive>_{queue,strata}.csv`, `data/survey/full/gen_K/audit/snippets/`, `results/survey/audit_full_gen<K>*.csv` |
| `src/survey/build_viz.py` | 13 visualise | full-survey tables, audit/presence results, microphone GeoPackages | `results/survey/viz/full_gen2.html` (local only) |
| `src/survey/review.py` | 8 review, 12 audit | detections, manifest, clips (or audit snippets) | `data/survey/review/` |
| `src/survey/make_generation.py` | 9 train | train judgments, XC clips | `generations/gen_K/` |
| `src/survey/evaluate.py` | 9, 10 compare | test judgments or XC test labels, generations | `results/survey/`, `gen_K/{test,xc}_scores.csv` |
| `src/common/config.py` | all | `config/project.yaml` | – |

---

## Documentation site

Every module, function and class has a Google-style docstring (arguments, return values,
errors). To browse them as a website with search, from `code/`:

```bash
pdoc --docformat google src/common/config.py src/survey/*.py src/tools/*.py src/training/*.py
```

This opens http://localhost:8080; the page reloads when a file is saved, and Ctrl+C
stops the server. Add `-o docs/` to write a static HTML copy instead. Only Python files
are documented: the YAML, shell scripts and data are not.

---

## Good to know

- **Run from `code/`**: all default paths are relative to it.
- **Label files:** lines starting with `\` (Audacity frequency ranges) are skipped, and
  decimal commas are accepted. Keep the label names consistent, since they become class
  folders.
- **Frozen plans and queues:** `data/survey/plan/<drive>_*.csv` and
  `data/survey/review/<queue>_queue.csv` are reused as they are. Delete them on purpose
  to resample.
- **heldout sites** are only used by the test queue and `evaluate.py`;
  `make_generation.py` refuses their verdicts.
- **XC test recordists** (`data/xc_test/sample.csv`) are never annotated for training;
  `make_generation.py` refuses their clips.
- **Changing the detection model or overlap** invalidates `detections/raw/`: `detect.py`
  stops rather than mixing results (`--force` re-runs everything).
- **`confusion_matrix.py`** writes to its third argument, so give it a `.csv` path; it
  refuses anything else.
