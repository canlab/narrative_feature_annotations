# Project Contents & User Guide

A map of everything in this repository: the tools, datasets, and extracted
derivatives, what is included so far, and how to load, view, and inspect them.
For a hands-on tour run [`docs/walkthrough.m`](walkthrough.m) in MATLAB.

> **What this project is.** Infrastructure that turns a movie or audio story into
> **second-by-second, hierarchical, computational annotations** (visual, audio,
> language, social, situational, affective), stored in a constant-shape format that
> loads into MATLAB, plus tools to analyze the annotation structure across a corpus
> and to design high-variance / low-redundancy stimulus sets.

## Status at a glance

| Phase | What | State |
|-------|------|-------|
| 1 Scoping review | best-in-class tools per feature class | ✅ `docs/scoping_review/` |
| 2 Pipeline | 24 extractors → constant-shape annotations + MATLAB reader | ✅ `src/nfe/`, `matlab/` |
| 3 Corpus | manifest + batch runner; **105 stimuli annotated** (75 audiovisual + 29 audio stories + 1 text) | ✅ `annotations/corpus/` |
| 4 Analysis | corpus reader, viewer, correlation/PCA/network, design tool, web search, draft paper | ✅ `matlab/`, `analysis/web/`, [`REVIEW_PAPER.md`](REVIEW_PAPER.md) |

---

## Folder map

```
README.md                  project overview + quickstart
requirements.txt           Python deps (pipeline)
data/
  manifest.csv / .json     stimulus catalog (105 stimuli) — tools/build_manifest.py
  movies/spacetop/         lab fMRI stimulus clips (49)        [internal]
  movies/open/             CC-BY Blender films (3) + SOURCES.md
  lexicons/                optional psycholinguistic norm CSVs (README.md)
  stories/narratives/      29 Narratives spoken-story audio clips (+ SOURCES.md)
  stories/samples/         pure-text sample story (.txt)
  movies/hcp/, movies/camcan/  placeholders for credentialed stimuli (see EXTERNAL_STIMULI.md)
schema/
  channel_template.json    the 103-channel constant-shape template
  annotation_schema.json   v0.1 pure-JSON profile schema
  example_annotation.json  tiny worked example
src/nfe/                   the Python annotation pipeline (see "Tools")
matlab/                    the MATLAB readers + analysis (see "Tools")
annotations/
  corpus/<id>/<id>.h5      DERIVATIVES: one annotation per stimulus (+ .manifest.json)
  corpus/corpus_index.csv  batch status/index
analysis/figures/          generated analysis figures (PNG)
analysis/corpus_stats.json corpus summary numbers the docs cite (refreshAnalysis)
analysis/web/              interactive segment search (index.html + segments.json + README)
docs/
  CONTENTS.md              ← this file
  walkthrough.m            runnable MATLAB tour
  scoping_review/          Phase 1 review (hierarchy, recommendations, ...)
  design/                  format spec, plans, per-phase status
tools/                     helper scripts (manifest, template, review assembly)
```

---

## Datasets

- **Corpus manifest: 105 stimuli** (`data/manifest.csv`) — 49 `spacetop` audiovisual clips + 4 short
  films (3 CC-BY Blender open films + *Kung Fury*) + **29 Narratives spoken-story audio clips**
  (`data/stories/narratives/`, ~5.3 h; OpenNeuro ds002345, Nastase et al.) + 1 pure-text sample
  story. 75 audiovisual, 29 audio-only, 1 text-only. Add media under `data/movies/<source>/`
  (movies/audio) or `data/stories/<source>/` (audio/text stories) and re-run
  `tools/build_manifest.py`. See [`ADDING_MOVIES.md`](ADDING_MOVIES.md); credentialed sets (HCP,
  CamCAN) in [`EXTERNAL_STIMULI.md`](EXTERNAL_STIMULI.md). All 105 are annotated; corpus-wide
  analysis focuses on the audio/language channels shared across modalities, with visual/social/
  affective structure on the 75-stimulus audiovisual subset (see [`REVIEW_PAPER.md`](REVIEW_PAPER.md) §5–6).
- **Lexicons (optional):** drop `data/lexicons/<field>.csv` (valence, arousal,
  dominance, concreteness, aoa) to light up those per-word channels; absent → NaN.

---

## Extracted derivatives

One annotation per stimulus at `annotations/corpus/<id>/`:

- **`<id>.h5`** — canonical HDF5. Hierarchical groups mirror the feature taxonomy:
  ```
  /time/        common 1 Hz grid (rate_hz, t_start_sec, n_samples, time_sec)
  /stimulus/    id, title, modality, duration, source, sha256
  /features/    visual/ audio/ language/ social/ situation/ affect/
                  → each leaf = one channel dataset [n_samples (× dim)] + attrs
                    (dtype, applicable, units, model, version, native_rate_hz, resample)
  /human/       reserved, empty — slots for later human annotation
  /provenance/  per-channel model registry
  ```
- **`<id>.manifest.json`** — readable sidecar: same hierarchy + metadata, no bulk arrays.

**Constant shape.** Every file has the **same 103 channels** (the template). Channels
not produced for a stimulus (a class that doesn't apply to the modality, or a pass not
run) are present with `applicable=false` and all-`NaN`, so the corpus stacks into
rectangular matrices. Full spec: [`design/ANNOTATION_FORMAT.md`](design/ANNOTATION_FORMAT.md).

The 103 channels span: **visual** (37 — low-level, semantic SigLIP2/DINOv2, motion,
depth, action, faces, pose, saliency), **audio** (21 — low-level, AudioSet/CLAP,
speech), **language** (22 — lexical, syntax, surprisal), **affect** (14 — text emotion/
sentiment, vocal, EmoNet image emotion, facial affect, VLM depicted), **situation**
(5 — incl. GSBS event boundaries), **social** (4). What each pass computes:
[`design/PHASE2_STATUS.md`](design/PHASE2_STATUS.md).

---

## Tools

### Python pipeline — `src/nfe/` (annotate media)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# annotate ONE movie (CPU passes always on; add MPS/VLM passes as flags):
PYTHONPATH=src .venv/bin/python -m nfe.run data/movies/open/BigBuckBunny.mp4 \
    --vision --audio-hl --events --template schema/channel_template.json
# annotate the WHOLE corpus (resumable, crash-safe):
PYTHONPATH=src .venv/bin/python -m nfe.batch --manifest data/manifest.csv \
    --out annotations/corpus --template schema/channel_template.json \
    --vision --audio-hl --events
```

Pass flags: `--vision` (SigLIP2/DINOv2/RAFT/depth/VideoMAE/faces/pose/saliency),
`--audio-hl` (AST/CLAP/vocal-affect/text-emotion/surprisal), `--reason` (Qwen2.5-VL —
slow), `--events` (GSBS). Modules: `ingest` (PyAV decode), `extractors/` (the 20
passes), `emit` (HDF5+JSON), `pipeline`/`run`/`batch`, `schema_registry` (skeleton
fill). Helpers in `tools/`: `build_manifest.py`, `build_channel_template.py`,
`build_search_index.py` (segment index for the web search interface).

### Web — `analysis/web/` (interactive segment search)

Browser tool to rank segments by any combination of features and play the matching
moment. Serve from the project root, then open the page:

```bash
python3 tools/serve.py           # from the project root (Range-enabled, so video seeking works)
# open http://localhost:8000/analysis/web/index.html
```

Rebuild the index after annotating more stimuli:
`PYTHONPATH=src .venv/bin/python tools/build_search_index.py --seglen 5`.
Details: [`../analysis/web/README.md`](../analysis/web/README.md).

### MATLAB — `matlab/` (load, view, analyze)

| Function | Purpose |
|----------|---------|
| `readAnnotations(path)` | load one `.h5`/folder/JSON → struct (stimulus, time_sec, features) |
| `getFeature(ann, "audio/low_level/mfcc")` | one channel + metadata by hierarchical path |
| `featuresToTimetable(ann)` | scalar channels → timetable on the common grid |
| `readAnnotationCorpus(folder)` | stack the whole corpus → `C.X [timepoints × channels]` (**scalar** channels only) |
| `annotationMovieViewer(movie, ann)` | play movie with synced annotation time series + marker |
| `analyzeCorpus(C)` | correlation heatmap, PCA, channel + class network graphs |
| `selectStimulusSet(C)` | D-optimal high-variance / low-redundancy segment selection |
| `featureInfo()` | label/category table for **all** expanded variables (class, subclass, level, model, color) |
| `featuresToTable(ann)` | one clip → wide table with **every vector expanded** into per-component columns (~2.7k vars) |
| `readAnnotationCorpusFull(folder)` | stack the whole corpus with vectors expanded → `C.X [timepoints × ~2.7k vars]` + `C.info` |
| `plotFeatureMatrix(C)` | heatmap of the full feature time series, color-coded by category |
| `factorAnalysisCorpus(C)` | exploratory factor analysis (EFA / `factoran`) across all features + color-coded loadings plot |
| `extractCategoryFactors(C)` | factor analysis **within each model/category** → `C.extracted_factors` (per-model factor time series), saveable to `.mat` |
| `plotFactorScores()` | load the saved factors and visualize 6 ways (time series, mango heatmap, force-directed graph, correlation matrix, t-SNE, UMAP) with CANlab tools; saves svg+png to `matlab/figures/` |

---

## How to load, view, inspect (MATLAB quick recipes)

```matlab
addpath matlab

% 1) Inspect ONE stimulus
ann = readAnnotations("annotations/corpus/ses-01_run-01_order-04_content-parkour");
tt  = featuresToTimetable(ann);          % scalars as a timetable
stackedplot(tt(:, ["visual__low_level_static__luminance","audio__low_level__rms"]))
mf  = getFeature(ann, "audio/low_level/mfcc");   % a vector channel [n × 13]

% 2) WATCH a movie with its annotations scrolling underneath
m = "data/movies/spacetop/videos/ses-01/ses-01_run-01_order-04_content-parkour.mp4";
annotationMovieViewer(m, "annotations/corpus/ses-01_run-01_order-04_content-parkour")

% 3) Analyze the whole CORPUS (scalar channels — correlation / PCA / design tool)
C   = readAnnotationCorpus("annotations/corpus");
res = analyzeCorpus(C);                   % figures → analysis/figures/
sel = selectStimulusSet(C, "K", 20);      % design a stimulus set; sel.table

% 4) Load the FULL expanded feature set (~2.7k variables: every vector expanded),
%    visualize it color-coded by category, and extract factors with EFA
F   = readAnnotationCorpusFull("annotations/corpus");   % F.X [timepoints × ~2768], F.info labels
plotFeatureMatrix(F, "Clip", "BigBuckBunny");           % heatmap, color-coded by class
fa  = factorAnalysisCorpus(F, "NumFactors", 10);        % EFA + loadings plot; fa.scores = per-timepoint factors

% Python inspection (alternative): h5py / pandas on <id>.h5 and corpus_index.csv
```

> **Two corpus readers.** `readAnnotationCorpus` returns only the **scalar** channels
> (what `analyzeCorpus`/`selectStimulusSet` expect). `readAnnotationCorpusFull` expands
> every **vector** channel (SigLIP/DINOv2/CLAP embeddings, AudioSet/action posteriors,
> EmoNet, MFCC, …) into one column per component — the full ~2,768-variable matrix with a
> companion `featureInfo` label table (class / subclass / level / model / color) for
> color-coded plotting and factor analysis. See the
> [feature map](FEATURE_MAP.md) for how the variables are organized.

```bash
# 4) SEARCH segments by feature in the browser (serve from project root)
python3 tools/serve.py           # then open http://localhost:8000/analysis/web/index.html
```

See [`walkthrough.m`](walkthrough.m) for the same steps, runnable section by section.

---

## Documentation index

- **Feature map (all 95 channels):** [`FEATURE_MAP.md`](FEATURE_MAP.md) — a graphical map of
  the feature hierarchy; editable SVG at `analysis/figures/feature_map.svg`
- **Adding movies/stories:** [`ADDING_MOVIES.md`](ADDING_MOVIES.md) — how to grow the
  corpus and refresh all derivatives + corpus-specific doc numbers
- **External stimuli (HCP, CamCAN):** [`EXTERNAL_STIMULI.md`](EXTERNAL_STIMULI.md) — how to
  obtain the credentialed/copyrighted neuroimaging movie stimuli and integrate them
- **Review paper (draft):** [`REVIEW_PAPER.md`](REVIEW_PAPER.md) — models/algorithms +
  the empirical structure of the annotation space (correlation, dimensionality, networks,
  contingency on stimulus type, design tool)
- **Scoping review (Phase 1):** [`scoping_review/`](scoping_review/README.md)
- **Format spec:** [`design/ANNOTATION_FORMAT.md`](design/ANNOTATION_FORMAT.md)
- **Plan & per-phase status:** [`design/IMPLEMENTATION_PLAN.md`](design/IMPLEMENTATION_PLAN.md),
  [`PHASE2_STATUS.md`](design/PHASE2_STATUS.md), [`PHASE3_CORPUS.md`](design/PHASE3_CORPUS.md),
  [`PHASE4_ANALYSIS.md`](design/PHASE4_ANALYSIS.md)
- **Frozen feature set:** [`design/PHASE2_CORE_SET_PROPOSAL.md`](design/PHASE2_CORE_SET_PROPOSAL.md)
- **Deployment:** [`design/DEPLOYMENT_FEASIBILITY.md`](design/DEPLOYMENT_FEASIBILITY.md)
