# Phase 4 — Analysis & Dissemination

MATLAB tools that load the annotated corpus and analyze the structure of the
annotations across stimuli, plus interactive viewing. Builds on the constant-shape
corpus from Phase 3 (`annotations/corpus/`).

## Done

### `readAnnotationCorpus(folder)`
Loads every `<id>/<id>.h5` under `folder` into one analysis structure:
- `C.ids` / `C.ann` — stimulus ids and full annotation structs
- `C.channels` — shared scalar-channel set (constant-shape → identical across stimuli)
- `C.X` `[sumT x P]` — all timepoints of all stimuli stacked into one rectangular matrix
- `C.stim`, `C.time_sec`, `C.nT` — stimulus id and within-stimulus time per row of `X`

Robust to partially-written files (skips + warns), so it can run while the Phase 3
batch is still filling the corpus. `X` is the direct input to correlation / PCA / the
design tool. Example:

```matlab
C = readAnnotationCorpus("annotations/corpus");
R = corr(C.X, 'rows','pairwise');          % cross-feature correlation across the corpus
[coeff,score,~,~,expl] = pca(normalize(C.X)); % principal components of the annotation set
```

### `annotationMovieViewer(movieFile, annPath)`
Plays a movie with its annotation time series synced below and a red marker on each
series tracking the playback position. Play/Pause + scrub slider; `"Channels"` and
`"Speed"` name-value options. Example:

```matlab
m = "data/movies/spacetop/videos/ses-01/ses-01_run-01_order-04_content-parkour.mp4";
annotationMovieViewer(m, "annotations/corpus/ses-01_run-01_order-04_content-parkour")
```

### `analyzeCorpus(C)`
Structure analysis over the stacked corpus matrix → figures in `analysis/figures/`:
clustered cross-feature **correlation heatmap**, **PCA** scree/cumulative variance,
a channel **correlation network** (nodes colored by feature class, edge `|r|≥thresh`),
and a **feature-class network** (edge = mean `|r|` between classes). Returns `R`,
cluster `order`, PCA `coeff/score/explained`, and `classR`. NaN-aware z-scoring before
PCA. NaN-aware. On the 83-stimulus modality-mixed corpus, 12 PCs reach 80% over the 26
audio/language channels shared across modalities; on the 75-stimulus audiovisual subset,
18 PCs reach 80%, with modest class couplings — strongest affect↔social (mean |r| 0.15), then
visual↔social and audio↔social (0.11) — roughly stable across content. (The depicted-affect,
social, and situational scalars come from the VLM reasoning pass, now run across the whole
audiovisual subset; the language class stays under-represented in the dialogue-light AV clips;
for the visual/social structure analyze the audiovisual subset — see REVIEW_PAPER §5–6.)

### `selectStimulusSet(C)`
Experimental-design / stimulus-selection tool. Splits every stimulus into fixed-length
candidate segments and **greedily maximizes `log det(cov)`** of the concatenated
annotation time series (projected onto the leading PCs) — jointly rewarding high
variance across the major feature dimensions and independence of the feature time
series (D-optimality over the annotation space). Returns `sel.table` (selected
segments) and `objTrace` vs a random baseline. Candidate segments dominated by
missing/imputed data are excluded (they cannot win the objective by being blank). On the
83-stimulus corpus, selecting 20 × 10 s segments reaches `log det(cov)` 12.8 vs 2.5 for
random — well over 5× the generalized variance — drawing from 16 distinct stimuli across
modalities.

### `refreshAnalysis(corpusFolder)`
One-command refresh matching the docs: full-corpus stats + design figure, audiovisual-subset
structural figures (Figs 1–5 of the review paper) and class couplings, the speech-rich vs
speech-sparse comparison, and `analysis/corpus_stats.json` with both full and AV-subset
numbers (missing classes export as NaN, never empty).

See also [`../CONTENTS.md`](../CONTENTS.md) (full guide) and [`../walkthrough.m`](../walkthrough.m).

### Interactive web search — `analysis/web/`
Static browser app over a precomputed segment index (`tools/build_search_index.py` →
`analysis/web/segments.json`; 5684 × 5 s segments, 174 searchable channels — index-like
channels and not-applicable skeleton fills are excluded, and interpretable vector channels are
expanded into per-component columns so e.g. EmoNet's 20 emotion categories and the 8 facial
expressions are individually searchable). Pick any combination of features,
toggle High/Low, and segments are ranked by mean z-score (with a features-covered count when
a segment lacks some selected channels); **▶ Play** seeks the actual clip to that segment
(audio stories play as sound; text-only has no playback). Serve from the project root
(`python3 tools/serve.py` → `analysis/web/index.html`; the bundled server adds HTTP
Range support so video seeking works). Rankings validate well —
high flow surfaces action scenes, high word-rate the dialogue clips, EmoNet "Aesthetic
Appreciation" the beach-sunset clips, etc. Launch +
rebuild notes: [`../../analysis/web/README.md`](../../analysis/web/README.md).

### Review paper
[`../REVIEW_PAPER.md`](../REVIEW_PAPER.md) — draft review covering the models/algorithms
behind each annotation and the empirical structure of the annotation space (redundancy,
dimensionality: 12–18 PCs→80%, class networks — couplings are modest, strongest
affect↔social ≈ 0.15, and roughly stable across speech density — and the D-optimal design
tool). The group-conditioned figure is `analysis/figures/class_coupling_by_group.png`.

## Phase 4 complete
Corpus reader, synced viewer, correlation/PCA/network analysis, experimental-design tool,
web search interface, and the draft review paper are all in place. Future work
(per the paper's §9): swap in production models, scale the VLM reasoning pass, add
elicited-affect + diarization, validate against human ratings / brain data.
