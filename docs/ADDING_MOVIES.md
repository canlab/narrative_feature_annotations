# Adding Movies and Stories to the Dataset

How to grow the corpus and refresh everything that depends on the specific set of
stimuli (annotations, search index, analysis figures, and the corpus-specific numbers
in the docs). The extractor code, schema/template, scoping review, and format spec do
**not** change when you only add media.

---

## For the user

### 1. Add the media
Drop files under a source subfolder of `data/movies/`:

```
data/movies/<source>/...        e.g. data/movies/myset/clip01.mp4     (movies / audio)
data/stories/<source>/...       e.g. data/stories/myset/story01.txt   (stories)
```

- **Movies:** any video container (`.mp4 .mkv .mov .avi .webm .m4v`). Modality is
  detected automatically (audiovisual / video-only).
- **Audio stories:** audio files (`.wav .mp3 .m4a .flac .aac .ogg`) — detected as
  `audio-only`; visual channels are emitted as `NaN` (not applicable).
- **Pure-text stories:** plain-text files (`.txt .text .story`) — detected as
  `text-only`. The text is placed on an estimated **reading-rate timeline** (~3 words/s)
  so it shares the 1 Hz grid; the language, text-emotion, surprisal, and event passes run,
  and visual/audio channels are `NaN`. (Markdown `.md` is treated as documentation, not a
  story.) Put stories under `data/stories/<source>/` (or `data/movies/` — both are scanned).
- Use one subfolder per **source** (keeps provenance/rights tidy). If the material has
  licensing terms, add a `SOURCES.md` in that subfolder (see `data/movies/open/SOURCES.md`).
- **Credentialed neuroimaging stimuli** (HCP 7T movies, CamCAN Hitchcock clip): these need a
  data-access request — see [`EXTERNAL_STIMULI.md`](EXTERNAL_STIMULI.md). Placeholder folders
  `data/movies/hcp/` and `data/movies/camcan/` are ready for them.

### 2. Refresh — two options

**Option A (easiest):** tell Claude *"I added media under `data/movies/<source>/`,
refresh the dataset and docs"*. Claude runs the runbook below and updates everything.

**Option B (do it yourself):**
```bash
# 1) manifest + annotate NEW stimuli (resumable) + rebuild search index
tools/refresh_corpus.sh                 # add --reason for the VLM pass; --no-skip to redo all
# 2) regenerate figures + the corpus stats the docs use (in MATLAB)
matlab -batch "addpath matlab; refreshAnalysis('annotations/corpus')"
# 3) update the corpus-specific numbers in the docs from analysis/corpus_stats.json
#    (ask Claude, or edit the files listed in the runbook below)
```

### 3. Verify
- `annotations/corpus/corpus_index.csv` — one row per stimulus, `status` = ok/skipped/error.
- `analysis/figures/*.png` and `analysis/corpus_stats.json` — regenerated.
- Web search: rebuilt automatically by step 1; reopen `analysis/web/index.html`.
- Spot-check a new stimulus in the viewer: `annotationMovieViewer(movie, annDir)`.

### Notes
- **Resumable:** existing annotations are skipped; only new stimuli are processed. Long
  films take a while (especially `--reason`); run overnight if needed.
- **Template is stable:** adding media does **not** change the 103-channel template. Only
  rebuild it (`tools/build_channel_template.py`) if you change the *extractor set* — then
  re-annotate everything with `--no-skip`.
- Hardware/runtime guidance: `docs/design/PHASE3_CORPUS.md`, `design/DEPLOYMENT_FEASIBILITY.md`.

---

## Agent runbook (for Claude)

When asked to refresh after media was added:

1. **Annotate + index** — `bash tools/refresh_corpus.sh` (forward `--reason` / `--source`
   / `--no-skip` as requested). Resumable. For many/long stimuli run it in the
   background and monitor `annotations/corpus/corpus_index.csv` (count `*/*.h5`, watch
   for `status=error`). This rebuilds `data/manifest.csv` and `analysis/web/segments.json`.

2. **Analysis + stats** — in MATLAB (via MCP):
   `addpath matlab; cd <root>; refreshAnalysis("annotations/corpus")`. Regenerates
   `analysis/figures/*.png` and writes **`analysis/corpus_stats.json`**.

3. **Update corpus-dependent docs** from `analysis/corpus_stats.json` (key → where).
   `refreshAnalysis` exports full-corpus stats under `full.*`, audiovisual-subset stats under
   `av_subset.*` (the paper's structural figures are AV-subset), and NaN for any class that
   drops out of the analysis — a NaN here means "not enough data", not "update the doc to NaN".
   | stat key | update in |
   |---|---|
   | `n_stimuli`, `total_minutes`, `by_source`, `by_modality` | `README.md` (Phase 3 row), `docs/CONTENTS.md` (Datasets + folder map count), `docs/REVIEW_PAPER.md` §5, `docs/design/PHASE3_CORPUS.md` |
   | `n_timepoints`, `full.n_channels_analyzed`, `av_subset.n_channels_analyzed` | `REVIEW_PAPER.md` §5/§6 |
   | `full.pcs_to_80`, `av_subset.pcs_to_80` (+ `pc1_5_pct`) | `REVIEW_PAPER.md` §6.2, `PHASE4_ANALYSIS.md` |
   | `class_visual_social`, `class_audio_social`, `class_audio_visual` | `REVIEW_PAPER.md` §6.3, `PHASE4_ANALYSIS.md` |
   | `contingency.visual_social_*`, `contingency.audio_social_*` | `REVIEW_PAPER.md` §6.4 |
   | `design_greedy`, `design_random`, `design_k`, `design_n_source_stimuli` | `REVIEW_PAPER.md` §7, `PHASE4_ANALYSIS.md` |
   | searchable segment/channel counts | from `build_search_index.py` stdout → `PHASE4_ANALYSIS.md` |

4. **Surface results** — `SendUserFile` the regenerated `analysis/figures/*.png` if useful.
5. **Memory** — update the project memory with the new corpus size and date.
6. **Sanity checks** — `corpus_index.csv` has 0 errors; `readAnnotationCorpus` loads the
   expected N; the constant 103-channel shape still holds across all files; if a new
   `SOURCES.md`/rights are needed for the added source, create it.

### Edge cases
- **New modality** (audio-only / text-only): supported. Confirm inapplicable classes are
  present as `applicable=false` NaN skeletons (the template fill handles this). Text
  stories run the language/affect/event passes on an estimated reading-rate timeline
  (`--audio-hl --events` reaches text-emotion + surprisal; lexical/syntax are always on).
- **Extractor set changed** (not just new media): rebuild the template
  (`tools/build_channel_template.py` on a full-stack short clip) → re-annotate all with
  `--no-skip` → then steps 2–6.
- **Failures:** per-stimulus errors are isolated and logged in `corpus_index.csv`
  (`status=error`, message); investigate individually.

### Does NOT need updating when only media is added
`src/nfe/**`, `matlab/**`, `schema/**` (template stable), `docs/scoping_review/**`,
`docs/design/ANNOTATION_FORMAT.md`, `docs/design/PHASE2_*`, and this file.
