# Narrative Feature Extraction

### 📖 [**Browse the documentation book →**](https://canlab.github.io/narrative_feature_annotations/)

*The full documentation — scoping review, annotation format, pipeline, corpus, the review
paper with figures, and the interactive segment browser — is published as a book at
**[canlab.github.io/narrative_feature_annotations](https://canlab.github.io/narrative_feature_annotations/)**.
It is the recommended entry point. (Build/deploy notes: [`docs/DEPLOYING.md`](docs/DEPLOYING.md).)*

---

Infrastructure for producing **second-by-second computational annotations of movies and audio stories** across the full range of stimulus, perceptual, linguistic, social, situational, and affective features — for use in cognitive science and naturalistic-neuroimaging research.

A human supplies a movie or story; the pipeline returns a hierarchical, semantically organized, time-resolved set of annotations from a curated set of best-in-class models. Annotations that do not apply to a given stimulus (e.g. visual features for an audio-only story) are returned as explicit nulls (`NaN`). The output reserves space for later human annotations and ships with a lightweight MATLAB reader.

## Project phases

| Phase | Goal | Status |
|-------|------|--------|
| **1. Scoping review** | Survey computational annotation tools across all feature classes; organize in a semantic hierarchy; recommend best-in-class vs redundant features. | ✅ Draft complete — see [`docs/scoping_review/`](docs/scoping_review/README.md) |
| **2. Pipeline build** | Select final feature/model set (with human input); deploy models locally or via API; produce hierarchical second-by-second annotations + MATLAB reader. | ✅ 23 extractors across all classes; constant-shape output; see [`docs/design/PHASE2_STATUS.md`](docs/design/PHASE2_STATUS.md) |
| **3. Corpus annotation** | Assemble movies/stories (user-supplied + sourced); run annotations across the whole corpus. | ✅ manifest + resumable batch runner; **83 stimuli annotated** (movies, films, 29 spoken stories, a text story). [`docs/design/PHASE3_CORPUS.md`](docs/design/PHASE3_CORPUS.md) |
| **4. Analysis & dissemination** | MATLAB analysis across corpus; review paper; interactive web search interface; experimental-design / stimulus-selection tool. | ✅ corpus reader, synced viewer, correlation/PCA/network, design tool, web search, [draft review paper](docs/REVIEW_PAPER.md). [`docs/design/PHASE4_ANALYSIS.md`](docs/design/PHASE4_ANALYSIS.md) |

## Repository layout

```
docs/
  scoping_review/      # Phase 1 output: the hierarchical scoping review
  design/              # plans, schema design, deployment feasibility
schema/                # annotation output format (spec + JSON schema + example)
src/nfe/               # the pipeline package
  base.py              #   types, common-grid TimeGrid, resampler
  ingest.py            #   PyAV decode + audio extraction
  emit.py              #   HDF5 + JSON-sidecar writer
  pipeline.py, run.py  #   orchestration + CLI
  extractors/          #   one module per feature pass
matlab/                # MATLAB reader (readAnnotations / getFeature / featuresToTimetable)
data/
  movies/              # input movies (spacetop/ = lab stimuli; open/ = CC-BY test films)
  stories/             # input audio stories
annotations/
  corpus/<id>/         # corpus derivatives: <id>.h5 + <id>.manifest.json per stimulus
                       #   + corpus_index.csv (batch status)
analysis/
  figures/             # generated analysis figures
  corpus_stats.json    # corpus summary numbers the docs cite
  web/                 # interactive segment-search app
tools/                 # helper scripts (manifest, template, search index, refresh)
```

## Quickstart (Phase 2 pipeline)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m nfe.run data/movies/open/BigBuckBunny.mp4 --rate 1.0
```

See [`docs/design/PHASE2_STATUS.md`](docs/design/PHASE2_STATUS.md) for what runs today and what's next.

## Design principles

- **Common time grid.** Every feature has a native rate (frame, sub-second, per-second, per-utterance); all are resampled onto a configurable common grid (default 1 Hz) with native-rate metadata preserved.
- **Explicit nulls.** Inapplicable annotations are `NaN`, never silently omitted, so downstream analysis can distinguish "not measured" from "measured zero."
- **Hierarchical + readable.** Output is organized by the semantic hierarchy in `docs/scoping_review/` so a human can read it and a program can traverse it.
- **Local-first.** Prefer open, locally deployable models; fall back to hosted large models only where they are clearly superior and irreplaceable.
- **Reproducible provenance.** Each feature records the model name, version, and parameters that produced it.

## Start here

- **[`docs/CONTENTS.md`](docs/CONTENTS.md)** — full guide to the folder, tools, datasets, the
  extracted derivatives, and how to load / view / inspect them.
- **[`docs/walkthrough.m`](docs/walkthrough.m)** — runnable MATLAB tour of the common operations.
- **[`docs/REVIEW_PAPER.md`](docs/REVIEW_PAPER.md)** — draft review: the models/algorithms and the
  empirical structure of the annotation space across the corpus.
- **[`docs/ADDING_MOVIES.md`](docs/ADDING_MOVIES.md)** — how to add movies/stories and refresh the
  whole dataset (annotations, search index, figures, and the corpus-specific doc numbers).

## Key documents

- **Scoping review (Phase 1 output):** [`docs/scoping_review/`](docs/scoping_review/README.md) —
  146 tools across 15 feature subclasses, the semantic hierarchy, redundancy analysis, and best-in-class
  recommendations.
- **Annotation output format:** [`docs/design/ANNOTATION_FORMAT.md`](docs/design/ANNOTATION_FORMAT.md) —
  authoritative v0.2 spec (HDF5 canonical + readable JSON sidecar; common grid; explicit nulls; reserved
  human-annotation slots; MATLAB reader interface). `schema/annotation_schema.json` is the v0.1 pure-JSON
  profile used for small/demo files, and `matlab/*.m` is the reference reader for that profile.
- **Phase 2 core-set freeze sheet:** [`docs/design/PHASE2_CORE_SET_PROPOSAL.md`](docs/design/PHASE2_CORE_SET_PROPOSAL.md) —
  fillable decision doc; sign-off freezes the Phase 2 build scope.
- **Deployment feasibility:** [`docs/design/DEPLOYMENT_FEASIBILITY.md`](docs/design/DEPLOYMENT_FEASIBILITY.md).
- **Full plan:** [`docs/design/IMPLEMENTATION_PLAN.md`](docs/design/IMPLEMENTATION_PLAN.md).
