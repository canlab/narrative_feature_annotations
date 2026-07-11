# Implementation Plan — Narrative Feature Extraction

This document is the master plan. It is intentionally concrete about Phase 1–2 (buildable now) and
directional about Phase 3–4 (depend on human selection, real movies, and Phase 1 outcomes).

---

## Phase 1 — Scoping review *(automated; in progress)*

**Deliverable:** a complete scoping review of computational annotation tools, organized in a logical
semantic hierarchy, with descriptions and recommendations (best-in-class vs likely-redundant).

**Method:** a fan-out of research agents, one per feature subclass, each grounding a structured
catalog of current (2023–2026) tools in web sources, followed by cross-cutting analyses (hierarchy,
redundancy, output schema, deployment feasibility) and synthesis.

**Feature classes covered**

- **Visual**
  - Low-level static (luminance, contrast, color, spatial frequency, edges)
  - High-level static (objects, scenes/places, attributes)
  - Faces, bodies, gaze, facial expression
  - Dynamic / motion (optical flow, motion energy, cut detection)
  - Action & event recognition (video understanding)
  - Saliency, aesthetics, depth
- **Audio**
  - Low-level acoustic (loudness, pitch, spectral, MFCC, onsets, tempo)
  - High-level audio (sound events, scenes, music vs speech, MIR)
  - Speech (ASR + timestamps, diarization, prosody, vocal affect)
- **Language / semantic**
  - Low-level lexical & word-level (frequency, concreteness, affect norms, surprisal, POS, NER)
  - Syntax & grammar (parsing, complexity, coreference, dialogue acts)
  - High-level semantics, discourse & narrative (embeddings, topics, coherence, story arc)
- **Social / situational / affective**
  - Social & interpersonal (agents, who-talks-to-whom, interaction type, theory of mind)
  - Situations, schemas, scripts & event segmentation (event boundaries, situation models)
  - Emotion & affect (multimodal valence/arousal + categorical; depicted vs elicited)

**Outputs (written under `docs/scoping_review/`):**
`00_overview.md`, `01_hierarchy.md`, per-class catalog sections, `08_redundancy.md`, `09_recommendations.md`.
Schema & deployment outputs land in `schema/` and `docs/design/`.

---

## Phase 2 — Pipeline build *(pending Phase 1 human sign-off)*

1. **Feature selection checkpoint** — human developer reviews Phase 1 recommendations and freezes the
   final feature/model set (core vs extended tiers).
2. **Model deployment** — for each selected model: download + pin locally (HF/torch), or build a thin
   API client where local is infeasible. Each lives in `src/nfe/extractors/` behind a uniform
   interface: `extract(stimulus, time_grid) -> {feature: array_over_time, metadata}`.
3. **Ingest** — accept a movie or story; demux video frames (configurable sampling) and audio;
   generate a timestamped transcript (ASR or supplied) that feeds language/social/situational models.
4. **Common-grid resampling** — map every native-rate feature onto the configured grid (default 1 Hz),
   recording the native rate and resampling method per feature.
5. **Emit** — write one hierarchical annotation file per stimulus (see `schema/`), with `NaN` for
   inapplicable features and reserved slots for later human annotations.
6. **MATLAB reader** — `matlab/readAnnotations.m` + helpers load annotations into MATLAB structs/tables.

**Applicability rule:** stimulus modality (video / audio / text) determines which classes run; the rest
emit `NaN` so the output shape is constant across stimuli.

---

## Phase 3 — Corpus annotation

- Assemble a corpus from user-supplied media and findable/downloadable sources (respecting licensing).
- Maintain a stimulus manifest (id, source, modality, duration, rights).
- Run the Phase 2 pipeline over the corpus; cache per-model outputs; store annotations in `annotations/output/`.

---

## Phase 4 — Analysis & dissemination

- **MATLAB analysis** loading annotations across all stimuli: descriptive structure, cross-feature
  correlations conditioned on stimulus type, principal components of the annotation set, and a network
  graph of how feature classes relate over time, with visualizations.
- **Review paper** documenting the models/algorithms and the empirical relationships among measures.
- **Interactive web interface** to search for movie/story segments high in particular features.
- **Experimental-design tool**: select stimulus segments that maximize variance across the major
  annotation principal components while maximizing independence of feature time series — formally, an
  optimal-design objective (e.g. maximize the determinant / D-optimality of the leading components over
  the selected set), applied to the corpus to produce a high-variance, low-redundancy stimulus set.

---

## Open decisions for the human developer (Phase 1 → 2 gate)

- Common grid rate (default 1 Hz) and whether multiple rates are emitted.
- Core vs extended feature tier membership.
- Output container format (recommendation produced in `schema/`).
- Budget/appetite for API-only large models vs local-only constraint.
- Licensing policy for Phase 3 corpus sourcing.
