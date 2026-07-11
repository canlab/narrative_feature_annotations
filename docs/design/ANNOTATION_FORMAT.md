# Annotation Output Data Format

**Status:** v0.2 design (supersedes the `schema/annotation_schema.json` v0.1 draft; reconciles it
with the Phase 1 catalogs).
**Scope:** the on-disk format the Phase 2 pipeline emits for **one stimulus** (one movie or one audio
story), and the MATLAB reader interface that loads it.

This document defines: (1) the container format and why; (2) the timeline / alignment convention;
(3) the hierarchical feature layout; (4) per-feature (per-channel) metadata; (5) how nulls and
reserved human-annotation slots are represented; (6) a worked 2-timepoint example; (7) the MATLAB
reader interface. It is the authoritative spec; `schema/annotation_schema.json` is the machine-checkable
encoding of it, and `matlab/*.m` is the reference reader.

---

## 1. Format recommendation: HDF5 canonical + JSON sidecar manifest

### 1.1 The requirement, sized

A single feature-rich movie produces, per second:

- ~hundreds of **scalar** channels (luminance, RMS contrast, F0, flow magnitude, valence, surprisal …);
- several **high-dimensional vector** channels — AudioSet tags (527-dim), CLIP/SigLIP probe sets
  (tens–hundreds of prompts), Places365 (365-dim), DINO/CLIP/LLM embeddings (384–4096-dim),
  pymoten motion energy (hundreds–thousands-dim), pose keypoints (17–133 × 3);
- **categorical / label** channels (scene category, speaker id, topic id, dialogue act, narrative stage);
- **sparse-event** channels (cuts, beats, turning points) carried as both a per-bin flag and an onset list.

A 2-hour movie at 1 Hz is 7200 timepoints. With a few thousand embedding dimensions per timepoint
across several encoders, the dense payload is **tens of MB to low GB per stimulus**. Inline-array
JSON (the v0.1 draft) is unworkable at that size: it is ~3–5× larger than binary, parses slowly,
holds everything in memory, and has no native typed n-D array. So JSON stays the *readable metadata*
layer, not the bulk container.

### 1.2 Decision

> **Canonical container: HDF5 (`.h5`), one file per stimulus.**
> **Sidecar: a human-readable JSON manifest (`.manifest.json`)** holding the same hierarchy and all
> metadata but **no bulk numeric arrays** — pointers (`data_ref`) into the `.h5` instead.

Rationale, against the requirements:

| Requirement | How HDF5 + JSON sidecar meets it |
|---|---|
| Second-by-second, configurable rate | One shared time axis dataset; every channel is `[n_samples × …]` aligned to it. |
| Semantically readable, coherent, **hierarchical** | HDF5 groups *are* a filesystem-like hierarchy; the JSON sidecar mirrors it and is the human-readable view. |
| Irrelevant features → null/NaN | Float datasets use `NaN`; an `applicable` attribute + a fill value make "not measured" explicit and self-describing. |
| Reserved human-annotation slots | A top-level `/human/` group, pre-created empty with the same shape contract; never written by machines. |
| **Loadable into MATLAB** | HDF5 is a first-class MATLAB format (`h5read`, `h5info`, `h5readatt`) with **no toolbox required**; the JSON sidecar loads via `jsondecode`. |
| Reproducible provenance | Per-channel attributes (model, version, native rate, units, resample op) live next to the data; global provenance in `/`. |
| Scales to dense embeddings | Chunked + gzip-compressed datasets; lazy/partial reads; no whole-file parse. |

**Parquet** is a strong alternative *for the scalar/tabular slice* (columnar, great with
pandas/Arrow), but it is awkward for ragged high-dimensional per-timepoint vectors and variable-length
event lists, and MATLAB support (`parquetread`) is newer and tabular-only. We therefore choose HDF5 as
the single canonical container and **optionally** emit a flat `*_scalars.parquet` projection of just
the scalar channels as a convenience export (see §3.5).

**Pure-JSON profile (small/demo only).** For tiny clips, tests, and documentation, the entire payload
*may* be inlined into the JSON file (the v0.1 layout). The reader auto-detects: if a channel has inline
`value`, it uses it; if it has `data_ref` only, it reads the `.h5`. This keeps the v0.1 example valid
while making the dense case feasible.

### 1.3 On-disk artifacts per stimulus

```
annotations/output/<stimulus_id>/
  <stimulus_id>.h5             # canonical: time axis + all feature datasets + attributes
  <stimulus_id>.manifest.json  # readable hierarchy + metadata + data_ref pointers (no bulk arrays)
  <stimulus_id>_scalars.parquet  # OPTIONAL flat export of scalar channels (convenience)
```

Self-contained alternative: a single `.h5` is sufficient on its own (it carries all metadata as
attributes); the sidecar JSON exists for grep/diff/readability and for the pure-JSON profile.

---

## 2. Timeline & alignment convention

### 2.1 The common grid

There is exactly **one common time grid per file**, defined by three numbers and materialized as one
dataset:

```
/time/rate_hz        scalar   e.g. 1.0   (configurable; default 1 Hz)
/time/t_start_sec    scalar   e.g. 0.0   (time of the *center* of sample 0; see §2.2)
/time/n_samples      scalar   e.g. 7200
/time/time_sec       [n_samples] float64   center timestamp of each grid bin (always written, even
                                           though derivable, so MATLAB/Python never recompute it)
```

Invariant: `time_sec[i] == t_start_sec + i / rate_hz`. **Every** feature dataset has leading dimension
`n_samples` and shares this axis — that is what makes the whole file a single aligned matrix family.

A file may also declare an alternate grid (e.g. a 0.5 Hz coarse grid, or a per-TR grid for a specific
fMRI study) under `/time/alt/<name>/…`; channels resampled to an alternate grid point at it via a
`grid` attribute. Default and common case: a single grid.

### 2.2 Bin convention (center-referenced, half-open)

Grid bin `i` covers the continuous interval

```
[ t_start_sec + (i - 0.5)/rate_hz ,  t_start_sec + (i + 0.5)/rate_hz )
```

and `time_sec[i]` is its **center**. All resampling (§2.4) maps native samples into these bins. Center-
referencing (not left-edge) keeps a feature's grid time aligned to the middle of the interval it
summarizes, which is the convention most neuroimaging analyses expect and avoids a systematic half-bin
lag. The choice is recorded once in `/time/bin_reference = "center"`.

### 2.3 Native rates feeding the grid

Each extractor has a **native rate** (from the Phase 1 catalogs), one of:

| Native rate | Examples | Typical native step |
|---|---|---|
| `frame` | luminance, optical flow, CLIP/DINO probes, depth, saliency, pose | 1/fps (e.g. 25 Hz, or the analysis 2 fps) |
| `subsecond` | librosa/openSMILE LLDs, Praat F0, CREPE, audio taggers on short hops | ~10–100 Hz |
| `second` | windowed VLM/affect/action windows | 1 Hz |
| `window` | sliding clip models (VideoMAE 16-frame, CLAP 10 s) | window-dependent |
| `utterance` | ASR segments, dialogue acts, per-utterance affect | irregular |
| `word` | surprisal, lexical norms, POS, LLM hidden states | irregular (from forced alignment) |
| `shot` / `scene` | per-shot VLM captions, scene labels, situational tags | irregular |
| `event` | cuts, beats, turning points, event-segment boundaries | irregular/sparse |

The **native rate is preserved as metadata** on every channel (`native_rate_hz`, which may be a number
or one of the strings above), and the **raw pre-resampling output is cached** (per DEPLOYMENT_FEASIBILITY
§2.6) so the grid can be regenerated at a different rate without re-running models.

### 2.4 Resampling onto the grid (per dtype)

One resampling rule per channel, chosen by dtype and recorded in the `resample` attribute. These match
DEPLOYMENT_FEASIBILITY §2.7:

| Channel kind | Default resample op (`resample`) | Notes |
|---|---|---|
| Continuous scalar (luminance, F0, flow, valence) | `mean` | Anti-aliased / area average of native samples within the bin. Optionally also emit `_std`, `_max` companion channels. |
| Probability / score **vector** (AudioSet 527, CLIP probes, emotion posteriors) | `mean` | Element-wise mean-pool within bin. |
| **Embedding** vector (DINO/CLIP/LLM/Qwen3) | `mean` | Mean-pool of native vectors in bin (record `embed_pool="mean"`). |
| Categorical / `label` (scene, speaker, topic, dialogue act, narrative stage) | `mode` | Most-frequent native label in bin; ties broken by longest dwell. Companion `*_change` boundary flag emitted. |
| `bool` / presence (speech present, face present) | `any` or `frac` | `any` = ≥1 native true in bin; or `frac` = fraction of bin true (record which). |
| Sparse `event` (cut, beat, turning point) | `count` + onset list | Per-bin integer count/flag **and** an exact onset-time list (§5.3). |
| Per-`word` feature (surprisal, concreteness, hidden state) | `mean` | Assign by word timestamp to its bin, average within bin; companion `word_rate`/`word_onset` channels. |

A native sample with no bin (out of range) is dropped; a bin with no native sample is `NaN` (continuous)
/ empty-label (categorical) / `0` (count) — see §5.

### 2.5 Why a single grid (not per-feature native arrays in the file)

Keeping every channel on one shared, regular grid is what makes the file a coherent design matrix:
MATLAB loads it straight into a `timetable`, cross-feature correlations and PCA (Phase 4) need no
re-interpolation, and "inapplicable = `NaN` of the right length" gives a **constant output shape across
all stimuli** (the README's core principle). Native-rate arrays still exist — in the cache, and
optionally archived under `/native/<channel>/…` for power users — but the grid is the contract.

---

## 3. Hierarchical feature layout

### 3.1 Top-level structure (HDF5 groups == hierarchy)

```
/                                      (root; global attributes = schema_version, ids, provenance)
├── time/                              the common grid (§2)
├── stimulus/                          (attrs: id, title, modality, duration_sec, source, media_file, sha256)
├── features/                          MACHINE annotations, mirroring the Phase 1 semantic hierarchy
│   ├── visual/
│   │   ├── low_level_static/          luminance, rms_contrast, colorfulness, edge_density, fft_slope, clutter…
│   │   ├── high_level_static/         scene_category, object_presence, clip_probe…, places365, panoptic_fractions
│   │   ├── faces_bodies_gaze/         n_faces, au_intensity, gaze_yaw/pitch, head_pose, pose_keypoints, expression
│   │   ├── dynamic_motion/            flow_magnitude, camera_motion, residual_motion, motion_energy, cut (event)
│   │   ├── action/                    action_posteriors, action_probe, action_segment (event)
│   │   └── saliency_aesthetics_depth/ saliency_entropy, depth_mean, fg_fraction, aesthetic, quality, memorability
│   ├── audio/
│   │   ├── low_level/                 rms, loudness, spectral_centroid, mfcc, chroma, f0, onset_strength, tempo
│   │   ├── high_level/                audioset_tags, clap_probe, scene, speech_music_noise, key, beat (event)
│   │   └── speech/                    asr_text, speaker_id, vad, speech_rate, prosody_f0, ser_arousal/valence/dominance
│   ├── language/
│   │   ├── lexical/                   word, lemma, pos, freq_zipf, concreteness, aoa, valence_norm, emolex_*, surprisal
│   │   ├── syntax/                    dep_depth, clauses_per_tunit, tree_depth, coref_chain_len, dialogue_act, readability
│   │   └── semantics_discourse/       embedding, coherence, drift, novelty, topic_id, topic_vector, narrative_stage, turning_point (event)
│   ├── social/                        n_agents, characters_present, active_speaker, mutual_gaze, joint_attention,
│   │                                  proximity, interaction_type, dominance, affiliation, tom_intention
│   ├── situation/                     location, time_of_day, indoor_outdoor, script_label, event_id, event_boundary (event),
│   │                                  space/time/causation/intention/protagonist (Event-Indexing dims)
│   └── affect/
│       ├── depicted/                  face_valence/arousal, voice_v/a/d, text_valence, music_v/a, categorical_emotion
│       └── elicited/                  induced_valence, induced_arousal, fear_flag   (separate stream from depicted)
├── human/                             RESERVED for later human annotation (§5.4); same shape contract; empty at emit
│   ├── visual/ … affect/              (mirrors features/ subgroups; created empty)
│   └── _free/                         free-form human channels not in the machine taxonomy
└── provenance/                        per-class model registry, env locks, params hashes (§4.3)
```

The top-level classes and subclasses are exactly the Phase 1 hierarchy (Visual / Audio / Language /
Social / Situation / Affect). A program traverses groups; a human reads the JSON sidecar, which has the
identical tree.

### 3.2 A leaf = one channel = one HDF5 dataset + attributes

Each leaf feature is **one dataset** named by the channel, shaped `[n_samples]` (scalar/label/bool/event)
or `[n_samples × D]` (vector/embedding) or `[n_samples × K × C]` (e.g. pose `K` keypoints × `C` coords),
with metadata carried as **HDF5 attributes on that dataset** (§4). The dataset's HDF5 path *is* its
hierarchical path, e.g. `/features/visual/low_level_static/luminance`.

### 3.3 Dtype encodings

| `dtype` | HDF5 storage | Null encoding |
|---|---|---|
| `scalar` | float64 `[n]` | `NaN` |
| `vector` | float32/float64 `[n × D]` | whole row `NaN` (not measured) |
| `bool` | int8 `[n]` (0/1) | `-1` (or float `NaN` if stored as float) |
| `categorical` | int32 `[n]` **code** + `categories` attr (string array) | code `-1` ↔ `<undefined>` |
| `label` | variable-length UTF-8 string `[n]` | empty string `""` ↔ not measured |
| `event` | see §5.3 (per-bin `count` int32 `[n]` + onset list group) | count `0` |
| `text` (e.g. ASR words) | variable-length UTF-8 string `[n]` | `""` |

Categorical channels store **integer codes + a `categories` attribute** (the label vocabulary) so MATLAB
reconstructs a `categorical` array directly and analyses stay numeric; `label`/`text` channels store
strings directly for the free-vocabulary cases (ASR words, LLM free-text tags).

### 3.4 Vector channels carry a component axis

A `[n × D]` vector dataset gets a `components` attribute: a length-`D` string array naming each column
(e.g. AudioSet class names, CLIP prompt strings, MFCC indices, keypoint names). Embeddings whose
dimensions are not individually meaningful set `components = []` and rely on `dim`/`model` metadata. This
makes every vector self-describing without an external codebook.

### 3.5 Optional flat scalar export

For quick tabular work, the pipeline may also emit `<id>_scalars.parquet`: one row per timepoint, one
column per **scalar** channel, column names = the slash-path with `/`→`__`
(`visual__low_level_static__luminance`), plus a `time_sec` column. This is a lossy convenience
projection (no vectors/labels/events); the `.h5` remains authoritative.

---

## 4. Per-feature (per-channel) metadata

### 4.1 Required + optional attributes on every channel

Carried as HDF5 attributes on the dataset (and as JSON keys in the sidecar). Required marked **R**.

| Attribute | Type | Meaning |
|---|---|---|
| `dtype` **R** | string | one of §3.3. |
| `applicable` **R** | bool | `false` ⇒ feature does not apply to this stimulus modality; value is all-null (§5.2). |
| `units` | string | e.g. `"0-1"`, `"Hz"`, `"dB"`, `"bits"`, `"1-9 (Warriner)"`, `"-1..1"`, `"deg"`. `""`/absent for unitless or categorical. |
| `model` **R** | string | producing tool/model, e.g. `"scikit-image"`, `"SigLIP2-so400m"`, `"faster-whisper-large-v3"`. |
| `version` **R** | string | model checkpoint + code version, e.g. `"0.24.0"`, `"siglip2-so400m-patch14-384"`. |
| `native_rate_hz` **R** | number or string | numeric Hz, or one of `frame|subsecond|second|window|utterance|word|shot|scene|event`. |
| `resample` **R** | string | op used to map native→grid (§2.4): `mean|mode|any|frac|count|sum|max|nearest`. |
| `components` | string[] | column names for vector dtypes (§3.4); empty for opaque embeddings. |
| `categories` | string[] | label vocabulary for `categorical` (code↔label map); index = code. |
| `dim` | int | `D` for vector/embedding channels. |
| `grid` | string | `"default"` or an alternate-grid name (§2.1). |
| `tier` | string | provenance: `cpu|gpu|heavy-gpu|api` (from DEPLOYMENT_FEASIBILITY). |
| `params_hash` | string | content hash of model params/config that produced this channel. |
| `notes` | string | free text (e.g. "null where no speech present"). |
| `confidence_ref` | string | optional path to a companion `[n]` confidence channel. |

### 4.2 Companion channels (uncertainty & dispersion)

Where a model exposes it, a channel `X` may have siblings:

- `X_conf` — per-timepoint confidence/probability (e.g. ASR avg_logprob, detection score, voicing prob);
- `X_std`, `X_max` — within-bin dispersion when many native samples collapsed into one grid bin;
- `X_change` — boundary/onset flag for categorical/label channels.

These are ordinary channels with their own metadata; the parent points to them via `confidence_ref`.

### 4.3 Global provenance (`/provenance` + root attributes)

```
/  (root attributes)
   schema_version   = "0.2"
   pipeline_version = "<git-describe>"
   generated_utc    = "2026-06-13T12:00:00Z"
   common_grid_rate_hz = 1.0

/provenance/
   models/        attr-per-model: name → {version, tier, env, params_hash, citation}
   env_locks/     names of the locked envs used (core-cv, torch-vision, speech, llm, …)
   stimulus_sha256
   transcript_source = "whisperx" | "supplied"
```

This satisfies the README's "reproducible provenance": every channel ties to a model entry, which ties
to a locked environment and a params hash.

---

## 5. Nulls, applicability, and reserved human slots

### 5.1 Three distinct "missing" states — never conflated

| State | Meaning | Encoding |
|---|---|---|
| **Not applicable** | feature class can't apply to this modality (visual on audio-only story) | `applicable=false` attr **and** value all-`NaN`/`""`/`-1` for full length |
| **Not measured here** | applicable, but no native sample fell in this bin (e.g. silence → no word, no speech) | per-element `NaN`/`""`/`code=-1`/`count=0`; `applicable=true` |
| **Measured zero / absent** | the model ran and reports zero/absence (e.g. 0 faces detected, loudness ≈ 0) | the actual numeric value (`0`, low float) — **not** null |

The `applicable` flag plus the element-wise fill value lets downstream analysis distinguish *"not
measured"* from *"measured zero"* — the README's explicit requirement. Float channels use IEEE `NaN`
(reads into MATLAB as `NaN` natively); categorical uses code `-1`; labels/text use `""`; events use `0`.

### 5.2 Inapplicable features are still present, full-length, all-null

When a stimulus is audio-only, every `visual/*` channel is still emitted with the correct
`[n_samples × …]` shape, filled with `NaN`/`""`/`-1`, and `applicable=false`. This guarantees a
**constant output shape and hierarchy across the whole corpus**, so cross-stimulus matrices, PCA, and
the Phase-4 design tool never have to special-case which classes a given stimulus has.

Modality → applicable-class gate (from IMPLEMENTATION_PLAN "applicability rule"):

| Stimulus modality | Applicable top-level classes |
|---|---|
| `audiovisual` | visual, audio, language, social, situation, affect (all) |
| `video-only` | visual, situation, affect.depicted, social (no speech-derived language/affect) |
| `audio-only` | audio, language, social*, situation, affect (no visual) |
| `text-only` | language, situation, social*, affect (text branch only) |

(*social/situation degrade to the subset derivable from the available modalities; per-channel
`applicable` flags carry the exact truth.)

### 5.3 Sparse events

An `event` channel is stored as a per-bin **count** dataset plus an **onset group** holding exact times:

```
/features/visual/dynamic_motion/cut            int32 [n]   per-bin cut count (resample="count")
/features/visual/dynamic_motion/cut__onsets/
    time_sec     [m] float64      exact onset times (m = total events)
    value        [m] (optional)   per-event payload (e.g. transition type, confidence)
```

So an event is both griddable (the `[n]` count/flag, usable as a regressor) and exact (the onset list,
for precise timing). Beats, turning points, scene boundaries, action segments follow the same pattern;
segment-type events add a paired `__offsets` list.

### 5.4 Reserved human-annotation slots (`/human/`)

A top-level `/human/` group mirrors the `/features/` subgroup tree but is **created empty** at machine
emit time — no datasets, just the group skeleton — plus a `/human/_free/` group for human channels that
don't fit the machine taxonomy. Properties:

- **Same channel contract.** A later human (or human-in-the-loop tool) writes a channel into
  `/human/<class>/<sub>/<name>` using the *identical* schema: `[n_samples]` aligned to `/time`, with
  `dtype`, `applicable`, `model="human"`, `version=<rater id / protocol>`, `resample`, `notes`.
- **Never auto-populated.** Machine runs only (re)create the empty skeleton; they never write under
  `/human/`, so re-running the pipeline never clobbers human work. (Writers should append, and the
  pipeline must not overwrite an existing `.h5`'s `/human/` group — emit a new file or merge.)
- **Discoverable.** `/human` has an attribute `populated = false` until a human channel is added; the
  reader exposes `ann.human` as an (initially empty) struct of the same shape as `ann.features`.
- **Provenance for humans.** A human channel records rater id, instructions/protocol version, and date
  in its attributes exactly as a model channel records `model`/`version`, so human and machine
  annotations are first-class and equally traceable.

In the **JSON sidecar / pure-JSON profile**, this is the `human_annotations` object: empty `{}` at emit,
later filled with the same hierarchical shape as `features`.

---

## 6. Worked example (~2 timepoints)

Below is the **JSON sidecar** view (the readable layer) for an *audiovisual* clip at 1 Hz, showing two
grid timepoints and a representative channel of each kind. Bulk vectors point into the `.h5` via
`data_ref`; small channels inline `value` for readability. A null at `t=0` for a language feature shows
"applicable but not measured" (no speech yet); the whole `affect/elicited` example shows a *separate*
stream from depicted affect.

```jsonc
{
  "schema_version": "0.2",
  "stimulus": {
    "id": "demo_movie_001", "title": "Demo clip", "modality": "audiovisual",
    "duration_sec": 2.0, "source": "user-supplied",
    "media_file": "data/movies/demo_movie_001.mp4", "sha256": "ab12…"
  },
  "time": {
    "rate_hz": 1.0, "t_start_sec": 0.0, "n_samples": 2,
    "bin_reference": "center", "time_sec": [0.0, 1.0]
  },
  "features": {
    "visual": {
      "low_level_static": {
        "luminance": {
          "dtype": "scalar", "applicable": true, "units": "0-1",
          "model": "scikit-image", "version": "0.24.0",
          "native_rate_hz": 25, "resample": "mean",
          "value": [0.41, 0.43]
        }
      },
      "high_level_static": {
        "scene_category": {
          "dtype": "categorical", "applicable": true,
          "model": "SigLIP2-so400m", "version": "siglip2-so400m-patch14-384",
          "native_rate_hz": "frame", "resample": "mode",
          "categories": ["kitchen", "hallway", "street"],
          "value": [0, 1]                        // codes -> "kitchen", "hallway"
        },
        "clip_probe": {
          "dtype": "vector", "applicable": true, "dim": 3,
          "model": "SigLIP2-so400m", "version": "siglip2-so400m-patch14-384",
          "native_rate_hz": "frame", "resample": "mean",
          "components": ["a kitchen", "a person cooking", "an empty hallway"],
          "data_ref": "/features/visual/high_level_static/clip_probe"   // [2 x 3] in .h5
        }
      },
      "dynamic_motion": {
        "cut": {
          "dtype": "event", "applicable": true, "units": "count",
          "model": "TransNetV2", "version": "1.0",
          "native_rate_hz": "frame", "resample": "count",
          "value": [0, 1],                         // one cut landed in bin t=1
          "onsets": { "time_sec": [1.04], "value": ["hard"] }
        }
      }
    },
    "audio": {
      "low_level": {
        "loudness": {
          "dtype": "scalar", "applicable": true, "units": "LUFS-rel",
          "model": "librosa", "version": "0.10.2",
          "native_rate_hz": 100, "resample": "mean",
          "value": [0.62, 0.71]
        }
      },
      "high_level": {
        "audioset_tags": {
          "dtype": "vector", "applicable": true, "dim": 527,
          "model": "BEATs", "version": "iter3+",
          "native_rate_hz": "window", "resample": "mean",
          "components": ["Speech", "Music", "..."],         // 527 names (truncated)
          "data_ref": "/features/audio/high_level/audioset_tags"   // [2 x 527] in .h5
        }
      }
    },
    "language": {
      "lexical": {
        "valence_norm": {
          "dtype": "scalar", "applicable": true, "units": "1-9 (Warriner)",
          "model": "Warriner-norms", "version": "2013",
          "native_rate_hz": "word", "resample": "mean",
          "value": [null, 6.2],                    // t=0 NaN: no speech in that bin (not measured)
          "notes": "null where no word onset falls in the bin"
        }
      },
      "semantics_discourse": {
        "embedding": {
          "dtype": "vector", "applicable": true, "dim": 1024,
          "model": "Qwen3-Embedding-0.6B", "version": "0.6B",
          "native_rate_hz": "utterance", "resample": "mean",
          "components": [],                         // opaque embedding dims
          "data_ref": "/features/language/semantics_discourse/embedding"  // [2 x 1024]
        },
        "narrative_stage": {
          "dtype": "categorical", "applicable": true,
          "model": "Llama-3.1-8B-Instruct", "version": "4bit",
          "native_rate_hz": "scene", "resample": "mode",
          "categories": ["setup", "conflict", "climax", "resolution"],
          "value": [0, 0]                           // both bins "setup"
        }
      }
    },
    "social": {
      "n_agents": {
        "dtype": "scalar", "applicable": true, "units": "count",
        "model": "InsightFace-buffalo_l", "version": "0.7",
        "native_rate_hz": "frame", "resample": "mean",
        "value": [2.0, 2.0]                          // measured 2 (not null)
      }
    },
    "affect": {
      "depicted": {
        "face_valence": {
          "dtype": "scalar", "applicable": true, "units": "-1..1",
          "model": "HSEmotion-EfficientNet-B2", "version": "va_mtl",
          "native_rate_hz": "frame", "resample": "mean",
          "value": [0.10, 0.30]
        }
      },
      "elicited": {
        "induced_valence": {
          "dtype": "scalar", "applicable": true, "units": "-1..1",
          "model": "LIRIS-ACCEDE-regressor", "version": "1.0",
          "native_rate_hz": "second", "resample": "mean",
          "notes": "viewer-elicited; distinct stream from affect.depicted",
          "value": [0.05, 0.12]
        }
      }
    }
  },
  "human_annotations": {},     // empty skeleton mirrors features/ (see §5.4)
  "provenance": {
    "pipeline_version": "0.2.0", "generated_utc": "2026-06-13T12:00:00Z",
    "models": {
      "SigLIP2-so400m": { "version": "siglip2-so400m-patch14-384", "tier": "gpu", "env": "torch-vision" },
      "BEATs": { "version": "iter3+", "tier": "gpu", "env": "audio-tag" }
      /* … */
    }
  }
}
```

For an **audio-only** story, the entire `features.visual` subtree is still present and full-length, with
every channel `applicable=false` and `value` all-`NaN` (scalars) / `-1` codes (categoricals) / `""`
(labels). Shape is identical to the audiovisual file; only `applicable` and the fill values differ.

---

## 7. MATLAB reader interface

MATLAB reads HDF5 natively (`h5read`, `h5info`, `h5readatt`) and JSON via `jsondecode` — **no toolbox
required** for the core reader. The reference implementation lives in `matlab/`.

> **Implementation status.** IMPLEMENTED and verified: `readAnnotations` (.h5, folder, or JSON),
> `getFeature`, `featuresToTimetable(ann)` (no name-value options; scalar/bool/event channels only,
> not-applicable channels as NaN), plus `readAnnotationCorpus`, `analyzeCorpus`, `selectStimulusSet`,
> `refreshAnalysis`, `annotationMovieViewer`. The signatures below marked *(planned)* —
> `listFeatures`, `getFeatureMatrix`, `writeHumanChannel`, the `"Lazy"` option, and
> featuresToTimetable's name-value options — are design targets, not yet implemented.

High-level signatures:

```matlab
% --- Load -----------------------------------------------------------------
ann = readAnnotations(path)
%   READANNOTATIONS  Load one annotation into a MATLAB struct.
%   PATH may be the .h5, the .manifest.json, or the stimulus folder.
%   If given JSON-with-data_ref, reads bulk arrays from the sibling .h5;
%   if given pure-JSON, uses inline values; if given the .h5, reads everything
%   from it (metadata from attributes). Returns:
%     ann.stimulus  struct   (id, title, modality, duration_sec, source, sha256)
%     ann.time      struct   (rate_hz, t_start_sec, n_samples, bin_reference)
%     ann.time_sec  double[n] column   (the common grid; always materialized)
%     ann.features  struct   nested groups mirroring the hierarchy; each leaf is
%                            a channel struct (see below)
%     ann.human     struct   same shape as features (empty until humans populate)
%     ann.provenance struct
%   JSON null / HDF5 fill -> NaN (numeric), "" (label/text), <undefined> (categorical).

% A leaf channel struct:
%   ch.value         numeric [n] | [n x D] | categorical [n] | string [n] | int [n]
%   ch.dtype, ch.applicable, ch.units, ch.model, ch.version,
%   ch.native_rate_hz, ch.resample, ch.components (string[]), ch.categories,
%   ch.notes, ch.onsets (struct .time_sec/.value for event dtypes)

% --- Navigate / select ----------------------------------------------------
ch = getFeature(ann, "visual/low_level_static/luminance")
%   GETFEATURE  Retrieve one channel by hierarchical (slash) path. Errors if the
%   path is not a leaf channel. Works for features/ and human/ paths.

paths = listFeatures(ann, namevalue)
%   LISTFEATURES  Return the slash-paths of all channels, filterable by:
%     "Class","visual" | "Dtype","scalar" | "Applicable",true |
%     "Modality",... | "Pattern","*valence*"
%   Use to discover what a file contains without manual traversal.

% --- Reshape for analysis -------------------------------------------------
tt = featuresToTimetable(ann, namevalue)
%   FEATURESTOTIMETABLE  Collect channels into a MATLAB timetable on the common
%   grid (RowTimes = seconds(ann.time_sec)). Options:
%     "Dtypes",["scalar","bool"]   which kinds to include (default scalars)
%     "Class","audio"              restrict to one branch
%     "ExpandVectors",true         explode [n x D] vectors into D named columns
%     "IncludeNaNApplicable",false drop all-NaN inapplicable channels
%   Variable names are the slash-path with "/"->"__". Vector/label/event channels
%   are skipped unless ExpandVectors / explicit Dtypes request them.

M = getFeatureMatrix(ann, paths)
%   GETFEATUREMATRIX  Stack a list of channel paths into one [n x P] numeric
%   matrix (scalars) for PCA / correlation / the Phase-4 design tool; returns the
%   column->path map and propagates NaN.

% --- Corpus level (Phase 4) ----------------------------------------------
C = readAnnotationCorpus(folder, namevalue)
%   READANNOTATIONCORPUS  Load many annotation files into a struct array (or a
%   stacked timetable keyed by stimulus_id), enforcing the constant-shape
%   contract so cross-stimulus matrices align. Options mirror featuresToTimetable.

% --- Write-back (human-in-the-loop) --------------------------------------
writeHumanChannel(path, "social/interaction_type", values, namevalue)
%   WRITEHUMANCHANNEL  Append a human annotation channel under /human/ in the .h5
%   (and update the JSON sidecar), validating length == n_samples and stamping
%   model="human", version=<RaterId>, plus protocol/date attributes. Never
%   touches /features/.
```

Design notes for the reader:

- **No toolbox dependency** for read: `h5read`/`h5info`/`h5readatt` + `jsondecode` are base MATLAB.
  `timetable`/`categorical` are base MATLAB. (Parquet export read needs no extra toolbox via
  `parquetread`, which is base since R2019a.)
- **Lazy option.** `readAnnotations(...,"Lazy",true)` reads only metadata + the time axis and defers
  bulk dataset reads until `getFeature`/`getFeatureMatrix` touches them — important for the multi-GB
  embedding channels.
- **Categorical round-trips.** Integer-code + `categories` attribute reconstructs a MATLAB `categorical`
  directly, so code `-1` becomes `<undefined>` and grid math stays numeric.
- **Constant shape guarantee** is what lets `readAnnotationCorpus` stack heterogeneous stimuli without
  per-file branching: inapplicable channels are present and `NaN`, so the corpus matrix is rectangular.

---

## 8. Summary of decisions

1. **HDF5 canonical, one file per stimulus**, with a **readable JSON sidecar manifest**; optional flat
   `*_scalars.parquet` export; a pure-JSON profile for small/demo files. All load into MATLAB with no
   toolbox.
2. **One shared, configurable common grid** (default 1 Hz), center-referenced half-open bins,
   materialized `time_sec`; every channel shares the `n_samples` leading axis. Native rates preserved as
   metadata; raw native outputs cached for re-gridding.
3. **Hierarchical groups** mirroring the Phase 1 semantic hierarchy (visual / audio / language / social /
   situation / affect, with subclasses); each leaf is one dataset + self-describing attributes.
4. **Per-channel metadata**: model, version, native rate, resample op, units, components/categories,
   tier, params hash; global provenance ties channels to locked environments.
5. **Three explicit missing-states** — not-applicable (`applicable=false`, all-null), not-measured
   (element `NaN`/`""`/`-1`/`0`), measured-zero (real value) — never conflated.
6. **Reserved `/human/` group** mirroring the machine tree, empty at emit, never auto-clobbered, with
   identical channel contract and human-rater provenance.
7. **MATLAB reader** (`readAnnotations`, `getFeature`, `listFeatures`, `featuresToTimetable`,
   `getFeatureMatrix`, `readAnnotationCorpus`, `writeHumanChannel`) loading the file into structs and
   timetables on the common grid.
