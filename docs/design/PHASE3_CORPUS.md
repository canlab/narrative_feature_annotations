# Phase 3 — Corpus Annotation

Assemble the stimulus corpus and run annotations across it, producing one
constant-shape annotation file per stimulus for the Phase 4 analysis.

## Corpus & manifest

`tools/build_manifest.py` scans `data/movies/**` and `data/stories/**` and writes
**`data/manifest.csv`** (+ `.json`): `id, path, source, modality, duration_sec, fps,
width, height, has_video, has_audio, rights`. Rebuild any time media is added.

Current corpus: **83 stimuli / ~470.6 min (~7.8 h)** — 49 lab `spacetop` clips + 4 short films
(3 CC-BY Blender open films + *Kung Fury*) + **29 Narratives spoken-story audio clips**
(`data/stories/narratives/`, ~5.3 h) + 1 pure-text sample story (53 audiovisual, 29 audio-only,
1 text-only). To grow it:
drop files under `data/movies/<source>/` (movies/audio) or `data/stories/<source>/`
(audio/text stories) and re-run the manifest (see [`../ADDING_MOVIES.md`](../ADDING_MOVIES.md);
credentialed sets in [`../EXTERNAL_STIMULI.md`](../EXTERNAL_STIMULI.md)).

## Batch runner

`python -m nfe.batch` annotates manifest entries:

```bash
PYTHONPATH=src .venv/bin/python -m nfe.batch \
  --manifest data/manifest.csv --out annotations/corpus \
  --template schema/channel_template.json \
  --vision --audio-hl --events --source spacetop --max-dur 65
```

- **Constant shape:** `--template schema/channel_template.json` → every file has the
  same 95-channel hierarchy (un-run/inapplicable channels are `applicable=false` NaN),
  so Phase 4 stacks them into rectangular matrices.
- **Resumable:** skips a stimulus whose `<id>.h5` already exists (`--no-skip` to force).
- **Crash-safe:** rewrites `annotations/corpus/corpus_index.csv` (id, status,
  n_measured/n_skeleton, seconds, error) after every stimulus.
- **Isolated:** one stimulus erroring doesn't stop the batch.
- Filters: `--source`, `--ids`, `--max-dur`, `--limit`. Passes: `--vision --audio-hl
  --reason --events`. `--fps` (default 2 for batch).

Output: `annotations/corpus/<id>/<id>.h5` (+ `.manifest.json`) per stimulus.

## Runtime guidance (this M1 Ultra, MPS)

Per-pass cost (≈, at 2 fps): CPU low-level + ASR + language are cheap; `pose` and the
frame VLM/depth/action passes dominate; **`--reason` (Qwen2.5-VL) is ~50 s/window** and
is impractical for long films here. Recommended tiers:

| Goal | Config | Notes |
|------|--------|-------|
| Broad, fast corpus | `--audio-hl --events` (CPU+audio) | many stimuli; visual = NaN skeleton |
| Rich per-stimulus | `--vision --audio-hl --events` | all classes measured; ~minutes/clip |
| + high-level reasoning | add `--reason` | only on short clips / sampled windows here |

Scale-up: run the full corpus in the background (overnight); the index + skip-existing
make it safely resumable. For production-scale `--reason`, use the 7B on a CUDA GPU or
shot-sampled windows.
