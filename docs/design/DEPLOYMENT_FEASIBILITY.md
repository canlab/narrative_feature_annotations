# Deployment Feasibility

How the recommended Phase 1 feature set is realistically deployed by a small lab. Grounded in the
scoping-review catalogs (`docs/scoping_review/`) and aligned with the extractor contract in
`IMPLEMENTATION_PLAN.md` and the output format in `ANNOTATION_FORMAT.md`. Design anchors: local-first,
explicit `NaN` nulls, one common time grid, reproducible provenance.

## 1. Hardware tiers

Four tiers, derived from each tool's catalog deployability tag (`yes-cpu` / `yes-gpu` /
`yes-heavy-gpu` / `api-only`) and sanity-checked against real VRAM:

- **Tier A — CPU backbone (no GPU).** All low-level visual and audio statistics, lexical and syntactic
  norms, and event-segmentation algorithms: librosa, openSMILE, Praat/parselmouth, scikit-image/OpenCV,
  spaCy. Runs on any laptop/server.
- **Tier B — single 24 GB GPU workhorse.** SigLIP / DINOv2, SEA-RAFT (flow), Whisper (ASR),
  BEATs (audio tagging), Qwen3-Embedding, RTMPose, Depth-Anything, and a 7B VLM (Qwen2.5-VL-7B at
  4-bit). This tier covers the great majority of high-level features.
- **Tier C — 48 GB+ GPU.** Heavier models (InternVideo2, ViTPose-G, Q-Align, 70B/72B LLMs). Each is
  paired with a Tier-B substitute that should be preferred unless the accuracy gain is needed.
- **Tier D — API-only (optional).** Frontier embeddings and frontier multimodal judges. None are
  required; each has a local default and is used as an accuracy ceiling / validation oracle.

## 2. Environment & orchestration

The real blocker is **incompatible dependency worlds** (TensorFlow vs PyTorch vs CTranslate2 vs mmcv,
plus MATLAB-origin SHINE/Rosenholtz code). Prescription:

- **5–6 locked `uv`/conda environments**, each with a resident GPU worker behind the uniform
  `Extractor` interface (`src/nfe/base.py`).
- A **Snakemake/Prefect DAG** orchestrating per-stimulus extraction, with **transcript-as-hub** ordering
  (ASR runs early because language/social/situational models consume the timestamped transcript).
- **Ingest specifics:** frame sampling at ~2 fps plus per-shot keyframes; audio extracted to 16 kHz
  (speech) and 44.1 kHz (music/acoustic) WAVs.
- **Batching rules per model family**; **content-addressed caching at stage boundaries** for idempotency
  and provenance (raw native-rate output cached so the common grid can be regenerated without re-running
  models).
- A **single common-grid resampler** with per-feature-type rules (continuous / probability / categorical
  / per-word / sparse-event) emitting constant-shape, `NaN`-padded output (see `ANNOTATION_FORMAT.md` §2.4).

## 3. Runtime & storage

- Per-hour-of-stimulus real-time factor (RTF): a **core config runs ~0.5–1× real-time** — a 2-hour movie
  annotates overnight on Tier A+B.
- Cost drivers: the per-shot VLM, dense per-frame depth/saliency/segmentation, pymoten motion energy,
  and benepar constituency parsing.
- Storage budget: **low tens of GB per hour** of stimulus (dominated by dense embedding/vector channels;
  HDF5 chunked+gzip keeps this in check).

## 4. Features that benefit from hosted large models

Only **3–4** classes realistically gain from large hosted models, each with a local `include` default:

1. High-level **social** semantics / theory-of-mind tagging.
2. Transcript-level **narrative / Event-Indexing** situational tagging.
3. **Multimodal affect** with rationale.
4. (Optional) top-end **embeddings**.

Interface guidance: a thin API client behind the same `Extractor` interface; JSON-schema-validated
structured output; per-shot/per-scene granularity; batch + cache + retry; an opt-in `allow_hosted`
privacy gate; and using APIs mainly to **validate/distill into the local 7–8B model** rather than as a
runtime dependency. See the `claude-api` skill reference for Anthropic model IDs, structured output, and
pricing if wiring a hosted judge.

## 5. Recommended default configs

- **Core (zero-API):** Tier A + Tier B only. Fully local, overnight per feature-length film, covers the
  recommended core feature set.
- **Extended:** add Tier C models where the accuracy gain is justified.
- **Hosted (opt-in):** enable Tier D for the 3–4 classes above, gated by `allow_hosted`.
