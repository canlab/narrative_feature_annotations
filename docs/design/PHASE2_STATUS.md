# Phase 2 — Build Status

The pipeline package lives in `src/nfe/`. It ingests a stimulus, runs the applicable
extractors, resamples every signal onto the common grid, and emits the canonical
HDF5 + JSON-sidecar annotation (`docs/design/ANNOTATION_FORMAT.md`).

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m nfe.run path/to/movie.mp4 --out annotations/output --rate 1.0
```

Output: `annotations/output/<id>/<id>.h5` (+ `<id>.manifest.json`). Load in MATLAB:

```matlab
ann = readAnnotations("annotations/output/<id>");   % folder, .h5, or .json
tt  = featuresToTimetable(ann);                       % scalars on the common grid
ch  = getFeature(ann, "audio/low_level/mfcc");        % one channel + metadata
```

## Pipeline core — DONE & verified

- `base.py` — types, common-grid `TimeGrid` (center-referenced bins), `FeatureChannel`,
  and the native-rate→grid resampler (mean/max/sum/nearest/mode/count + event onsets).
- `ingest.py` — PyAV decode (downscaled RGB frames at an analysis fps) + audio extraction
  via the imageio-ffmpeg static binary. **No system ffmpeg, no cv2** (avoids the
  PyAV/OpenCV ffmpeg-dylib clash on macOS).
- `emit.py` — HDF5 writer (hierarchical groups, per-channel attrs, reserved `/human/`
  group, `/provenance` model registry) + JSON sidecar manifest with `data_ref` pointers.
- `pipeline.py` / `run.py` — orchestration + CLI, per-extractor error isolation,
  modality-driven applicability.
- MATLAB reader extended to the canonical `.h5` path (`matlab/readAnnotations.m`),
  verified end-to-end on generated output (scalars, vectors, event onsets, timetable).

## Extractors

| Status | Extractor | Frozen core pick | Notes |
|--------|-----------|------------------|-------|
| ✅ runs (CPU) | `visual_lowlevel` | scikit-image + OpenCV | luminance, contrast, colorfulness, edges, entropy, color means, FFT slope |
| ✅ runs (CPU) | `visual_shots` | TransNetV2 + PySceneDetect | **substitute**: color-histogram cut detector; swap in TransNetV2 on GPU tier |
| ✅ runs (CPU) | `audio_lowlevel` | librosa + openSMILE + Parselmouth | RMS, spectral, MFCC, chroma, F0 (pyin), onset, tempo. openSMILE/Praat to add |
| ✅ runs (CPU) | `asr` (transcript hub) | faster-whisper + WhisperX | word/segment timestamps; attaches Transcript to Ingest; emits speech_present, word_rate, asr_text. Default model `small` (CT2 = CPU on Apple Silicon); pass `--asr-model large-v3` for production. Diarization (pyannote) is a later Social pass |
| ✅ runs (CPU) | `language_lexical`, `language_syntax` | spaCy + wordfreq + norms + minicons | freq_zipf, word_length, VAD/concreteness/AoA norms (NaN unless `data/lexicons/*.csv` present); tree_depth, dependency distance, content/noun/verb fractions. Uses `en_core_web_sm` (→ trf for production); LLM surprisal (minicons) is a later torch pass |
| ✅ runs (MPS) | `visual_semantic` (SigLIP2 + DINOv2) | SigLIP 2 + DINOv2 | per-frame image embedding (768-d), zero-shot probe scores (16-d), DINOv2 CLS embedding (384-d). `--vision`. Default base/small checkpoints (→ so400m / dinov2-large for production) |
| ✅ runs (MPS) | `visual_motion` (RAFT), `visual_depth` (Depth-Anything-V2), `visual_action` (VideoMAE) | SEA-RAFT, Depth-Anything, VideoMAE V2 | flow magnitude / camera / residual motion; depth mean/range/fg/entropy; Kinetics-400 posteriors + top label. RAFT substitutes SEA-RAFT; VideoMAE has a benign q/v-bias key-name mismatch in transformers 5.x (outputs verified correct) |
| ✅ runs (MPS) | `audio_events` (AST), `audio_clap` (CLAP), `vocal_affect` (wav2vec2-dim) | BEATs + CLAP, audEERING | AudioSet-527 tags + top; open-vocab audio embedding + probes; voice arousal/dominance/valence (depicted stream). `--audio-hl`. AST substitutes BEATs |
| ✅ runs (CPU/MPS) | `faces` (MTCNN), `pose` (Keypoint R-CNN) | InsightFace/OpenFace, MMPose RTMPose | n_faces/present/max_face_frac/det_prob + `social/n_agents`; n_persons/present/kp_score + `social/min_pair_distance`. cv2-free substitutes (facenet-pytorch, torchvision). Identity/AUs/gaze + 133-kpt whole-body are later (OpenFace isolated; RTMPose) |
| ✅ runs (MPS) | `qwen_reasoning` (Qwen2.5-VL-3B) | Qwen2.5-VL-7B | one VLM pass per window → JSON populating Social/Situation/Affect: scene_description, setting, indoor/outdoor, interaction_type, dominance, depicted emotion + valence/arousal. `--reason` (slow). 3B default (→ 7B for production) |
| ✅ runs (MPS) | `text_emotion` (GoEmotions), `text_sentiment` (CardiffNLP), `language_surprisal` (GPT-2) | RoBERTa-GoEmotions, twitter-roberta-sentiment, minicons | 28-emotion vector + top; neg/neu/pos sentiment + polarity scalar (affect/depicted); per-segment surprisal + entropy (bits). All consume the transcript; in `--audio-hl` |
| ✅ runs (MPS) | `visual_emonet` (EmoNet, Kragel 2019) | EmoNet (AlexNet, Sci Adv 2019) | 20-way image emotion-schema distribution per frame → affect/depicted/emonet(+top). Vendored port (ecco-laboratory), weights from OSF. `--vision` |
| ✅ runs (CPU/MPS) | `face_emotion` (HSEmotion `enet_b0_8_va_mtl`) | HSEmotion / EmotiEffLib (AffectNet SOTA) | 8 facial expressions + face valence/arousal per MTCNN face, averaged per frame → affect/depicted/face_emotion(+top), face_valence, face_arousal. Needs `timm==0.9.16`. `--vision` |
| ✅ runs (CPU) | `visual_saliency` (spectral-residual) | ViNet | saliency mean/peak/entropy + salient-area fraction. Model-free attention proxy; in `--vision` (ViNet for production) |
| ✅ runs (CPU) | `event_segmentation` (GSBS) — **post-pass** | GSBS | runs after all extractors over the assembled scalar matrix → situation/event_id (state per bin) + event_boundary (onsets). `--events` |
| ⬜ next | `speech_diarization` (pyannote), elicited affect (LIRIS/MuSe) | pyannote, LIRIS-ACCEDE | pyannote needs a HF token + accepted terms; no off-the-shelf elicited-affect checkpoint |
| ⬜ | `vlm_reasoning` (Qwen2.5-VL) | one consolidated VLM pass | Social/Situation/Affect/narrative |
| ⬜ | `event_segmentation`, `text_emotion`, `elicited_affect` | GSBS, GoEmotions, LIRIS/MuSe | |
| ⬜ | hosted (opt-in) | OpenAI text-embedding-3-large, GPT-5.x | gated by `allow_hosted`; per sign-off |

## Constant-shape contract — DONE
Every stimulus can yield an identical channel set via an auto-generated channel
template (`schema/channel_template.json`, built by `tools/build_channel_template.py`
from a real full run — no hand-maintained spec lists). Run with `--template
schema/channel_template.json`: channels not produced (class inapplicable to the
modality, or pass disabled) are filled as `applicable=false`, all-`NaN` skeletons with
the right dtype/dim/components. Verified: a CPU-only run + template = the same 95-channel
hierarchy as the full stack (e.g. kungfury: 78 measured + 8 skeleton; a CPU-only run has
correspondingly more skeleton channels). This unblocks Phase 3/4 stacking.

## Code-review fixes (2026-07)
A verified review pass fixed: `event_id` dtype consistency across the degenerate GSBS
branch (constant-shape); extractor **failures now tracked** (`n_failed`/`failed` in the
summary + `status=partial` in `corpus_index.csv`) instead of being silently relabelled
"not applicable"; `faces_present`/`pose_present` computed as any-in-bin with NaN for
unmeasured bins (were derived from the mean, erasing missingness); temp workdirs cleaned
up; zero-length media/text now errors instead of writing an empty "ok" file; the
`nearest` resampler returns nearest-to-bin-center; MATLAB reader always orients vectors
time-first; `featuresToTimetable` excludes categorical class-code channels from the
analysis matrix; `analyzeCorpus` tolerates constant channels; and the batch index merges
across filtered runs. **Shared frame decode DONE:** `Ingest` caches decoded frames (capped
for very long films) so the ~9 visual passes share one decode instead of re-decoding.

## Second review pass (2026-07-08)
Fixed: skeleton event/bool fill values no longer leak into analyses as real data
(`featuresToTimetable` + the search-index builder treat `applicable=false` as NaN/absent);
the design tool excludes candidate segments dominated by imputed/missing cells (they could
previously win the objective *because* data were missing); `refreshAnalysis` now produces the
full documented artifact set in one command (full-corpus + AV-subset stats/figures, NaN-safe
class lookups, crash-guarded contingency split); web search fixes (rounded-zero std wipeout,
High/Low toggle ignored on check, text-only play dead-end, index-like channels excluded,
missing-feature coverage shown); viewer works for audio/text-only stimuli with clean timer
teardown; walkthrough root detection; **SigLIP2/CLAP text towers precomputed** (verified
bit-identical to the combined forward; visual-semantic pass ~3× faster).

## Known follow-ups
- **F0 (pyin) is the runtime bottleneck** (~0.85× audio duration); make optional / coarser.
- CLAP is fed 22 kHz audio (loses >11 kHz); optionally extract a 48 kHz stream for it.
- Replace CPU substitutes (histogram cuts) with the frozen GPU picks.
- Planned MATLAB APIs not yet built: `listFeatures`, `getFeatureMatrix`, `writeHumanChannel`,
  lazy reads (see ANNOTATION_FORMAT §7 status note).
