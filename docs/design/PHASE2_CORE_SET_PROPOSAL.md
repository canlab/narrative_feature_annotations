# Phase 2 Core-Set Proposal — freeze sheet

**Purpose:** the minimal, tractable, best-in-class feature set to build in Phase 2. Derived from
`docs/scoping_review/09_recommendations.md`. Mark each row **keep ✓ / drop ✗ / → extended** and edit
freely; your sign-off at the bottom freezes the Phase 2 scope. Everything here runs **locally** on
Tier A (CPU) + Tier B (one 24–48 GB GPU). Default answer for every row is **keep**.

Note: keep all i didn't answer. x = selected (i.e., x by keep means keep it. x by change means change)
---

## 1. Global parameters

| Parameter | Proposed default | Decision |
|---|---|---|
| Common grid rate | **1 Hz**, center-referenced bins | x keep ☐ change: ____ |
| Emit alternate grids | No (single grid; native rates cached for re-gridding) | x keep ☐ also emit: ____ |
| Output container | **HDF5 canonical + JSON sidecar** (+ optional scalar Parquet) | x keep ☐ change: ____ |
| Hosted (API) models | **Off by default**; local-only core | ☐ keep x allow opt-in for: OpenAI text-embedding-3-large, ChatGPT5.x and later____ |
| Depicted vs elicited emotion | **Two separate streams** (non-negotiable) | x keep ☐ change: ____ |

---

## 2. Core extraction passes (extract once → route to all classes)

The core set is ~18 model passes, not 146 tools, because signals are shared. Each pass below is one
deployable extractor; "feeds" shows which feature classes consume it.

### Visual
| Pass | Model | Tier | Feeds | Key per-timepoint outputs | Keep? |
|---|---|---|---|---|---|
| Low-level image stats | scikit-image + OpenCV + Hasler colorfulness | CPU | Visual | luminance, RMS contrast, color means (RGB/HSV/Lab), colorfulness, edge density, entropy, FFT slope | x |
| Orientation/SF bank | pyrtools / plenoptic | CPU | Visual | scale×orientation energy vector | x |
| Semantic image probe | **SigLIP 2** | GPU | Visual, Situation | image embedding + per-prompt object/scene/place/attribute scores | x |
| Label-free embedding | **DINOv2** | GPU | Visual | CLS embedding (for RSA / probes) | x |
| Faces + identity | **InsightFace** (RetinaFace+ArcFace) | GPU | Visual, Social | face count, identity tracks, landmarks, head pose, 512-d id embedding | x |
| Face AUs + expression | **OpenFace 3.0** | CPU | Visual, Affect | AUs, gaze, head pose, discrete emotion | x |
| Body pose | **MMPose RTMPose** (+ByteTrack) | GPU | Visual, Social | 17/133 keypoints → posture, orientation, proximity, contact proxy | x |
| Optical flow | **SEA-RAFT** | GPU | Visual | flow magnitude, orientation hist, camera-vs-object motion | x |
| Motion energy | **pymoten** | CPU | Visual | brain-aligned motion-energy vector | x |
| Shot boundaries | **TransNetV2 + PySceneDetect** | GPU/CPU | Visual, Situation, Action | per-frame cut prob, shot segments | x |
| Action recognition | **VideoMAE V2** + X-CLIP | GPU | Visual | Kinetics posteriors + zero-shot action-phrase scores | x |
| Saliency + depth | **ViNet + Depth-Anything-V2** | GPU | Visual | saliency entropy/peak, depth mean/range, foreground fraction | x |

### Audio / speech
| Pass | Model | Tier | Feeds | Key outputs | Keep? |
|---|---|---|---|---|---|
| Low-level acoustic | **librosa + openSMILE eGeMAPS + Parselmouth** | CPU | Audio, Speech, Affect | RMS, spectral, MFCC, chroma, F0, jitter/shimmer/HNR, tempo/beats | ☐ |
| Audio events/scenes | **BEATs + CLAP** | GPU | Audio | 527-d AudioSet probs + open-vocab prompt similarity | ☐ |
| Speech/music/noise | **inaSpeechSegmenter** | CPU | Audio | timestamped speech/music/noise segments | ☐ |
| ASR (the hub) | **faster-whisper large-v3 + WhisperX** | GPU | Speech, Language, Social, Situation, Affect | word/segment text + timestamps + speaking rate | ☐ |
| Diarization + VAD | **pyannote + Silero VAD** | GPU/CPU | Speech, Social | who-spoke-when, active speaker, overlap, speech activity | ☐ |
| Vocal affect | **audEERING wav2vec2-dim** | GPU | Speech, Affect | voice valence/arousal/dominance + embedding | ☐ |

### Language (consumes the transcript)
| Pass | Model | Tier | Feeds | Key outputs | Keep? |
|---|---|---|---|---|---|
| Lexical + syntax | **spaCy trf** + wordfreq/SUBTLEX + concreteness/AoA/VAD norms + NRC EmoLex | CPU | Language, Affect | POS, dep, NER; freq, concreteness, AoA, valence, emotion per word | ☐ |
| Syntactic complexity | **benepar + L2SCA + Maverick (coref)** | GPU/CPU | Language | tree depth, clause ratios, complexity indices, coref chains | ☐ |
| LLM surprisal/hidden | **minicons (GPT-2/Pythia)** | GPU | Language | per-token surprisal, entropy, hidden-state vector | ☐ |
| Semantic embedding | **Qwen3-Embedding** + sliding-window coherence + BERTopic | GPU/CPU | Language, Situation | embedding, coherence, drift, novelty, topic series | ☐ |

### High-level reasoning (one consolidated VLM/LLM pass)
| Pass | Model | Tier | Feeds | Key outputs | Keep? |
|---|---|---|---|---|---|
| Per-shot video-LLM | **Qwen2.5-VL 7B** (unified JSON schema) | GPU | Social, Situation, Affect, Language | interaction type, dominance/affiliation, ToM/intention, addressee; Event-Indexing fields (space/time/cause/intent/protagonist), script labels, event boundaries; depicted emotion+V/A/D; narrative stage/turning points | ☐ |
| Active-speaker | **Light-ASD** | GPU | Social | per-face speaking prob → speaker/listener roles | ☐ |
| Gaze target | **Gaze-LLE** | GPU | Social | mutual gaze, joint attention | ☐ |

### Situation / affect (algorithmic + dedicated)
| Pass | Model | Tier | Feeds | Key outputs | Keep? |
|---|---|---|---|---|---|
| Event segmentation | **GSBS / statesegmentation** | CPU | Situation | per-timepoint state label, ranked boundaries, optimal K | ☐ |
| Dialogue-text emotion | **RoBERTa-GoEmotions** (+NRC-VAD) | CPU | Affect | 28 emotion scores → V/A proxy per utterance | ☐ |
| Elicited affect | **LIRIS-ACCEDE / MuSe regressor** | GPU | Affect (separate stream) | viewer-induced valence/arousal per second | ☐ |

---

## 3. Deliberately keep-both (distinct signals, not redundant)
DINOv2 (RSA) vs SigLIP 2 (interpretable); pymoten (brain-aligned) vs SEA-RAFT (interpretable) vs
VideoMAE (semantic action); BEATs (fixed 527) vs CLAP (open-vocab); inaSpeechSegmenter (partition) vs
pyannote (who-spoke); **depicted vs elicited emotion**. ☐ accept all  ☐ edit: ____

## 4. Notable items left to the *extended* tier (off by default)
Grounding-DINO/Mask2Former/SAM2 (object boxes/masks), Places365 (calibrated taxonomy), Q-Align
(aesthetics), Essentia/madmom (music key/beat/mood), DMRST (RST discourse), turning-point model,
API embeddings, frontier-API VLM judge (gold-label validation). ☐ ok  ☐ promote to core: ____

## 5. Modality applicability (auto-null rule)
Audio-only story → all `visual` channels `applicable=false` (NaN). Text-only → visual+audio null,
language/social/situation/affect run on text. ☐ accept

---

## Sign-off
- Approved by: __Tor Wager__  Date: __6/19/2026__
- Common grid rate frozen at: ___1___ Hz
- Hosted models: ☐ none x enabled for: _OpenAI ChatGPT models___________________
- Notes / edits: ________________________________________________

_On sign-off, Phase 2 builds one `Extractor` (`src/extractors/<class>/`) per kept pass behind the
`base.py` interface, wired around the shared passes above._
