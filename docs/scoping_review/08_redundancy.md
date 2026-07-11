# Redundancy & Parsimony Analysis

# Redundancy & Parsimony Analysis: Computational Narrative Feature Extraction

This analysis identifies redundant or highly-correlated feature extractors across the 16 catalogs, recommends a single best-in-class option per redundancy group, flags cases where keeping two is justified by genuinely distinct information, and ends with a MINIMAL tractable set vs. a FULL set.

---

## A. WITHIN-SUBCLASS REDUNDANCY

These are pairs/groups producing near-identical outputs. Keep one; the rest are interchangeable variants.

| # | Redundant group | Why correlated | KEEP (best-in-class) | Drop / conditional |
|---|---|---|---|---|
| 1 | scikit-image vs OpenCV (color/edge/FFT) | Same pixel-level operators | **scikit-image + OpenCV together** as one engine (OpenCV for decode/speed, skimage for stats) | Treat as a single tool, not two features |
| 2 | OpenCV Gabor bank vs pyrtools vs GIST | All = oriented spatial-frequency energy | **pyrtools** (principled steerable pyramid) | GIST only if a compact fixed-length holistic vector is wanted; drop OpenCV Gabor |
| 3 | Global entropy / edge density vs Rosenholtz clutter | Weak vs strong clutter proxies | **Rosenholtz Feature-Congestion + Subband-Entropy** | Entropy/edge density redundant as clutter |
| 4 | OpenCLIP vs SigLIP 2 | Same open-vocab text-probe role | **SigLIP 2** | Drop OpenCLIP (run one) |
| 5 | Places365 / ImageNet CNNs vs CLIP/SigLIP probing | Same scene/object categories | **SigLIP 2 zero-shot probe** | Keep Places365 ONLY for a frozen, reproducible fixed-taxonomy vector |
| 6 | Qwen2.5/3-VL vs InternVL3 vs LLaVA-OneVision vs VideoLLaMA3 | Interchangeable captioning/VQA VLMs | **Qwen-VL (2.5 image / 3 video)** | Drop others as primary; one API judge for validation only |
| 7 | YOLO-World vs Grounding DINO | Same open-vocab detection | **Grounding DINO** (accuracy) | YOLO-World only when per-frame throughput is the bottleneck |
| 8 | SAM 2 vs Mask2Former (static panoptic) | Overlapping segmentation | **Mask2Former** for composition | SAM 2 only for video-consistent tracking / Grounded-SAM |
| 9 | OpenFace 2.0 vs OpenFace 3.0 | Superseded | **OpenFace 3.0** | OpenFace 2.0 legacy/AU-scale reproduction only |
| 10 | Standalone RetinaFace vs InsightFace | InsightFace bundles RetinaFace | **InsightFace buffalo_l** | Drop standalone RetinaFace |
| 11 | ViTPose vs MMPose RTMPose/RTMW | Same keypoints | **MMPose RTMPose/RTMW** | ViTPose only for max-accuracy on strong GPU |
| 12 | EmoNet vs HSEmotion (face V/A) | Same face valence-arousal | **HSEmotion/EmotiEffLib** | EmoNet citable cross-check only |
| 13 | RAFT / MemFlow / VideoFlow / Farneback vs SEA-RAFT | Same dense flow | **SEA-RAFT** | MemFlow if temporal smoothing needed; Farneback = no-GPU fallback |
| 14 | AutoShot vs TransNetV2 | Same shot detection | **TransNetV2** (+ PySceneDetect CPU prior) | Drop AutoShot |
| 15 | TimeSformer/VideoSwin/SlowFast vs VideoMAE V2 / InternVideo2 | Superseded clip classifiers | **VideoMAE V2** (+ InternVideo2) | Legacy backbones for TAL features only |
| 16 | X-CLIP vs InternVideo2 zero-shot | Same open-vocab action role | **X-CLIP** (cheap default) | InternVideo2 zero-shot if accuracy needed |
| 17 | MiDaS/DPT/ZoeDepth vs Depth-Anything-V2 | Superseded depth | **Depth-Anything-V2** | Apple Depth Pro only if metric (meters) needed |
| 18 | TASED-Net / UNISAL vs ViNet | Same video saliency | **ViNet** (ViNet-A for audio-visual) | UNISAL if lightest needed |
| 19 | DeepGaze (static) vs ViNet (video) | Static superseded for video | **ViNet** | DeepGaze only as pure static attention proxy |
| 20 | LAION-Aesthetics / CLIP-IQA vs Q-Align | Subsumed by Q-Align IAA/IQA | **Q-Align/OneAlign** | Lightweight fallbacks when 7B LMM too heavy |
| 21 | torchaudio / Essentia spectral/MFCC vs librosa | Same DSP descriptors | **librosa** (primary MIR) | Essentia for roughness/beat; torchaudio if GPU/differentiable |
| 22 | librosa pyin / torchaudio pitch vs CREPE/Praat | Weaker F0 | **Praat (Parselmouth)** + **CREPE** | Drop pyin/torchaudio F0 |
| 23 | openSMILE jitter/shimmer/HNR vs Parselmouth | Same voice quality | **Parselmouth** (canonical) | openSMILE for standardized 88-dim eGeMAPS summary |
| 24 | AST / PANNs / YAMNet vs BEATs / PaSST | Highly correlated 527-class | **BEATs** (one primary tagger) | PANNs for framewise SED; YAMNet CPU-only |
| 25 | Essentia tempo vs madmom/Beat This! | Overlapping beat tracking | **madmom / Beat This!** | Essentia tempo redundant |
| 26 | openai-whisper / whisper.cpp / insanely-fast vs faster-whisper | Same weights | **faster-whisper + WhisperX** | Drop variants |
| 27 | pyannote 3.1 vs community-1 | Superseded | **pyannote community-1** (3.1 in social catalog) | Use newest |
| 28 | WebRTC VAD / audiotok vs Silero/pyannote VAD | Superseded | **Silero VAD** | Drop old VADs |
| 29 | NeMo ASR / emotion2vec+ vs Whisper/audEERING | Overlapping defaults | **faster-whisper + audEERING wav2vec2-dim** | NeMo for multilingual; emotion2vec+ for categorical |
| 30 | NLTK vs spaCy | Superseded | **spaCy (trf)** | NLTK for WordNet synsets only |
| 31 | `surprisal` pkg vs minicons | Direct overlap | **minicons** | Pick one |
| 32 | ANEW / MRC / Glasgow vs Warriner / Brysbaert / Kuperman | Smaller-coverage norm sets | **Warriner VAD + Brysbaert + Kuperman + NRC-VAD** | Glasgow for imageability/size only |
| 33 | Trankit / supar vs spaCy / Stanza / benepar | Overlapping parsers | **spaCy + Stanza + benepar** | Trankit for multilingual seg; supar for CRF marginals |
| 34 | fastcoref vs Maverick | Superseded on accuracy | **Maverick** | fastcoref for high-throughput batch |
| 35 | textstat vs L2SCA | Surface vs deep complexity | **L2SCA** (+ Yngve/Frazier) | textstat cheap sliding-window scalar |
| 36 | multilingual-e5 / GTE / BGE / API embeddings vs Qwen3-Embedding | Interchangeable encoders | **Qwen3-Embedding** | Pick one; API only if leaderboard-topping required |
| 37 | DMRST / TRIPOD turning-point vs instruction-tuned LLM | Subsumed by LLM extractor | **Instruction-tuned LLM** | Supervised models only when reproducible labels required |
| 38 | BrainIAK HMM vs GSBS | Overlapping boundary detectors | **GSBS** (auto-K, faster) | HMM for soft event posteriors / reactivation |
| 39 | MovieCLIP vs zero-shot CLIP / Places365 | Overlapping setting taxonomy | **SigLIP/CLIP zero-shot** | MovieCLIP if 179-cinematic taxonomy fits |
| 40 | TalkNet-ASD vs Light-ASD/LR-ASD | Same task, 23x params | **Light-ASD/LR-ASD** | Drop TalkNet |
| 41 | OpenPose vs RTMPose/ViTPose | Legacy | **RTMPose** | Drop OpenPose |
| 42 | Emotion-LLaMA vs AffectGPT vs general MLLM | Overlapping fusion/reasoning | **One MLLM (Qwen-VL or GPT-4o)** | Emotion-LLaMA for benchmarked labels; AffectGPT for open-vocab — not all three |

---

## B. CROSS-SUBCLASS REDUNDANCY

These span catalogs and are the easiest savings to miss. The same underlying model or signal appears in multiple subclasses.

| # | Appears in | Redundant signal | Resolution |
|---|---|---|---|
| C1 | Faces, Affect, Social | **Face emotion** (Py-Feat / OpenFace 3.0 / HSEmotion / EmoNet) | Run **HSEmotion** once; feed its output to both the Affect and Social streams. Do not extract face emotion separately per catalog. |
| C2 | Faces, Speech, Affect | **Vocal affect** (HSEmotion audio? no — audEERING wav2vec2-dim) appears in Audio-Speech AND Affect | Run **audEERING wav2vec2-dim** once; shared by both |
| C3 | Speech, Social | **Diarization** (pyannote) | One pyannote run feeds ASR-speaker-attribution AND social speaker-turn graph |
| C4 | Speech, Low-level acoustic, Affect | **Prosody / F0 / voice quality** (Parselmouth, openSMILE) | One Parselmouth + openSMILE pass; resample for all three uses |
| C5 | Faces, Social, Dynamic visual | **Body pose** (MMPose RTMPose) | One pose run feeds gesture (Faces), proximity/touch (Social) |
| C6 | Faces, Social | **Gaze** (OpenFace 3.0 / L2CS-Net / Gaze-LLE) | L2CS-Net or Gaze-LLE once; mutual-gaze/joint-attention derived for Social |
| C7 | Faces, Social, High-level objects | **Face detect + identity** (InsightFace) | Single InsightFace track feeds character count/identity everywhere |
| C8 | Dynamic visual, Situation, Action | **Shot/cut boundaries** (TransNetV2 / PySceneDetect) | Run once; shared event-boundary prior |
| C9 | Action, Situation, Social, Affect | **Video-LLM** (Qwen-VL) | A **single Qwen-VL pass per shot** with a combined multi-field prompt schema (action + situation + social + emotion) replaces 4 separate VLM passes. This is the single biggest compute saving. |
| C10 | High-level visual, Situation | **Scene/place recognition** (CLIP / Places365) | One SigLIP probe with merged label set |
| C11 | Semantics, Affect (text), Lexical | **Text/dialogue affect** (RoBERTa-GoEmotions, NRC-VAD, Warriner) | Lexical-VAD lookups and GoEmotions shared across Language-Lexical and Affect |
| C12 | Semantics, Affect | **Autoregressive LLM surprisal** (minicons GPT-2/Llama) | One extractor; surprisal used as both a comprehension feature (Semantics) and arousal proxy |
| C13 | Audio high-level, Audio low-level, Affect | **Music emotion / MIR** (Essentia, MERT) | Essentia run once feeds tempo/key (MIR) and valence-arousal (Affect) |
| C14 | Low-level visual, High-level visual | **CLIP early-layer / orientation energy** | Analytic features (pyrtools) for low-level; CLIP reserved for mid/high-level — do not duplicate |

---

## C. CASES WHERE KEEPING TWO IS JUSTIFIED (distinct information)

These are NOT redundant despite surface similarity — log as separate streams.

| Pair | Why both are needed |
|---|---|
| **DINOv2/DINOv3 + SigLIP 2** | Label-free self-supervised dense embedding (RSA, no human labels) vs. text-probed interpretable concept scores — genuinely different representations |
| **VideoMAE V2 + V-JEPA 2** | Both self-supervised, but V-JEPA 2 is motion/temporal-biased; complementary, not duplicate |
| **pymoten (motion energy) + SEA-RAFT (optical flow)** | Brain-aligned filter-bank regressor vs. interpretable camera-vs-object flow decomposition — different scientific uses |
| **Action recognition (VideoMAE) + low-level motion (SEA-RAFT/pymoten)** | Semantic "what action" vs. pre-semantic "how much motion" — deliberately separate layers |
| **Depicted emotion (face/voice/text models) + ELICITED emotion (LIRIS-ACCEDE/MuSe regressor)** | Character-expressed affect vs. viewer-induced affect — correlated but NOT interchangeable; must be distinct streams |
| **Per-modality affect (wav2vec2 + HSEmotion + GoEmotions) + fusion MLLM** | Interpretable unimodal signals as primary features; MLLM as an added reasoned/fused layer, not a replacement |
| **CREPE + Parselmouth F0** | CREPE cleaner on noisy/creaky speech; Praat canonical for voice-science jitter/shimmer/HNR — keep Praat for voice quality, CREPE for robust pitch contour |
| **openSMILE eGeMAPS + Parselmouth** | openSMILE's standardized 88-dim affective-computing summary vs. Praat's per-frame canonical contours |
| **Mask2Former (panoptic composition) + Grounding DINO (object boxes/counts)** | Area-fraction scene composition vs. discrete object presence/counts/positions |
| **BEATs (AudioSet events) + CLAP (open-vocab prompts)** | Fixed 527-class calibrated tagging vs. arbitrary custom text-prompt similarity |
| **inaSpeechSegmenter + pyannote diarization** | Speech/music/noise segmentation vs. who-spoke-when speaker turns — different partitions |
| **Lexical surprisal (LLM) + lexical norms (frequency/concreteness/AoA/VAD)** | Contextual prediction-error vs. context-free word properties |
| **spaCy (CLEAR/fast) + Stanza (true UD)** | Justified only if cross-tool UD-scheme complexity metrics are needed; otherwise pick one |
| **GSBS + BrainIAK HMM** | GSBS for hard boundaries/auto-K; HMM only if soft posteriors / reactivation modeling is the goal |

---

## D. RECOMMENDED FEATURE SETS

### MINIMAL TRACTABLE SET (single consumer GPU + CPU; one-pass-per-modality)

The design principle: **one extractor per signal, one shared VLM pass, maximal cross-subclass reuse.**

**Visual — low-level**
- scikit-image + OpenCV core (luminance, RMS contrast, RGB/HSV/CIELAB color, entropy, edges, FFT slope, Hasler-Susstrunk colorfulness)
- pyrtools steerable pyramid (orientation × scale energy)

**Visual — dynamic**
- SEA-RAFT (flow magnitude, camera-vs-object decomposition)
- pymoten (brain-aligned motion energy regressor)
- TransNetV2 + PySceneDetect (shot boundaries — shared with Situation/Action)

**Visual — high-level / faces / action / saliency**
- SigLIP 2 (object/scene/place/attribute probes — shared with Situation)
- DINOv2 (label-free embedding for RSA)
- InsightFace (face count/identity — shared with Social)
- OpenFace 3.0 (landmarks + AUs + gaze + emotion in one model — shared with Affect/Social)
- MMPose RTMPose (pose — shared with Social)
- VideoMAE V2 (sliding-window action posteriors + embeddings) + X-CLIP (zero-shot action)
- ViNet (video saliency) + Depth-Anything-V2 (depth)

**Audio**
- librosa (spectral/MFCC/chroma/onset/tempo)
- openSMILE eGeMAPS (standardized 88-dim LLDs)
- Parselmouth (F0, formants, jitter/shimmer/HNR)
- BEATs (AudioSet events) + CLAP (open-vocab prompts)
- inaSpeechSegmenter (speech/music/noise)
- faster-whisper + WhisperX (ASR + word alignment)
- pyannote community-1 (diarization — shared with Social)
- audEERING wav2vec2-dim (vocal A/V/D — shared with Affect)

**Language**
- spaCy trf (POS/dep/NER/morph)
- wordfreq + Brysbaert + Kuperman + Warriner/NRC-VAD + NRC-EmoLex (lexical norms — shared with Affect)
- minicons GPT-2/Llama (surprisal + hidden states — shared with Semantics)
- benepar + L2SCA (syntactic complexity)
- Maverick (coreference)
- Qwen3-Embedding + sliding-window coherence/drift + BERTopic (semantics)
- RoBERTa-GoEmotions (dialogue affect — shared with Affect)

**Situation / Social / Affect (mostly via shared passes)**
- GSBS (data-driven event segmentation on the embedding timeline)
- Light-ASD (active speaker)
- Gaze-LLE (joint/mutual attention)
- HSEmotion (face V/A — or reuse OpenFace 3.0 emotion)
- **One Qwen2.5/3-VL pass per shot** with a combined JSON schema covering: dense action/event description + situational dimensions (space/time/causation/intention/protagonist) + social interaction-type/dominance/affiliation/ToM + depicted emotion. **This single pass replaces four separate VLM deployments.**
- LIRIS-ACCEDE/MuSe regressor (viewer-ELICITED valence/arousal — kept separate from all depicted-emotion streams)

This set covers every subclass, runs on one GPU, and eliminates all Section A/B redundancy.

### FULL SET (add when resources / specific accuracy needs justify)

Add to the minimal set, only where a distinct signal or best-in-class accuracy is wanted:

- **Visual low-level:** SHINE (literature-matched canonical defs), Rosenholtz clutter, GIST (compact holistic vector)
- **Visual high-level:** Grounding DINO (object boxes/counts), Mask2Former (panoptic composition), SAM 2 (video-tracked masks/Grounded-SAM), Places365 (frozen fixed taxonomy)
- **Dynamic:** MemFlow (temporally smooth flow)
- **Action:** InternVideo2-1B (stronger zero-shot/features), V-JEPA 2 (motion-biased complement), ActionFormer/OpenTAD (timestamped TAL segments), VideoLLaMA3 (alt video-LLM)
- **Saliency/depth:** Q-Align (aesthetic + quality), ResMem (memorability), Apple Depth Pro (metric depth), DeepGaze IIE (static fixation)
- **Faces:** Py-Feat (validated AU detector set), L2CS-Net (best-in-class 360° gaze), MediaPipe (CPU redundancy layer), InsightFace age/sex
- **Audio:** Essentia + madmom/Beat This! (roughness/beat/MIR + calibrated mood/genre/valence-arousal), CREPE (robust F0), mosqito (standards-based psychoacoustic roughness/loudness/sharpness), MERT music-emotion regressor
- **Speech:** emotion2vec+ (categorical SER), NeMo Parakeet/Canary (multilingual ASR cross-check)
- **Language:** Stanza (true UD scheme), GLiNER (custom entity types), LIWC-22 (validated pronoun/function-word scheme), DMRST (reproducible RST discourse labels), supervised TRIPOD turning-points, textstat (cheap readability), dialogue-act classifier
- **Semantics:** API embeddings (OpenAI/Voyage/Gemini) if leaderboard-topping required
- **Situation:** BrainIAK HMM (soft event posteriors / reactivation), MovieCLIP (cinematic-taxonomy settings), Llama-3-70B/GPT-4 transcript event segmentation
- **Social:** InternVL3 or a frontier API as a second VLM for agreement/gold-label validation; Social-IQ 2.0 / Social Genome schemas for prompt construction
- **Affect:** Emotion-LLaMA OR AffectGPT (benchmarked / open-vocab depicted-emotion reasoning), EmoNet (cross-check), EmoBank-VAD text regressor, AttendAffectNet (fused induced V/A)

---

## E. KEY TAKEAWAYS

1. **The single largest saving is consolidating all high-level reasoning into ONE video-LLM pass** (C9). Action, situation, social, and depicted-emotion semantics are all currently realized by Qwen-VL-class models — run it once per shot with a unified multi-field prompt rather than four times.
2. **Cross-subclass sharing (Section B) saves more than within-subclass pruning.** Face emotion, gaze, pose, diarization, prosody, shot boundaries, surprisal, and lexical-VAD each appear in 2-4 catalogs — extract once, route everywhere.
3. **Foundation models supersede most fixed-taxonomy CNNs** (Places365, ImageNet, YAMNet, AST) — keep the latter only when a frozen, calibrated, reproducible probability vector is explicitly required.
4. **The one non-negotiable "keep two" is depicted vs. elicited emotion** — these are different constructs and conflating them is a scientific error, not a redundancy.
