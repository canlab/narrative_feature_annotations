# Recommendations — Best-in-Class Feature Set

This section translates the feature catalogs and redundancy analysis into actionable per-class recommendations. For each feature class (and its subclasses), it names the best-in-class tool(s), the key features to extract, whether the tool runs locally, and a priority tier: **core** (the minimal tractable set — one extractor per signal, maximal cross-subclass reuse, single consumer GPU + CPU) or **extended** (added only when resources or a specific accuracy/distinct-signal need justifies it).

## Recommendation table

| Class | Subclass | Recommended tool | Output features | Local? | Priority |
|---|---|---|---|---|---|
| Visual | Low-level static visual | scikit-image + OpenCV + NumPy | Mean luminance, RMS contrast, RGB/HSV/CIELAB color means+stds, Shannon entropy, edge density, FFT power-spectrum slope, GLCM/LBP texture | Yes (CPU) | core |
| Visual | Low-level static visual | Hasler-Susstrunk colorfulness (direct impl) | Colorfulness index M, mean a*/b* (warmth), HSV saturation, dominant hue | Yes (CPU) | core |
| Visual | Low-level static visual | pyrtools / plenoptic steerable pyramid | Per-subband (scale x orientation) energy; orientation-energy and spatial-frequency-band vectors | Yes (CPU) | core |
| Visual | Low-level static visual | SHINE (+ SHINE_color) | Canonical luminance, RMS contrast, 1-D rotationally averaged Fourier amplitude spectrum + slope | Yes (CPU) | extended |
| Visual | Low-level static visual | Rosenholtz Feature-Congestion + Subband-Entropy | Global clutter scalars (+ per-pixel map), color/contrast/orientation clutter sub-components | Yes (CPU) | extended |
| Visual | Low-level static visual | GIST / spatial envelope | Fixed-length holistic orientation-energy descriptor (e.g. 512-d) | Yes (CPU) | extended |
| Visual | High-level static visual (objects/scenes/places/attributes) | SigLIP 2 | Image embedding (768-1152-d) + per-prompt sigmoid scores for object/scene/place/attribute label sets | Yes (GPU) | core |
| Visual | High-level static visual | DINOv2 / DINOv3 | Label-free CLS embedding (384-1536-d) + dense patch maps for RSA / linear probes | Yes (GPU) | core |
| Visual | High-level static visual | Grounding DINO 1.5/1.6 | Per-object boxes + free-text labels + confidence (presence/counts/positions) | Yes (GPU) | extended |
| Visual | High-level static visual | Mask2Former / OneFormer (panoptic) | Per-pixel class map + instance masks; per-category area fractions, object counts, composition vector | Yes (GPU) | extended |
| Visual | High-level static visual | SAM 2 | Video-tracked instance masks (sizes, counts, track IDs); label via Grounded-SAM | Yes (GPU) | extended |
| Visual | High-level static visual | Places365 ResNet-50 | Frozen 365-d scene posterior + indoor/outdoor + scene attributes (reproducible fixed taxonomy) | Yes (CPU) | extended |
| Visual | Faces / bodies / gaze / expression | OpenFace 3.0 | Landmarks + FACS AUs + eye-gaze + head pose + discrete emotion (one multitask model) | Yes (CPU) | core |
| Visual | Faces / bodies / gaze / expression | InsightFace buffalo_l (RetinaFace + ArcFace) | Face bbox + score, 5/106/68 landmarks, head pose, age/sex, 512-d identity embedding | Yes (GPU) | core |
| Visual | Faces / bodies / gaze / expression | MMPose RTMPose / RTMW | 17 body or 133 whole-body keypoints (x,y,conf) per person -> posture, gesture, orientation | Yes (GPU) | core |
| Visual | Faces / bodies / gaze / expression | Py-Feat | Validated AU probabilities/intensities + 7 emotions + head pose (Fex time series) | Yes (GPU) | extended |
| Visual | Faces / bodies / gaze / expression | L2CS-Net | High-accuracy gaze yaw/pitch -> 3D gaze vector | Yes (GPU) | extended |
| Visual | Faces / bodies / gaze / expression | MediaPipe Tasks | 478 face-mesh landmarks, 52 blendshapes, 33 body + 21 hand landmarks (CPU redundancy layer) | Yes (CPU) | extended |
| Visual | Dynamic visual (motion/flow) | SEA-RAFT | Dense (u,v) flow -> mean flow magnitude, orientation histogram, camera-vs-object motion decomposition | Yes (GPU) | core |
| Visual | Dynamic visual | pymoten | High-dim motion-energy filter-bank vector (brain-aligned encoding regressor) | Yes (CPU) | core |
| Visual | Dynamic visual | TransNetV2 + PySceneDetect | Per-frame cut probability, shot boundaries/segments, per-frame visual-change score | Yes (GPU/CPU) | core |
| Visual | Dynamic visual | MemFlow | Temporally coherent (u,v) flow (lower frame-to-frame jitter) | Yes (GPU) | extended |
| Visual | Action / activity recognition | VideoMAE V2 | Per-window Kinetics-400/710 posteriors + pooled embedding (sliding 16-frame window) | Yes (GPU) | core |
| Visual | Action / activity recognition | X-CLIP | Per-window zero-shot similarity scores over arbitrary action-phrase vocabulary | Yes (GPU) | core |
| Visual | Action / activity recognition | InternVideo2-1B | Stronger zero-shot action similarity + features for TAL | Yes (heavy GPU) | extended |
| Visual | Action / activity recognition | V-JEPA 2 | Motion/temporal-biased self-supervised window embeddings + action posteriors | Yes (GPU) | extended |
| Visual | Action / activity recognition | ActionFormer / OpenTAD | (start, end, action_class, confidence) timestamped action segments | Yes (GPU) | extended |
| Visual | Saliency / attention / aesthetics / depth | ViNet (ViNet-A audio-visual) | Per-frame saliency map -> concentration/entropy, peak, salient-area fraction, temporal shift | Yes (GPU) | core |
| Visual | Saliency / attention / aesthetics / depth | Depth-Anything-V2 | Per-frame depth map -> mean/range depth, foreground fraction, depth entropy/gradient | Yes (GPU) | core |
| Visual | Saliency / attention / aesthetics / depth | Q-Align / OneAlign | Per-frame aesthetic + technical-quality scalars | Yes (heavy GPU) | extended |
| Visual | Saliency / attention / aesthetics / depth | ResMem | Per-frame memorability scalar (0-1) | Yes (CPU) | extended |
| Visual | Saliency / attention / aesthetics / depth | Apple Depth Pro | Per-frame metric (meters) depth map + true-scale foreground distance | Yes (GPU) | extended |
| Audio | Low-level acoustic | librosa | RMS, spectral centroid/bandwidth/rolloff/flatness/contrast, ZCR, MFCC+deltas, chroma, tonnetz, onset envelope, tempo/beats, pYIN F0, HPSS | Yes (CPU) | core |
| Audio | Low-level acoustic | openSMILE eGeMAPS (opensmile-python) | ~10 ms LLDs (F0, loudness, jitter, shimmer, HNR, spectral, MFCC1-4, formants) + 88-d eGeMAPS functionals | Yes (CPU) | core |
| Audio | Low-level acoustic | Parselmouth (Praat) | F0/intensity/formant contours; jitter/shimmer/HNR voice quality; spectral moments, CPP | Yes (CPU) | core |
| Audio | Low-level acoustic | Essentia | Dissonance/roughness, inharmonicity, HPCP, spectral complexity, robust beat tracking | Yes (CPU) | extended |
| Audio | Low-level acoustic | CREPE / torchcrepe | High-accuracy per-frame F0 + voicing confidence + salience matrix | Yes (GPU) | extended |
| Audio | Low-level acoustic | mosqito | Standards-based loudness (sones), sharpness (acum), roughness (asper), fluctuation strength | Yes (CPU) | extended |
| Audio | High-level audio (events/scenes/music/speech) | BEATs | 527-d AudioSet event/scene probabilities per window + 768-d embeddings | Yes (GPU) | core |
| Audio | High-level audio | CLAP (general + music_and_speech) | 512-d joint audio/text embeddings; per-prompt cosine-similarity time series; zero-shot labels | Yes (GPU) | core |
| Audio | High-level audio | inaSpeechSegmenter | Timestamped speech/music/noise segments (+ optional speaker sex) | Yes (CPU) | core |
| Audio | High-level audio | Essentia + TF models | BPM/tempo, key+scale, genre/mood tags, valence-arousal regressors, instrument tags | Yes (CPU) | extended |
| Audio | High-level audio | madmom / Beat This! | Beat + downbeat timestamps, per-beat tempo curve, meter, chord labels | Yes (CPU) | extended |
| Audio | High-level audio | PANNs CNN14 | Framewise (tens-of-ms) SED for onset/offset localization | Yes (GPU) | extended |
| Audio | Speech (ASR/diarization/prosody/affect) | faster-whisper (large-v3) + WhisperX | Segment + word-level text and timestamps, confidences, language, speaking rate | Yes (GPU) | core |
| Audio | Speech | pyannote diarization (community-1 / 3.1) | Speaker-turn segments, per-frame speech-activity, overlap flags, speaker embeddings | Yes (GPU) | core |
| Audio | Speech | Parselmouth (Praat) | Per-frame F0, intensity, formants F1-F4, HNR, jitter, shimmer; pause structure | Yes (CPU) | core |
| Audio | Speech | audEERING wav2vec2-large-robust-12-ft-emotion-msp-dim | Per-window arousal/valence/dominance scalars + 1024-d voice-affect embedding | Yes (GPU) | core |
| Audio | Speech | Silero VAD v5 | ~30 ms speech probability; speech/non-speech boundaries; total speech time | Yes (CPU) | core |
| Audio | Speech | emotion2vec+ | 9-class categorical emotion posteriors + emotion embeddings | Yes (GPU) | extended |
| Audio | Speech | NeMo Parakeet-TDT / Canary + Sortformer | Word/segment ASR + timestamps; per-frame speaker activity (multilingual cross-check) | Yes (GPU) | extended |
| Audio | Speech | CREPE / torchcrepe | Robust per-frame F0 + voicing on noisy/creaky speech | Yes (GPU) | extended |
| Language | Low-level lexical / word-level | spaCy (en_core_web_trf) | Per-token lemma, UPOS + fine POS, dependency, NER, is_stop/punct flags | Yes (CPU) | core |
| Language | Low-level lexical | wordfreq + SUBTLEX-US | Zipf/log frequency, contextual diversity, %known | Yes (CPU) | core |
| Language | Low-level lexical | Brysbaert concreteness + Kuperman AoA + Warriner/NRC-VAD | Per-word concreteness, AoA, valence/arousal/dominance | Yes (CPU) | core |
| Language | Low-level lexical | NRC EmoLex | 8 binary emotion associations + pos/neg sentiment per word | Yes (CPU) | core |
| Language | Low-level lexical | minicons (GPT-2-medium / Pythia) | Per-token surprisal (bits), next-word entropy, log-prob (whole-word aggregated) | Yes (GPU) | core |
| Language | Low-level lexical | LIWC-22 | Validated pronoun/person-reference + function-word categories; segment summary scores | Yes (CPU) | extended |
| Language | Low-level lexical | GLiNER | User-defined entity-type spans + confidence | Yes (GPU) | extended |
| Language | Syntactic / grammatical structure | spaCy (en_core_web_trf) | Dependency labels+heads, POS, morphology (tense/aspect/mood), noun chunks | Yes (CPU) | core |
| Language | Syntactic structure | benepar | PTB constituency trees -> tree depth, clause counts, phrase-type counts | Yes (GPU) | core |
| Language | Syntactic structure | L2SCA (+ Yngve/Frazier from trees) | 14 complexity indices (clause/T-unit ratios, dependent clauses, complex nominals) | Yes (CPU) | core |
| Language | Syntactic structure | Maverick | Coreference clusters (who/what each mention refers to) | Yes (GPU) | core |
| Language | Syntactic structure | Stanza | True UD-scheme dependencies + UFeats + constituency (cross-tool complexity metrics) | Yes (GPU) | extended |
| Language | Syntactic structure | textstat | Flesch-Kincaid/Gunning Fog/SMOG readability over sliding windows | Yes (CPU) | extended |
| Language | Syntactic structure | Dialogue-act classifier (DialogTag/SwDA) | Per-utterance DA label (statement/question/backchannel/...) + probabilities | Yes (CPU) | extended |
| Language | High-level semantics / discourse / narrative | Qwen3-Embedding (0.6B; scale to 4B/8B) | Per-window L2-normalized embedding (1024-4096-d, Matryoshka-truncatable) | Yes (GPU) | core |
| Language | Semantics / discourse / narrative | Sliding-window cosine pipeline | Per-timepoint local coherence, semantic drift, novelty, segmentation-boundary score | Yes (CPU) | core |
| Language | Semantics / discourse / narrative | BERTopic (topics-over-time) | Per-window topic id + soft topic distribution; per-bin topic prevalence series | Yes (CPU) | core |
| Language | Semantics / discourse / narrative | Autoregressive LLM hidden-state extractor (GPT-2 / Llama-3.1-8B) | Per-token hidden-state vector + surprisal/entropy, resampled per TR (neuroimaging standard) | Yes (GPU) | core |
| Language | Semantics / discourse / narrative | Instruction-tuned LLM (Llama-3.1-8B / Qwen2.5-7B-Instruct) | Structured JSON: narrative stage, turning points, discourse relations, tension, summary | Yes (GPU) | core |
| Language | Semantics / discourse / narrative | DMRST (neural RST parser) | EDU boundaries + discourse-tree relations/nuclearity (reproducible labels) | Yes (GPU) | extended |
| Language | Semantics / discourse / narrative | Supervised turning-point model (TRIPOD-style) | 5 TP-type probabilities, normalized story position, sentiment-arc value | Yes (GPU) | extended |
| Language | Semantics / discourse / narrative | API embeddings (OpenAI/Voyage/Gemini) | Leaderboard-topping per-window embedding (only if off-site transfer acceptable) | API-only | extended |
| Social | Social & interpersonal | pyannote diarization (3.1 / community-1) | Speaker turns, active-speaker id, speaker count, overlap, turn-taking transitions | Yes (GPU) | core |
| Social | Social & interpersonal | Light-ASD / LR-ASD | Per-face per-frame speaking probability -> active speaker, speaker-vs-listener roles | Yes (GPU) | core |
| Social | Social & interpersonal | InsightFace (SCRFD + ArcFace) | Face count, identity track, characters-present set, co-presence matrix (social-network seed) | Yes (GPU) | core |
| Social | Social & interpersonal | Gaze-LLE | Per-person gaze target + heatmap -> mutual-gaze and joint-attention signals | Yes (GPU) | core |
| Social | Social & interpersonal | MMPose RTMPose + ByteTrack | Tracked skeletons -> inter-personal distance, body orientation, contact/touch proxy, approach/retreat | Yes (GPU) | core |
| Social | Social & interpersonal | Qwen2.5-VL (7B) | Per-shot interaction-type, dominance/affiliation, ToM/intention, addressee, social-network edges | Yes (heavy GPU) | core |
| Social | Social & interpersonal | Social-IQ 2.0 / Social Genome (schema) | Label schemas, few-shot exemplars, held-out evaluation items for the social extractor | Yes (CPU) | core |
| Social | Social & interpersonal | InternVL3 or frontier API judge | Second VLM for inter-rater agreement / gold-label validation | GPU / API | extended |
| Situation | Situations / schemas / event segmentation | GSBS / statesegmentation | Per-timepoint state label, ranked boundaries, optimal K, hierarchical boundaries | Yes (CPU) | core |
| Situation | Situations / event segmentation | TransNetV2 + PySceneDetect | Shot-boundary timestamps + per-frame change score (event-boundary prior) | Yes (GPU/CPU) | core |
| Situation | Situations / event segmentation | SigLIP 2 / OpenCLIP zero-shot | Per-frame location/time-of-day/setting/script attribute scores (custom taxonomy) | Yes (GPU) | core |
| Situation | Situations / event segmentation | Qwen2.5-VL (7B) | Timestamped event boundaries + Event-Indexing fields (space/time/causation/intention/protagonist), script labels | Yes (GPU) | core |
| Situation | Situations / event segmentation | Instruction-tuned / API LLM (transcript) | Transcript event boundaries + per-event situational fields (esp. audio stories) | Yes (heavy GPU) / API | core |
| Situation | Situations / event segmentation | Places365 ResNet-50 | Calibrated fixed-taxonomy location/setting + scene attributes | Yes (GPU) | extended |
| Situation | Situations / event segmentation | BrainIAK EventSegment HMM | Soft event posteriors / reactivation modeling | Yes (CPU) | extended |
| Affect | Emotion & affect (multimodal) | audEERING wav2vec2-dim | Per-window voice arousal/valence/dominance + 1024-d embedding | Yes (GPU) | core |
| Affect | Emotion & affect | HSEmotion / EmotiEffLib | Per-face categorical emotion + continuous valence/arousal + 1280-d embedding | Yes (CPU) | core |
| Affect | Emotion & affect | RoBERTa-GoEmotions (+ NRC-VAD mapping) | Per-utterance 28 emotion scores -> valence/arousal proxy | Yes (CPU) | core |
| Affect | Emotion & affect | Qwen2.5-VL (7B) | Per-window JSON: categorical emotion, V/A/D, intensity, confidence, justification (depicted vs viewer) | Yes (GPU) | core |
| Affect | Emotion & affect | LIRIS-ACCEDE / MuSe-trained continuous regressor | Per-second viewer-ELICITED valence/arousal (kept as a SEPARATE stream) | Yes (GPU) | core |
| Affect | Emotion & affect | MERT-based V/A regressor | Per-window soundtrack/music valence/arousal + mood tags | Yes (GPU) | extended |
| Affect | Emotion & affect | Emotion-LLaMA or AffectGPT | Per-clip benchmarked / open-vocabulary depicted-emotion label + multimodal-cue explanation | Yes (heavy GPU) | extended |
| Affect | Emotion & affect | EmoNet | Per-face valence/arousal + expression (citable cross-check) | Yes (GPU) | extended |

## Tradeoffs

**Cross-subclass reuse dominates within-subclass pruning.** Several signals appear in multiple classes and must be extracted once and routed everywhere, not re-run per catalog. The clearest cases: face detection/identity (InsightFace) feeds Faces, Social, and High-level visual; face emotion (HSEmotion/OpenFace 3.0) feeds Faces, Social, and Affect; body pose (MMPose RTMPose) feeds Faces, Social, and Dynamic visual; diarization (pyannote) feeds Speech and Social; prosody (Parselmouth + openSMILE) feeds Speech, Low-level acoustic, and Affect; shot boundaries (TransNetV2/PySceneDetect) feed Dynamic visual, Situation, and Action; vocal affect (audEERING wav2vec2-dim) feeds Speech and Affect; LLM surprisal and hidden states (minicons) feed Lexical and Semantics; and lexical-VAD/GoEmotions feed Lexical and Affect. Wiring the pipeline around shared passes, rather than per-class extractors, is what makes the whole set tractable on a single consumer GPU.

**The single largest compute saving is consolidating high-level reasoning into one video-LLM pass.** Action description, situational dimensions (space/time/causation/intention/protagonist), social interaction-type/dominance/ToM, and depicted emotion are all realized by Qwen2.5/3-VL-class models. Running one Qwen-VL pass per shot with a unified multi-field JSON schema replaces four separate VLM deployments. InternVL3, VideoLLaMA3, LLaVA-OneVision, Emotion-LLaMA, and AffectGPT are interchangeable for this role; keep at most one primary plus, optionally, one frontier-API judge reserved for gold-label validation rather than routine annotation.

**Foundation models supersede most fixed-taxonomy CNNs, but not always.** SigLIP 2 zero-shot probing covers the object/scene/place/attribute role that Places365 and ImageNet backbones (and YAMNet/AST in audio) used to fill. Keep the fixed-taxonomy models only when a frozen, calibrated, reproducible probability vector is explicitly required for interpretability. Likewise, classic flow (RAFT/Farneback), depth (MiDaS/ZoeDepth), saliency (TASED-Net/UNISAL), and shot detection (AutoShot) are superseded by SEA-RAFT, Depth-Anything-V2, ViNet, and TransNetV2 respectively; the older tools survive only as no-GPU fallbacks.

**Some surface-similar pairs are genuinely distinct and must both be kept.** DINOv2 (label-free dense embedding for RSA) versus SigLIP 2 (text-probed interpretable scores) are different representations. pymoten (brain-aligned motion-energy regressor) versus SEA-RAFT (interpretable camera-vs-object flow), and semantic action recognition (VideoMAE) versus pre-semantic motion (SEA-RAFT/pymoten), are deliberately separate layers. BEATs (calibrated 527-class) versus CLAP (arbitrary open-vocabulary prompts), inaSpeechSegmenter (speech/music/noise partition) versus pyannote (who-spoke-when), Mask2Former (area-fraction composition) versus Grounding DINO (object counts/positions), and CREPE (robust pitch contour) versus Parselmouth (canonical voice-quality) all carry complementary information.

**The one non-negotiable "keep two" is depicted versus elicited emotion.** Face/voice/text emotion models predict character-expressed (depicted) affect, whereas LIRIS-ACCEDE/MuSe-trained regressors predict viewer-induced (elicited) affect. These are correlated but not interchangeable; logging them as a single stream would be a scientific error, not a parsimony gain. Similarly, interpretable per-modality affect signals (wav2vec2 + HSEmotion + GoEmotions) should remain the primary features, with any fusion MLLM as an added reasoned layer rather than a replacement.

**Local-first is feasible throughout.** Every core recommendation runs locally; most low-level audio and lexical tools are CPU-only, and the heaviest core components (Qwen-VL, VideoMAE, SigLIP 2, DINOv2, pyannote, Whisper) fit on a single 24-48 GB GPU. API embeddings and frontier-API VLM judges are the only API-only entries and are confined to the extended tier, used only when leaderboard-topping accuracy is required and off-site transfer of transcripts/frames is acceptable.
