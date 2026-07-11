# Semantic Feature Hierarchy

This semantic hierarchy organizes all annotation features into a coherent tree. The natural organizing axes are MODALITY (visual / audio / language / multimodal) crossed with LEVEL (low-level signal → mid → high-level semantic/social/situational/affective).

# Semantic Hierarchy for Naturalistic-Stimulus Annotation Features

The tree is organized first by **modality** (the signal source: visual pixels, audio waveform, linguistic transcript) and second by **representational level** (low-level perceptual signal → mid-level structure → high-level semantic/social/situational meaning). Cross-modal classes (Social, Situation, Affect) sit at the top level because they are defined by *what is represented*, not by a single signal source, and they consume features from multiple modalities.

```
ANNOTATION FEATURES
│
├── 1. VISUAL  — features derived from the image/video stream
│   │
│   ├── 1.1 Low-level static (per-frame signal statistics)
│   │   ├── 1.1.1 Luminance & contrast        (mean luminance, RMS/Michelson contrast, luminance histogram)
│   │   ├── 1.1.2 Color                        (RGB/HSV/CIELAB means+SDs, colorfulness, warmth, saturation, dominant hue)
│   │   ├── 1.1.3 Spatial frequency & orientation energy  (FFT power spectrum/slope, steerable-pyramid subband energy, Gabor banks)
│   │   ├── 1.1.4 Texture & edges             (edge density, GLCM, LBP, Laplacian sharpness/focus)
│   │   ├── 1.1.5 Complexity & clutter        (Shannon entropy, Feature-Congestion, Subband-Entropy)
│   │   └── 1.1.6 Holistic spatial envelope   (GIST descriptor — bridges to mid-level)
│   │
│   ├── 1.2 Low-level dynamic (per-frame-pair motion signal)
│   │   ├── 1.2.1 Optical flow                (dense u,v field; mean/median magnitude, orientation histogram, divergence/curl)
│   │   ├── 1.2.2 Camera vs object motion     (global-motion estimate + residual object-motion magnitude)
│   │   ├── 1.2.3 Motion energy               (pymoten spatiotemporal Gabor-channel energies for fMRI encoding)
│   │   └── 1.2.4 Shot/cut boundaries & visual change  (cut probabilities, scene segments, frame-to-frame novelty score)
│   │
│   ├── 1.3 Mid-level perceptual attention & geometry
│   │   ├── 1.3.1 Visual saliency             (static & spatiotemporal fixation maps; concentration, entropy, peak, salient-area)
│   │   ├── 1.3.2 Monocular depth             (relative/metric depth maps; mean/range depth, foreground fraction, layering)
│   │   └── 1.3.3 Image aesthetics & quality  (aesthetic score, technical IQA, memorability)
│   │
│   ├── 1.4 High-level static semantics (what is depicted, per frame)
│   │   ├── 1.4.1 Vision-language embeddings & probes  (CLIP/SigLIP image embeddings; open-vocab object/scene/attribute scores)
│   │   ├── 1.4.2 Self-supervised dense embeddings     (DINOv2/v3 CLS + patch features for RSA / linear probes)
│   │   ├── 1.4.3 Objects (localized)          (open-vocab detection boxes, labels, counts, positions)
│   │   ├── 1.4.4 Scenes/places/attributes     (scene-category posteriors, indoor/outdoor, scene attributes)
│   │   ├── 1.4.5 Scene composition (panoptic) (per-class area fractions, instance masks/counts, segmentation tracks)
│   │   └── 1.4.6 Frame description (VLM)       (free-text captions + structured VQA fields per keyframe)
│   │
│   ├── 1.5 High-level dynamic semantics (what happens, over time)
│   │   ├── 1.5.1 Action recognition          (Kinetics/SSv2 posteriors + pooled spatiotemporal embeddings, sliding window)
│   │   ├── 1.5.2 Zero-shot open-vocab actions (X-CLIP/InternVideo2 similarity to custom action phrases)
│   │   ├── 1.5.3 Temporal action localization (start–end action segments with confidence)
│   │   └── 1.5.4 Dense event description (video-LLM)  (timestamped open-vocabulary event captions / temporal grounding)
│   │
│   └── 1.6 Person-centric visual (faces, bodies, gaze, expression)
│       ├── 1.6.1 Face detection & identity    (bboxes, landmarks, ArcFace identity embeddings, age/sex)
│       ├── 1.6.2 Head pose & gaze             (yaw/pitch/roll, gaze vectors, gaze targets)
│       ├── 1.6.3 Facial expression            (FACS action units, discrete emotion, valence/arousal — depicted)
│       └── 1.6.4 Body & hand pose             (2D/3D/whole-body keypoints → posture, gesture, orientation)
│
├── 2. AUDIO  — features derived from the soundtrack/waveform
│   │
│   ├── 2.1 Low-level acoustic (per-frame signal descriptors)
│   │   ├── 2.1.1 Loudness & energy           (RMS, loudness in sones, intensity dB, zero-crossing rate)
│   │   ├── 2.1.2 Spectral descriptors        (centroid, rolloff, flux, flatness, bandwidth, contrast, MFCC, mel-spectrogram)
│   │   ├── 2.1.3 Pitch / F0                   (F0 contour, voiced flags, salience — CREPE/pYIN/Praat)
│   │   ├── 2.1.4 Harmonic & psychoacoustic    (harmonicity/HNR, inharmonicity, roughness/dissonance, sharpness, tonality)
│   │   └── 2.1.5 Voice quality                (jitter, shimmer, formants F1–F4, spectral tilt — segment-level)
│   │
│   ├── 2.2 Mid-level musical structure
│   │   ├── 2.2.1 Tonal/harmonic content      (chroma, tonnetz, key/mode + strength, chords)
│   │   └── 2.2.2 Rhythm & meter              (onsets, tempo/BPM, beat & downbeat times, time signature)
│   │
│   └── 2.3 High-level audio semantics
│       ├── 2.3.1 Sound events / tagging       (527-class AudioSet posteriors, framewise SED, embeddings)
│       ├── 2.3.2 Acoustic scene               (scene classification, ambience)
│       ├── 2.3.3 Speech/music/noise segmentation  (time-stamped speech/music/noise; VAD)
│       ├── 2.3.4 Open-vocab audio probes (CLAP)   (per-prompt audio-text similarity time series)
│       └── 2.3.5 Music semantics (MIR)        (genre, mood/theme, instrument tags, danceability, music valence/arousal)
│
├── 3. LANGUAGE  — features derived from the (time-aligned) transcript
│   │
│   ├── 3.1 Speech-to-language interface (paralinguistic bridge from audio)
│   │   ├── 3.1.1 ASR transcript & timing      (word/segment text + timestamps, confidence, speaking rate)
│   │   ├── 3.1.2 Speaker diarization & turns   (who-spoke-when, speaker count, overlap, turn transitions)
│   │   ├── 3.1.3 Prosody contours             (per-second F0/intensity/HNR resampled from acoustic layer)
│   │   └── 3.1.4 Vocal affect                 (dimensional A/V/D + categorical emotion from voice — depicted)
│   │
│   ├── 3.2 Low-level lexical (per-word psycholinguistic features)
│   │   ├── 3.2.1 Token annotations            (lemma, POS, NER, stop/punct flags)
│   │   ├── 3.2.2 Frequency & length           (Zipf/log frequency, contextual diversity, AoA, word length)
│   │   ├── 3.2.3 Semantic norms               (concreteness, imageability, valence/arousal/dominance per word)
│   │   ├── 3.2.4 Emotion lexicons             (NRC EmoLex 8-emotion flags, LIWC affect categories)
│   │   └── 3.2.5 Surprisal & entropy          (per-word LLM surprisal in bits, next-word entropy)
│   │
│   ├── 3.3 Mid-level syntactic structure (per-utterance)
│   │   ├── 3.3.1 Dependency & morphology      (dep relations+heads, UD morph feats: tense/aspect/mood)
│   │   ├── 3.3.2 Constituency trees           (PTB phrase-structure trees, nesting depth)
│   │   ├── 3.3.3 Syntactic complexity         (tree depth, Yngve/Frazier, L2SCA clause/T-unit indices, readability)
│   │   ├── 3.3.4 Coreference                  (mention clusters; who/what each pronoun refers to)
│   │   └── 3.3.5 Dialogue acts                (per-utterance speech-act categories)
│   │
│   └── 3.4 High-level semantics, discourse & narrative
│       ├── 3.4.1 Passage embeddings           (dense sentence/window embeddings — the semantic backbone)
│       ├── 3.4.2 Contextual LLM hidden states (per-word last-layer states — fMRI encoding-model standard)
│       ├── 3.4.3 Coherence/drift/novelty      (sliding-window cosine coherence, drift, segmentation troughs)
│       ├── 3.4.4 Topic structure              (topic assignment + topics-over-time prevalence)
│       ├── 3.4.5 Discourse relations          (RST/EDU segmentation, relation labels, nuclearity)
│       └── 3.4.6 Narrative structure (LLM)    (narrative stage, turning points, structured tags, summaries)
│
├── 4. SOCIAL  — depicted interpersonal content (cross-modal: vision + audio + language)
│   │
│   ├── 4.1 Social primitives (deterministic, per-second tracks)
│   │   ├── 4.1.1 Agent presence & identity     (character count, identity tracks, co-presence matrix)
│   │   ├── 4.1.2 Active speaker / addressee     (who-talks-on-screen, speaker↔face binding, listener roles)
│   │   ├── 4.1.3 Gaze & joint attention         (mutual gaze, shared-target / joint-attention signals)
│   │   └── 4.1.4 Proximity, orientation & touch  (pairwise distance, facing toward/away, contact, approach/retreat)
│   │
│   └── 4.2 Social semantics (VLM/LLM-inferred, per-shot/scene)
│       ├── 4.2.1 Interaction type             (cooperation/conflict, dialogue, joint action categories)
│       ├── 4.2.2 Dominance & affiliation       (relational ratings, social-network edges)
│       └── 4.2.3 Theory-of-mind & intention    (mental-state, goal, and intention inferences)
│
├── 5. SITUATION  — setting, schemas, scripts & event structure (cross-modal)
│   │
│   ├── 5.1 Event segmentation                 (data-driven boundaries on feature timelines: HMM/GSBS; + shot priors)
│   ├── 5.2 Setting & location                 (scene type, indoor/outdoor, time-of-day, spatial layout)
│   ├── 5.3 Scripts & schemas                  (recognized situational scripts / schema labels)
│   └── 5.4 Event-indexing dimensions          (space, time, causation, intention, protagonist per event/segment)
│
└── 6. AFFECT  — emotion & affect (cross-modal; depicted vs elicited kept distinct)
    │
    ├── 6.1 Depicted affect (expressed by characters, by source modality)
    │   ├── 6.1.1 Facial affect                (face valence/arousal + categorical — from 1.6.3)
    │   ├── 6.1.2 Vocal affect                 (voice A/V/D + categorical — from 3.1.4)
    │   ├── 6.1.3 Textual/dialogue affect       (utterance emotion + VAD mapping — from 3.2.3/3.4)
    │   └── 6.1.4 Music/soundtrack affect       (soundtrack valence/arousal, mood — from 2.3.5)
    │
    ├── 6.2 Multimodal fused affect            (MLLM window-level categorical + V/A/D + reasoned justification)
    │
    └── 6.3 Elicited (viewer) affect           (induced V/A timeseries, emotional-impact/fear flags — SEPARATE stream)
```

## Node-by-node rationale (design notes)

**Top-level split (1–3 modality, 4–6 cross-modal).** Classes 1–3 are pure *signal-source* modalities: every feature traces to pixels, the waveform, or the transcript. Classes 4–6 are *representational targets* (social content, situational structure, affect) that are intrinsically multimodal and reuse outputs from 1–3 — so they are organized by what they represent, not by signal source. This keeps each leaf feature in exactly one home while letting cross-modal classes reference contributing leaves (the "from X.Y.Z" pointers).

**Level ordering within each modality.** Every modality is internally ordered low-level signal → mid-level structure → high-level semantics, and within visual/audio the static-vs-dynamic distinction is preserved as a parallel axis (1.1/1.4 static; 1.2/1.5 dynamic). This is the scoping-review backbone: a reviewer can walk any modality from "what the sensor measures" to "what it means."

**Language as a bridge modality.** Subclass 3.1 (Speech-to-language interface) is the deliberate seam between Audio and Language: ASR, diarization, prosody, and vocal affect are computed from the waveform but produce the time-aligned linguistic substrate, so they live at the head of Language. Acoustic prosody itself stays in 2.1 (signal) and is *referenced*, not duplicated, by 3.1.3.

**Depicted vs elicited affect.** Class 6 explicitly separates depicted/expressed affect (6.1, organized by contributing modality) from viewer-elicited/induced affect (6.3) because they have different ground truth and must be logged as distinct feature streams — a recurring caution in the affect catalog.

## Implications for the output data format

This tree maps directly onto a hierarchical schema. Each leaf is a **feature group** with a stable dotted path (e.g. `visual.lowlevel_static.color`, `language.semantics.coherence_drift`, `affect.depicted.vocal`) and a uniform record envelope:

- `path` (semantic address), `modality`, `level` (low/mid/high), `temporal_unit` (frame / window / utterance / event / shot), `onset`, `duration`, `value(s)`, plus provenance (`extractor`, `model`, `version`, `recommendation_tier`).
- Cross-modal classes (4–6) store **references** to the contributing leaf paths rather than copying values, encoding the "from X.Y.Z" pointers as first-class lineage.
- The `level` and `static/dynamic` axes become filterable facets — the same metadata that structures the scoping review structures query and resampling (everything resamples to a common per-second / per-TR movie clock).

Paths worth flagging for the format spec: `temporal_unit` and a `depicted_vs_elicited` flag (on class 6) are load-bearing — they are the two fields most likely to be conflated downstream and should be required, not optional.
