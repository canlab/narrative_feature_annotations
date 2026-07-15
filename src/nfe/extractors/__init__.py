"""Extractor registry.

CORE_CPU are the passes runnable today on this machine (no GPU/CUDA needed).
GPU/VLM/hosted passes from the frozen core set (PHASE2_CORE_SET_PROPOSAL.md) are
added here as they are implemented; the pipeline runs whatever is registered and
applies to the stimulus modality.
"""

from __future__ import annotations

from .asr import ASR
from .audio_clap import AudioClap
from .audio_events import AudioEvents
from .audio_lowlevel import AudioLowLevel
from .event_segmentation import EventSegmentation
from .face_emotion import FaceEmotion
from .faces import Faces
from .language_hl import LanguageHL
from .language_lexical import LanguageLexical
from .language_surprisal import LanguageSurprisal
from .language_syntax import LanguageSyntax
from .pose import Pose
from .qwen_reasoning import QwenReasoning
from .text_emotion import TextEmotion
from .text_sentiment import TextSentiment
from .visual_action import VisualAction
from .visual_depth import VisualDepth
from .visual_emonet import VisualEmoNet
from .visual_lowlevel import VisualLowLevel
from .visual_motion import VisualMotion
from .visual_saliency import VisualSaliency
from .visual_semantic import VisualSemantic
from .visual_shots import VisualShots
from .vocal_affect import VocalAffect


def core_cpu_extractors(asr_model: str = "small"):
    # ASR runs before the language passes: it attaches the transcript to the Ingest
    # object that LanguageLexical / LanguageSyntax then consume.
    return [
        VisualLowLevel(), VisualShots(), AudioLowLevel(),
        ASR(model_size=asr_model), LanguageLexical(), LanguageSyntax(),
    ]


def vision_mps_extractors():
    """GPU/MPS (+ cheap CPU saliency) visual passes. Opt in via the `vision` flag."""
    return [VisualSemantic(), VisualMotion(), VisualDepth(), VisualAction(),
            Faces(), Pose(), VisualSaliency(), VisualEmoNet(), FaceEmotion()]


def audio_hl_mps_extractors():
    """GPU/MPS audio + transcript-affect/surprisal passes. Opt in via `audio_hl`."""
    return [AudioEvents(), AudioClap(), VocalAffect(), TextEmotion(), TextSentiment(),
            LanguageSurprisal(), LanguageHL()]


def reasoning_extractors():
    """Consolidated VLM reasoning (Qwen2.5-VL). Heaviest; opt in via `reason`."""
    return [QwenReasoning()]


def event_extractors():
    """Post-pass over the assembled feature matrix (GSBS). Opt in via `events`."""
    return [EventSegmentation()]


def default_extractors(asr_model: str = "small", vision: bool = False, audio_hl: bool = False,
                       reason: bool = False, events: bool = False):
    exts = core_cpu_extractors(asr_model)
    if vision:
        exts += vision_mps_extractors()
    if audio_hl:
        exts += audio_hl_mps_extractors()
    if reason:
        exts += reasoning_extractors()
    if events:
        exts += event_extractors()      # runs last (post-pass over everything above)
    return exts
