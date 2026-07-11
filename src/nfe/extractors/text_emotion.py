"""Dialogue/narration emotion from text (RoBERTa-GoEmotions, MPS/CPU).

Per transcript segment -> 28 GoEmotions sigmoid scores, assigned to the bins the
segment spans. Channels under features/affect/depicted/: text_emotion (28-d vector)
and text_emotion_top (categorical). Frozen core pick; a depicted (content) affect
stream distinct from voice/face/elicited affect.
"""

from __future__ import annotations

import numpy as np

from ..base import (LANGUAGE_MODALITIES, Extractor, FeatureChannel, Stimulus, TimeGrid)
from ..torch_util import get_device

MODEL = "SamLowe/roberta-base-go_emotions"
GOEMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]


class TextEmotion(Extractor):
    feature_class = "affect"
    name = "text_emotion"
    applicable_modalities = LANGUAGE_MODALITIES
    tier = "gpu"

    def __init__(self, device: str = "auto"):
        self.device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.dev = get_device(self.device)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(self.dev).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self._loaded = True

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        tr = ingest.get_transcript()
        meta = dict(model=MODEL, version="go_emotions", native_rate_hz="utterance", tier="gpu")
        if tr is None or not tr.has_speech:
            post = np.full((grid.n_samples, len(GOEMOTIONS)), np.nan, np.float32)
            top = np.full(grid.n_samples, -1, int)
            note = "no speech" if tr else "no transcript"
            return [
                FeatureChannel(path="affect/depicted/text_emotion", value=post, dtype="vector",
                               components=GOEMOTIONS, resample="mean", notes=note, **meta),
                FeatureChannel(path="affect/depicted/text_emotion_top", value=top, dtype="categorical",
                               categories=GOEMOTIONS, resample="mode", notes=note, **meta)]
        self._load()
        torch = self.torch
        sums = np.zeros((grid.n_samples, len(self.labels)))
        counts = np.zeros(grid.n_samples)
        for seg in tr.segments:
            if not seg.text.strip():
                continue
            with torch.no_grad():
                enc = self.tok(seg.text, return_tensors="pt", truncation=True, max_length=128).to(self.dev)
                probs = self.model(**enc).logits.sigmoid()[0].cpu().numpy()
            b0, b1 = int(grid.bin_index(seg.start)), int(grid.bin_index(seg.end))
            for b in range(max(b0, 0), min(b1, grid.n_samples - 1) + 1):
                sums[b] += probs
                counts[b] += 1
        post = np.divide(sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0)
        top = np.full(grid.n_samples, -1, int)
        valid = counts > 0
        top[valid] = post[valid].argmax(1)
        return [
            FeatureChannel(path="affect/depicted/text_emotion", value=post.astype(np.float32),
                           dtype="vector", components=self.labels, units="sigmoid",
                           resample="mean", **meta),
            FeatureChannel(path="affect/depicted/text_emotion_top", value=top, dtype="categorical",
                           categories=self.labels, resample="mode", **meta),
        ]
