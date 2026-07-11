"""Audio events / scenes (AST, AudioSet-527, MPS/CPU). Sliding window -> grid.

Channels under features/audio/high_level/: audioset_tags (527-d posterior vector,
components = AudioSet class names) and audioset_top (categorical top class per bin).
Frozen core pick is BEATs; AST is the HF-native stand-in (both AudioSet-527).
For framewise localization, PANNs CNN14 is the extended upgrade.
"""

from __future__ import annotations

import numpy as np

from ..base import (AUDIO_MODALITIES, Extractor, FeatureChannel, Stimulus, TimeGrid)
from ..torch_util import get_device

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"


class AudioEvents(Extractor):
    feature_class = "audio"
    name = "audio_events"
    applicable_modalities = AUDIO_MODALITIES
    tier = "gpu"

    def __init__(self, win_sec: float = 4.0, hop_sec: float = 1.0, device: str = "auto"):
        self.win_sec = win_sec
        self.hop_sec = hop_sec
        self.device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.torch = torch
        self.dev = get_device(self.device)
        self.fe = AutoFeatureExtractor.from_pretrained(MODEL)
        self.model = AutoModelForAudioClassification.from_pretrained(MODEL).to(self.dev).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self._loaded = True

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        import librosa
        self._load()
        torch = self.torch
        wav = ingest.audio_wav()
        if wav is None:
            return []
        y, _ = librosa.load(wav, sr=16000, mono=True)
        win, hop = int(self.win_sec * 16000), int(self.hop_sec * 16000)
        n_cls = len(self.labels)
        sums = np.zeros((grid.n_samples, n_cls))
        counts = np.zeros(grid.n_samples)
        for s in range(0, max(len(y) - win, 0) + 1, hop):
            seg = y[s:s + win]
            if seg.size < hop:
                break
            with torch.no_grad():
                inp = self.fe(seg, sampling_rate=16000, return_tensors="pt").to(self.dev)
                probs = self.model(**inp).logits.sigmoid()[0].cpu().numpy()   # multi-label
            c = int(grid.bin_index((s + win / 2) / 16000))
            if 0 <= c < grid.n_samples:
                sums[c] += probs
                counts[c] += 1
        post = np.divide(sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0)
        top = np.full(grid.n_samples, -1, int)
        valid = counts > 0
        top[valid] = post[valid].argmax(1)
        meta = dict(model=f"AST(BEATs-substitute):{MODEL}", version="audioset",
                    native_rate_hz=f"window~{self.win_sec}s", tier="gpu")
        return [
            FeatureChannel(path="audio/high_level/audioset_tags", value=post.astype(np.float32),
                           dtype="vector", components=self.labels, resample="mean",
                           units="sigmoid", **meta),
            FeatureChannel(path="audio/high_level/audioset_top", value=top, dtype="categorical",
                           categories=self.labels, resample="mode",
                           notes="argmax AudioSet class per bin (-1 = none)", **meta),
        ]
