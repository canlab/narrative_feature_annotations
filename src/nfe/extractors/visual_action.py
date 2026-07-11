"""Action recognition (VideoMAE V2, MPS/CPU). Sliding 16-frame clips -> grid.

Channel under features/visual/action/: action_posteriors (Kinetics-400 softmax
vector, components = class names), plus action_top (categorical, top class per bin).
Each clip's posterior is assigned to the grid bins its time span covers.
"""

from __future__ import annotations

import numpy as np

from ..base import (VISUAL_MODALITIES, Extractor, FeatureChannel, Stimulus, TimeGrid)
from ..torch_util import get_device


class VisualAction(Extractor):
    feature_class = "visual"
    name = "visual_action"
    applicable_modalities = VISUAL_MODALITIES
    tier = "gpu"

    def __init__(self, model_name="MCG-NJU/videomae-base-finetuned-kinetics",
                 clip_len: int = 16, stride_sec: float = 2.0, device: str = "auto"):
        self.model_name = model_name
        self.clip_len = clip_len
        self.stride_sec = stride_sec
        self.device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
        self.torch = torch
        self.dev = get_device(self.device)
        self.proc = VideoMAEImageProcessor.from_pretrained(self.model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(self.model_name).to(self.dev).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self._loaded = True

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        self._load()
        torch = self.torch
        frames = list(ingest.iter_frames())
        if len(frames) < self.clip_len:
            return []
        step = max(1, round(self.stride_sec * ingest.analysis_fps))
        starts = list(range(0, len(frames) - self.clip_len + 1, step))

        n_cls = len(self.labels)
        sums = np.zeros((grid.n_samples, n_cls))
        counts = np.zeros(grid.n_samples)
        for s in starts:
            window = frames[s:s + self.clip_len]
            clip = [rgb for _, rgb in window]
            with torch.no_grad():
                inp = self.proc(clip, return_tensors="pt").to(self.dev)
                probs = self.model(**inp).logits.softmax(-1)[0].cpu().numpy()
            b0, b1 = int(grid.bin_index(window[0][0])), int(grid.bin_index(window[-1][0]))
            for b in range(max(b0, 0), min(b1, grid.n_samples - 1) + 1):
                sums[b] += probs
                counts[b] += 1
        post = np.divide(sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0)
        top = np.full(grid.n_samples, -1, int)
        valid = counts > 0
        top[valid] = post[valid].argmax(1)
        meta = dict(model=self.model_name, version="kinetics-400",
                    native_rate_hz="window", tier="gpu")
        return [
            FeatureChannel(path="visual/action/action_posteriors", value=post.astype(np.float32),
                           dtype="vector", components=self.labels, resample="mean",
                           units="softmax", **meta),
            FeatureChannel(path="visual/action/action_top", value=top, dtype="categorical",
                           categories=self.labels, resample="mode",
                           notes="argmax Kinetics class per bin (-1 = no clip)", **meta),
        ]
