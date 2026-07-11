"""Monocular depth (Depth-Anything-V2, MPS/CPU). Per frame -> grid mean.

Channels under features/visual/saliency_aesthetics_depth/: depth_mean, depth_range,
foreground_frac, depth_entropy. Depth-Anything outputs relative inverse depth
(larger = nearer); stats are per-frame min-max normalized.
"""

from __future__ import annotations

import numpy as np

from ..base import (VISUAL_MODALITIES, Extractor, FeatureChannel, Stimulus,
                    TimeGrid, grid_reduce_scalar)
from ..torch_util import batched, get_device


def _stats(d: np.ndarray) -> tuple[float, float, float, float]:
    rng = float(d.max() - d.min())
    dn = (d - d.min()) / (rng + 1e-6)
    fg = float((dn > 0.75).mean())                       # nearest quartile
    hist = np.histogram(dn, bins=16, range=(0, 1))[0].astype(float)
    p = hist / max(hist.sum(), 1)
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return float(dn.mean()), rng, fg, ent


class VisualDepth(Extractor):
    feature_class = "visual"
    name = "visual_depth"
    applicable_modalities = VISUAL_MODALITIES
    tier = "gpu"

    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf",
                 batch_size: int = 8, device: str = "auto"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        self.dev = get_device(self.device)
        self.proc = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name).to(self.dev).eval()
        self._loaded = True

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        from PIL import Image
        self._load()
        torch = self.torch
        times, rows = [], []
        for chunk in batched(ingest.iter_frames(), self.batch_size):
            imgs = [Image.fromarray(rgb) for _, rgb in chunk]
            with torch.no_grad():
                inp = self.proc(images=imgs, return_tensors="pt").to(self.dev)
                pred = self.model(**inp).predicted_depth.cpu().numpy()   # [N,h,w]
            for k, (t, _) in enumerate(chunk):
                times.append(t); rows.append(_stats(pred[k]))
        if not times:
            return []
        times = np.array(times)
        rows = np.array(rows)
        meta = dict(model=self.model_name, version="hf", native_rate_hz=ingest.analysis_fps,
                    resample="mean", tier="gpu")
        names = ["depth_mean", "depth_range", "foreground_frac", "depth_entropy"]
        units = {"depth_entropy": "bits"}
        return [FeatureChannel(path=f"visual/saliency_aesthetics_depth/{n}",
                               value=grid_reduce_scalar(times, rows[:, i], grid, "mean"),
                               dtype="scalar", units=units.get(n, ""), **meta)
                for i, n in enumerate(names)]
