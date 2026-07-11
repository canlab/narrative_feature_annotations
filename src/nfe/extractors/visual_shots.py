"""Shot-boundary detection from the frame stream (CPU, cv2-free).

Color-histogram delta between consecutive sampled frames; a cut is flagged when
the normalized-histogram L1 distance exceeds a threshold and the minimum shot
length has elapsed. This is a lightweight stand-in for the frozen core pick
(TransNetV2 + PySceneDetect); swap in TransNetV2 when the GPU tier is wired.

Channels under features/visual/dynamic_motion/: cut (event), shot_index (scalar).
"""

from __future__ import annotations

import numpy as np

from ..base import (VISUAL_MODALITIES, Extractor, FeatureChannel, Stimulus,
                    TimeGrid, grid_events)

BINS = 8


def _hist(rgb: np.ndarray) -> np.ndarray:
    q = (rgb.astype(np.int32) * BINS // 256).clip(0, BINS - 1)
    idx = (q[..., 0] * BINS + q[..., 1]) * BINS + q[..., 2]
    h = np.bincount(idx.ravel(), minlength=BINS ** 3).astype(np.float64)
    return h / max(h.sum(), 1)


class VisualShots(Extractor):
    feature_class = "visual"
    name = "visual_shots"
    applicable_modalities = VISUAL_MODALITIES
    tier = "cpu"

    def __init__(self, threshold: float = 0.6, min_shot_sec: float = 0.5):
        self.threshold = threshold
        self.min_shot_sec = min_shot_sec

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        prev = None
        last_cut_t = -1e9
        cut_times = []
        for t, rgb in ingest.iter_frames():
            h = _hist(rgb)
            if prev is not None:
                dist = float(np.abs(h - prev).sum())   # L1 in [0, 2]
                if dist > self.threshold and (t - last_cut_t) >= self.min_shot_sec:
                    cut_times.append(t)
                    last_cut_t = t
            prev = h
        cut_times = np.array(cut_times, float)
        shot_idx = np.searchsorted(cut_times, grid.time_sec, side="right").astype(np.float64)
        return [
            FeatureChannel(
                path="visual/dynamic_motion/cut", value=grid_events(cut_times, grid),
                dtype="event", units="count", model="histogram-cut(TransNetV2-substitute)",
                version="slice-0.2", native_rate_hz="event", resample="count", tier="cpu",
                onsets=cut_times, notes=f"L1 color-hist delta > {self.threshold}"),
            FeatureChannel(
                path="visual/dynamic_motion/shot_index", value=shot_idx, dtype="scalar",
                units="index", model="histogram-cut", version="slice-0.2",
                native_rate_hz="shot", resample="step", tier="cpu",
                notes="0-based shot number active at each grid bin"),
        ]
