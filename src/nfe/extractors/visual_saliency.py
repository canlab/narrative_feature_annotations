"""Visual saliency (spectral-residual, numpy/scipy, CPU). Per frame -> grid mean.

Channels under features/visual/saliency_aesthetics_depth/: saliency_mean,
saliency_peak, saliency_entropy, salient_area_frac. Spectral-residual saliency
(Hou & Zhang 2007) is a fast, model-free attention proxy; the frozen core pick is
ViNet (audio-visual deep video saliency) — swap it in on the GPU tier for production.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from ..base import (VISUAL_MODALITIES, Extractor, FeatureChannel, Stimulus,
                    TimeGrid, grid_reduce_scalar)


def _spectral_residual(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(gray)
    log_amp = np.log(np.abs(f) + 1e-8)
    phase = np.angle(f)
    sr = log_amp - uniform_filter(log_amp, size=3)
    sal = np.abs(np.fft.ifft2(np.exp(sr + 1j * phase))) ** 2
    sal = gaussian_filter(sal, sigma=3)
    rng = sal.max() - sal.min()
    return (sal - sal.min()) / (rng + 1e-8)


def _features(rgb: np.ndarray) -> tuple[float, float, float, float]:
    gray = rgb[..., :3].mean(2)
    sal = _spectral_residual(gray)
    hist = np.histogram(sal, bins=16, range=(0, 1))[0].astype(float)
    p = hist / max(hist.sum(), 1)
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return float(sal.mean()), float(sal.max()), ent, float((sal > 3 * sal.mean()).mean())


class VisualSaliency(Extractor):
    feature_class = "visual"
    name = "visual_saliency"
    applicable_modalities = VISUAL_MODALITIES
    tier = "cpu"

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        times, rows = [], []
        for t, rgb in ingest.iter_frames():
            times.append(t)
            rows.append(_features(rgb))
        if not times:
            return []
        times, rows = np.array(times), np.array(rows)
        meta = dict(model="spectral-residual(ViNet-substitute)", version="hou2007",
                    native_rate_hz=ingest.analysis_fps, resample="mean", tier="cpu")
        names = ["saliency_mean", "saliency_peak", "saliency_entropy", "salient_area_frac"]
        units = {"saliency_entropy": "bits"}
        return [FeatureChannel(path=f"visual/saliency_aesthetics_depth/{n}",
                               value=grid_reduce_scalar(times, rows[:, i], grid, "mean"),
                               dtype="scalar", units=units.get(n, ""), **meta)
                for i, n in enumerate(names)]
