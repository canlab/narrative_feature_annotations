"""Visual low-level static properties (CPU). Per-frame -> grid mean.

Channels under features/visual/low_level_static/: luminance, rms_contrast,
colorfulness, edge_density, entropy, {r,g,b}_mean, {hue,sat,val}_mean, fft_slope.
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2hsv
from skimage.feature import canny

from ..base import (VISUAL_MODALITIES, Extractor, FeatureChannel, Stimulus,
                    TimeGrid, grid_reduce_scalar)

_SCALARS = ["luminance", "rms_contrast", "colorfulness", "edge_density", "entropy",
            "r_mean", "g_mean", "b_mean", "hue_mean", "sat_mean", "val_mean", "fft_slope"]
_UNITS = {"luminance": "0-1", "rms_contrast": "0-1", "entropy": "bits",
          "hue_mean": "0-1", "sat_mean": "0-1", "val_mean": "0-1", "edge_density": "0-1"}


def _radial_fft_slope(gray: np.ndarray) -> float:
    """Slope of the log-log radially-averaged power spectrum (natural-image ~ -2)."""
    g = gray.astype(np.float32) - gray.mean()
    p = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = p.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.hypot(y - cy, x - cx).astype(int)
    nbin = min(cy, cx)
    if nbin < 8:
        return np.nan
    radial = np.bincount(r.ravel(), p.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    f = np.arange(1, nbin)
    pw = radial[1:nbin]
    good = pw > 0
    if good.sum() < 4:
        return np.nan
    return float(np.polyfit(np.log(f[good]), np.log(pw[good]), 1)[0])


def _frame_features(rgb: np.ndarray) -> dict:
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    # Hasler-Susstrunk colorfulness
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    gray_u8 = gray.astype(np.uint8)
    edges = canny(gray / 255.0, sigma=1.0)
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    hsv = rgb2hsv(rgb)   # H,S,V all in 0-1
    return {
        "luminance": float(gray.mean() / 255.0),
        "rms_contrast": float(gray.std() / 255.0),
        "colorfulness": colorfulness,
        "edge_density": float(edges.mean()),
        "entropy": entropy,
        "r_mean": float(r.mean() / 255.0), "g_mean": float(g.mean() / 255.0), "b_mean": float(b.mean() / 255.0),
        "hue_mean": float(hsv[..., 0].mean()), "sat_mean": float(hsv[..., 1].mean()),
        "val_mean": float(hsv[..., 2].mean()),
        "fft_slope": _radial_fft_slope(gray),
    }


class VisualLowLevel(Extractor):
    feature_class = "visual"
    name = "visual_lowlevel"
    applicable_modalities = VISUAL_MODALITIES
    tier = "cpu"

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        times, rows = [], []
        for t, rgb in ingest.iter_frames():
            times.append(t)
            rows.append(_frame_features(rgb))
        if not times:
            return []
        times = np.array(times)
        out = []
        for name in _SCALARS:
            vals = np.array([row[name] for row in rows], float)
            gridded = grid_reduce_scalar(times, vals, grid, "mean")
            out.append(FeatureChannel(
                path=f"visual/low_level_static/{name}", value=gridded, dtype="scalar",
                units=_UNITS.get(name, ""), model="scikit-image", version="slice-0.2",
                native_rate_hz=ingest.analysis_fps, resample="mean", tier="cpu"))
        return out
