"""Core types and the common-grid resampler for the annotation pipeline.

Implements the contracts in docs/design/ANNOTATION_FORMAT.md:
- one shared, center-referenced common time grid per stimulus,
- a uniform Extractor interface,
- per-channel feature objects carrying value + provenance,
- the native-rate -> grid resampling rules (§2.4).
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

# All-null grid bins (e.g. unvoiced f0) collapse to NaN by design, not by error.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

Modality = Literal["audiovisual", "video-only", "audio-only", "text-only"]
Dtype = Literal["scalar", "vector", "categorical", "label", "bool", "event"]
ResampleOp = Literal["mean", "max", "sum", "nearest", "mode", "any", "frac", "count"]

VISUAL_MODALITIES = ("audiovisual", "video-only")
AUDIO_MODALITIES = ("audiovisual", "audio-only")
# Transcript-consuming passes also apply to pure-text stories.
LANGUAGE_MODALITIES = ("audiovisual", "audio-only", "text-only")


@dataclass
class Stimulus:
    """A movie or story handed to the pipeline."""

    id: str
    media_file: str
    modality: Modality
    duration_sec: float
    title: str = ""
    source: str = ""
    sha256: str = ""

    @property
    def has_video(self) -> bool:
        return self.modality in VISUAL_MODALITIES

    @property
    def has_audio(self) -> bool:
        return self.modality in AUDIO_MODALITIES


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    """Timestamped transcript — the hub feeding language/social/situational passes."""

    language: str = ""
    text: str = ""
    segments: list[Segment] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)

    @property
    def has_speech(self) -> bool:
        return len(self.words) > 0


@dataclass
class TimeGrid:
    """The common grid every feature is resampled onto (center-referenced bins)."""

    rate_hz: float = 1.0
    t_start_sec: float = 0.0
    n_samples: int = 0
    bin_reference: str = "center"

    @classmethod
    def from_duration(cls, duration_sec: float, rate_hz: float = 1.0) -> "TimeGrid":
        n = int(np.floor(duration_sec * rate_hz + 1e-9))
        return cls(rate_hz=rate_hz, t_start_sec=0.0, n_samples=max(n, 0))

    @property
    def time_sec(self) -> np.ndarray:
        return self.t_start_sec + np.arange(self.n_samples) / self.rate_hz

    def bin_index(self, t: np.ndarray) -> np.ndarray:
        """Map continuous times to grid-bin indices (center-referenced)."""
        return np.round((np.asarray(t, float) - self.t_start_sec) * self.rate_hz).astype(int)


@dataclass
class FeatureChannel:
    """One leaf channel: a value series on the common grid + provenance.

    ``value`` length == grid.n_samples. NaN / "" / -1 / 0 encode not-measured
    per dtype. When the feature does not apply to the stimulus modality,
    ``applicable=False`` and the value is the all-null fill.
    """

    path: str
    value: np.ndarray
    dtype: Dtype = "scalar"
    applicable: bool = True
    units: str = ""
    labels: Sequence[str] | None = None        # for dtype == "label"
    categories: Sequence[str] | None = None     # vocabulary for dtype == "categorical"
    components: Sequence[str] | None = None      # column names for vector dtypes
    model: str = ""
    version: str = ""
    native_rate_hz: float | str = ""
    resample: str = ""
    tier: str = ""
    notes: str = ""
    onsets: np.ndarray | None = None             # exact onset times for dtype == "event"


@dataclass
class ExtractorResult:
    channels: list[FeatureChannel] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Resampling: native (times, values) -> regular grid
# --------------------------------------------------------------------------- #
def grid_reduce_scalar(times, values, grid: TimeGrid, op: ResampleOp = "mean") -> np.ndarray:
    """Reduce per-sample scalar/vector values into grid bins.

    times: [N] seconds; values: [N] or [N, D]. Empty bins -> NaN.
    Supported ops: mean, max, sum, nearest.
    """
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    vec = values.ndim == 2
    D = values.shape[1] if vec else 1
    out = np.full((grid.n_samples, D) if vec else grid.n_samples, np.nan, float)
    if times.size == 0:
        return out
    idx = grid.bin_index(times)
    keep = (idx >= 0) & (idx < grid.n_samples) & np.isfinite(times)
    idx, v = idx[keep], values[keep]
    if idx.size == 0:
        return out
    if op == "nearest":
        centers = grid.t_start_sec + idx / grid.rate_hz
        order = np.argsort(-np.abs(times[keep] - centers))  # farthest first; nearest wins last
        for i in order:
            out[idx[i]] = v[i]
        return out
    for b in np.unique(idx):
        sel = v[idx == b]
        if op == "mean":
            out[b] = np.nanmean(sel, axis=0)
        elif op == "max":
            out[b] = np.nanmax(sel, axis=0)
        elif op == "sum":
            out[b] = np.nansum(sel, axis=0)
        else:
            raise ValueError(f"unsupported scalar op {op!r}")
    return out


def grid_reduce_labels(times, labels, grid: TimeGrid) -> list[str]:
    """Mode-reduce string labels into grid bins (empty bin -> '')."""
    times = np.asarray(times, float)
    out = [""] * grid.n_samples
    if times.size == 0:
        return out
    idx = grid.bin_index(times)
    buckets: dict[int, list[str]] = {}
    for i, lab in zip(idx, labels):
        if 0 <= i < grid.n_samples:
            buckets.setdefault(i, []).append(str(lab))
    for b, labs in buckets.items():
        out[b] = max(set(labs), key=labs.count)
    return out


def grid_events(onset_times, grid: TimeGrid) -> np.ndarray:
    """Per-bin event counts (int) from a list of onset times."""
    times = np.asarray(onset_times, float)
    out = np.zeros(grid.n_samples, int)
    if times.size == 0:
        return out
    idx = grid.bin_index(times)
    for i in idx[(idx >= 0) & (idx < grid.n_samples)]:
        out[i] += 1
    return out


class Extractor(ABC):
    """Base class for all feature-class extractors."""

    feature_class: str = ""              # top-level hierarchy node, e.g. "visual"
    name: str = ""                        # human label, e.g. "visual_lowlevel"
    applicable_modalities: tuple[Modality, ...] = ()
    tier: str = "cpu"

    def applies_to(self, stim: Stimulus) -> bool:
        return stim.modality in self.applicable_modalities

    @abstractmethod
    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        """Produce grid-aligned feature channels. ``ingest`` is a media.Ingest."""
        raise NotImplementedError
