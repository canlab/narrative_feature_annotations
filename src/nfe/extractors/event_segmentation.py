"""Event segmentation (GSBS, CPU) — a POST-pass over the assembled feature matrix.

Unlike the per-media extractors, this consumes the channels already produced for the
stimulus: it stacks the scalar channels into a [time x features] matrix and runs
Greedy State Boundary Search (Geerligs et al.) to find data-driven event states.

Channels under features/situation/: event_id (state index per bin) and
event_boundary (event onsets). The pipeline calls post_extract() after the normal
extractors. Frozen core pick: GSBS / statesegmentation.
"""

from __future__ import annotations

import numpy as np

from ..base import Extractor, FeatureChannel, Stimulus, TimeGrid, grid_events


class EventSegmentation(Extractor):
    feature_class = "situation"
    name = "event_segmentation"
    applicable_modalities = ("audiovisual", "video-only", "audio-only", "text-only")
    tier = "cpu"

    def __init__(self, kmax: int | None = None):
        self.kmax = kmax

    # Not a media extractor; the pipeline dispatches on post_extract().
    def extract(self, stim, grid, ingest):
        return []

    def _matrix(self, channels, n):
        cols = []
        for ch in channels:
            if ch.dtype != "scalar" or not ch.applicable:
                continue
            v = np.asarray(ch.value, float)
            if v.shape != (n,) or np.all(np.isnan(v)):
                continue
            sd = np.nanstd(v)
            if sd < 1e-9:
                continue
            cols.append(np.nan_to_num((v - np.nanmean(v)) / sd, nan=0.0))
        return np.array(cols).T if cols else np.empty((n, 0))

    def post_extract(self, stim: Stimulus, grid: TimeGrid,
                     channels: list[FeatureChannel]) -> list[FeatureChannel]:
        n = grid.n_samples
        x = self._matrix(channels, n)
        meta = dict(model="GSBS(statesegmentation)", version="1.x",
                    native_rate_hz="derived", tier="cpu")
        if x.shape[1] < 2 or n < 6:
            # Same dtypes as the normal branch (constant-shape contract): scalar + event.
            return [
                FeatureChannel(path="situation/event_id", value=np.full(n, np.nan),
                               dtype="scalar", units="state-index", resample="mode",
                               notes="too few features/timepoints for GSBS", **meta),
                FeatureChannel(path="situation/event_boundary", value=np.zeros(n, int),
                               dtype="event", units="count", resample="count",
                               onsets=np.array([]), **meta)]
        from statesegmentation import GSBS
        kmax = self.kmax or min(max(n // 3, 2), 25)
        g = GSBS(x=x, kmax=kmax)
        g.fit()
        states = np.asarray(g.states, int)
        bounds = np.asarray(g.bounds, int)
        onsets = grid.time_sec[bounds > 0]
        return [
            FeatureChannel(path="situation/event_id", value=states.astype(np.float64),
                           dtype="scalar", units="state-index", resample="mode",
                           notes=f"GSBS optimal k={int(getattr(g, 'nstates', states.max() + 1))}", **meta),
            FeatureChannel(path="situation/event_boundary", value=grid_events(onsets, grid),
                           dtype="event", units="count", resample="count", onsets=onsets, **meta),
        ]
