"""Constant-shape support: fill missing channels with NaN skeletons from a template.

For corpus runs (Phase 3/4) every stimulus must yield the SAME channel set so the
annotations stack into rectangular matrices. A *channel template* (schema/
channel_template.json) lists every channel the full pipeline can emit, with its
dtype/dim/components. After a run, any template channel that was not produced —
because its class doesn't apply to the stimulus modality, or the pass was disabled —
is added as an all-null channel with applicable=False. The template is generated
from a real full run (tools/build_channel_template.py), so it never drifts from code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import FeatureChannel, TimeGrid


def build_template(channels: list[FeatureChannel]) -> dict:
    specs = []
    for c in channels:
        v = np.asarray(c.value)
        specs.append({
            "path": c.path, "dtype": c.dtype, "units": c.units, "model": c.model,
            "dim": int(v.shape[1]) if (c.dtype == "vector" and v.ndim == 2) else None,
            "components": list(c.components) if c.components is not None else None,
            "categories": list(c.categories) if c.categories is not None else None,
        })
    return {"channels": specs}


def load_template(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def _skeleton(spec: dict, grid: TimeGrid) -> FeatureChannel:
    n, dt = grid.n_samples, spec["dtype"]
    common = dict(path=spec["path"], dtype=dt, applicable=False, units=spec.get("units", ""),
                  model=spec.get("model", ""), resample="", native_rate_hz="",
                  notes="skeleton: feature not applicable to this stimulus / pass not run")
    if dt == "vector":
        dim = spec.get("dim") or (len(spec["components"]) if spec.get("components") else 1)
        return FeatureChannel(value=np.full((n, dim), np.nan, np.float32),
                              components=spec.get("components"), **common)
    if dt == "label":
        return FeatureChannel(value=np.array([]), labels=[""] * n, **common)
    if dt == "categorical":
        return FeatureChannel(value=np.full(n, -1, int), categories=spec.get("categories"), **common)
    if dt == "event":
        return FeatureChannel(value=np.zeros(n, int), **common)
    # scalar / bool
    return FeatureChannel(value=np.full(n, np.nan), **common)


def apply_template(channels: list[FeatureChannel], grid: TimeGrid, template: dict) -> list[FeatureChannel]:
    emitted = {c.path for c in channels}
    extra = [_skeleton(s, grid) for s in template["channels"] if s["path"] not in emitted]
    return channels + extra
