"""Write annotations as HDF5 (canonical) + JSON sidecar manifest.

Follows docs/design/ANNOTATION_FORMAT.md: one shared time grid; hierarchical
groups under /features mirroring the semantic hierarchy; per-channel metadata as
HDF5 attributes; a /human group reserved (empty) for later human annotation; the
JSON sidecar mirrors the hierarchy with metadata + data_ref pointers (no bulk arrays).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .base import FeatureChannel, Stimulus, TimeGrid

SCHEMA_VERSION = "0.2"
STR = h5py.string_dtype(encoding="utf-8")


def _feat_path(ch: FeatureChannel) -> str:
    return f"/features/{ch.path}"


def _encode(ch: FeatureChannel):
    """Return (numpy_array_for_h5, fill_note) for a channel's value."""
    v = ch.value
    if ch.dtype in ("scalar", "vector"):
        arr = np.asarray(v, np.float32 if ch.dtype == "vector" else np.float64)
    elif ch.dtype == "bool":
        a = np.asarray(v, float)
        arr = np.where(np.isnan(a), -1, a).astype(np.int8)
    elif ch.dtype == "event":
        arr = np.asarray(v, np.int32)
    elif ch.dtype == "categorical":
        arr = np.asarray(v, np.int32)            # codes; -1 == <undefined>
    elif ch.dtype == "label":
        arr = np.array([s if s is not None else "" for s in (ch.labels or [])], dtype=STR)
    else:
        raise ValueError(f"unknown dtype {ch.dtype!r}")
    return arr


def _set_attrs(dset, ch: FeatureChannel):
    dset.attrs["dtype"] = ch.dtype
    # int8 (not Python bool) so MATLAB reads a number, not an enum cell.
    dset.attrs["applicable"] = np.int8(1 if ch.applicable else 0)
    dset.attrs["model"] = ch.model
    dset.attrs["version"] = ch.version
    dset.attrs["resample"] = ch.resample
    dset.attrs["tier"] = ch.tier
    if ch.units:
        dset.attrs["units"] = ch.units
    if ch.notes:
        dset.attrs["notes"] = ch.notes
    # native_rate_hz may be numeric or a string code
    nr = ch.native_rate_hz
    dset.attrs["native_rate_hz"] = float(nr) if isinstance(nr, (int, float)) else str(nr)
    if ch.components is not None:
        dset.attrs["components"] = np.array(list(ch.components), dtype=STR)
        dset.attrs["dim"] = int(len(ch.components))
    elif ch.dtype == "vector":
        dset.attrs["dim"] = int(np.asarray(ch.value).shape[1])
    if ch.categories is not None:
        dset.attrs["categories"] = np.array(list(ch.categories), dtype=STR)


def write_annotation(stim: Stimulus, grid: TimeGrid, channels: list[FeatureChannel],
                     out_dir: str, pipeline_version: str = "0.2.0",
                     generated_utc: str = "") -> dict:
    """Write <id>.h5 + <id>.manifest.json under out_dir/<id>/. Returns paths."""
    out = Path(out_dir) / stim.id
    out.mkdir(parents=True, exist_ok=True)
    h5_path = out / f"{stim.id}.h5"
    json_path = out / f"{stim.id}.manifest.json"

    classes = sorted({ch.path.split("/")[0] for ch in channels})
    models = {ch.model: {"version": ch.version, "tier": ch.tier}
              for ch in channels if ch.model}

    with h5py.File(h5_path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["stimulus_id"] = stim.id
        f.attrs["pipeline_version"] = pipeline_version
        f.attrs["generated_utc"] = generated_utc

        tg = f.create_group("time")
        tg.attrs["rate_hz"] = float(grid.rate_hz)
        tg.attrs["t_start_sec"] = float(grid.t_start_sec)
        tg.attrs["n_samples"] = int(grid.n_samples)
        tg.attrs["bin_reference"] = grid.bin_reference
        tg.create_dataset("time_sec", data=grid.time_sec.astype(np.float64))

        sg = f.create_group("stimulus")
        for k in ("id", "title", "modality", "source", "sha256", "media_file"):
            sg.attrs[k] = getattr(stim, k)
        sg.attrs["duration_sec"] = float(stim.duration_sec)

        for ch in channels:
            dset = f.create_dataset(
                _feat_path(ch), data=_encode(ch),
                compression="gzip" if ch.dtype in ("vector",) else None,
            )
            _set_attrs(dset, ch)
            if ch.dtype == "event" and ch.onsets is not None:
                grp = f.require_group(f"{_feat_path(ch)}__onsets")
                grp.create_dataset("time_sec", data=np.asarray(ch.onsets, np.float64))

        human = f.create_group("human")          # reserved, empty
        for cls in classes:
            human.create_group(cls)
        human.create_group("_free")

        pg = f.create_group("provenance")
        pg.attrs["pipeline_version"] = pipeline_version
        for m, meta in models.items():
            mg = pg.require_group(m)
            mg.attrs["version"] = meta["version"]
            mg.attrs["tier"] = meta["tier"]

    manifest = _build_manifest(stim, grid, channels, pipeline_version, generated_utc, models)
    json_path.write_text(json.dumps(manifest, indent=2))
    return {"h5": str(h5_path), "manifest": str(json_path)}


def _build_manifest(stim, grid, channels, pipeline_version, generated_utc, models) -> dict:
    features: dict = {}
    for ch in channels:
        parts = ch.path.split("/")
        node = features
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        leaf = {
            "dtype": ch.dtype,
            "applicable": bool(ch.applicable),
            "units": ch.units,
            "model": ch.model,
            "version": ch.version,
            "native_rate_hz": ch.native_rate_hz,
            "resample": ch.resample,
            "tier": ch.tier,
            "data_ref": _feat_path(ch),
        }
        if ch.notes:
            leaf["notes"] = ch.notes
        if ch.components is not None:
            leaf["dim"] = len(ch.components)
        if ch.categories is not None:
            leaf["categories"] = list(ch.categories)
        node[parts[-1]] = leaf
    return {
        "schema_version": SCHEMA_VERSION,
        "stimulus": {
            "id": stim.id, "title": stim.title, "modality": stim.modality,
            "duration_sec": stim.duration_sec, "source": stim.source, "sha256": stim.sha256,
            "media_file": stim.media_file,
        },
        "timeline": {
            "rate_hz": grid.rate_hz, "t_start_sec": grid.t_start_sec,
            "n_samples": grid.n_samples, "bin_reference": grid.bin_reference,
        },
        "features": features,
        "human_annotations": {},
        "provenance": {
            "pipeline_version": pipeline_version, "generated_utc": generated_utc,
            "models": models,
        },
    }
