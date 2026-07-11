#!/usr/bin/env python3
"""Build the segment search index for the web interface (Phase 4).

Splits every annotated stimulus into fixed-length windows, summarizes each window
by the mean of every numeric (scalar/bool/event) channel, and writes a compact JSON
(analysis/web/segments.json) that the static web app ranks and filters client-side.

    PYTHONPATH=src .venv/bin/python tools/build_search_index.py [--seglen 5]
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NUMERIC = {"scalar", "bool", "event"}
# scalar-typed channels whose values are arbitrary indices, not magnitudes — never rankable
EXCLUDE = {"situation/event_id", "visual/dynamic_motion/shot_index"}
# Expand interpretable vector channels (named semantic components, not opaque embeddings,
# and not too high-dimensional) into per-component searchable columns "path[component]".
# This surfaces e.g. EmoNet's 20 emotion categories and the 8 facial expressions in search.
EMBED_RE = re.compile(r"^d\d+$")
MAX_VECTOR_DIM = 30


def load_manifest():
    m = {}
    p = ROOT / "data" / "manifest.csv"
    if p.exists():
        for r in csv.DictReader(open(p, newline="")):
            m[r["id"]] = r
    return m


def scalar_channels(f):
    """Return dict path -> array for numeric channels in one annotation file.

    Skips index-like channels (EXCLUDE) and channels marked applicable=false —
    skeleton event/bool fills (0 / -1) are "not measured", not real values.
    """
    out = {}

    def walk(g):
        for k in g:
            it = g[k]
            if isinstance(it, h5py.Group):
                walk(it)
                continue
            path = it.name.split("/features/")[1]
            if path in EXCLUDE or not int(it.attrs.get("applicable", 1)):
                continue
            dtype = it.attrs.get("dtype")
            if dtype in NUMERIC and it.ndim == 1:
                out[path] = it[:].astype(float)
            elif dtype == "vector" and it.ndim == 2:
                comps = [c.decode() if isinstance(c, bytes) else str(c)
                         for c in it.attrs.get("components", [])]
                if not comps or len(comps) > MAX_VECTOR_DIM or any(EMBED_RE.match(c) for c in comps):
                    continue                       # skip opaque embeddings / very wide vectors
                data = it[:].astype(float)
                for j, comp in enumerate(comps):
                    out[f"{path}[{comp}]"] = data[:, j]
    walk(f["features"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="annotations/corpus")
    ap.add_argument("--seglen", type=int, default=5, help="window length (s, = samples at 1 Hz)")
    ap.add_argument("--out", default="analysis/web/segments.json")
    args = ap.parse_args()

    man = load_manifest()
    files = sorted((ROOT / args.corpus).glob("*/*.h5"))
    if not files:
        sys.exit(f"no annotation files under {args.corpus}")

    # pass 1: load every file's channels; channel set = union across files (modalities differ)
    loaded = []
    for hf in files:
        with h5py.File(hf, "r") as f:
            sid = f.attrs["stimulus_id"]
            modality = f["stimulus"].attrs.get("modality", "")
            rate = float(f["time"].attrs["rate_hz"])
            n = int(f["time"].attrs["n_samples"])
            chans = scalar_channels(f)
        loaded.append((sid, modality, rate, n, chans))
    chan_names = sorted({name for *_, chans in loaded for name in chans})

    segments = []
    for sid, modality, rate, n, chans in loaded:
        meta = man.get(sid, {})
        media = "/" + meta["path"] if meta.get("path") else ""
        step = max(1, args.seglen)
        for w in range(0, n, step):
            idx = slice(w, min(w + step, n))
            v = []
            for name in chan_names:
                col = chans.get(name)
                seg = col[idx] if col is not None else np.array([np.nan])
                m = np.nanmean(seg) if np.any(~np.isnan(seg)) else np.nan
                v.append(None if np.isnan(m) else round(float(m), 4))
            segments.append({"stim": sid, "src": meta.get("source", ""),
                             "modality": modality, "media": media,
                             "t0": round(w / rate, 2), "t1": round(min(w + step, n) / rate, 2),
                             "v": v})

    # per-channel mean/std over all segments (for client-side z-scoring); keep only
    # channels with real variance (drops skeleton/NaN + constant channels).
    arr = np.array([[np.nan if x is None else x for x in s["v"]] for s in segments], float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    # filter on the ROUNDED std (what actually ships) so the client never divides by 0
    keep = np.isfinite(std) & (np.round(std, 4) > 0)
    ki = np.where(keep)[0]
    channels = [{"name": chan_names[i], "class": chan_names[i].split("/")[0],
                 "mean": round(float(mean[i]), 4), "std": round(float(std[i]), 4)} for i in ki]
    for s in segments:
        s["v"] = [s["v"][i] for i in ki]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seglen": args.seglen, "channels": channels,
                               "segments": segments}, separators=(",", ":")))
    print(f"wrote {out}: {len(segments)} segments, {len(channels)} searchable channels "
          f"from {len(files)} stimuli")


if __name__ == "__main__":
    main()
