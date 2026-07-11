#!/usr/bin/env python3
"""Generate schema/channel_template.json from a real full-stack run.

Runs every extractor (Qwen limited to 1 window for speed) on a short audiovisual
clip, then reads the produced HDF5 and records each channel's path/dtype/dim/
components/categories. The pipeline uses this template to fill NaN skeletons so
every stimulus yields an identical channel set (constant-shape contract).

    PYTHONPATH=src .venv/bin/python tools/build_channel_template.py <short_clip.mp4>
"""
import json
import sys
import tempfile
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nfe.extractors import (audio_hl_mps_extractors, core_cpu_extractors,  # noqa: E402
                            event_extractors, vision_mps_extractors)
from nfe.extractors.qwen_reasoning import QwenReasoning  # noqa: E402
from nfe.pipeline import annotate  # noqa: E402


def specs_from_h5(h5_path: str) -> list[dict]:
    out = []
    with h5py.File(h5_path, "r") as f:
        feats = f["features"]

        def walk(g):
            for k in g:
                it = g[k]
                if isinstance(it, h5py.Group):
                    walk(it)
                    continue
                a = it.attrs
                path = it.name.split("/features/")[1]
                comp = a.get("components")
                cats = a.get("categories")
                out.append({
                    "path": path,
                    "dtype": a.get("dtype", "scalar"),
                    "units": a.get("units", ""),
                    "model": a.get("model", ""),
                    "dim": int(it.shape[1]) if it.ndim == 2 else None,
                    "components": [c.decode() if isinstance(c, bytes) else str(c) for c in comp]
                    if comp is not None else None,
                    "categories": [c.decode() if isinstance(c, bytes) else str(c) for c in cats]
                    if cats is not None else None,
                })
        walk(feats)
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    clip = sys.argv[1]
    exts = (core_cpu_extractors() + vision_mps_extractors() + audio_hl_mps_extractors()
            + [QwenReasoning(max_windows=1)] + event_extractors())
    tmp = tempfile.mkdtemp(prefix="nfe_template_")
    summary = annotate(clip, out_dir=tmp, extractors=exts, source="template-build")
    specs = specs_from_h5(summary["h5"])
    # drop the *__onsets helper datasets (not feature channels)
    specs = [s for s in specs if not s["path"].endswith("__onsets/time_sec")]
    template = {"n_channels": len(specs), "channels": sorted(specs, key=lambda s: s["path"])}
    out = ROOT / "schema" / "channel_template.json"
    out.write_text(json.dumps(template, indent=1))
    print(f"wrote {out} with {len(specs)} channels")


if __name__ == "__main__":
    main()
