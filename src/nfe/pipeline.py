"""Pipeline orchestration: ingest -> run applicable extractors -> emit annotation."""

from __future__ import annotations

import shutil
import tempfile
import time
import traceback
from datetime import datetime, timezone

from .base import TimeGrid
from .emit import write_annotation
from .extractors import default_extractors
from .ingest import Ingest, make_stimulus
from .schema_registry import apply_template, load_template

PIPELINE_VERSION = "0.2.0"


def annotate(media_path: str, out_dir: str = "annotations/output", *,
             rate_hz: float = 1.0, analysis_fps: float = 4.0, max_side: int = 320,
             audio_sr: int = 22050, extractors=None, asr_model: str = "small",
             vision: bool = False, audio_hl: bool = False, reason: bool = False,
             events: bool = False, template_path: str | None = None, source: str = "",
             workdir: str | None = None, verbose: bool = True) -> dict:
    """Annotate one stimulus and write HDF5 + JSON sidecar. Returns a summary dict."""
    stim = make_stimulus(media_path, source=source)
    grid = TimeGrid.from_duration(stim.duration_sec, rate_hz=rate_hz)
    if grid.n_samples < 1:
        raise ValueError(f"{stim.id}: zero-length grid (duration {stim.duration_sec:.2f}s) — "
                         "unreadable/empty media or empty text file.")
    owns_wd = workdir is None
    wd = workdir or tempfile.mkdtemp(prefix=f"nfe_{stim.id}_")
    try:
        ingest = Ingest(stim, wd, analysis_fps=analysis_fps, max_side=max_side, audio_sr=audio_sr)
        exts = (extractors if extractors is not None
                else default_extractors(asr_model, vision=vision, audio_hl=audio_hl,
                                        reason=reason, events=events))
        main_exts = [e for e in exts if not hasattr(e, "post_extract")]
        post_exts = [e for e in exts if hasattr(e, "post_extract")]

        if verbose:
            print(f"[nfe] {stim.id}: modality={stim.modality} dur={stim.duration_sec:.1f}s "
                  f"grid={grid.n_samples}@{rate_hz}Hz")
        channels, ran, failed = [], [], []
        for ext in main_exts:
            if not ext.applies_to(stim):
                if verbose:
                    print(f"[nfe]   skip {ext.name} (n/a for {stim.modality})")
                continue
            t0 = time.time()
            try:
                chs = ext.extract(stim, grid, ingest)
                channels.extend(chs)
                ran.append(ext.name)
                if verbose:
                    print(f"[nfe]   {ext.name}: {len(chs)} channels ({time.time()-t0:.1f}s)")
            except Exception:
                failed.append(ext.name)
                print(f"[nfe]   ERROR in {ext.name}:\n{traceback.format_exc()}")

        # post-extractors consume the channels produced above (e.g. event segmentation)
        for ext in post_exts:
            if not ext.applies_to(stim):
                continue
            t0 = time.time()
            try:
                chs = ext.post_extract(stim, grid, channels)
                channels.extend(chs)
                ran.append(ext.name)
                if verbose:
                    print(f"[nfe]   {ext.name} (post): {len(chs)} channels ({time.time()-t0:.1f}s)")
            except Exception:
                failed.append(ext.name)
                print(f"[nfe]   ERROR in {ext.name} (post):\n{traceback.format_exc()}")

        n_measured = len(channels)
        n_skeleton = 0
        if template_path:
            tmpl = load_template(template_path)
            if tmpl:
                channels = apply_template(channels, grid, tmpl)
                n_skeleton = len(channels) - n_measured
                if verbose:
                    print(f"[nfe]   +{n_skeleton} NaN-skeleton channels (constant-shape template)")

        paths = write_annotation(
            stim, grid, channels, out_dir, pipeline_version=PIPELINE_VERSION,
            generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if verbose:
            msg = f"[nfe] wrote {len(channels)} channels -> {paths['h5']}"
            print(msg + (f"  ({len(failed)} pass(es) FAILED: {','.join(failed)})" if failed else ""))
        return {"stimulus": stim.id, "modality": stim.modality, "n_channels": len(channels),
                "n_measured": n_measured, "n_skeleton": n_skeleton, "extractors": ran,
                "n_failed": len(failed), "failed": failed, **paths}
    finally:
        if owns_wd:
            shutil.rmtree(wd, ignore_errors=True)   # remove temp WAV + transcript cache
