#!/usr/bin/env python3
"""Run the Qwen2.5-VL reasoning pass on audiovisual corpus stimuli and patch its
social/situation/affect channels into the existing annotation .h5 files.

The vision/audio/language channels are already correct; this only replaces the 8
NaN-skeleton reasoning channels (scene_description, setting, indoor_outdoor,
interaction_type, dominance, vlm_emotion, vlm_valence, vlm_arousal) with real values
and sets applicable=true. Resumable per stimulus (skips ones already patched).

    PYTHONPATH=src .venv/bin/python tools/patch_reason.py [--ids a,b] [--limit N]
                                                          [--window-sec 5] [--out annotations/corpus]
"""
import argparse
import csv
import shutil
import sys
import tempfile
import time
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nfe.base import TimeGrid                       # noqa: E402
from nfe.emit import _encode, _set_attrs             # noqa: E402
from nfe.extractors.qwen_reasoning import QwenReasoning  # noqa: E402
from nfe.ingest import Ingest, make_stimulus         # noqa: E402

PROBE = "features/affect/depicted/vlm_valence"       # channel used to detect "already patched"


def already_patched(h5: Path) -> bool:
    with h5py.File(h5, "r") as f:
        d = f.get(PROBE)
        return d is not None and int(d.attrs.get("applicable", 0)) == 1


def patch_h5(h5: Path, channels) -> None:
    with h5py.File(h5, "r+") as f:
        for ch in channels:
            path = "/features/" + ch.path
            if path in f:
                del f[path]                          # replace the skeleton dataset
            dset = f.create_dataset(path, data=_encode(ch))
            _set_attrs(dset, ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="annotations/corpus")
    ap.add_argument("--ids", help="comma-separated id filter (default: all audiovisual)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--window-sec", type=float, default=5.0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(ROOT / "data" / "manifest.csv")))
    av = [r for r in rows if r["modality"] == "audiovisual"]
    if args.ids:
        want = set(args.ids.split(","))
        av = [r for r in av if r["id"] in want]
    if args.limit:
        av = av[:args.limit]

    ext = QwenReasoning(window_sec=args.window_sec)   # model loads once, reused across stimuli
    t_start = time.time()
    done = failed = 0
    for i, r in enumerate(av, 1):
        sid = r["id"]
        h5 = ROOT / args.out / sid / f"{sid}.h5"
        if not h5.exists():
            print(f"[reason] ({i}/{len(av)}) skip {sid} (no annotation)", flush=True)
            continue
        if already_patched(h5):
            print(f"[reason] ({i}/{len(av)}) skip {sid} (already patched)", flush=True)
            done += 1
            continue
        print(f"[reason] ({i}/{len(av)}) {sid} ({float(r['duration_sec'])/60:.1f} min) ...", flush=True)
        wd = tempfile.mkdtemp(prefix=f"nfe_reason_{sid}_")
        t0 = time.time()
        try:
            stim = make_stimulus(str(ROOT / r["path"]), source=r["source"])
            grid = TimeGrid.from_duration(stim.duration_sec, rate_hz=1.0)
            ing = Ingest(stim, wd, analysis_fps=2.0, max_side=320)
            chs = ext.extract(stim, grid, ing)
            patch_h5(h5, chs)
            done += 1
            print(f"[reason]   patched {len(chs)} channels ({(time.time()-t0)/60:.1f} min)", flush=True)
        except Exception as e:
            failed += 1
            print(f"[reason]   ERROR {sid}: {e}", flush=True)
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    print(f"[reason] done: {done} ok, {failed} failed, {len(av)} total "
          f"({(time.time()-t_start)/3600:.1f} h)", flush=True)


if __name__ == "__main__":
    main()
