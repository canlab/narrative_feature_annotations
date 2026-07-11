#!/usr/bin/env python3
"""Add the new affect channels (EmoNet, HSEmotion face affect, CardiffNLP sentiment) to
the existing corpus annotations WITHOUT recomputing the correct vision/audio/language/
reasoning channels already stored.

Per stimulus: run ASR (to get a transcript for text_sentiment), then whichever of the
three new extractors apply, and patch their channels into <id>.h5. The 6 vision channels
(EmoNet + face) get applicable=false NaN skeletons for audio-only / text-only stimuli, so
every stimulus keeps the constant-shape contract. Resumable (skips already-patched files).

    PYTHONPATH=src .venv/bin/python tools/patch_new_affect.py [--ids a,b] [--limit N]
                                                              [--out annotations/corpus]
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

from nfe.base import TimeGrid                          # noqa: E402
from nfe.emit import _encode, _set_attrs                # noqa: E402
from nfe.extractors.asr import ASR                      # noqa: E402
from nfe.extractors.face_emotion import FaceEmotion     # noqa: E402
from nfe.extractors.text_sentiment import TextSentiment  # noqa: E402
from nfe.extractors.visual_emonet import VisualEmoNet    # noqa: E402
from nfe.ingest import Ingest, make_stimulus            # noqa: E402
from nfe.schema_registry import _skeleton, load_template  # noqa: E402

PROBE = "features/affect/depicted/text_sentiment"        # present in every patched stimulus
NEW_PATHS = [
    "affect/depicted/emonet", "affect/depicted/emonet_top",
    "affect/depicted/face_emotion", "affect/depicted/face_emotion_top",
    "affect/depicted/face_valence", "affect/depicted/face_arousal",
    "affect/depicted/text_sentiment", "affect/depicted/text_sentiment_polarity",
    "affect/depicted/text_sentiment_top"]


def already_patched(h5: Path) -> bool:
    with h5py.File(h5, "r") as f:
        return PROBE in f


def patch_h5(h5: Path, channels) -> None:
    with h5py.File(h5, "r+") as f:
        for ch in channels:
            path = "/features/" + ch.path
            if path in f:
                del f[path]
            comp = "gzip" if ch.dtype == "vector" else None
            dset = f.create_dataset(path, data=_encode(ch), compression=comp)
            _set_attrs(dset, ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="annotations/corpus")
    ap.add_argument("--ids", help="comma-separated id filter (default: all stimuli)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--template", default=str(ROOT / "schema" / "channel_template.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(ROOT / "data" / "manifest.csv")))
    if args.ids:
        want = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in want]
    if args.limit:
        rows = rows[:args.limit]

    template = load_template(args.template)
    tmpl_by_path = {s["path"]: s for s in template["channels"]}

    asr = ASR(model_size="small")
    emonet = VisualEmoNet()
    face = FaceEmotion()
    sentiment = TextSentiment()
    t_start = time.time()
    done = failed = 0
    for i, r in enumerate(rows, 1):
        sid = r["id"]
        h5 = ROOT / args.out / sid / f"{sid}.h5"
        if not h5.exists():
            print(f"[affect] ({i}/{len(rows)}) skip {sid} (no annotation)", flush=True)
            continue
        if already_patched(h5):
            print(f"[affect] ({i}/{len(rows)}) skip {sid} (already patched)", flush=True)
            done += 1
            continue
        mod = r["modality"]
        print(f"[affect] ({i}/{len(rows)}) {sid} [{mod}] "
              f"({float(r['duration_sec'])/60:.1f} min) ...", flush=True)
        wd = tempfile.mkdtemp(prefix=f"nfe_affect_{sid}_")
        t0 = time.time()
        try:
            stim = make_stimulus(str(ROOT / r["path"]), source=r["source"])
            grid = TimeGrid.from_duration(stim.duration_sec, rate_hz=1.0)
            ing = Ingest(stim, wd, analysis_fps=2.0, max_side=320)
            if stim.has_audio:
                asr.extract(stim, grid, ing)                 # attaches transcript for sentiment
            chs = []
            if stim.has_video:
                chs += emonet.extract(stim, grid, ing)
                chs += face.extract(stim, grid, ing)
            chs += sentiment.extract(stim, grid, ing)        # applies to all modalities
            # NaN skeletons for whichever of the 9 new channels this modality didn't produce
            produced = {c.path for c in chs}
            chs += [_skeleton(tmpl_by_path[p], grid) for p in NEW_PATHS if p not in produced]
            patch_h5(h5, chs)
            done += 1
            print(f"[affect]   patched {len(chs)} channels ({(time.time()-t0)/60:.1f} min)", flush=True)
        except Exception as e:
            failed += 1
            import traceback
            print(f"[affect]   ERROR {sid}: {e}\n{traceback.format_exc()}", flush=True)
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    print(f"[affect] done: {done} ok, {failed} failed, {len(rows)} total "
          f"({(time.time()-t_start)/3600:.1f} h)", flush=True)


if __name__ == "__main__":
    main()
