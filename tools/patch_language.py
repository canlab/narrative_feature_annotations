#!/usr/bin/env python3
"""Add the high-level language channels (Qwen3 + Llama embeddings + derived surprisal/
coherence) to the existing corpus annotations, and (re)populate the psycholinguistic
word norms now that data/lexicons/ is installed — WITHOUT recomputing the other channels.

Per stimulus: run ASR (transcript + word timings; text-only builds its transcript from
the .txt), then LanguageHL (Qwen3 sentence embedding, Llama-3.1-8B AR hidden states,
semantic coherence/drift/novelty/surprise, LLM narrative expectedness) and LanguageLexical
(freq/length + valence/arousal/dominance/concreteness/AoA), and patch their channels into
<id>.h5. Non-speech stimuli get all-NaN language channels (constant shape preserved).
Resumable (skips already-patched files).

    PYTHONPATH=src .venv/bin/python tools/patch_language.py [--ids a,b] [--limit N]
        [--out annotations/corpus] [--no-narrative]
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

from nfe.base import TimeGrid                             # noqa: E402
from nfe.emit import _encode, _set_attrs                  # noqa: E402
from nfe.extractors.asr import ASR                        # noqa: E402
from nfe.extractors.language_hl import LanguageHL         # noqa: E402
from nfe.extractors.language_lexical import LanguageLexical  # noqa: E402
from nfe.ingest import Ingest, make_stimulus              # noqa: E402
from nfe.schema_registry import _skeleton, load_template  # noqa: E402

PROBE = "features/language/embedding/qwen3"
NEW_PATHS = [
    "language/embedding/qwen3", "language/embedding/llama_ar",
    "language/hl/semantic_coherence", "language/hl/semantic_drift",
    "language/hl/semantic_novelty", "language/hl/semantic_surprise",
    "language/hl/narrative_expectedness", "language/hl/narrative_surprise",
    "language/lexical/valence", "language/lexical/arousal", "language/lexical/dominance",
    "language/lexical/concreteness", "language/lexical/aoa",
    "language/lexical/freq_zipf", "language/lexical/word_length"]


def already_patched(h5: Path) -> bool:
    # require ALL new channels present (a partial/failed patch must be redone)
    with h5py.File(h5, "r") as f:
        return all(("features/" + p) in f for p in NEW_PATHS)


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
    ap.add_argument("--no-narrative", action="store_true", help="skip the LLM expectedness pass")
    ap.add_argument("--force", action="store_true", help="re-patch even if already complete")
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
    lang_hl = LanguageHL(narrative=not args.no_narrative)
    lexical = LanguageLexical()
    t_start = time.time()
    done = failed = 0
    for i, r in enumerate(rows, 1):
        sid = r["id"]
        h5 = ROOT / args.out / sid / f"{sid}.h5"
        if not h5.exists():
            print(f"[lang] ({i}/{len(rows)}) skip {sid} (no annotation yet)", flush=True)
            continue
        if already_patched(h5) and not args.force:
            print(f"[lang] ({i}/{len(rows)}) skip {sid} (already patched)", flush=True)
            done += 1
            continue
        mod = r["modality"]
        print(f"[lang] ({i}/{len(rows)}) {sid} [{mod}] ({float(r['duration_sec'])/60:.1f} min) ...",
              flush=True)
        wd = tempfile.mkdtemp(prefix=f"nfe_lang_{sid}_")
        t0 = time.time()
        try:
            stim = make_stimulus(str(ROOT / r["path"]), source=r["source"])
            grid = TimeGrid.from_duration(stim.duration_sec, rate_hz=1.0)
            ing = Ingest(stim, wd, analysis_fps=2.0, max_side=320)
            if stim.has_audio:
                asr.extract(stim, grid, ing)                 # attaches transcript (word timings)
            chs = lang_hl.extract(stim, grid, ing)           # get_transcript() handles text-only
            chs += lexical.extract(stim, grid, ing)
            produced = {c.path for c in chs}
            chs += [_skeleton(tmpl_by_path[p], grid) for p in NEW_PATHS
                    if p not in produced and p in tmpl_by_path]
            patch_h5(h5, chs)
            done += 1
            print(f"[lang]   patched {len(chs)} channels ({(time.time()-t0)/60:.1f} min)", flush=True)
        except Exception as e:
            failed += 1
            import traceback
            print(f"[lang]   ERROR {sid}: {e}\n{traceback.format_exc()}", flush=True)
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    print(f"[lang] done: {done} ok, {failed} failed, {len(rows)} total "
          f"({(time.time()-t_start)/3600:.1f} h)", flush=True)


if __name__ == "__main__":
    main()
