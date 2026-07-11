"""Corpus batch annotation (Phase 3).

Reads the stimulus manifest and annotates each entry with the chosen passes and a
channel template (constant shape). Resumable (skips stimuli whose .h5 already
exists), crash-safe (rewrites the corpus index after every stimulus), and isolated
(one stimulus failing does not stop the batch).

    PYTHONPATH=src .venv/bin/python -m nfe.batch --manifest data/manifest.csv \\
        --out annotations/output --template schema/channel_template.json \\
        --vision --audio-hl --events --source spacetop --max-dur 70
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from .pipeline import annotate

INDEX_FIELDS = ["id", "source", "modality", "duration_sec", "status",
                "n_channels", "n_measured", "n_skeleton", "n_failed", "seconds", "error"]


def read_manifest(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def run_corpus(manifest, out_dir="annotations/output", template_path=None, *,
               vision=False, audio_hl=False, reason=False, events=False,
               asr_model="small", analysis_fps=2.0, sources=None, ids=None,
               max_dur=None, limit=None, skip_existing=True, verbose=True):
    rows = read_manifest(manifest)
    if sources:
        rows = [r for r in rows if r["source"] in sources]
    if ids:
        rows = [r for r in rows if r["id"] in ids]
    if max_dur:
        rows = [r for r in rows if float(r["duration_sec"]) <= max_dur]
    if limit:
        rows = rows[:limit]

    out = Path(out_dir)
    index_path = out / "corpus_index.csv"
    out.mkdir(parents=True, exist_ok=True)
    # merge with any existing index so a filtered run doesn't drop other stimuli's rows
    index = {row["id"]: row for row in read_manifest(index_path)} if index_path.exists() else {}
    print(f"[batch] {len(rows)} stimuli "
          f"({sum(float(r['duration_sec']) for r in rows)/60:.1f} min) -> {out_dir}")

    for i, r in enumerate(rows, 1):
        sid = r["id"]
        h5 = out / sid / f"{sid}.h5"
        base = {"id": sid, "source": r["source"], "modality": r["modality"],
                "duration_sec": r["duration_sec"], "error": ""}
        if skip_existing and h5.exists():
            print(f"[batch] ({i}/{len(rows)}) skip {sid} (exists)")
            if sid not in index:            # keep a prior richer row if we have one
                index[sid] = {**base, "status": "skipped", "n_channels": "", "n_measured": "",
                              "n_skeleton": "", "n_failed": "", "seconds": ""}
            _write_index(index_path, index)
            continue
        print(f"[batch] ({i}/{len(rows)}) {sid} ({r['duration_sec']}s) ...")
        t0 = time.time()
        try:
            s = annotate(str(Path(r["path"])) if Path(r["path"]).is_absolute()
                         else str((Path(manifest).resolve().parent.parent / r["path"])),
                         out_dir=out_dir, asr_model=asr_model, analysis_fps=analysis_fps,
                         vision=vision, audio_hl=audio_hl, reason=reason, events=events,
                         template_path=template_path, source=r["source"], verbose=verbose)
            nf = s.get("n_failed", 0)
            index[sid] = {**base, "status": "partial" if nf else "ok",
                          "n_channels": s["n_channels"], "n_measured": s["n_measured"],
                          "n_skeleton": s["n_skeleton"], "n_failed": nf,
                          "seconds": round(time.time() - t0, 1),
                          "error": ("failed: " + ",".join(s.get("failed", []))) if nf else ""}
        except Exception as e:
            print(f"[batch]   ERROR {sid}: {e}")
            index[sid] = {**base, "status": "error", "n_channels": "", "n_measured": "",
                          "n_skeleton": "", "n_failed": "", "seconds": round(time.time() - t0, 1),
                          "error": str(e)[:200]}
        _write_index(index_path, index)

    vals = list(index.values())
    ok = sum(1 for r in vals if r["status"] == "ok")
    print(f"[batch] done: {ok} ok / {len(vals)} total ({sum(r['status']=='partial' for r in vals)} "
          f"partial, {sum(r['status']=='error' for r in vals)} error). index -> {index_path}")
    return vals


def _write_index(path: Path, index):
    rows = list(index.values()) if isinstance(index, dict) else index
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:                       # backfill any missing new columns
            w.writerow({k: row.get(k, "") for k in INDEX_FIELDS})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Batch-annotate the corpus (Phase 3).")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="annotations/output")
    ap.add_argument("--template", default=None)
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--audio-hl", action="store_true")
    ap.add_argument("--reason", action="store_true")
    ap.add_argument("--events", action="store_true")
    ap.add_argument("--asr-model", default="small")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--source", action="append", help="filter by source (repeatable)")
    ap.add_argument("--ids", help="comma-separated id filter")
    ap.add_argument("--max-dur", type=float, help="skip stimuli longer than this (sec)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-skip", action="store_true", help="re-annotate even if output exists")
    a = ap.parse_args(argv)
    run_corpus(a.manifest, out_dir=a.out, template_path=a.template, vision=a.vision,
               audio_hl=a.audio_hl, reason=a.reason, events=a.events, asr_model=a.asr_model,
               analysis_fps=a.fps, sources=a.source, ids=a.ids.split(",") if a.ids else None,
               max_dur=a.max_dur, limit=a.limit, skip_existing=not a.no_skip, verbose=False)


if __name__ == "__main__":
    main()
