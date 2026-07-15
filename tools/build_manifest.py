#!/usr/bin/env python3
"""Catalog all media under data/movies into a stimulus manifest (Phase 3).

Scans for video/audio files, probes each, and writes data/manifest.csv (+ .json)
with id, path, source, modality, duration, fps, and rights. The batch runner
(nfe.batch) consumes this to annotate the corpus.

    PYTHONPATH=src .venv/bin/python tools/build_manifest.py
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nfe.ingest import infer_modality, probe  # noqa: E402

MEDIA_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
             ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}
from nfe.ingest import TEXT_EXT, make_stimulus  # noqa: E402
RIGHTS = {"open": "CC-BY (Blender open movies)", "spacetop": "lab-internal",
          "stories": "lab-internal", "narratives": "OpenNeuro ds002345 (CC0)",
          "hcp": "HCP (credentialed; not redistributable)", "imensa": "lab-internal",
          "kungfury": "third-party (research use)"}

# Path fragments to skip. For HCP we annotate only the Pre_20140821_version clips
# (more fMRI subjects); the Post_20140821_version clips are near-duplicates.
SKIP_DIRS = {"Post_20140821_version"}


def main():
    roots = [ROOT / "data" / "movies", ROOT / "data" / "stories"]
    rows = []
    for base in roots:
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            ext = f.suffix.lower()
            if not f.is_file() or ext not in (MEDIA_EXT | TEXT_EXT):
                continue
            if f.stem.lower() in {"readme", "sources", "contents", "license"}:
                continue                                   # skip stray doc files
            if SKIP_DIRS & set(f.parts):
                continue                                   # e.g. HCP Post_ version (dedup)
            rel = f.relative_to(ROOT)
            parts = f.relative_to(base).parts
            source = parts[0] if len(parts) > 1 else base.name
            try:
                if ext in TEXT_EXT:                      # pure-text story
                    stim = make_stimulus(str(f), source=source)
                    modality, dur = stim.modality, stim.duration_sec
                    fps = width = height = 0
                    has_video = has_audio = False
                else:
                    info = probe(str(f))
                    modality = infer_modality(info)
                    dur, fps, width, height = info.duration_sec, info.fps, info.width, info.height
                    has_video, has_audio = info.has_video, info.has_audio
            except Exception as e:
                print(f"  WARN probe failed: {rel} ({e})")
                continue
            rows.append({
                "id": f.stem, "path": str(rel), "source": source, "modality": modality,
                "duration_sec": round(dur, 2), "fps": round(fps, 2),
                "width": width, "height": height,
                "has_video": has_video, "has_audio": has_audio,
                "rights": RIGHTS.get(source, "unknown"),
            })

    if not rows:
        sys.exit("no media/story files found under data/movies or data/stories")

    # duplicate ids are fatal: output folders and the search index key on id
    seen = {}
    for r in rows:
        seen.setdefault(r["id"], []).append(r["path"])
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    if dups:
        sys.exit(f"duplicate stimulus ids (rename files so stems are unique): {dups}")

    out_csv = ROOT / "data" / "manifest.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (ROOT / "data" / "manifest.json").write_text(json.dumps(rows, indent=1))
    tot = sum(r["duration_sec"] for r in rows)
    print(f"wrote {out_csv} : {len(rows)} stimuli, {tot/60:.1f} min total")
    from collections import Counter
    print("  by source:", dict(Counter(r["source"] for r in rows)))
    print("  by modality:", dict(Counter(r["modality"] for r in rows)))


if __name__ == "__main__":
    main()
