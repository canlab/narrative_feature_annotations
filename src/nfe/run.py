"""CLI: annotate one media file.

    PYTHONPATH=src .venv/bin/python -m nfe.run <media> [--out DIR] [--rate HZ] [--fps N]
"""

from __future__ import annotations

import argparse
import json

from .pipeline import annotate


def main(argv=None):
    ap = argparse.ArgumentParser(description="Annotate a movie/story (Phase 2 core CPU pipeline).")
    ap.add_argument("media", help="path to a video/audio file")
    ap.add_argument("--out", default="annotations/output", help="output directory")
    ap.add_argument("--rate", type=float, default=1.0, help="common-grid rate (Hz)")
    ap.add_argument("--fps", type=float, default=4.0, help="frame analysis rate (Hz)")
    ap.add_argument("--max-side", type=int, default=320, help="downscale frames to this max side")
    ap.add_argument("--asr-model", default="small",
                    help="faster-whisper model (small | distil-large-v3 | large-v3 | ...)")
    ap.add_argument("--vision", action="store_true",
                    help="also run MPS/GPU visual passes (SigLIP2, DINOv2, RAFT, depth, action)")
    ap.add_argument("--audio-hl", action="store_true",
                    help="also run MPS/GPU audio high-level passes (AST, CLAP, vocal affect)")
    ap.add_argument("--reason", action="store_true",
                    help="also run the Qwen2.5-VL reasoning pass (social/situation/affect) — slow")
    ap.add_argument("--events", action="store_true",
                    help="also run GSBS event segmentation (post-pass over the feature matrix)")
    ap.add_argument("--template", default=None,
                    help="channel template (e.g. schema/channel_template.json) for constant-shape "
                         "output: missing channels are filled as applicable=false NaN skeletons")
    ap.add_argument("--source", default="", help="provenance source label")
    args = ap.parse_args(argv)

    summary = annotate(args.media, out_dir=args.out, rate_hz=args.rate,
                       analysis_fps=args.fps, max_side=args.max_side, asr_model=args.asr_model,
                       vision=args.vision, audio_hl=args.audio_hl, reason=args.reason,
                       events=args.events, template_path=args.template, source=args.source)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
