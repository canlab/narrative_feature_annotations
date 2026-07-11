"""Media ingest: probe a stimulus and provide decoded frames + extracted audio.

Uses PyAV (bundled ffmpeg libs) for video decoding and the imageio-ffmpeg static
ffmpeg binary for audio extraction, so no system ffmpeg install is required.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np

from .base import Modality, Segment, Stimulus, Transcript, Word

TEXT_EXT = {".txt", ".text", ".story"}   # NB: not .md (that's documentation here)
DEFAULT_READING_RATE = 3.0   # words/sec — narration pace; sets the pseudo-timeline for text


def build_text_transcript(text: str, rate: float = DEFAULT_READING_RATE) -> Transcript:
    """Turn a plain-text story into a Transcript on an estimated timeline.

    Words are placed on a uniform words/sec clock so text stories land on the same
    1 Hz grid as audio; sentences become segments. This is what lets the language /
    affect / event passes annotate pure text with no audio.
    """
    text = text.strip()
    dt = 1.0 / rate
    t = 0.0
    words_all, segments = [], []
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = s.strip()
        if not s:
            continue
        sw, seg_start = [], t
        for w in s.split():
            wobj = Word(text=w, start=t, end=t + dt, prob=1.0)
            sw.append(wobj)
            words_all.append(wobj)
            t += dt
        segments.append(Segment(text=s, start=seg_start, end=t, words=sw))
    return Transcript(language="en", text=text, segments=segments, words=words_all)


@dataclass
class MediaInfo:
    duration_sec: float
    has_video: bool
    has_audio: bool
    fps: float
    width: int
    height: int


def probe(path: str) -> MediaInfo:
    with av.open(path) as c:
        vstreams = c.streams.video
        astreams = c.streams.audio
        dur = float(c.duration / av.time_base) if c.duration else 0.0
        fps = w = h = 0
        if vstreams:
            vs = vstreams[0]
            fps = float(vs.average_rate) if vs.average_rate else 0.0
            w, h = vs.codec_context.width, vs.codec_context.height
            if not dur and vs.duration and vs.time_base:
                dur = float(vs.duration * vs.time_base)
        if not dur and astreams:
            a = astreams[0]
            if a.duration and a.time_base:
                dur = float(a.duration * a.time_base)
        return MediaInfo(dur, bool(vstreams), bool(astreams), fps, w, h)


def infer_modality(info: MediaInfo) -> Modality:
    if info.has_video and info.has_audio:
        return "audiovisual"
    if info.has_video:
        return "video-only"
    if info.has_audio:
        return "audio-only"
    return "text-only"


def sha256_short(path: str, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))   # hash first 1 MB (provenance id, not integrity)
    return h.hexdigest()[:16]


def make_stimulus(path: str, stim_id: str | None = None, source: str = "",
                  reading_rate: float = DEFAULT_READING_RATE) -> Stimulus:
    p = Path(path)
    if p.suffix.lower() in TEXT_EXT:                      # pure-text story
        n_words = len(p.read_text(encoding="utf-8", errors="ignore").split())
        return Stimulus(
            id=stim_id or p.stem, media_file=str(path), modality="text-only",
            duration_sec=n_words / reading_rate, title=p.stem, source=source,
            sha256=sha256_short(path))
    info = probe(path)
    return Stimulus(
        id=stim_id or p.stem,
        media_file=str(path),
        modality=infer_modality(info),
        duration_sec=info.duration_sec,
        title=p.stem,
        source=source,
        sha256=sha256_short(path),
    )


class Ingest:
    """Lazy access to a stimulus's frames and audio for extractors."""

    def __init__(self, stim: Stimulus, workdir: str,
                 analysis_fps: float = 4.0, max_side: int = 320, audio_sr: int = 22050,
                 reading_rate: float = DEFAULT_READING_RATE):
        self.stim = stim
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.analysis_fps = analysis_fps
        self.max_side = max_side
        self.audio_sr = audio_sr
        self.reading_rate = reading_rate
        self._audio_path: str | None = None
        self._transcript = None   # set by the ASR pass, read by language/social/situation
        self._frame_cache: list | None = None   # decoded frames, shared across visual passes

    # Cap the in-RAM frame cache so very long films fall back to per-pass streaming
    # instead of exhausting memory (~0.3 MB/frame at max_side=320).
    MAX_CACHE_FRAMES = 6000

    def set_transcript(self, tr) -> None:
        self._transcript = tr

    def get_transcript(self):
        # For a text-only story there is no ASR pass; build the transcript from the
        # text (on an estimated timeline) the first time a language pass asks for it.
        if self._transcript is None and self.stim.modality == "text-only":
            text = Path(self.stim.media_file).read_text(encoding="utf-8", errors="ignore")
            self._transcript = build_text_transcript(text, self.reading_rate)
        return self._transcript

    def iter_frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (t_sec, RGB uint8 [h,w,3]) downscaled to max_side, at ~analysis_fps.

        Decoded frames are cached on the first *complete* pass so the many visual
        extractors share one decode instead of re-decoding the video each. If the
        video is too long to cache safely, it falls back to streaming per call.
        """
        if not self.stim.has_video:
            return
        if self._frame_cache is not None:
            yield from self._frame_cache
            return
        cache: list | None = []
        for t, rgb in self._decode_frames():
            if cache is not None:
                cache.append((t, rgb))
                if len(cache) > self.MAX_CACHE_FRAMES:
                    cache = None            # too big: stop caching, free it, keep streaming
            yield t, rgb
        if cache is not None:               # only set on a fully-consumed pass
            self._frame_cache = cache

    def _decode_frames(self) -> Iterator[tuple[float, np.ndarray]]:
        step = 1.0 / self.analysis_fps
        next_t = 0.0
        with av.open(self.stim.media_file) as c:
            vs = c.streams.video[0]
            vs.thread_type = "AUTO"
            w0, h0 = vs.codec_context.width, vs.codec_context.height
            scale = min(1.0, self.max_side / max(w0, h0))
            w2, h2 = max(1, round(w0 * scale)), max(1, round(h0 * scale))
            for frame in c.decode(vs):
                if frame.pts is None:
                    continue
                t = float(frame.pts * vs.time_base)
                if t + 1e-6 < next_t:
                    continue
                next_t = t + step
                # PyAV handles downscale + RGB conversion (no cv2 -> avoids ffmpeg dylib clash)
                rgb = frame.reformat(width=w2, height=h2, format="rgb24").to_ndarray()
                yield t, rgb

    def audio_wav(self) -> str | None:
        """Extract (once) a mono wav at audio_sr; return path or None if no audio."""
        if not self.stim.has_audio:
            return None
        if self._audio_path and Path(self._audio_path).exists():
            return self._audio_path
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        out = self.workdir / f"{self.stim.id}_a{self.audio_sr}.wav"
        cmd = [ff, "-y", "-i", self.stim.media_file, "-vn",
               "-ac", "1", "-ar", str(self.audio_sr), "-f", "wav", str(out)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._audio_path = str(out)
        return self._audio_path
