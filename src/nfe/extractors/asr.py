"""ASR pass (faster-whisper) — the transcript hub.

Transcribes speech with word + segment timestamps, attaches the Transcript to the
Ingest object (so language/social/situation extractors can consume it), and emits
speech channels under features/audio/speech/.

Frozen core pick: faster-whisper large-v3 + WhisperX. CTranslate2 has no Metal
backend, so this runs on CPU on Apple Silicon; default model is "small" for fast
iteration — pass model_size="large-v3" (or "distil-large-v3") for production quality.
WhisperX forced-alignment / pyannote diarization are a separate (Social) pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..base import (AUDIO_MODALITIES, Extractor, FeatureChannel, Segment, Stimulus,
                    TimeGrid, Transcript, Word, grid_events)


class ASR(Extractor):
    feature_class = "audio"
    name = "asr"
    applicable_modalities = AUDIO_MODALITIES
    tier = "cpu"

    def __init__(self, model_size: str = "small", device: str = "cpu",
                 compute_type: str = "int8", vad_filter: bool = True):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.vad_filter = vad_filter

    def _transcribe(self, stim: Stimulus, ingest) -> Transcript:
        cache = Path(ingest.workdir) / f"{stim.id}_transcript_{self.model_size}.json"
        if cache.exists():
            d = json.loads(cache.read_text())
            words = [Word(**w) for w in d["words"]]
            segs = [Segment(s["text"], s["start"], s["end"],
                            [Word(**w) for w in s["words"]]) for s in d["segments"]]
            return Transcript(d["language"], d["text"], segs, words)

        from faster_whisper import WhisperModel
        model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        seg_iter, info = model.transcribe(
            ingest.audio_wav(), word_timestamps=True, vad_filter=self.vad_filter)
        segments, words = [], []
        for s in seg_iter:
            sw = [Word(w.word.strip(), float(w.start), float(w.end), float(w.probability or 1.0))
                  for w in (s.words or []) if w.start is not None]
            segments.append(Segment(s.text.strip(), float(s.start), float(s.end), sw))
            words.extend(sw)
        tr = Transcript(info.language or "", " ".join(s.text for s in segments).strip(),
                        segments, words)
        cache.write_text(json.dumps({
            "language": tr.language, "text": tr.text,
            "segments": [{"text": s.text, "start": s.start, "end": s.end,
                          "words": [vars(w) for w in s.words]} for s in tr.segments],
            "words": [vars(w) for w in tr.words]}))
        return tr

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        tr = self._transcribe(stim, ingest)
        ingest.set_transcript(tr)
        meta = dict(model=f"faster-whisper-{self.model_size}", version="ct2",
                    native_rate_hz="word", tier="cpu")

        # speech_present: bin covered by any word
        present = np.zeros(grid.n_samples, float)
        for w in tr.words:
            b0, b1 = int(grid.bin_index(w.start)), int(grid.bin_index(w.end))
            for b in range(max(b0, 0), min(b1, grid.n_samples - 1) + 1):
                present[b] = 1.0

        # word_rate: words per second
        starts = np.array([w.start for w in tr.words], float)
        word_rate = grid_events(starts, grid).astype(float) * grid.rate_hz

        # asr_text: words whose onset falls in each bin
        texts = [""] * grid.n_samples
        buckets: dict[int, list[str]] = {}
        for w in tr.words:
            b = int(grid.bin_index(w.start))
            if 0 <= b < grid.n_samples:
                buckets.setdefault(b, []).append(w.text)
        for b, ws in buckets.items():
            texts[b] = " ".join(ws)

        return [
            FeatureChannel(path="audio/speech/speech_present", value=present, dtype="bool",
                           resample="any", units="0/1", notes="bin overlapped by any word", **meta),
            FeatureChannel(path="audio/speech/word_rate", value=word_rate, dtype="scalar",
                           resample="count", units="words/s", **meta),
            FeatureChannel(path="audio/speech/asr_text", value=np.array([]), labels=texts,
                           dtype="label", resample="concat", units="",
                           notes=f"language={tr.language}", **meta),
        ]
