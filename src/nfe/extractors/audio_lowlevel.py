"""Audio low-level acoustic features (librosa, CPU). Frame-wise -> grid mean.

Channels under features/audio/low_level/: rms, spectral_centroid/bandwidth/
rolloff/flatness, zcr, onset_strength, f0 (scalars); mfcc, chroma (vectors);
tempo (whole-stimulus scalar). Feeds Audio, Speech, Affect per the core-set design.
"""

from __future__ import annotations

import numpy as np

from ..base import (AUDIO_MODALITIES, Extractor, FeatureChannel, Stimulus,
                    TimeGrid, grid_reduce_scalar)

HOP = 512


class AudioLowLevel(Extractor):
    feature_class = "audio"
    name = "audio_lowlevel"
    applicable_modalities = AUDIO_MODALITIES
    tier = "cpu"

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        import librosa
        wav = ingest.audio_wav()
        if wav is None:
            return []
        y, sr = librosa.load(wav, sr=ingest.audio_sr, mono=True)
        if y.size == 0:
            return []

        def t_of(n):
            return librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP)

        S = np.abs(librosa.stft(y, hop_length=HOP))
        scal = {
            "rms": librosa.feature.rms(S=S, hop_length=HOP)[0],
            "spectral_centroid": librosa.feature.spectral_centroid(S=S, sr=sr)[0],
            "spectral_bandwidth": librosa.feature.spectral_bandwidth(S=S, sr=sr)[0],
            "spectral_rolloff": librosa.feature.spectral_rolloff(S=S, sr=sr)[0],
            "spectral_flatness": librosa.feature.spectral_flatness(S=S)[0],
            "zcr": librosa.feature.zero_crossing_rate(y, hop_length=HOP)[0],
            "onset_strength": librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP),
        }
        units = {"spectral_centroid": "Hz", "spectral_bandwidth": "Hz",
                 "spectral_rolloff": "Hz", "zcr": "rate"}
        out: list[FeatureChannel] = []
        for name, vals in scal.items():
            vals = np.asarray(vals, float)
            g = grid_reduce_scalar(t_of(len(vals)), vals, grid, "mean")
            out.append(FeatureChannel(
                path=f"audio/low_level/{name}", value=g, dtype="scalar",
                units=units.get(name, ""), model="librosa", version="0.11",
                native_rate_hz=sr / HOP, resample="mean", tier="cpu"))

        # f0 (pyin) — NaN where unvoiced
        try:
            f0, _, _ = librosa.pyin(y, sr=sr, fmin=65, fmax=2093, hop_length=HOP)
            g = grid_reduce_scalar(t_of(len(f0)), np.asarray(f0, float), grid, "mean")
            out.append(FeatureChannel(
                path="audio/low_level/f0", value=g, dtype="scalar", units="Hz",
                model="librosa.pyin", version="0.11", native_rate_hz=sr / HOP,
                resample="mean", tier="cpu", notes="NaN where unvoiced"))
        except Exception:
            pass

        # vector channels
        for name, mat, comps in [
            ("mfcc", librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP),
             [f"mfcc{i}" for i in range(13)]),
            ("chroma", librosa.feature.chroma_stft(S=S, sr=sr),
             ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]),
        ]:
            mat = mat.T  # [frames, D]
            g = grid_reduce_scalar(t_of(mat.shape[0]), mat, grid, "mean")
            out.append(FeatureChannel(
                path=f"audio/low_level/{name}", value=g, dtype="vector", components=comps,
                model="librosa", version="0.11", native_rate_hz=sr / HOP,
                resample="mean", tier="cpu"))

        # tempo (whole-stimulus constant)
        try:
            tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)[0])[0])
            out.append(FeatureChannel(
                path="audio/low_level/tempo", value=np.full(grid.n_samples, tempo),
                dtype="scalar", units="BPM", model="librosa.beat", version="0.11",
                native_rate_hz="whole-stimulus", resample="nearest", tier="cpu"))
        except Exception:
            pass
        return out
