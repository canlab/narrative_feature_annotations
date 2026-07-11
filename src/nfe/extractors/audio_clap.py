"""CLAP open-vocabulary audio (MPS/CPU). Sliding window -> grid mean.

Channels under features/audio/high_level/: clap_embedding (512-d audio embedding)
and clap_probe (softmax over an audio prompt set). Kept distinct from AST/BEATs
(fixed AudioSet taxonomy) because CLAP scores arbitrary text prompts.
"""

from __future__ import annotations

import numpy as np

from ..base import (AUDIO_MODALITIES, Extractor, FeatureChannel, Stimulus,
                    TimeGrid, grid_reduce_scalar)
from ..torch_util import get_device

MODEL = "laion/clap-htsat-unfused"
SR = 48000
DEFAULT_PROMPTS = ["speech", "music", "singing", "silence", "background noise",
                   "laughter", "applause", "footsteps", "a vehicle engine",
                   "nature sounds", "a crowd of people", "wind"]


class AudioClap(Extractor):
    feature_class = "audio"
    name = "audio_clap"
    applicable_modalities = AUDIO_MODALITIES
    tier = "gpu"

    def __init__(self, win_sec: float = 4.0, hop_sec: float = 1.0, prompts=None,
                 device: str = "auto"):
        self.win_sec = win_sec
        self.hop_sec = hop_sec
        self.prompts = list(prompts) if prompts is not None else DEFAULT_PROMPTS
        self.device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.dev = get_device(self.device)
        self.proc = ClapProcessor.from_pretrained(MODEL)
        self.model = ClapModel.from_pretrained(MODEL).to(self.dev).eval()
        # Precompute normalized text embeddings ONCE (verified identical to the combined
        # forward); avoids re-running the text tower on every 1-s audio window.
        tin = self.proc(text=self.prompts, return_tensors="pt", padding=True).to(self.dev)
        with torch.no_grad():
            t = self.model.get_text_features(**tin)
        t = t if torch.is_tensor(t) else t.pooler_output
        self.text_emb = torch.nn.functional.normalize(t, dim=-1)
        self._loaded = True

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        import librosa
        self._load()
        torch = self.torch
        wav = ingest.audio_wav()
        if wav is None:
            return []
        y, _ = librosa.load(wav, sr=SR, mono=True)
        win, hop = int(self.win_sec * SR), int(self.hop_sec * SR)
        times, emb, probe = [], [], []
        for s in range(0, max(len(y) - win, 0) + 1, hop):
            seg = y[s:s + win]
            if seg.size < hop:
                break
            with torch.no_grad():
                inp = self.proc(audio=seg, sampling_rate=SR, return_tensors="pt",
                                padding=True).to(self.dev)
                ae = self.model.get_audio_features(input_features=inp["input_features"])
                ae = ae if torch.is_tensor(ae) else ae.pooler_output
                ae_n = torch.nn.functional.normalize(ae, dim=-1)
                logits = ae_n @ self.text_emb.T * self.model.logit_scale_a.exp()
                p = logits.softmax(-1)
            times.append((s + win / 2) / SR)
            emb.append(ae_n[0].cpu().numpy())
            probe.append(p[0].cpu().numpy())
        if not times:
            return []
        times, emb, probe = np.array(times), np.array(emb), np.array(probe)
        meta = dict(model=MODEL, version="htsat-unfused", native_rate_hz=f"window~{self.win_sec}s",
                    resample="mean", tier="gpu")
        return [
            FeatureChannel(path="audio/high_level/clap_embedding",
                           value=grid_reduce_scalar(times, emb, grid, "mean"), dtype="vector",
                           components=[f"d{i}" for i in range(emb.shape[1])],
                           notes="L2-normalized audio embedding", **meta),
            FeatureChannel(path="audio/high_level/clap_probe",
                           value=grid_reduce_scalar(times, probe, grid, "mean"), dtype="vector",
                           components=list(self.prompts), units="softmax", **meta),
        ]
