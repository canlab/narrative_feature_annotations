"""LLM surprisal from the transcript (GPT-2, MPS/CPU). Per segment -> grid mean.

Channels under features/language/lexical/: surprisal (bits) and entropy (bits),
the standard neuroimaging language-model predictability regressors. Frozen core
pick is minicons (GPT-2-medium / Pythia); this uses GPT-2 directly. Segment-level
mean for now (per-word alignment to whisper timestamps is a later refinement).
"""

from __future__ import annotations

import numpy as np

from ..base import (LANGUAGE_MODALITIES, Extractor, FeatureChannel, Stimulus, TimeGrid)
from ..torch_util import get_device

MODEL = "gpt2"


class LanguageSurprisal(Extractor):
    feature_class = "language"
    name = "language_surprisal"
    applicable_modalities = LANGUAGE_MODALITIES
    tier = "gpu"

    def __init__(self, device: str = "auto"):
        self.device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.dev = get_device(self.device)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL).to(self.dev).eval()
        self._loaded = True

    def _segment_stats(self, text: str) -> tuple[float, float]:
        torch = self.torch
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.dev)
        if ids.shape[1] < 2:
            return np.nan, np.nan
        with torch.no_grad():
            logits = self.model(ids).logits[0]                  # [T, V]
        logp = torch.log_softmax(logits[:-1], dim=-1)           # predict token t+1
        tgt = ids[0, 1:]
        surp = -(logp[torch.arange(len(tgt)), tgt]) / np.log(2)  # bits
        p = logp.exp()
        ent = -(p * logp).sum(-1) / np.log(2)                    # bits
        return float(surp.mean().cpu()), float(ent.mean().cpu())

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        tr = ingest.get_transcript()
        meta = dict(model=MODEL, version="hf", native_rate_hz="utterance",
                    resample="mean", units="bits", tier="gpu")

        def ch(name, vals, note=""):
            return FeatureChannel(path=f"language/lexical/{name}", value=vals,
                                  dtype="scalar", notes=note, **meta)
        if tr is None or not tr.has_speech:
            nan = np.full(grid.n_samples, np.nan)
            note = "no speech" if tr else "no transcript"
            return [ch("surprisal", nan.copy(), note), ch("entropy", nan.copy(), note)]

        self._load()
        sums = {"surprisal": np.zeros(grid.n_samples), "entropy": np.zeros(grid.n_samples)}
        counts = np.zeros(grid.n_samples)
        for seg in tr.segments:
            if not seg.text.strip():
                continue
            s, e = self._segment_stats(seg.text)
            if np.isnan(s):
                continue
            b0, b1 = int(grid.bin_index(seg.start)), int(grid.bin_index(seg.end))
            for b in range(max(b0, 0), min(b1, grid.n_samples - 1) + 1):
                sums["surprisal"][b] += s
                sums["entropy"][b] += e
                counts[b] += 1
        out = []
        for name in ("surprisal", "entropy"):
            vals = np.divide(sums[name], counts, out=np.full(grid.n_samples, np.nan), where=counts > 0)
            out.append(ch(name, vals))
        return out
