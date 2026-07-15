"""High-level language features from the transcript (LLM embeddings + surprisal).

Adds a rich, dense representation of *what is said*, mirroring what SigLIP/DINOv2 do
for vision and CLAP for audio:

  language/embedding/qwen3     Qwen3-Embedding-0.6B sentence embedding (1024-d), per
                               utterance, L2-normalized, binned to the grid (mean).
  language/embedding/llama_ar  Llama-3.1-8B autoregressive hidden states (4096-d) at a
                               mid layer — the standard fMRI/MEG language-encoding
                               feature — token-time-spread within each utterance, binned.

Plus interpretable derived scalars at several levels (all under language/hl/):
  semantic_coherence   cos(utterance_t, utterance_{t-1})        — local meaning continuity
  semantic_drift       cos(utterance_t, running-mean context)   — integration / topic drift
  semantic_novelty     1 - max cos to any earlier utterance     — recurrence / new content
  semantic_surprise    1 - semantic_coherence                   — semantic prediction error
  narrative_expectedness   LLM-rated predictability of the line given prior context (0-1)
  narrative_surprise       1 - narrative_expectedness           — high-level narrative surprise

Utterance timing comes from the whisper word/segment timestamps (ASR pass); everything is
downsampled to the common 1 Hz grid. Non-speech stimuli yield all-NaN (constant shape).
"""
from __future__ import annotations

import re
import numpy as np

from ..base import (LANGUAGE_MODALITIES, Extractor, FeatureChannel, Stimulus, TimeGrid,
                    grid_reduce_scalar)
from ..torch_util import get_device

QWEN = "Qwen/Qwen3-Embedding-0.6B"
LLAMA = "NousResearch/Meta-Llama-3.1-8B-Instruct"
AR_LAYER = 18                      # mid layer (of 32) — best brain-fit region empirically
CONTEXT_SEGS = 3                   # prior utterances shown to the narrative-rating LLM


class LanguageHL(Extractor):
    feature_class = "language"
    name = "language_hl"
    applicable_modalities = LANGUAGE_MODALITIES
    tier = "gpu"

    def __init__(self, device: str = "auto", narrative: bool = True, ar_layer: int = AR_LAYER):
        self.device = device
        self.narrative = narrative
        self.ar_layer = ar_layer
        self._loaded = False

    # ------------------------------------------------------------------ models
    def _load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.dev = get_device(self.device)
        self.qtok = AutoTokenizer.from_pretrained(QWEN)
        self.qmodel = AutoModel.from_pretrained(QWEN, dtype=torch.float32).to(self.dev).eval()
        self.ltok = AutoTokenizer.from_pretrained(LLAMA)
        self.lmodel = AutoModelForCausalLM.from_pretrained(
            LLAMA, dtype=torch.bfloat16, output_hidden_states=True).to(self.dev).eval()
        if self.ltok.pad_token is None:
            self.ltok.pad_token = self.ltok.eos_token
        self._loaded = True

    def _qwen_embed(self, texts: list[str]) -> np.ndarray:
        """Last-token-pooled, L2-normalized Qwen3 embeddings. [N, 1024]."""
        torch = self.torch
        out = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i + 16]
            enc = self.qtok(batch, padding=True, truncation=True, max_length=512,
                            return_tensors="pt").to(self.dev)
            with torch.no_grad():
                h = self.qmodel(**enc).last_hidden_state           # [B, T, 1024]
            last = enc.attention_mask.sum(1) - 1                    # last real token per row
            emb = h[torch.arange(h.shape[0]), last]                # [B, 1024]
            emb = torch.nn.functional.normalize(emb, dim=-1)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, 1024))

    def _llama_hidden(self, text: str) -> np.ndarray:
        """Per-token hidden states at self.ar_layer for one utterance. [T, 4096]."""
        torch = self.torch
        enc = self.ltok(text, return_tensors="pt", truncation=True, max_length=512).to(self.dev)
        with torch.no_grad():
            hs = self.lmodel(**enc).hidden_states                  # tuple(len 33) of [1,T,4096]
        return hs[self.ar_layer][0].float().cpu().numpy()

    def _rate_expectedness(self, prior: str, line: str) -> float:
        """Ask Llama how predictable `line` is given `prior`. Returns 0..1 or NaN."""
        torch = self.torch
        msg = [
            {"role": "system", "content":
             "You rate how expected or predictable a line of dialogue or narration is given the "
             "preceding context, from 0.00 (completely surprising/unexpected) to 1.00 (completely "
             "expected/predictable). Reply with only the number."},
            {"role": "user", "content":
             f"Context: {prior if prior else '(start of story)'}\nNext line: \"{line}\"\n"
             "Expectedness (0.00-1.00):"}]
        enc = self.ltok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True).to(self.dev)
        with torch.no_grad():
            gen = self.lmodel.generate(**enc, max_new_tokens=6, do_sample=False,
                                       pad_token_id=self.ltok.eos_token_id)
        txt = self.ltok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"[01](?:\.\d+)?|\.\d+", txt)
        if not m:
            return np.nan
        return float(np.clip(float(m.group()), 0.0, 1.0))

    # ------------------------------------------------------------------ extract
    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        tr = ingest.get_transcript()
        meta = dict(version="hf", native_rate_hz="utterance", tier="gpu")
        n = grid.n_samples

        def vec_ch(name, val, dim, model, note=""):
            # Opaque embeddings: no per-dimension component names (d0..dN are implied by
            # position; 4096 name strings also exceed HDF5's attribute-size limit).
            return FeatureChannel(path=f"language/embedding/{name}", value=val, dtype="vector",
                                  components=None, resample="mean", model=model, notes=note, **meta)

        def sc_ch(name, val, model, note=""):
            return FeatureChannel(path=f"language/hl/{name}", value=val, dtype="scalar",
                                  resample="mean", model=model, units="", notes=note,
                                  version="hf", native_rate_hz="utterance", tier="gpu")

        if tr is None or not tr.has_speech:
            note = "no speech" if tr else "no transcript"
            nanv = np.full(n, np.nan)
            chs = [vec_ch("qwen3", np.full((n, 1024), np.nan), 1024, QWEN, note),
                   vec_ch("llama_ar", np.full((n, 4096), np.nan), 4096, LLAMA, note)]
            for nm in ("semantic_coherence", "semantic_drift", "semantic_novelty", "semantic_surprise"):
                chs.append(sc_ch(nm, nanv.copy(), QWEN, note))
            for nm in ("narrative_expectedness", "narrative_surprise"):
                chs.append(sc_ch(nm, nanv.copy(), LLAMA, note))
            return chs

        self._load()
        segs = [s for s in tr.segments if s.text.strip()]
        texts = [s.text.strip() for s in segs]
        emb = self._qwen_embed(texts)                              # [S, 1024]

        # Fill EVERY grid second an utterance spans (not just its midpoint), so speech
        # seconds are densely covered; overlapping utterances are averaged.
        def span_fill(values):
            values = np.asarray(values, float)
            times, vals = [], []
            for i, s in enumerate(segs):
                b0, b1 = int(grid.bin_index(s.start)), int(grid.bin_index(s.end))
                for b in range(max(b0, 0), min(b1, grid.n_samples - 1) + 1):
                    times.append(grid.t_start_sec + b / grid.rate_hz)
                    vals.append(values[i])
            if not times:
                shape = (grid.n_samples,) if values.ndim == 1 else (grid.n_samples, values.shape[1])
                return np.full(shape, np.nan)
            return grid_reduce_scalar(np.array(times), np.array(vals), grid, "mean")

        qgrid = span_fill(emb)

        # ---- Llama AR hidden states -> grid (token-time spread within each utterance)
        tok_times, tok_vecs = [], []
        for s in segs:
            h = self._llama_hidden(s.text)                         # [T, 4096]
            if h.shape[0] == 0:
                continue
            tt = s.start + (np.arange(h.shape[0]) + 0.5) / h.shape[0] * max(s.end - s.start, 1e-3)
            tok_times.append(tt)
            tok_vecs.append(h)
        if tok_vecs:
            lgrid = grid_reduce_scalar(np.concatenate(tok_times), np.concatenate(tok_vecs, 0), grid, "mean")
        else:
            lgrid = np.full((n, 4096), np.nan)

        # ---- semantic coherence / drift / novelty / surprise (per utterance)
        coh = np.full(len(segs), np.nan)
        drift = np.full(len(segs), np.nan)
        nov = np.full(len(segs), np.nan)
        run = None
        for i in range(len(segs)):
            e = emb[i]
            if i > 0:
                coh[i] = float(emb[i] @ emb[i - 1])
                sims = emb[:i] @ e
                nov[i] = float(1.0 - np.max(sims))
                drift[i] = float(e @ (run / np.linalg.norm(run) + 1e-9))
            run = e.copy() if run is None else run + e
        surprise = 1.0 - coh

        out = [
            vec_ch("qwen3", qgrid, 1024, QWEN),
            vec_ch("llama_ar", lgrid, 4096, LLAMA),
            sc_ch("semantic_coherence", span_fill(coh), QWEN),
            sc_ch("semantic_drift", span_fill(drift), QWEN),
            sc_ch("semantic_novelty", span_fill(nov), QWEN),
            sc_ch("semantic_surprise", span_fill(surprise), QWEN),
        ]

        # ---- narrative expectedness (LLM-rated), optional/heavier
        if self.narrative:
            exp = np.full(len(segs), np.nan)
            for i, s in enumerate(segs):
                prior = " ".join(texts[max(0, i - CONTEXT_SEGS):i])
                exp[i] = self._rate_expectedness(prior, texts[i])
            out.append(sc_ch("narrative_expectedness", span_fill(exp), LLAMA))
            out.append(sc_ch("narrative_surprise", span_fill(1.0 - exp), LLAMA))
        else:
            nanv = np.full(n, np.nan)
            out.append(sc_ch("narrative_expectedness", nanv.copy(), LLAMA, "disabled"))
            out.append(sc_ch("narrative_surprise", nanv.copy(), LLAMA, "disabled"))
        return out
