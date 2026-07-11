"""Syntactic / grammatical structure from the transcript (spaCy, CPU).

Per-segment parse -> complexity aggregates (tree depth, mean dependency distance,
content-word fraction, noun/verb fractions), assigned to the grid bins each segment
spans (mean across overlapping segments). Frozen core pick: spaCy trf + benepar +
L2SCA + Maverick coref; this slice uses en_core_web_sm (swap to trf for production).
"""

from __future__ import annotations

import numpy as np

from ..base import (LANGUAGE_MODALITIES, Extractor, FeatureChannel, Stimulus, TimeGrid)

_CONTENT = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
_METRICS = ["tree_depth", "mean_dep_distance", "content_frac", "noun_frac", "verb_frac"]


def _tok_depth(tok) -> int:
    d, seen = 0, 0
    while tok.head != tok and seen < 200:
        tok = tok.head
        d += 1
        seen += 1
    return d


def _segment_metrics(doc) -> dict:
    toks = [t for t in doc if not t.is_space]
    if not toks:
        return {m: np.nan for m in _METRICS}
    n = len(toks)
    pos = [t.pos_ for t in toks]
    return {
        "tree_depth": float(max(_tok_depth(t) for t in toks)),
        "mean_dep_distance": float(np.mean([abs(t.i - t.head.i) for t in toks if t.head != t] or [0.0])),
        "content_frac": sum(p in _CONTENT for p in pos) / n,
        "noun_frac": sum(p in ("NOUN", "PROPN") for p in pos) / n,
        "verb_frac": pos.count("VERB") / n,
    }


class LanguageSyntax(Extractor):
    feature_class = "language"
    name = "language_syntax"
    applicable_modalities = LANGUAGE_MODALITIES
    tier = "cpu"

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.spacy_model = spacy_model
        self._nlp = None

    def _load(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self.spacy_model, disable=["ner"])
        return self._nlp

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        tr = ingest.get_transcript()
        meta = dict(model=f"spaCy:{self.spacy_model}", version="3.8", native_rate_hz="utterance",
                    resample="mean", tier="cpu")

        def channel(name, vals, notes=""):
            return FeatureChannel(path=f"language/syntax/{name}", value=vals, dtype="scalar",
                                  notes=notes, **meta)

        if tr is None or not tr.has_speech:
            nan = np.full(grid.n_samples, np.nan)
            note = "no speech" if tr else "no transcript"
            return [channel(m, nan.copy(), note) for m in _METRICS]

        nlp = self._load()
        sums = {m: np.zeros(grid.n_samples) for m in _METRICS}
        counts = np.zeros(grid.n_samples)
        for seg in tr.segments:
            if not seg.text.strip():
                continue
            mvals = _segment_metrics(nlp(seg.text))
            b0, b1 = int(grid.bin_index(seg.start)), int(grid.bin_index(seg.end))
            for b in range(max(b0, 0), min(b1, grid.n_samples - 1) + 1):
                counts[b] += 1
                for m in _METRICS:
                    sums[m][b] += mvals[m]
        out = []
        for m in _METRICS:
            vals = np.divide(sums[m], counts, out=np.full(grid.n_samples, np.nan), where=counts > 0)
            out.append(channel(m, vals))
        return out
