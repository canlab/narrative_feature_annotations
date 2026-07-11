"""Low-level lexical / word-level features from the transcript (CPU).

Per-word values (Zipf frequency, length, affective/concreteness/AoA norms) mapped
to grid bins by word onset (mean). Frozen core pick: spaCy + wordfreq/SUBTLEX +
Brysbaert/Kuperman/Warriner norms + NRC EmoLex; LLM surprisal (minicons) is a
later torch pass. Norms are NaN unless data/lexicons/ CSVs are present (see norms.py).
"""

from __future__ import annotations

import re

import numpy as np
from wordfreq import zipf_frequency

from ..base import (LANGUAGE_MODALITIES, Extractor, FeatureChannel, Stimulus,
                    TimeGrid, grid_reduce_scalar)
from ..norms import FIELDS, load_norms

_WORD = re.compile(r"[^a-z']+")


def _clean(token: str) -> str:
    return _WORD.sub("", token.lower())


class LanguageLexical(Extractor):
    feature_class = "language"
    name = "language_lexical"
    applicable_modalities = LANGUAGE_MODALITIES
    tier = "cpu"

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        tr = ingest.get_transcript()
        norms = load_norms()
        meta = dict(native_rate_hz="word", resample="mean", tier="cpu")

        def channel(name, vals, model, units="", notes=""):
            return FeatureChannel(path=f"language/lexical/{name}", value=vals, dtype="scalar",
                                  units=units, model=model, version="slice-0.2", notes=notes, **meta)

        if tr is None or not tr.has_speech:
            nan = np.full(grid.n_samples, np.nan)
            out = [channel("freq_zipf", nan.copy(), "wordfreq", "zipf"),
                   channel("word_length", nan.copy(), "len", "chars")]
            out += [channel(f, nan.copy(), "norms:" + f,
                            notes="no speech" if tr else "no transcript") for f in FIELDS]
            return out

        times = np.array([w.start for w in tr.words], float)
        clean = [_clean(w.text) for w in tr.words]
        freq = np.array([zipf_frequency(c, "en") if c else np.nan for c in clean])
        length = np.array([len(c) if c else np.nan for c in clean])

        out = [
            channel("freq_zipf", grid_reduce_scalar(times, freq, grid, "mean"),
                    "wordfreq", "zipf", "mean Zipf frequency of words onsetting in bin"),
            channel("word_length", grid_reduce_scalar(times, length, grid, "mean"),
                    "len", "chars"),
        ]
        have = set(norms.available())
        for field in FIELDS:
            vals = np.array([norms.get(c, field) if c else np.nan for c in clean])
            note = "" if field in have else f"no data/lexicons/{field}.csv -> NaN"
            out.append(channel(field, grid_reduce_scalar(times, vals, grid, "mean"),
                               f"norms:{field}", notes=note))
        return out
