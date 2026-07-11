"""Sentiment polarity of dialogue/narration (CardiffNLP twitter-roberta, MPS/CPU).

Per transcript segment -> 3-way sentiment (negative/neutral/positive) softmax, assigned
to the bins the segment spans, plus a signed polarity scalar P(pos)-P(neg) in [-1,1].
Complements text_emotion (GoEmotions, 28 fine-grained *emotions*) with a clean valence
*polarity* signal. Model trained on ~124M tweets + TweetEval (Barbieri et al.).

Channels under features/affect/depicted/: text_sentiment (3-d vector),
text_sentiment_polarity (scalar), text_sentiment_top (categorical).
"""

from __future__ import annotations

import numpy as np

from ..base import (LANGUAGE_MODALITIES, Extractor, FeatureChannel, Stimulus, TimeGrid)
from ..torch_util import get_device

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT = ["negative", "neutral", "positive"]


def _normalize(text: str) -> str:
    # Match the model's training preprocessing (usernames -> @user, links -> http).
    toks = []
    for tok in text.split():
        if tok.startswith("@") and len(tok) > 1:
            tok = "@user"
        elif tok.startswith("http"):
            tok = "http"
        toks.append(tok)
    return " ".join(toks)


class TextSentiment(Extractor):
    feature_class = "affect"
    name = "text_sentiment"
    applicable_modalities = LANGUAGE_MODALITIES
    tier = "gpu"

    def __init__(self, device: str = "auto"):
        self.device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.dev = get_device(self.device)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(self.dev).eval()
        # Order the model's own labels to our fixed [negative, neutral, positive].
        id2 = {i: self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)}
        self.col = [next(i for i, lab in id2.items() if lab.startswith(s[:3])) for s in SENTIMENT]
        self._loaded = True

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        tr = ingest.get_transcript()
        meta = dict(model=MODEL, version="tweeteval-latest", native_rate_hz="utterance", tier="gpu")

        def channels(post, top):
            polarity = post[:, 2] - post[:, 0]                    # P(pos) - P(neg) in [-1,1]
            return [
                FeatureChannel(path="affect/depicted/text_sentiment", value=post.astype(np.float32),
                               dtype="vector", components=list(SENTIMENT), units="softmax",
                               resample="mean", **meta),
                FeatureChannel(path="affect/depicted/text_sentiment_polarity",
                               value=polarity.astype(np.float32), dtype="scalar", units="[-1,1]",
                               resample="mean", notes="P(positive)-P(negative)", **meta),
                FeatureChannel(path="affect/depicted/text_sentiment_top", value=top,
                               dtype="categorical", categories=list(SENTIMENT), resample="mode", **meta),
            ]

        if tr is None or not tr.has_speech:
            post = np.full((grid.n_samples, len(SENTIMENT)), np.nan, np.float32)
            top = np.full(grid.n_samples, -1, int)
            return channels(post, top)

        self._load()
        torch = self.torch
        sums = np.zeros((grid.n_samples, len(SENTIMENT)))
        counts = np.zeros(grid.n_samples)
        for seg in tr.segments:
            if not seg.text.strip():
                continue
            with torch.no_grad():
                enc = self.tok(_normalize(seg.text), return_tensors="pt",
                               truncation=True, max_length=128).to(self.dev)
                probs = self.model(**enc).logits.softmax(-1)[0].cpu().numpy()[self.col]
            b0, b1 = int(grid.bin_index(seg.start)), int(grid.bin_index(seg.end))
            for b in range(max(b0, 0), min(b1, grid.n_samples - 1) + 1):
                sums[b] += probs
                counts[b] += 1
        post = np.divide(sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0)
        top = np.full(grid.n_samples, -1, int)
        valid = counts > 0
        top[valid] = post[valid].argmax(1)
        return channels(post, top)
