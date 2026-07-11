"""Vocal affect — audEERING wav2vec2 dimensional emotion (MPS/CPU).

Per sliding window -> grid mean: voice arousal / dominance / valence in 0-1.
This is the DEPICTED (speaker-expressed) affect stream from the voice; it is kept
separate from viewer-elicited affect. Feeds Speech + Affect per the core set.

The checkpoint uses a custom regression head (per the model card), defined below.
"""

from __future__ import annotations

import numpy as np

from ..base import (AUDIO_MODALITIES, Extractor, FeatureChannel, Stimulus,
                    TimeGrid, grid_reduce_scalar)
from ..torch_util import get_device

MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


def _build_model_classes():
    import torch.nn as nn
    from transformers.models.wav2vec2.modeling_wav2vec2 import (Wav2Vec2Model,
                                                                Wav2Vec2PreTrainedModel)

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, x):
            import torch
            x = self.dropout(x)
            x = torch.tanh(self.dense(x))
            return self.out_proj(self.dropout(x))

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values).last_hidden_state.mean(1)
            return self.classifier(hidden)        # [B, 3] = arousal, dominance, valence

    return EmotionModel


class VocalAffect(Extractor):
    feature_class = "audio"
    name = "vocal_affect"
    applicable_modalities = AUDIO_MODALITIES
    tier = "gpu"

    def __init__(self, win_sec: float = 2.0, hop_sec: float = 1.0, device: str = "auto"):
        self.win_sec = win_sec
        self.hop_sec = hop_sec
        self.device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2Config, Wav2Vec2Processor
        self.torch = torch
        self.dev = get_device(self.device)
        self.proc = Wav2Vec2Processor.from_pretrained(MODEL)
        # Build from config + load weights manually: transformers 5.x meta-device
        # finalize doesn't support this custom regression-head architecture.
        config = Wav2Vec2Config.from_pretrained(MODEL)
        model = _build_model_classes()(config)
        try:
            from safetensors.torch import load_file
            sd = load_file(hf_hub_download(MODEL, "model.safetensors"))
        except Exception:
            sd = torch.load(hf_hub_download(MODEL, "pytorch_model.bin"), map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not [k for k in missing if "classifier" in k], f"head not loaded: {missing}"
        self.model = model.to(self.dev).eval()
        self._loaded = True

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        import librosa
        self._load()
        torch = self.torch
        wav = ingest.audio_wav()
        if wav is None:
            return []
        y, _ = librosa.load(wav, sr=16000, mono=True)
        win, hop = int(self.win_sec * 16000), int(self.hop_sec * 16000)
        times, vad = [], []
        for s in range(0, max(len(y) - win, 0) + 1, hop):
            seg = y[s:s + win]
            if seg.size < hop:
                break
            with torch.no_grad():
                inp = self.proc(seg, sampling_rate=16000, return_tensors="pt").input_values.to(self.dev)
                out = self.model(inp)[0].cpu().numpy()      # arousal, dominance, valence
            times.append((s + win / 2) / 16000)
            vad.append(out)
        if not times:
            return []
        times, vad = np.array(times), np.array(vad)
        meta = dict(model=MODEL, version="msp-dim", native_rate_hz=f"window~{self.win_sec}s",
                    resample="mean", tier="gpu", units="0-1")
        names = ["arousal", "dominance", "valence"]
        return [FeatureChannel(path=f"audio/speech/voice_{n}",
                               value=grid_reduce_scalar(times, vad[:, i], grid, "mean"),
                               dtype="scalar", notes="depicted (speaker) affect", **meta)
                for i, n in enumerate(names)]
