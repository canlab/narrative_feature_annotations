"""EmoNet (Kragel et al., 2019, Science Advances) — 20 emotion-schema probabilities
per video frame (AlexNet-based, MPS/CPU).

EmoNet maps a natural image to a 20-way normative-emotion distribution ("emotion
schemas embedded in the human visual system"). This is *whole-image* depicted affect,
complementary to the face-level (face_emotion), voice-level (vocal_affect), text-level
(text_emotion/text_sentiment), and scene-level (Qwen) affect streams.

Per frame -> grid mean:
  affect/depicted/emonet       20-d softmax over the Cowen&Keltner emotion categories
  affect/depicted/emonet_top   argmax category

Model architecture + weights are the PyTorch port of Kragel's Caffe/MATLAB EmoNet from
ecco-laboratory/emonet-pytorch (MIT); weights are fetched once from OSF (osf.io/amdju).
Input follows that port's replicate script exactly: RGB, raw 0-255 values (NO ImageNet
normalization), resized to 227x227.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..base import (VISUAL_MODALITIES, Extractor, FeatureChannel, Stimulus,
                    TimeGrid, grid_reduce_scalar)
from ..torch_util import batched, get_device

WEIGHTS_URL = "https://osf.io/amdju/download"
WEIGHTS_CACHE = Path.home() / ".cache" / "nfe" / "emonet_kragel2019_weights.pt"

# Cowen & Keltner (2017) emotion categories, in EmoNet's output order (Kragel 2019).
EMONET_CLASSES = [
    "Adoration", "Aesthetic Appreciation", "Amusement", "Anxiety", "Awe", "Boredom",
    "Confusion", "Craving", "Disgust", "Empathic Pain", "Entrancement", "Excitement",
    "Fear", "Horror", "Interest", "Joy", "Romance", "Sadness", "Sexual Desire", "Surprise"]


def _build_emonet(num_classes: int = 20):
    """AlexNet-style EmoNet, ported from ecco-laboratory/emonet-pytorch (MIT)."""
    import torch
    from torch import nn

    lrn_alpha = 9.999999747378752e-05

    class EmoNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_0 = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4), nn.ReLU(),
                nn.LocalResponseNorm(size=5, alpha=lrn_alpha, beta=0.75, k=1),
                nn.MaxPool2d(kernel_size=3, stride=2))
            self.conv_1 = nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2), nn.ReLU(),
                nn.LocalResponseNorm(size=5, alpha=lrn_alpha, beta=0.75, k=1),
                nn.MaxPool2d(kernel_size=3, stride=2))
            self.conv_2 = nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), nn.ReLU())
            self.conv_3 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2), nn.ReLU())
            self.conv_4 = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2))
            self.conv_5 = nn.Sequential(nn.Conv2d(256, 4096, kernel_size=6, stride=1), nn.ReLU())
            self.conv_6 = nn.Sequential(nn.Conv2d(4096, 4096, kernel_size=1, stride=1), nn.ReLU())
            self.classifier = nn.Sequential(
                nn.Conv2d(4096, num_classes, kernel_size=1, stride=1),
                nn.Flatten(start_dim=-3, end_dim=-1), nn.Softmax(dim=-1))

        def forward(self, x):
            x = x.to(torch.float)
            for blk in (self.conv_0, self.conv_1, self.conv_2, self.conv_3,
                        self.conv_4, self.conv_5, self.conv_6, self.classifier):
                x = blk(x)
            return x

    return EmoNet()


class VisualEmoNet(Extractor):
    feature_class = "affect"
    name = "visual_emonet"
    applicable_modalities = VISUAL_MODALITIES
    tier = "gpu"

    def __init__(self, batch_size: int = 16, device: str = "auto"):
        self.batch_size = batch_size
        self.device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        self.torch = torch
        self.dev = get_device(self.device)
        if not WEIGHTS_CACHE.exists():
            WEIGHTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(WEIGHTS_URL, str(WEIGHTS_CACHE))
        model = _build_emonet(len(EMONET_CLASSES))
        model.load_state_dict(torch.load(str(WEIGHTS_CACHE), map_location="cpu", weights_only=True))
        self.model = model.to(self.dev).eval()
        self._loaded = True

    def _prep(self, rgb: np.ndarray):
        # RGB HxWx3 uint8 -> [3,227,227] float in 0-255 (no normalization), per the port.
        import torch
        from torchvision.transforms.functional import resize
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)   # [3,H,W] uint8
        return resize(t, [227, 227], antialias=True).float()

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        self._load()
        torch = self.torch
        times, probs = [], []
        for batch in batched(ingest.iter_frames(), self.batch_size):
            ts = [t for t, _ in batch]
            x = torch.stack([self._prep(rgb) for _, rgb in batch]).to(self.dev)
            with torch.no_grad():
                p = self.model(x).float().cpu().numpy()               # [B,20] softmax
            times.extend(ts)
            probs.append(p)
        if not times:
            return []
        times = np.array(times)
        probs = np.concatenate(probs)
        grid_probs = grid_reduce_scalar(times, probs, grid, "mean")   # [n_samples,20]
        top = np.full(grid.n_samples, -1, int)
        valid = np.isfinite(grid_probs).all(1)
        top[valid] = grid_probs[valid].argmax(1)
        meta = dict(model="Kragel2019-EmoNet(emonet-pytorch)", version="osf-amdju",
                    native_rate_hz=ingest.analysis_fps, tier="gpu")
        return [
            FeatureChannel(path="affect/depicted/emonet", value=grid_probs.astype(np.float32),
                           dtype="vector", components=list(EMONET_CLASSES), units="softmax",
                           resample="mean", notes="whole-image emotion schema (Kragel 2019)", **meta),
            FeatureChannel(path="affect/depicted/emonet_top", value=top, dtype="categorical",
                           categories=list(EMONET_CLASSES), resample="mode", **meta),
        ]
