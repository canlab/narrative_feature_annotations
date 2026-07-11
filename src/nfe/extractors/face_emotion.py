"""Facial affect/emotion on detected faces (HSEmotion EfficientNet, AffectNet — CPU/MPS).

Detects faces (MTCNN, as in faces.py) and reads each face's expression with HSEmotion's
`enet_b0_8_va_mtl` model (state-of-the-art on AffectNet): 8 discrete expressions plus
continuous valence and arousal. Per frame the faces are averaged, then reduced to the grid.
This is *face-level* depicted affect, complementary to whole-image (visual_emonet),
voice (vocal_affect), and text (text_emotion/text_sentiment) streams.

Per frame (mean over faces) -> grid mean:
  affect/depicted/face_emotion       8-d softmax over AffectNet expressions
  affect/depicted/face_emotion_top   argmax expression
  affect/depicted/face_valence       continuous valence in [-1, 1]
  affect/depicted/face_arousal       continuous arousal in [-1, 1]

HSEmotion (Savchenko), Apache-2.0; timm-based EfficientNet, no OpenCV. Bins with no
detected face are NaN (they do not contribute to the grid mean).
"""

from __future__ import annotations

import numpy as np

from ..base import (VISUAL_MODALITIES, Extractor, FeatureChannel, Stimulus,
                    TimeGrid, grid_reduce_scalar)
from ..torch_util import get_device

MODEL = "enet_b0_8_va_mtl"
AFFECTNET8 = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]


class FaceEmotion(Extractor):
    feature_class = "affect"
    name = "face_emotion"
    applicable_modalities = VISUAL_MODALITIES
    tier = "gpu"

    def __init__(self, min_prob: float = 0.90, min_side: int = 24, device: str = "cpu"):
        # MTCNN + the tiny b0 EfficientNet run comfortably on CPU; face crops are small.
        self.min_prob = min_prob
        self.min_side = min_side
        self.device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from facenet_pytorch import MTCNN
        from hsemotion.facial_emotions import HSEmotionRecognizer
        self.dev = get_device(self.device)
        self.mtcnn = MTCNN(keep_all=True, device=self.dev)
        # hsemotion pickles a full nn.Module; torch>=2.6 defaults weights_only=True and
        # would refuse to unpickle it. Force weights_only=False just for construction
        # (weights come from the trusted HSE-asavchenko release).
        orig_load = torch.load

        def _full_load(*a, **k):
            k.setdefault("weights_only", False)
            return orig_load(*a, **k)

        torch.load = _full_load
        try:
            self.fer = HSEmotionRecognizer(model_name=MODEL, device=str(self.dev))
        finally:
            torch.load = orig_load
        self._loaded = True

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        from PIL import Image
        self._load()
        n_emo = len(AFFECTNET8)
        times, emo, val, aro = [], [], [], []
        for t, rgb in ingest.iter_frames():
            H, W = rgb.shape[:2]
            boxes, probs = self.mtcnn.detect(Image.fromarray(rgb))
            times.append(t)
            crops = []
            if boxes is not None:
                for box, p in zip(boxes, probs):
                    if p is None or p < self.min_prob:
                        continue
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, W), min(y2, H)
                    if x2 - x1 >= self.min_side and y2 - y1 >= self.min_side:
                        crops.append(np.ascontiguousarray(rgb[y1:y2, x1:x2]))
            if not crops:
                emo.append(np.full(n_emo, np.nan)); val.append(np.nan); aro.append(np.nan)
                continue
            _, scores = self.fer.predict_multi_emotions(crops, logits=False)
            scores = np.asarray(scores, float)                    # [n_faces, 10] = 8 emo + val + aro
            emo.append(scores[:, :n_emo].mean(0))
            val.append(float(scores[:, n_emo].mean()))
            aro.append(float(scores[:, n_emo + 1].mean()))
        if not times:
            return []
        times = np.array(times)
        emo = np.asarray(emo, float)                              # [T, 8], NaN where no face
        grid_emo = grid_reduce_scalar(times, emo, grid, "mean")
        top = np.full(grid.n_samples, -1, int)
        valid = np.isfinite(grid_emo).all(1)
        top[valid] = grid_emo[valid].argmax(1)
        meta = dict(model=f"HSEmotion-{MODEL}", version="affectnet-8-va",
                    native_rate_hz=ingest.analysis_fps, tier="gpu")
        return [
            FeatureChannel(path="affect/depicted/face_emotion", value=grid_emo.astype(np.float32),
                           dtype="vector", components=list(AFFECTNET8), units="softmax",
                           resample="mean", notes="mean over detected faces", **meta),
            FeatureChannel(path="affect/depicted/face_emotion_top", value=top, dtype="categorical",
                           categories=list(AFFECTNET8), resample="mode", **meta),
            FeatureChannel(path="affect/depicted/face_valence",
                           value=grid_reduce_scalar(times, np.array(val), grid, "mean"),
                           dtype="scalar", units="[-1,1]", resample="mean", **meta),
            FeatureChannel(path="affect/depicted/face_arousal",
                           value=grid_reduce_scalar(times, np.array(aro), grid, "mean"),
                           dtype="scalar", units="[-1,1]", resample="mean", **meta),
        ]
