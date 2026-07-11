"""Face detection (facenet-pytorch MTCNN, cv2-free). Per frame -> grid mean.

Channels: features/visual/faces_bodies_gaze/{n_faces, faces_present, max_face_frac,
face_det_prob} and features/social/n_agents (faces feed Social). Frozen core pick is
InsightFace/OpenFace 3.0 (full AUs/gaze/identity); MTCNN is the cv2-free stand-in for
presence/size. Identity embeddings + AUs are a later pass (needs OpenFace, isolated).
"""

from __future__ import annotations

import numpy as np

from ..base import (VISUAL_MODALITIES, Extractor, FeatureChannel, Stimulus,
                    TimeGrid, grid_reduce_scalar)
from ..torch_util import get_device


class Faces(Extractor):
    feature_class = "visual"
    name = "faces"
    applicable_modalities = VISUAL_MODALITIES
    tier = "gpu"

    def __init__(self, min_prob: float = 0.90, device: str = "cpu"):
        # MTCNN's NMS/ops are most stable on CPU; embeddings/pose use MPS elsewhere.
        self.min_prob = min_prob
        self.device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        from facenet_pytorch import MTCNN
        self.dev = get_device(self.device)
        self.mtcnn = MTCNN(keep_all=True, device=self.dev)
        self._loaded = True

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        from PIL import Image
        self._load()
        times, n_faces, max_frac, prob = [], [], [], []
        for t, rgb in ingest.iter_frames():
            H, W = rgb.shape[:2]
            boxes, probs = self.mtcnn.detect(Image.fromarray(rgb))
            times.append(t)
            if boxes is None:
                n_faces.append(0.0); max_frac.append(0.0); prob.append(np.nan)
                continue
            keep = probs >= self.min_prob
            boxes, probs = boxes[keep], probs[keep]
            n_faces.append(float(len(boxes)))
            if len(boxes):
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                max_frac.append(float(areas.max() / (W * H)))
                prob.append(float(probs.mean()))
            else:
                max_frac.append(0.0); prob.append(np.nan)
        if not times:
            return []
        times = np.array(times)
        nf = grid_reduce_scalar(times, np.array(n_faces), grid, "mean")
        meta = dict(model="facenet-pytorch-MTCNN(InsightFace-substitute)", version="2.5",
                    native_rate_hz=ingest.analysis_fps, resample="mean", tier="gpu")

        def ch(path, vals, dtype="scalar", **kw):
            return FeatureChannel(path=path, value=vals, dtype=dtype, **{**meta, **kw})
        return [
            ch("visual/faces_bodies_gaze/n_faces", nf, units="count"),
            ch("visual/faces_bodies_gaze/faces_present",
               np.where(np.isnan(nf), np.nan, (nf > 0).astype(float)),
               dtype="bool", resample="any", units="0/1"),
            ch("visual/faces_bodies_gaze/max_face_frac",
               grid_reduce_scalar(times, np.array(max_frac), grid, "mean"), units="0-1"),
            ch("visual/faces_bodies_gaze/face_det_prob",
               grid_reduce_scalar(times, np.array(prob), grid, "mean")),
            ch("social/n_agents", nf, units="count", notes="face count (proxy for agents present)"),
        ]
