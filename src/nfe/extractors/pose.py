"""Body pose (torchvision Keypoint R-CNN, MPS/CPU). Per frame -> grid mean.

Channels: features/visual/faces_bodies_gaze/{n_persons, pose_present, mean_kp_score}
and features/social/min_pair_distance (proximity proxy). Frozen core pick is MMPose
RTMPose (133-kpt whole-body); torchvision Keypoint R-CNN (COCO-17) is the cv2-free,
dependency-light stand-in for person count / proximity.
"""

from __future__ import annotations

import numpy as np

from ..base import (VISUAL_MODALITIES, Extractor, FeatureChannel, Stimulus,
                    TimeGrid, grid_reduce_scalar)
from ..torch_util import get_device


class Pose(Extractor):
    feature_class = "visual"
    name = "pose"
    applicable_modalities = VISUAL_MODALITIES
    tier = "gpu"

    def __init__(self, min_score: float = 0.7, device: str = "auto"):
        self.min_score = min_score
        self.device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from torchvision.models.detection import (KeypointRCNN_ResNet50_FPN_Weights,
                                                   keypointrcnn_resnet50_fpn)
        self.torch = torch
        self.dev = get_device(self.device)
        self.model = keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(self.dev).eval()
        self._loaded = True

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        self._load()
        torch = self.torch
        times, n_persons, kp_score, min_dist = [], [], [], []
        for t, rgb in ingest.iter_frames():
            H, W = rgb.shape[:2]
            x = torch.from_numpy(rgb).permute(2, 0, 1).float().to(self.dev) / 255.0
            with torch.no_grad():
                out = self.model([x])[0]
            keep = out["scores"].cpu().numpy() >= self.min_score
            boxes = out["boxes"].cpu().numpy()[keep]
            kps = out["keypoints_scores"].cpu().numpy()[keep]
            times.append(t)
            n = len(boxes)
            n_persons.append(float(n))
            kp_score.append(float(kps.mean()) if n else np.nan)
            if n >= 2:
                cx = (boxes[:, 0] + boxes[:, 2]) / 2
                cy = (boxes[:, 1] + boxes[:, 3]) / 2
                d = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
                d[np.diag_indices(n)] = np.inf
                min_dist.append(float(d.min() / np.hypot(W, H)))   # normalized
            else:
                min_dist.append(np.nan)
        if not times:
            return []
        times = np.array(times)
        npn = grid_reduce_scalar(times, np.array(n_persons), grid, "mean")
        meta = dict(native_rate_hz=ingest.analysis_fps, resample="mean", tier="gpu")
        vmodel = "torchvision-KeypointRCNN(RTMPose-substitute)"

        def ch(path, vals, dtype="scalar", model=vmodel, **kw):
            return FeatureChannel(path=path, value=vals, dtype=dtype, model=model,
                                  version="coco17", **{**meta, **kw})
        return [
            ch("visual/faces_bodies_gaze/n_persons", npn, units="count"),
            ch("visual/faces_bodies_gaze/pose_present",
               np.where(np.isnan(npn), np.nan, (npn > 0).astype(float)),
               dtype="bool", resample="any", units="0/1"),
            ch("visual/faces_bodies_gaze/mean_kp_score",
               grid_reduce_scalar(times, np.array(kp_score), grid, "mean")),
            ch("social/min_pair_distance",
               grid_reduce_scalar(times, np.array(min_dist), grid, "mean"),
               units="0-1", notes="nearest person-pair distance / frame diagonal; NaN if <2 persons"),
        ]
