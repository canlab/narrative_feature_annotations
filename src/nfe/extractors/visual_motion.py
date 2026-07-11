"""Dense optical flow (torchvision RAFT, MPS/CPU). Per frame-pair -> grid mean.

Channels under features/visual/dynamic_motion/: flow_magnitude, camera_motion
(global median flow), residual_motion (object-motion proxy). Frozen core pick is
SEA-RAFT; torchvision RAFT is the cv2-free stand-in (swap SEA-RAFT for production).
"""

from __future__ import annotations

import numpy as np

from ..base import (VISUAL_MODALITIES, Extractor, FeatureChannel, Stimulus,
                    TimeGrid, grid_reduce_scalar)
from ..torch_util import batched, get_device


class VisualMotion(Extractor):
    feature_class = "visual"
    name = "visual_motion"
    applicable_modalities = VISUAL_MODALITIES
    tier = "gpu"

    def __init__(self, batch_size: int = 8, device: str = "auto"):
        self.batch_size = batch_size
        self.device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        self.torch = torch
        self.dev = get_device(self.device)
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(self.dev).eval()
        self._loaded = True

    def _to_tensor(self, rgb):
        x = self.torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0 * 2 - 1  # [-1,1]
        h, w = x.shape[1:]
        return x[:, : (h // 8) * 8, : (w // 8) * 8]   # RAFT needs dims divisible by 8

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        self._load()
        torch = self.torch
        frames = list(ingest.iter_frames())
        if len(frames) < 2:
            return []
        pairs = list(zip(frames[:-1], frames[1:]))
        times, mag, cam, resid = [], [], [], []
        for chunk in batched(pairs, self.batch_size):
            a = torch.stack([self._to_tensor(f0[1]) for f0, _ in chunk]).to(self.dev)
            b = torch.stack([self._to_tensor(f1[1]) for _, f1 in chunk]).to(self.dev)
            with torch.no_grad():
                fl = self.model(a, b)[-1].cpu().numpy()         # [N,2,H,W]
            u, v = fl[:, 0], fl[:, 1]
            m = np.hypot(u, v)
            mu = np.median(u.reshape(len(chunk), -1), axis=1)
            mv = np.median(v.reshape(len(chunk), -1), axis=1)
            res = np.hypot(u - mu[:, None, None], v - mv[:, None, None]).mean(axis=(1, 2))
            for k, (_, f1) in enumerate(chunk):
                times.append(f1[0]); mag.append(float(m[k].mean()))
                cam.append(float(np.hypot(mu[k], mv[k]))); resid.append(float(res[k]))
        times = np.array(times)
        meta = dict(model="torchvision-RAFT(SEA-RAFT-substitute)", version="raft_small",
                    native_rate_hz=ingest.analysis_fps, resample="mean", tier="gpu", units="px")

        def ch(name, vals, note=""):
            return FeatureChannel(path=f"visual/dynamic_motion/{name}",
                                  value=grid_reduce_scalar(times, np.array(vals), grid, "mean"),
                                  dtype="scalar", notes=note, **meta)
        return [ch("flow_magnitude", mag, "mean optical-flow magnitude"),
                ch("camera_motion", cam, "global median flow magnitude (camera proxy)"),
                ch("residual_motion", resid, "mean |flow - median| (object-motion proxy)")]
