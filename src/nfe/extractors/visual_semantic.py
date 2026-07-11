"""High-level static visual semantics: SigLIP2 + DINOv2 (PyTorch, MPS/CPU).

Per-frame -> grid mean:
  visual/high_level_static/siglip_embedding   image embedding (opaque)
  visual/high_level_static/siglip_probe        sigmoid scores over a prompt set
  visual/high_level_static/dino_embedding      label-free DINOv2 CLS embedding

Frozen core picks: SigLIP 2 (interpretable text-probed) and DINOv2 (RSA/probes) — kept
as distinct representations. Default checkpoints are the small/base tier for iteration;
swap to siglip2-so400m / dinov2-large for production.
"""

from __future__ import annotations

import numpy as np

from ..base import (VISUAL_MODALITIES, Extractor, FeatureChannel, Stimulus,
                    TimeGrid, grid_reduce_scalar)
from ..torch_util import batched, get_device

DEFAULT_PROMPTS = [
    "people", "a single person", "a crowd of people", "a human face close-up",
    "an indoor scene", "an outdoor natural landscape", "a city or street scene",
    "an animal", "a vehicle", "food", "text or writing on screen",
    "a dark, low-light scene", "a bright and colorful scene",
    "fast motion or action", "a calm, static scene", "water or sky",
]


class VisualSemantic(Extractor):
    feature_class = "visual"
    name = "visual_semantic"
    applicable_modalities = VISUAL_MODALITIES
    tier = "gpu"

    def __init__(self, siglip="google/siglip2-base-patch16-224",
                 dino="facebook/dinov2-small", prompts=None,
                 batch_size: int = 16, device: str = "auto"):
        self.siglip_name = siglip
        self.dino_name = dino
        self.prompts = list(prompts) if prompts is not None else DEFAULT_PROMPTS
        self.batch_size = batch_size
        self.device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor
        self.torch = torch
        self.dev = get_device(self.device)
        self.sig = AutoModel.from_pretrained(self.siglip_name).to(self.dev).eval()
        self.sig_proc = AutoProcessor.from_pretrained(self.siglip_name)
        self.dino = AutoModel.from_pretrained(self.dino_name).to(self.dev).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained(self.dino_name)
        # Precompute normalized text embeddings ONCE (verified numerically identical to
        # the combined forward); avoids re-running the text tower on every frame batch.
        # transformers 5.x: get_text_features returns an output object -> pooler_output.
        tin = self.sig_proc(text=self.prompts, padding="max_length", return_tensors="pt").to(self.dev)
        with torch.no_grad():
            t = self.sig.get_text_features(**tin)
        t = t if torch.is_tensor(t) else t.pooler_output
        self.text_emb = torch.nn.functional.normalize(t, dim=-1)
        self._loaded = True

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        from PIL import Image
        self._load()
        torch = self.torch
        times, sig_emb, probes, dino_emb = [], [], [], []
        for batch in batched(ingest.iter_frames(), self.batch_size):
            ts = [t for t, _ in batch]
            imgs = [Image.fromarray(rgb) for _, rgb in batch]
            with torch.no_grad():
                spx = self.sig_proc(images=imgs, return_tensors="pt").to(self.dev)
                ie = self.sig.get_image_features(**spx)
                ie = ie if torch.is_tensor(ie) else ie.pooler_output
                ie_n = torch.nn.functional.normalize(ie, dim=-1)
                logits = ie_n @ self.text_emb.T * self.sig.logit_scale.exp() + self.sig.logit_bias
                p = logits.sigmoid()                            # [B, n_prompts]
                dpx = self.dino_proc(images=imgs, return_tensors="pt").to(self.dev)
                dcls = self.dino(**dpx).pooler_output
            times.extend(ts)
            sig_emb.append(ie_n.float().cpu().numpy())
            probes.append(p.float().cpu().numpy())
            dino_emb.append(torch.nn.functional.normalize(dcls, dim=-1).float().cpu().numpy())
        if not times:
            return []
        times = np.array(times)
        sig_emb = np.concatenate(sig_emb)
        probes = np.concatenate(probes)
        dino_emb = np.concatenate(dino_emb)
        meta = dict(version="hf", native_rate_hz=ingest.analysis_fps, resample="mean", tier="gpu")
        return [
            FeatureChannel(path="visual/high_level_static/siglip_embedding",
                           value=grid_reduce_scalar(times, sig_emb, grid, "mean"),
                           dtype="vector", components=[f"d{i}" for i in range(sig_emb.shape[1])],
                           model=self.siglip_name, notes="L2-normalized image embedding", **meta),
            FeatureChannel(path="visual/high_level_static/siglip_probe",
                           value=grid_reduce_scalar(times, probes, grid, "mean"),
                           dtype="vector", components=list(self.prompts),
                           model=self.siglip_name, units="sigmoid",
                           notes="independent per-prompt probabilities", **meta),
            FeatureChannel(path="visual/high_level_static/dino_embedding",
                           value=grid_reduce_scalar(times, dino_emb, grid, "mean"),
                           dtype="vector", components=[f"d{i}" for i in range(dino_emb.shape[1])],
                           model=self.dino_name, notes="L2-normalized DINOv2 CLS embedding", **meta),
        ]
