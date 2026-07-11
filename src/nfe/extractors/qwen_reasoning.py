"""Consolidated high-level reasoning (Qwen2.5-VL, MPS/CPU).

One VLM pass per time window over sampled frames, returning a unified JSON that
populates Social, Situation, and Affect channels at once (the core-set design's
single video-LLM pass). Each window's fields are assigned to the grid bins it spans.

Frozen core pick: Qwen2.5-VL-7B; default here is the 3B instruct model for MPS
tractability (set model_name to the 7B for production). Slow — this is the heaviest
pass; tune window_sec / frames_per_window / max_windows.
"""

from __future__ import annotations

import json
import re

import numpy as np

from ..base import (VISUAL_MODALITIES, Extractor, FeatureChannel, Stimulus, TimeGrid)
from ..torch_util import get_device

PROMPT = (
    "Analyze this short movie moment from the given frames. Respond with ONLY a compact JSON "
    "object, no prose, with exactly these keys:\n"
    '{"scene_description": "<one short sentence>", "setting": "<short location>", '
    '"indoor_outdoor": "indoor|outdoor|unknown", '
    '"interaction_type": "none|conversation|cooperation|conflict|affiliation|other", '
    '"dominance": <0..1 how dominant/assertive the main agent is>, '
    '"depicted_emotion": "<one word>", "valence": <-1..1>, "arousal": <0..1>}'
)
LABEL_KEYS = {"scene_description": "situation/scene_description",
              "setting": "situation/setting",
              "indoor_outdoor": "situation/indoor_outdoor",
              "interaction_type": "social/interaction_type",
              "depicted_emotion": "affect/depicted/vlm_emotion"}
SCALAR_KEYS = {"dominance": "social/dominance",
               "valence": "affect/depicted/vlm_valence",
               "arousal": "affect/depicted/vlm_arousal"}


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


class QwenReasoning(Extractor):
    feature_class = "social"
    name = "qwen_reasoning"
    applicable_modalities = VISUAL_MODALITIES
    tier = "vlm"

    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", window_sec: float = 5.0,
                 frames_per_window: int = 3, max_new_tokens: int = 256,
                 max_windows: int | None = None, device: str = "auto"):
        self.model_name = model_name
        self.window_sec = window_sec
        self.frames_per_window = frames_per_window
        self.max_new_tokens = max_new_tokens
        self.max_windows = max_windows
        self.device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.dev = get_device(self.device)
        self.proc = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.float16, attn_implementation="eager"
        ).to(self.dev).eval()
        self._loaded = True

    def _ask(self, pils) -> dict:
        from qwen_vl_utils import process_vision_info
        torch = self.torch
        messages = [{"role": "user", "content": [{"type": "image", "image": im} for im in pils]
                     + [{"type": "text", "text": PROMPT}]}]
        text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.proc(text=[text], images=image_inputs, padding=True,
                           return_tensors="pt").to(self.dev)
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        return _parse(self.proc.batch_decode(trimmed, skip_special_tokens=True)[0])

    def extract(self, stim: Stimulus, grid: TimeGrid, ingest) -> list[FeatureChannel]:
        from PIL import Image
        self._load()
        frames = list(ingest.iter_frames())
        if not frames:
            return []
        ft = np.array([t for t, _ in frames])
        labels = {p: [""] * grid.n_samples for p in LABEL_KEYS.values()}
        scalars = {p: np.full(grid.n_samples, np.nan) for p in SCALAR_KEYS.values()}

        n_win = int(np.ceil(stim.duration_sec / self.window_sec))
        if self.max_windows:
            n_win = min(n_win, self.max_windows)
        for w in range(n_win):
            t0, t1 = w * self.window_sec, (w + 1) * self.window_sec
            idx = np.where((ft >= t0) & (ft < t1))[0]
            if idx.size == 0:
                continue
            pick = idx[np.linspace(0, idx.size - 1, min(self.frames_per_window, idx.size)).astype(int)]
            pils = [Image.fromarray(frames[i][1]) for i in pick]
            try:
                d = self._ask(pils)
            except Exception:
                continue
            b0, b1 = int(grid.bin_index(t0)), int(grid.bin_index(min(t1, stim.duration_sec)))
            bins = range(max(b0, 0), min(b1, grid.n_samples - 1) + 1)
            for key, path in LABEL_KEYS.items():
                if key in d:
                    for b in bins:
                        labels[path][b] = str(d[key])[:200]
            for key, path in SCALAR_KEYS.items():
                try:
                    val = float(d[key])
                    for b in bins:
                        scalars[path][b] = val
                except (KeyError, TypeError, ValueError):
                    pass

        meta = dict(model=self.model_name, version="instruct",
                    native_rate_hz=f"window~{self.window_sec}s", tier="vlm")
        out = []
        for path, vals in labels.items():
            out.append(FeatureChannel(path=path, value=np.array([]), labels=vals, dtype="label",
                                      resample="vlm", **meta))
        for path, vals in scalars.items():
            out.append(FeatureChannel(path=path, value=vals, dtype="scalar", resample="vlm", **meta))
        return out
