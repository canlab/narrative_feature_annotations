"""Torch helpers shared by GPU/MPS extractors (Apple Silicon -> MPS, else CPU)."""

from __future__ import annotations

from itertools import islice
from typing import Iterable, Iterator


def get_device(prefer: str = "auto") -> str:
    import torch
    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def batched(it: Iterable, n: int) -> Iterator[list]:
    """Yield lists of up to n items from an iterable."""
    iterator = iter(it)
    while True:
        chunk = list(islice(iterator, n))
        if not chunk:
            return
        yield chunk
