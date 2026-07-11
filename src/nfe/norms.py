"""Psycholinguistic norm tables (optional, licensing-clean).

Affective and lexical norms (Warriner VAD, Brysbaert concreteness, Kuperman AoA)
are not redistributed here. Drop CSVs into data/lexicons/ as `<field>.csv` with a
`word,value` header and they light up automatically; absent tables yield NaN, so
the annotation shape is unchanged. See data/lexicons/README.md for sourcing.
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path

FIELDS = ("valence", "arousal", "dominance", "concreteness", "aoa")


class Norms:
    def __init__(self, lexdir: str = "data/lexicons"):
        self.lexdir = Path(lexdir)
        self.tables: dict[str, dict[str, float]] = {}
        for field in FIELDS:
            f = self.lexdir / f"{field}.csv"
            if f.exists():
                self.tables[field] = self._load(f)

    @staticmethod
    def _load(path: Path) -> dict[str, float]:
        out: dict[str, float] = {}
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) < 2:
                    continue
                w, v = row[0].strip().lower(), row[1].strip()
                try:
                    out[w] = float(v)
                except ValueError:
                    continue   # header / non-numeric
        return out

    def available(self) -> list[str]:
        return sorted(self.tables)

    def get(self, word: str, field: str) -> float:
        return self.tables.get(field, {}).get(word.lower(), math.nan)


@lru_cache(maxsize=8)
def load_norms(lexdir: str = "data/lexicons") -> Norms:
    return Norms(lexdir)
