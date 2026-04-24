"""Tokenizer wrapper with a fixed, per-batch padding policy.

Dynamic padding (`padding=True`) pads each batch to the longest sequence in
that batch rather than to `max_length`. This is the canonical throughput
optimization for variable-length inputs: short batches don't pay for
padding they don't need.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
from transformers import AutoTokenizer


class TokenizedBatch(TypedDict):
    input_ids: np.ndarray
    attention_mask: np.ndarray


class CachedTokenizer:
    """Loads a HF fast tokenizer once and reuses it across requests."""

    def __init__(self, tokenizer_dir: Path, max_seq_len: int = 128) -> None:
        self._tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
        self._max_seq_len = max_seq_len

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def encode(self, texts: list[str]) -> TokenizedBatch:
        enc = self._tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_seq_len,
            return_tensors="np",
        )
        return {
            "input_ids": enc["input_ids"].astype(np.int64, copy=False),
            "attention_mask": enc["attention_mask"].astype(np.int64, copy=False),
        }
