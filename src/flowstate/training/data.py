"""AG News dataset loading and tokenization.

Kept deliberately small: tokenization + a stratified train/validation split
carved from the original train partition. The original `test` split is
reserved for the final held-out evaluation reported in `eval_report.json`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from datasets import DatasetDict, load_dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


AG_NEWS_LABELS: tuple[str, ...] = ("World", "Sports", "Business", "Sci/Tech")


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "ag_news"
    max_seq_len: int = 128
    validation_fraction: float = 0.1
    seed: int = 42


def load_and_tokenize(
    tokenizer: PreTrainedTokenizerBase, cfg: DataConfig
) -> DatasetDict:
    """Return a DatasetDict with keys {train, validation, test}.

    AG News ships train (120k) + test (7.6k) only, so we stratify-split the
    train partition to produce a validation set used for checkpoint selection.
    The original test set is kept untouched for the final report.
    """
    raw = load_dataset(cfg.dataset_name)

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_seq_len,
        )

    tokenized = raw.map(_tokenize, batched=True, remove_columns=["text"])

    split = tokenized["train"].train_test_split(
        test_size=cfg.validation_fraction,
        seed=cfg.seed,
        stratify_by_column="label",
    )
    return DatasetDict(
        train=split["train"],
        validation=split["test"],
        test=tokenized["test"],
    )
