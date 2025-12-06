"""
Data loading and preprocessing for Atlas.

This module handles dataset loading, tokenization, batching,
and data preprocessing pipelines.
"""

from atlas.data.dataset import TextDataset
from atlas.data.loader import (
    create_dataloader,
    collate_batch,
    get_dataloader_stats,
    split_dataset,
)
from atlas.data.preprocessing import (
    clean_text,
    chunk_text,
    load_text_file,
    load_jsonl,
    iterate_documents,
    count_tokens,
    filter_by_length,
)

__all__ = [
    "TextDataset",
    "create_dataloader",
    "collate_batch",
    "get_dataloader_stats",
    "split_dataset",
    "clean_text",
    "chunk_text",
    "load_text_file",
    "load_jsonl",
    "iterate_documents",
    "count_tokens",
    "filter_by_length",
]
