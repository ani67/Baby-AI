"""Per-mind file paths.

Every mind lives in its own directory under data/. Paths derive from a
mind_name string. Multiple minds can run concurrently in separate
processes without sharing any filesystem state.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass


_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


@dataclass(frozen=True)
class MindPaths:
    mind_name: str

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.mind_name):
            raise ValueError(
                f"mind_name must match {_NAME_RE.pattern!r}; got {self.mind_name!r}"
            )

    @property
    def root(self) -> str:               return f"data/{self.mind_name}"
    @property
    def db(self) -> str:                 return f"{self.root}/mind.db"
    @property
    def language_head(self) -> str:      return f"{self.root}/language_head.pt"
    @property
    def vocab(self) -> str:              return f"{self.root}/language_head_vocab.json"
    @property
    def surprised_log(self) -> str:      return f"{self.root}/surprised_sentences.jsonl"
    @property
    def book_text_log(self) -> str:      return f"{self.root}/last_book.txt"
    @property
    def ingest_progress(self) -> str:    return f"{self.root}/ingest_progress.json"
    @property
    def curriculum_progress(self) -> str: return f"{self.root}/curriculum_progress.json"
    @property
    def dialogue_log(self) -> str:       return f"{self.root}/dialogue.jsonl"

    def ensure_dirs(self) -> None:
        os.makedirs(self.root, exist_ok=True)
