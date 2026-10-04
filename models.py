from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExtractedBlock:
    document_path: str
    page_number: int
    text: str
    block_type: str = "narrative"
    table_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    document_id: int | None
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    chunk_type: str = "narrative"
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Any | None = None

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class SearchResult:
    chunk: Chunk
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    distance: float | None = None


@dataclass(slots=True)
class Answer:
    text: str
    sources: list[SearchResult]
    refused: bool = False
    latency_ms: dict[str, float] = field(default_factory=dict)
