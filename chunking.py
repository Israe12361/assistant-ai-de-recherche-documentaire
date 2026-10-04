from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from models import Chunk, ExtractedBlock

_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ0-9«])")
_HEADING_RE = re.compile(
    r"^(?:[0-9]+(?:\.[0-9]+)*[.)]?\s+)?[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ][^.!?]{2,140}$"
)
_SECTION_RE = re.compile(
    r"^(?:SECTION|CHAPTER|CHAPITRE|PART|PARTIE|ANNEX|ANNEXE|APPENDIX|APPENDICE)\b",
    re.IGNORECASE,
)
_NUMBERED_RE = re.compile(
    r"^(?:\(?\d+(?:[.)]|(?:\.\d+)+)|[A-Za-z][.)]|[IVXLCDM]+[.)])\s+",
    re.IGNORECASE,
)
_VERB_LIKE_RE = re.compile(
    r"\b(?:is|are|was|were|will|would|has|have|had|does|did|can|could|should|must|"
    r"creates?|executes?|confirms?|informs?|requests?|schedules?|validates?|"
    r"communicates?|performs?|reviews?|follows?|provides?|contacts?)\b",
    re.IGNORECASE,
)

_MIN_REPEATS_FOR_SUBHEADING = 2


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [part.strip() for part in _SENTENCE_RE.split(normalized) if part.strip()]


def _is_heading(text: str) -> bool:
    compact = normalize_text(text)
    if len(compact) > 140 or not _HEADING_RE.match(compact):
        return False
    return not _VERB_LIKE_RE.search(compact)


def _is_numbered_item(text: str) -> bool:
    return bool(_NUMBERED_RE.match(normalize_text(text)))


def _heading_key(text: str) -> str:
    return normalize_text(text).casefold().strip(" :-–—")


def _is_top_level_heading(text: str) -> bool:
    return bool(_SECTION_RE.match(normalize_text(text)))


def _infer_section(text: str) -> str | None:
    match = re.search(
        r"\b(?:SECTION|CHAPTER|CHAPITRE|PART|PARTIE|ANNEX|ANNEXE|APPENDIX|APPENDICE)\s+"
        r"([0-9A-Za-z.-]+)\s*:?\s*([^\n]+)",
        text,
        re.IGNORECASE,
    )
    return match.group(0).strip() if match else None


def _collect_recurring_headings(blocks: list[ExtractedBlock]) -> set[str]:
    counts: Counter[str] = Counter()
    for block in blocks:
        if block.block_type != "narrative":
            continue
        for line in block.text.splitlines():
            line = normalize_text(line)
            if line and _is_heading(line):
                counts[_heading_key(line)] += 1
    return {heading for heading, count in counts.items() if count >= _MIN_REPEATS_FOR_SUBHEADING}


def _cosine(a: Any, b: Any) -> float:
    numerator = float((a * b).sum())
    denominator = float((a * a).sum() ** 0.5 * (b * b).sum() ** 0.5)
    return numerator / denominator if denominator else 0.0


def _scope_metadata(
    block: ExtractedBlock,
    hierarchy: dict[str, Any],
    subsection: str | None = None,
) -> dict[str, Any]:
    metadata = dict(block.metadata or {})
    metadata.update(
        {
            "document_name": Path(block.document_path).name,
            "source_page": block.page_number,
            "source_type": block.block_type,
            "section": hierarchy.get("section"),
            "parent_heading": hierarchy.get("parent_heading"),
            "subsection": subsection or hierarchy.get("subsection"),
            "heading": hierarchy.get("heading"),
            "table_id": block.table_id,
            "page_start": block.page_number,
            "page_end": block.page_number,
        }
    )
    return {key: value for key, value in metadata.items() if value is not None}


def _scope_prefix(metadata: dict[str, Any]) -> str:
    parts = [
        f"{label}={metadata[label]}"
        for label in ("section", "parent_heading", "subsection")
        if metadata.get(label)
    ]
    return f"[{' | '.join(parts)}] " if parts else ""


def _all_headings(units: list[str]) -> bool:
    return all(_is_heading(u) for u in units)


class SemanticChunker:
    def __init__(
        self,
        min_chars: int = 700,
        target_chars: int = 1400,
        max_chars: int = 2200,
        similarity_floor: float = 0.48,
        overlap_sentences: int = 0,
    ) -> None:
        if not 0 < min_chars <= target_chars <= max_chars:
            raise ValueError("Les tailles doivent respecter min <= cible <= max.")
        self.min_chars = min_chars
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.similarity_floor = similarity_floor
        self.overlap_sentences = overlap_sentences

    def _semantic_breaks(self, units: list[str], embedder: Any | None) -> set[int]:
        if not embedder or len(units) < 2:
            return set()
        probes = []
        for index in range(len(units) - 1):
            left = " ".join(units[max(0, index - 1) : index + 1])
            right = " ".join(units[index + 1 : index + 3])
            probes.extend([left, right])
        vectors = embedder.encode(probes, normalize_embeddings=True)
        similarities = [_cosine(vectors[i], vectors[i + 1]) for i in range(0, len(vectors), 2)]
        if not similarities:
            return set()
        sorted_values = sorted(similarities)
        median = sorted_values[len(sorted_values) // 2]
        threshold = max(self.similarity_floor, median - 0.13)
        return {index + 1 for index, value in enumerate(similarities) if value < threshold}

    def _make_chunks(
        self,
        units: list[str],
        block: ExtractedBlock,
        start_index: int,
        hierarchy: dict[str, Any],
        subsection: str | None,
        embedder: Any | None,
    ) -> list[Chunk]:
        if not units:
            return []
        breaks = self._semantic_breaks(units, embedder)
        groups: list[list[str]] = []
        group: list[str] = []
        group_len = 0
        for index, unit in enumerate(units):
            in_numbered_run = (
                _is_numbered_item(unit)
                and index + 1 < len(units)
                and (_is_numbered_item(units[index + 1]) or not _is_heading(units[index + 1]))
            )
            if group and (index in breaks and group_len >= self.min_chars) and not in_numbered_run:
                groups.append(group)
                group, group_len = [], 0
            if group and group_len + len(unit) + 1 > self.max_chars:
                groups.append(group)
                group, group_len = [], 0
            group.append(unit)
            group_len += len(unit) + (1 if group_len else 0)
            reached_target = group_len >= self.target_chars and index + 1 < len(units)
            if reached_target and not in_numbered_run:
                groups.append(group)
                group, group_len = [], 0
        if group:
            groups.append(group)

        chunks: list[Chunk] = []
        for offset, group in enumerate(groups):
            metadata = _scope_metadata(block, hierarchy, subsection)
            text = _scope_prefix(metadata) + " ".join(group).strip()
            chunks.append(
                Chunk(
                    document_id=None,
                    chunk_index=start_index + offset,
                    page_start=block.page_number,
                    page_end=block.page_number,
                    text=text,
                    chunk_type=block.block_type,
                    metadata=metadata,
                )
            )
        return chunks

    def chunk(self, blocks: Iterable[ExtractedBlock], embedder: Any | None = None) -> list[Chunk]:
        blocks = list(blocks)
        recurring_headings = _collect_recurring_headings(blocks)
        result: list[Chunk] = []
        hierarchy_by_document: dict[str, dict[str, Any]] = {}

        for block in blocks:
            text = normalize_text(block.text)
            if not text:
                continue
            hierarchy = hierarchy_by_document.setdefault(
                block.document_path,
                {"section": None, "parent_heading": None, "subsection": None, "heading": None},
            )

            if block.block_type != "narrative":
                table_title = (block.metadata or {}).get("table_title", "")
                metadata = _scope_metadata(block, hierarchy)
                prefix = _scope_prefix(metadata)
                prefixed_text = f"{prefix}[{table_title}] {text}" if table_title else f"{prefix}{text}"
                result.append(
                    Chunk(
                        document_id=None,
                        chunk_index=len(result),
                        page_start=block.page_number,
                        page_end=block.page_number,
                        text=prefixed_text,
                        chunk_type=block.block_type,
                        metadata=metadata,
                    )
                )
                continue

            units: list[str] = []
            active_subsection = hierarchy.get("subsection")

            def flush() -> None:
                nonlocal units
                if units:
                    result.extend(
                        self._make_chunks(units, block, len(result), dict(hierarchy), active_subsection, embedder)
                    )
                    units = []

            for raw_line in block.text.splitlines() or [text]:
                line = normalize_text(raw_line)
                if not line:
                    continue
                if _is_heading(line):
                    key = _heading_key(line)
                    is_recurring = key in recurring_headings
                    should_flush = bool(units) and not _all_headings(units)
                    if _is_top_level_heading(line) or not is_recurring:
                        if should_flush:
                            flush()
                            units = []
                        if _is_top_level_heading(line):
                            hierarchy["section"] = _infer_section(line) or line
                        hierarchy["parent_heading"] = line
                        hierarchy["heading"] = line
                        hierarchy["subsection"] = None
                        active_subsection = None
                        units.append(line)
                    else:
                        if should_flush:
                            flush()
                            units = []
                        active_subsection = line
                        hierarchy["subsection"] = line
                        hierarchy["heading"] = line
                        units.append(line)
                else:
                    units.extend(split_sentences(line) or [line])
            flush()

        for index, chunk in enumerate(result):
            chunk.chunk_index = index
        return result