from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from models import Chunk, SearchResult

_TOKEN_RE = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)

# Expansion lexicale générique pour les questions françaises sur des documents anglais.
_BILINGUAL_TERMS = {
    "urgence": "emergency",
    "urgente": "emergency",
    "modification": "change",
    "changements": "changes",
    "approbation": "approval approved",
    "approuver": "approve approval",
    "demande": "request",
    "immédiate": "immediate",
    "délai": "time deadline",
    "impact": "impact",
    "risque": "risk",
    "complexe": "complex complexly",
    "perturbation": "disruption",
    "opérations": "operations",
    "entreprise": "business",
    "notification": "notification",
    "utilisateurs": "users",
    "affectés": "affected",
    "saisie": "input",
    "validation": "validation validate",
    "erreur": "error errors",
    "documentation": "documentation",
    "documenter": "document",
    "retour": "rollback",
    "secours": "contingency",
    "application": "application",
    "infrastructure": "infrastructure",
    "mettre": "implement",
    "œuvre": "implementation",
    "clôture": "closure close",
}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Any,
        candidate_k: int = 50,
        final_k: int = 4,
        neighbor_radius: int = 1,
        rrf_k: int = 60,
    ) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.candidate_k = max(1, candidate_k)
        self.final_k = max(1, final_k)
        self.neighbor_radius = max(0, neighbor_radius)
        self.rrf_k = max(1, rrf_k)

        if not chunks:
            self.matrix = np.empty((0, 0), dtype=np.float32)
            self._tokens: list[list[str]] = []
            self._bm25 = None
            self._by_document = defaultdict(list)
            return

        self.matrix = np.asarray([chunk.embedding for chunk in chunks], dtype=np.float32)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != len(chunks):
            raise ValueError("Les chunks doivent avoir des embeddings de même dimension.")
        self._tokens = [tokenize(chunk.text) for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Installez rank-bm25 pour activer la recherche hybride.") from exc
        self._bm25 = BM25Okapi(self._tokens)

        self._by_document: dict[int, list[int]] = defaultdict(list)
        self._by_page: dict[tuple[int | None, int], list[int]] = defaultdict(list)
        for index, chunk in enumerate(chunks):
            self._by_page[(chunk.document_id, chunk.page_start)].append(index)
            if chunk.document_id is not None:
                self._by_document[chunk.document_id].append(index)
        for indices in self._by_document.values():
            indices.sort(key=lambda idx: self.chunks[idx].chunk_index)

        self._page_terms = {
            page: {term for index in indices for term in self._tokens[index]}
            for page, indices in self._by_page.items()
        }
        page_count = max(1, len(self._page_terms))
        page_frequency = Counter(term for terms in self._page_terms.values() for term in terms)
        self._page_idf = {
            term: np.log((1.0 + page_count) / (1.0 + frequency)) + 1.0
            for term, frequency in page_frequency.items()
        }

    def _query_variants(self, query: str) -> list[str]:
        query = query.strip()
        variants = [query]
        translated_terms = []
        for term in tokenize(query):
            translation = _BILINGUAL_TERMS.get(term)
            if translation:
                translated_terms.append(translation)
        if translated_terms:
            variants.append(f"{query} {' '.join(translated_terms)}")

        clauses = re.split(r"\s+(?:et|ou|ainsi que)\s+|[,;:]", query, flags=re.IGNORECASE)
        for clause in clauses:
            clause = clause.strip(" ?. !\n\t")
            if len(tokenize(clause)) >= 3 and clause not in variants:
                variants.append(clause)
        return variants[:4]

    def _vector_ranking(self, query: str) -> list[tuple[int, float]]:
        vector = np.asarray(self.embedder.encode_one(query), dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return []
        vector = vector / norm
        matrix_norms = np.linalg.norm(self.matrix, axis=1)
        safe = np.where(matrix_norms == 0, 1.0, matrix_norms)
        scores = (self.matrix @ vector) / safe
        order = np.argsort(-scores)[: self.candidate_k]
        return [(int(index), float(scores[index])) for index in order]

    def _bm25_ranking(self, query: str) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
        order = np.argsort(-scores)[: self.candidate_k]
        return [(int(index), float(scores[index])) for index in order]

    def _page_term_ranking(self, query: str) -> list[tuple[tuple[int | None, int], float]]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return []
        ranked = []
        for page, page_terms in self._page_terms.items():
            score = sum(
                (2.0 if any(character.isdigit() for character in term) else 1.0)
                * self._page_idf.get(term, 1.0)
                for term in query_terms
                if term in page_terms
            )
            if score:
                ranked.append((page, score))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def _exact_term_ranking(self, query: str) -> list[tuple[int, float]]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        def score(tokens: list[str]) -> float:
            document_terms = set(tokens)
            value = sum(
                2.0 if any(character.isdigit() for character in term) else 1.0
                for term in query_terms
                if term in document_terms
            )
            return value / max(1.0, len(query_terms) ** 0.5)

        scores = np.asarray([score(tokens) for tokens in self._tokens], dtype=np.float32)
        order = np.argsort(-scores)[: self.candidate_k]
        return [(int(index), float(scores[index])) for index in order if scores[index] > 0]

    @staticmethod
    def _is_numeric_query(query: str) -> bool:
        lowered = query.casefold()
        return bool(re.search(r"\d|pourcentage|taux|combien|nombre|chiffre|montant|hausse|baisse", lowered))

    @staticmethod
    def _is_explanatory_query(query: str) -> bool:
        lowered = query.casefold().strip()
        return lowered.startswith((
            "pourquoi", "comment", "explique", "décris", "décrivez",
            "quels facteurs", "quelles sont les raisons", "quel est l'effet",
        )) or any(term in lowered for term in ("causes de", "raisons de", "impact de", "effets de"))

    @staticmethod
    def _needs_continuation(text: str) -> bool:
        compact = text.strip().casefold()
        return (
            len(compact) < 850
            or compact.endswith((":", ";", ",", "-", "…"))
            or bool(re.search(r"\b(notamment|cependant|par ailleurs|en revanche|ainsi)\s*$", compact))
        )

    @staticmethod
    def _same_scope(first: Chunk, second: Chunk) -> bool:
        # Si les deux chunks possèdent une métadonnée hiérarchique, elle doit
        # rester identique. En l’absence de métadonnée, on ne bloque pas l’ancien corpus.
        for key in ("parent_heading", "section", "subsection", "topic", "system"):
            left = str(first.metadata.get(key) or "").strip().casefold()
            right = str(second.metadata.get(key) or "").strip().casefold()
            if left and right and left != right:
                return False
        return first.chunk_type == second.chunk_type

    def _adaptive_radius(self, query: str, seed: SearchResult, seed_position: int) -> int:
        if self.neighbor_radius <= 0 or seed.chunk.chunk_type == "table_row":
            return 0
        if seed_position > 0:
            return 0
        if self._is_numeric_query(query):
            compact = seed.chunk.text.strip().casefold()
            incomplete = compact.endswith((":", ";", ",", "-", "…"))
            return min(1, self.neighbor_radius) if incomplete else 0
        if self._is_explanatory_query(query):
            return min(1, self.neighbor_radius)
        if self._needs_continuation(seed.chunk.text):
            return min(1, self.neighbor_radius)
        return 0

    def _expand(self, seeds: list[SearchResult], query: str) -> list[SearchResult]:
        selected: dict[tuple[int | None, int], SearchResult] = {
            (item.chunk.document_id, item.chunk.chunk_index): item for item in seeds
        }
        for seed_position, seed in enumerate(seeds):
            radius = self._adaptive_radius(query, seed, seed_position)
            if radius <= 0 or seed.chunk.document_id is None:
                continue
            indices = self._by_document.get(seed.chunk.document_id, [])
            position = next(
                (i for i, idx in enumerate(indices) if self.chunks[idx].chunk_index == seed.chunk.chunk_index),
                None,
            )
            if position is None:
                continue
            for neighbor_index in indices[max(0, position - radius) : position + radius + 1]:
                neighbor = self.chunks[neighbor_index]
                if not self._same_scope(seed.chunk, neighbor):
                    continue
                key = (neighbor.document_id, neighbor.chunk_index)
                if key not in selected:
                    selected[key] = SearchResult(
                        chunk=neighbor,
                        rrf_score=seed.rrf_score * 0.85,
                    )
        return list(selected.values())

    def search(self, query: str, max_context_chars: int = 8000) -> list[SearchResult]:
        if not query.strip() or not self.chunks:
            return []

        fused: dict[int, SearchResult] = {}
        for variant_index, variant in enumerate(self._query_variants(query)):
            weight = 1.0 if variant_index == 0 else 0.75
            vector_ranking = self._vector_ranking(variant)
            bm25_ranking = self._bm25_ranking(variant)
            exact_ranking = self._exact_term_ranking(variant)

            for rank, (index, score) in enumerate(vector_ranking, start=1):
                result = fused.setdefault(index, SearchResult(chunk=self.chunks[index]))
                result.vector_rank = min(result.vector_rank or rank, rank)
                result.distance = max(result.distance or score, score)
                result.rrf_score += weight / (self.rrf_k + rank)
            for rank, (index, _score) in enumerate(bm25_ranking, start=1):
                result = fused.setdefault(index, SearchResult(chunk=self.chunks[index]))
                result.bm25_rank = min(result.bm25_rank or rank, rank)
                result.rrf_score += weight / (self.rrf_k + rank)
            for rank, (index, _score) in enumerate(exact_ranking, start=1):
                result = fused.setdefault(index, SearchResult(chunk=self.chunks[index]))
                result.rrf_score += (weight * 0.65) / (self.rrf_k + rank)

            # Réserve le meilleur chunk d’une des deux pages lexicalement les plus utiles.
            local_scores: dict[int, float] = {}
            for rank, (index, _score) in enumerate(vector_ranking, start=1):
                local_scores[index] = local_scores.get(index, 0.0) + 1.0 / (self.rrf_k + rank)
            for rank, (index, _score) in enumerate(bm25_ranking, start=1):
                local_scores[index] = local_scores.get(index, 0.0) + 1.0 / (self.rrf_k + rank)
            for page, _page_score in self._page_term_ranking(variant)[:2]:
                page_indexes = self._by_page.get(page, [])
                if not page_indexes:
                    continue
                best_index = max(page_indexes, key=lambda index: local_scores.get(index, 0.0))
                result = fused.setdefault(best_index, SearchResult(chunk=self.chunks[best_index]))
                result.rrf_score += weight * 0.01

        seeds = sorted(fused.values(), key=lambda item: item.rrf_score, reverse=True)[: self.final_k]
        expanded = self._expand(seeds, query)
        expanded.sort(key=lambda item: item.rrf_score, reverse=True)

        selected: list[SearchResult] = []
        selected_keys: set[tuple[int | None, int]] = set()
        total = 0
        for item in expanded:
            key = (item.chunk.document_id, item.chunk.chunk_index)
            if key in selected_keys or len(selected) >= self.final_k:
                continue
            addition = len(item.chunk.text) + 80
            if selected and total + addition > max_context_chars:
                continue
            selected.append(item)
            selected_keys.add(key)
            total += addition
        return selected
