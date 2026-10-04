from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SentenceTransformerEmbedder:
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    device: str | None = None

    def __post_init__(self) -> None:
        self._model: Any | None = None

    @property
    def model_version(self) -> str:
        return self.model_name

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "Installez sentence-transformers pour calculer les embeddings."
                ) from exc
            kwargs = {"device": self.device} if self.device else {}
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._load().encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def vector_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes, dimension: int) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32)
    if vector.size != dimension:
        raise ValueError(f"Dimension d’embedding invalide : {vector.size} != {dimension}.")
    return vector
