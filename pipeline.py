from __future__ import annotations
import hashlib
import time
from pathlib import Path
from typing import Any

from chunking import SemanticChunker
from embeddings import SentenceTransformerEmbedder
from extract import extract_pdf
from models import Answer, Chunk
from ollama_client import OllamaClient
from prompting import SYSTEM_PROMPT, build_user_prompt
from retrieval import HybridRetriever
from storage import SQLiteRepository

REFUSAL = "Je ne trouve pas cette information dans les documents indexés."

class AnswerValidator:
    def validate(self, text: str, has_sources: bool) -> tuple[str, bool]:
        if not has_sources:
            return REFUSAL, True
        cleaned = text.strip()
        if not cleaned:
            return REFUSAL, True
        # Llama 3.2 3B peut parfois être verbeux même s'il ne trouve pas.
        # On vérifie si la réponse est très courte et contient le refus.
        if len(cleaned) < 50 and REFUSAL.lower() in cleaned.lower():
            return REFUSAL, True
        return cleaned, False

class RAGPipeline:
    @classmethod
    def from_config(cls, config: Any) -> "RAGPipeline":
        return cls(
            database_path=config.database_file,
            embedding_model=config.embedding_model,
            ollama_model=config.ollama_model,
            ollama_base_url=config.ollama_base_url,
        )

    def __init__(
        self,
        database_path: str | Path = "rag.sqlite3",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        ollama_model: str = "llama3.2:3b",
        ollama_base_url: str = "http://127.0.0.1:11434",
        candidate_k: int = 80, 
        final_k: int = 5,     
        max_context_chars: int = 6000, # Réduit pour Llama 3.2 3B
    ) -> None:
        self.repository = SQLiteRepository(database_path)
        self.embedder = SentenceTransformerEmbedder(embedding_model)
        self.chunker = SemanticChunker()
        self.ollama = OllamaClient(base_url=ollama_base_url, model=ollama_model)
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.max_context_chars = max_context_chars
        self.validator = AnswerValidator()
        self._retriever: HybridRetriever | None = None

    @staticmethod
    def file_sha256(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _new_retriever(self) -> HybridRetriever:
        chunks = self.repository.load_chunks()
        self._retriever = HybridRetriever(
            chunks,
            self.embedder,
            candidate_k=self.candidate_k,
            final_k=self.final_k,
        )
        return self._retriever

    def index_pdf(
        self,
        pdf_path: str | Path,
        force: bool = False,
        sha256: str | None = None,
    ) -> dict[str, Any]:
        path = Path(pdf_path).expanduser().resolve()
        digest = sha256 or self.file_sha256(path)
        if not force and self.repository.document_id_by_hash(digest) is not None:
            return {"status": "skipped", "reason": "already_indexed", "sha256": digest}

        started = time.perf_counter()
        blocks = extract_pdf(path)
        page_count = max((block.page_number for block in blocks), default=0)
        chunks = self.chunker.chunk(blocks, embedder=self.embedder)
        
        if not chunks:
            raise ValueError(f"Aucun contenu textuel exploitable dans {path}.")
            
        vectors = self.embedder.encode([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector
            
        document_id = self.repository.save_document(
            path=str(path),
            sha256=digest,
            page_count=page_count,
            chunks=chunks,
            embedding_model=self.embedder.model_version,
        )
        self._retriever = None
        return {
            "status": "indexed",
            "document_id": document_id,
            "pages": page_count,
            "chunks": len(chunks),
            "seconds": round(time.perf_counter() - started, 2),
        }

    def search(self, question: str):
        retriever = self._retriever or self._new_retriever()
        started = time.perf_counter()
        results = retriever.search(question, max_context_chars=self.max_context_chars)
        return results, round((time.perf_counter() - started) * 1000, 1)

    def ask(self, question: str) -> Answer:
        total_started = time.perf_counter()
        sources, retrieval_ms = self.search(question)
        
        if not sources:
            return Answer(
                text=REFUSAL,
                sources=[],
                refused=True,
                latency_ms={"retrieval": retrieval_ms, "total": round((time.perf_counter() - total_started) * 1000, 1)},
            )
            
        prompt = build_user_prompt(question, sources)
        generation_started = time.perf_counter()
        
        # Appel Ollama avec le prompt système
        raw = self.ollama.chat(SYSTEM_PROMPT, prompt)
        
        print("\n" + "=" * 70)
        print(f"DEBUG - SOURCES: {len(sources)} | CONTEXTE: ~{sum(len(s.chunk.text) for s in sources)} chars")
        print("DEBUG - RÉPONSE BRUTE:")
        print(raw)
        print("=" * 70)

        text, refused = self.validator.validate(raw, bool(sources))
        return Answer(
            text=text,
            sources=sources,
            refused=refused,
            latency_ms={
                "retrieval": retrieval_ms,
                "generation": round((time.perf_counter() - generation_started) * 1000, 1),
                "total": round((time.perf_counter() - total_started) * 1000, 1),
            },
        )
