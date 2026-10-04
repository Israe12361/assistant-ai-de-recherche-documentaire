from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from embeddings import blob_to_vector, vector_to_blob
from models import Chunk


SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    page_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    embedding_model TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    chunk_type TEXT NOT NULL,
    text TEXT NOT NULL,
    text_normalized TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    char_count INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_document_page ON chunks(document_id, page_start);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
"""


class SQLiteRepository:
    def __init__(self, database_path: str | Path) -> None:
        database = Path(database_path).expanduser()
        database.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = str(database)
        # The desktop UI performs indexing in a worker thread. SQLite operations
        # remain serialized by the application workflow, while this option allows
        # the same repository object to be used by that worker safely.
        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def document_id_by_hash(self, sha256: str) -> int | None:
        row = self.connection.execute(
            "SELECT id FROM documents WHERE sha256 = ?", (sha256,)
        ).fetchone()
        return int(row["id"]) if row else None

    def document_id_by_path(self, path: str) -> int | None:
        row = self.connection.execute(
            "SELECT id FROM documents WHERE path = ?", (str(Path(path).expanduser().resolve()),)
        ).fetchone()
        return int(row["id"]) if row else None

    def document_hash_by_path(self, path: str) -> str | None:
        row = self.connection.execute(
            "SELECT sha256 FROM documents WHERE path = ?",
            (str(Path(path).expanduser().resolve()),),
        ).fetchone()
        return str(row["sha256"]) if row else None

    def list_document_paths(self) -> list[str]:
        rows = self.connection.execute("SELECT path FROM documents ORDER BY path").fetchall()
        return [str(row["path"]) for row in rows]

    def delete_document_by_path(self, path: str) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM documents WHERE path = ?",
                (str(Path(path).expanduser().resolve()),),
            )

    def save_document(
        self,
        path: str,
        sha256: str,
        page_count: int,
        chunks: Iterable[Chunk],
        embedding_model: str,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.connection:
            existing_ids = {
                document_id
                for document_id in (
                    self.document_id_by_hash(sha256),
                    self.document_id_by_path(path),
                )
                if document_id is not None
            }
            for existing in existing_ids:
                self.connection.execute("DELETE FROM documents WHERE id = ?", (existing,))
            cursor = self.connection.execute(
                """INSERT INTO documents
                (path, filename, sha256, page_count, created_at, indexed_at, embedding_model)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (path, Path(path).name, sha256, page_count, now, now, embedding_model),
            )
            document_id = int(cursor.lastrowid)
            for chunk in chunks:
                if chunk.embedding is None:
                    raise ValueError("Chaque chunk doit avoir un embedding avant la sauvegarde.")
                vector = np.asarray(chunk.embedding, dtype=np.float32)
                self.connection.execute(
                    """INSERT INTO chunks
                    (document_id, chunk_index, page_start, page_end, chunk_type, text,
                     text_normalized, embedding, embedding_dimension, char_count, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        document_id,
                        chunk.chunk_index,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.chunk_type,
                        chunk.text,
                        " ".join(chunk.text.lower().split()),
                        vector_to_blob(vector),
                        int(vector.size),
                        len(chunk.text),
                        json.dumps(chunk.metadata, ensure_ascii=False),
                    ),
                )
        return document_id

    def count_chunks(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"])

    def load_chunks(self) -> list[Chunk]:
        rows = self.connection.execute(
            """SELECT document_id, chunk_index, page_start, page_end, chunk_type,
                      text, embedding, embedding_dimension, metadata_json
               FROM chunks ORDER BY document_id, chunk_index"""
        ).fetchall()
        return [
            Chunk(
                document_id=int(row["document_id"]),
                chunk_index=int(row["chunk_index"]),
                page_start=int(row["page_start"]),
                page_end=int(row["page_end"]),
                text=row["text"],
                chunk_type=row["chunk_type"],
                metadata=json.loads(row["metadata_json"] or "{}"),
                embedding=blob_to_vector(row["embedding"], int(row["embedding_dimension"])),
            )
            for row in rows
        ]

    def get_neighbors(self, document_id: int, chunk_index: int, radius: int = 1) -> list[Chunk]:
        rows = self.connection.execute(
            """SELECT document_id, chunk_index, page_start, page_end, chunk_type,
                      text, embedding, embedding_dimension, metadata_json
               FROM chunks
               WHERE document_id = ? AND chunk_index BETWEEN ? AND ?
               ORDER BY chunk_index""",
            (document_id, chunk_index - radius, chunk_index + radius),
        ).fetchall()
        return [
            Chunk(
                document_id=int(row["document_id"]),
                chunk_index=int(row["chunk_index"]),
                page_start=int(row["page_start"]),
                page_end=int(row["page_end"]),
                text=row["text"],
                chunk_type=row["chunk_type"],
                metadata=json.loads(row["metadata_json"] or "{}"),
                embedding=blob_to_vector(row["embedding"], int(row["embedding_dimension"])),
            )
            for row in rows
        ]

    def document_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        return int(row["n"])
