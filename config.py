from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


APP_NAME = "assistant-rag-local"
DEFAULT_DOCUMENTS_FOLDER = Path.home() / "Documents" / "RAG-Documents"
DEFAULT_DATABASE_PATH = Path.home() / ".assistant-rag-local" / "rag.sqlite3"
DEFAULT_CONFIG_PATH = Path.home() / ".assistant-rag-local" / "config.json"
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"


def get_config_path(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get("RAG_CONFIG_PATH", DEFAULT_CONFIG_PATH)).expanduser()


@dataclass
class AppConfig:
    documents_folder: str = str(DEFAULT_DOCUMENTS_FOLDER)
    database_path: str = str(DEFAULT_DATABASE_PATH)
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ollama_base_url: str = "http://127.0.0.1:11434"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        config_path = get_config_path(path)
        if not config_path.is_file():
            config = cls()
            config.save(config_path)
            return config
        try:
            values = json.loads(config_path.read_text(encoding="utf-8"))
            return cls(**{key: value for key, value in values.items() if key in cls.__dataclass_fields__})
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            config = cls()
            config.save(config_path)
            return config

    def save(self, path: str | Path | None = None) -> Path:
        config_path = get_config_path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        return config_path

    @property
    def documents_path(self) -> Path:
        return Path(self.documents_folder).expanduser()

    @property
    def database_file(self) -> Path:
        return Path(self.database_path).expanduser()
