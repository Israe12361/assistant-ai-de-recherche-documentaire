from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import AppConfig
from pipeline import RAGPipeline
from sync import sync_folder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assistant RAG documentaire local")
    parser.add_argument("--config", type=Path, default=None, help="Fichier de configuration")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="Indexer un PDF ponctuellement")
    index.add_argument("pdf", type=Path)
    index.add_argument("--force", action="store_true")

    sync = sub.add_parser("sync", help="Synchroniser le dossier enregistré")
    sync.add_argument("--folder", type=Path, default=None, help="Surcharge ponctuelle du dossier configuré")
    sync.add_argument("--force", action="store_true")

    ask = sub.add_parser("ask", help="Poser une question sur l’index")
    ask.add_argument("question")

    sub.add_parser("stats", help="Afficher les statistiques de l’index")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AppConfig.load(args.config)
    pipeline = RAGPipeline.from_config(config)
    try:
        if args.command == "index":
            print(json.dumps(pipeline.index_pdf(args.pdf, force=args.force), ensure_ascii=False, indent=2))
        elif args.command == "sync":
            folder = args.folder or config.documents_path
            print(json.dumps(sync_folder(pipeline, folder, force=args.force), ensure_ascii=False, indent=2))
        elif args.command == "stats":
            print(json.dumps({
                "documents_folder": str(config.documents_path),
                "ollama_model": config.ollama_model,
                "documents": pipeline.repository.document_count(),
                "chunks": pipeline.repository.count_chunks(),
            }, ensure_ascii=False, indent=2))
        elif args.command == "ask":
            answer = pipeline.ask(args.question)
            print(answer.text)
            print("\nSources :")
            for index, source in enumerate(answer.sources, start=1):
                print(f"[S{index}] document_id={source.chunk.document_id}, page={source.chunk.page_start}")
            print(f"\nLatence : {answer.latency_ms}")
    finally:
        pipeline.repository.close()


if __name__ == "__main__":
    main()
