from __future__ import annotations

from pathlib import Path
from typing import Any

from config import AppConfig
from pipeline import RAGPipeline


def sync_folder(
    pipeline: RAGPipeline,
    folder: str | Path,
    force: bool = False,
    prune_missing: bool = True,
) -> dict[str, Any]:
    """Synchronise les PDF d'un dossier avec l'index SQLite.

    ``prune_missing=True`` est adapté à une application qui possède un seul
    dossier documentaire configuré : les documents absents du dossier courant
    sont supprimés, y compris ceux provenant de l'ancien chemin après un
    changement de dossier.

    Si plusieurs dossiers indépendants doivent coexister dans la même base,
    utiliser ``prune_missing=False`` et gérer chaque source séparément.
    """
    root = Path(folder).expanduser().resolve()
    created_folder = not root.exists()
    root.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(
        (
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: str(path).casefold(),
    )
    current_paths = {str(path) for path in pdfs}

    report: dict[str, Any] = {
        "folder": str(root),
        "created_folder": created_folder,
        "scanned": len(pdfs),
        "indexed": [],
        "skipped": [],
        "removed": [],
        "errors": [],
    }

    for pdf_path in pdfs:
        path_string = str(pdf_path)
        try:
            digest = pipeline.file_sha256(pdf_path)
            existing_hash = pipeline.repository.document_hash_by_path(path_string)

            if not force and existing_hash == digest:
                report["skipped"].append(path_string)
                continue

            # Le hash est déjà calculé ; il n'est pas recalculé dans index_pdf.
            result = pipeline.index_pdf(
                pdf_path,
                force=True,
                sha256=digest,
            )
            report["indexed"].append(result)
        except Exception as exc:
            report["errors"].append({"path": path_string, "error": str(exc)})

    if prune_missing:
        # Mode mono-dossier : le changement de chemin est traité correctement.
        # On supprime tout document qui n'est plus dans le dossier courant,
        # y compris les documents enregistrés sous l'ancien chemin.
        stored_paths = pipeline.repository.list_document_paths()
        for stored_path in stored_paths:
            normalized = str(Path(stored_path).expanduser().resolve())
            if normalized not in current_paths:
                pipeline.repository.delete_document_by_path(normalized)
                report["removed"].append(normalized)
    else:
        # Mode multi-dossiers : on ne supprime que les anciens documents du
        # dossier actuellement synchronisé.
        for stored_path in pipeline.repository.list_document_paths():
            candidate = Path(stored_path).expanduser().resolve()
            if candidate.is_relative_to(root) and str(candidate) not in current_paths:
                pipeline.repository.delete_document_by_path(str(candidate))
                report["removed"].append(str(candidate))

    # Le retriever contient une copie des chunks chargés en mémoire. Il doit
    # être reconstruit après toute insertion, modification ou suppression.
    if report["indexed"] or report["removed"]:
        pipeline._retriever = None

    return report


def sync_from_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = AppConfig.load(config_path)
    pipeline = RAGPipeline(
        database_path=config.database_file,
        embedding_model=config.embedding_model,
        ollama_model=config.ollama_model,
    )
    try:
        return sync_folder(
            pipeline,
            config.documents_path,
            prune_missing=True,
        )
    finally:
        pipeline.repository.close()
