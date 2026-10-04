from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from models import ExtractedBlock

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"[+]?\d[\d\s().-]{6,}\d")
_ENDS_SENTENCE_RE = re.compile(r"[.!?:]\s*$")


def _clean(value: object) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def _compress_row(row: list[object]) -> list[str]:
    """Retire les cellules vides/None d'une ligne, en ne gardant que les
    valeurs réelles, dans leur ordre d'apparition.

    pdfplumber peut retourner des lignes avec un nombre de colonnes qui
    varie d'une ligne à l'autre pour un même tableau (colonnes "fantômes"
    dues à des bordures internes, cellules fusionnées mal interprétées).
    Comparer les cellules par position brute dans ce cas décale toutes les
    valeurs d'une ligne dès qu'une cellule attendue est absente ailleurs
    que prévu. Compresser d'abord chaque ligne à ses seules valeurs non
    vides, puis aligner par position sur cette forme compressée, est
    beaucoup plus robuste face à ce bruit d'extraction — indépendamment du
    contenu ou du vocabulaire du tableau.
    """
    return [cleaned for cell in row if (cleaned := _clean(cell))]


def _looks_like_header_row(row: list[object]) -> bool:
    """Une ligne d'en-tête typique a peu de cellules vides et pas de longues
    phrases — contrairement à une ligne de données. Heuristique structurelle
    (longueur moyenne des cellules), indépendante du vocabulaire du document."""
    non_empty = _compress_row(row)
    if not non_empty:
        return False
    avg_len = sum(len(c) for c in non_empty) / len(non_empty)
    return avg_len <= 30


def _resolve_headers(rows: list[list[object]]) -> tuple[list[str], int]:
    header_row_index = 0
    for candidate_index in range(min(2, len(rows))):
        if _looks_like_header_row(rows[candidate_index]):
            header_row_index = candidate_index
            break
    headers = _compress_row(rows[header_row_index])
    return headers, header_row_index


def _looks_like_new_data_start(text: str) -> bool:
    """Heuristique positive : ce texte ressemble au début d'une nouvelle
    entrée structurée (email, téléphone, identifiant technique en
    majuscules, nom propre) plutôt qu'à la suite d'une phrase interrompue.
    Utilisée pour éviter de fusionner à tort une vraie nouvelle ligne de
    contact qui commencerait par une adresse email ou un numéro.
    """
    first_token = text.split(" ", 1)[0] if text else ""
    if _EMAIL_RE.match(text) or _PHONE_RE.match(text):
        return True
    # Identifiant technique probable : tout en majuscules avec chiffres/tirets
    # (ex: "MAMEK-CEDGE01", "10.50.200.75").
    if re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{2,}", first_token):
        return True
    return False


def _looks_like_continuation(previous_last_value: str, row: list[object]) -> bool:
    """A row is a continuation of the previous one, not a new logical row,
    when: its identifying first cell is empty, its content does not look
    like the start of a new structured entry (email, phone, technical id),
    and the previous cell's text did not already end on strong punctuation
    (a finished sentence rarely continues mid-thought on the next row).
    """
    if not row:
        return False
    first_cell = _clean(row[0])
    if first_cell:
        return False
    values = _compress_row(row)
    if not values:
        return False
    if _looks_like_new_data_start(values[0]):
        return False
    if previous_last_value and _ENDS_SENTENCE_RE.search(previous_last_value):
        return False
    return True


def _table_to_text(rows: list[list[object]]) -> list[str]:
    if not rows or len(rows) < 2:
        return []

    headers, header_row_index = _resolve_headers(rows)
    if not headers:
        return []

    logical_rows: list[list[str]] = []
    for row in rows[header_row_index + 1 :]:
        values = _compress_row(row)
        if not values:
            continue
        previous_last_value = logical_rows[-1][-1] if logical_rows and logical_rows[-1] else ""
        if logical_rows and _looks_like_continuation(previous_last_value, row):
            # Fusionne cette ligne avec la dernière valeur de la ligne
            # précédente plutôt que de créer une nouvelle entrée — c'est la
            # suite de la même phrase/cellule coupée par pdfplumber.
            merged = " ".join(values)
            logical_rows[-1][-1] = f"{logical_rows[-1][-1]} {merged}".strip()
        else:
            logical_rows.append(values)

    output: list[str] = []
    for row_number, values in enumerate(logical_rows, start=1):
        pairs = [f"{header} : {value}" for header, value in zip(headers, values)]
        if len(values) > len(headers):
            for extra_value in values[len(headers):]:
                pairs.append(f"(suite) : {extra_value}")
        if pairs:
            output.append(f"Ligne {row_number} — " + " ; ".join(pairs))
    return output


def extract_pdf(path: str | Path) -> list[ExtractedBlock]:
    """Extract page text and tables while keeping their provenance.

    Table rows are represented as explicit header/value pairs, aligned by
    compressed position (non-empty cells only) rather than raw column
    index, to stay robust against pdfplumber's inconsistent handling of
    merged or borderless cells across rows of the same table. Rows that
    look like a mid-sentence continuation of the previous row (a wrapped
    cell split across two physical table rows) are merged back together
    instead of becoming separate, disconnected fragments.
    """
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Installez pdfplumber pour indexer des PDF.") from exc

    pdf_path = Path(path).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    blocks: list[ExtractedBlock] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            tables = []
            try:
                tables = page.find_tables() or []
            except Exception:
                tables = []

            narrative = ""
            if tables:
                try:
                    outside = page
                    for table in tables:
                        outside = outside.outside_bbox(table.bbox)
                    narrative = outside.extract_text() or ""
                except Exception:
                    narrative = page.extract_text() or ""
            else:
                narrative = page.extract_text() or ""

            narrative = "\n".join(line.strip() for line in narrative.splitlines() if line.strip())
            if narrative.strip():
                blocks.append(
                    ExtractedBlock(
                        document_path=str(pdf_path),
                        page_number=page_number,
                        text=narrative.strip(),
                        block_type="narrative",
                    )
                )

            for table_index, table in enumerate(tables, start=1):
                try:
                    rows = table.extract() or []
                except Exception:
                    rows = []
                for row_text in _table_to_text(rows):
                    blocks.append(
                        ExtractedBlock(
                            document_path=str(pdf_path),
                            page_number=page_number,
                            text=row_text,
                            block_type="table_row",
                            table_id=f"p{page_number}-t{table_index}",
                            metadata={"table_index": table_index},
                        )
                    )
    return blocks


def iter_pdf_pages(path: str | Path) -> Iterable[ExtractedBlock]:
    """Compatibility alias for callers that want a simple iterable API."""
    yield from extract_pdf(path)