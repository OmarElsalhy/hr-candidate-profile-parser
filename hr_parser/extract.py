"""Read text out of uploaded resume files."""

from __future__ import annotations

import io
from pathlib import Path

from hr_parser.sections import clean_resume_text

SUPPORTED_TYPES = ("pdf", "txt")


def _pdf_links(reader) -> list[str]:
    """Collect hyperlink targets (e.g. a word 'GitHub' that links to a URL)."""
    links: list[str] = []
    for page in reader.pages:
        for annotation in page.get("/Annots") or []:
            try:
                action = annotation.get_object().get("/A")
                uri = action.get_object().get("/URI") if action is not None else None
            except Exception:  # noqa: BLE001 - malformed annotations are ignored
                continue
            if uri and str(uri) not in links:
                links.append(str(uri))
    return links


def read_uploaded_file(filename: str, data: bytes) -> str:
    """Return the plain text of a PDF or TXT resume."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        links = _pdf_links(reader)
        if links:
            text += "\n\nLinks found in the file:\n" + "\n".join(links)
    elif suffix == ".txt":
        text = data.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

    text = clean_resume_text(text)
    if not text:
        raise ValueError("No extractable text found (scanned PDFs need OCR first).")
    return text