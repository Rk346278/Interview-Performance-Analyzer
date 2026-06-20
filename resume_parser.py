"""Extract text from PDF resume files."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract all text from a PDF file on disk."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    reader = PdfReader(str(path))
    return _extract_text_from_reader(reader)


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """Extract all text from PDF bytes (e.g. uploaded file content)."""
    if not pdf_bytes:
        raise ValueError("PDF content is empty")

    reader = PdfReader(BytesIO(pdf_bytes))
    return _extract_text_from_reader(reader)


def _extract_text_from_reader(reader: PdfReader) -> str:
    if reader.is_encrypted:
        reader.decrypt("")

    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text.strip())

    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError(
            "No text could be extracted from the PDF. "
            "The file may be scanned/image-only or empty."
        )

    return text
