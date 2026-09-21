"""
Handles turning an uploaded file into plain text. Supports PDF
(via pypdf) and plain text / markdown files directly.
"""
from dataclasses import dataclass
from typing import List
from pypdf import PdfReader


@dataclass
class PageText:
    page_number: int  # 1-indexed; 0 for non-paginated formats (md/txt)
    text: str


def load_pdf(file_path: str) -> List[PageText]:
    """Extracts text page-by-page so we can cite page numbers later."""
    reader = PdfReader(file_path)
    pages: List[PageText] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(PageText(page_number=i + 1, text=text))

    if not pages:
        raise ValueError(
            "No extractable text found in this PDF. It may be a scanned "
            "image-only document that needs OCR, which this app doesn't "
            "currently support."
        )
    return pages


def load_text_file(file_path: str) -> List[PageText]:
    """Handles .md / .txt notes as a single 'page' of text."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()

    if not content:
        raise ValueError("This file appears to be empty.")

    return [PageText(page_number=0, text=content)]


def load_document(file_path: str, filename: str) -> List[PageText]:
    """Dispatches to the right loader based on file extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return load_pdf(file_path)
    elif lower.endswith(".md") or lower.endswith(".txt"):
        return load_text_file(file_path)
    else:
        raise ValueError(
            f"Unsupported file type for '{filename}'. "
            "Only .pdf, .md, and .txt are supported."
        )
