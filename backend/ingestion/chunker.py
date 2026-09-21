"""
Splits page text into overlapping chunks. Character-based (not
strict token-based) for simplicity and zero extra dependencies at
runtime — CHUNK_SIZE=2000 chars / CHUNK_OVERLAP=200 chars roughly
approximates 500 tokens / 50 token overlap, which is a solid default
for retrieval precision vs. context completeness.
"""
from dataclasses import dataclass
from typing import List
from .loader import PageText


@dataclass
class Chunk:
    text: str
    page: int
    chunk_index: int


def chunk_pages(
    pages: List[PageText],
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    """
    Chunks each page's text independently (rather than concatenating
    the whole document first) so page numbers stay accurate for
    citations. Long pages get split into multiple chunks; short pages
    may end up as a single chunk.
    """
    chunks: List[Chunk] = []
    global_index = 0

    for page in pages:
        text = page.text
        start = 0
        text_len = len(text)

        if text_len <= chunk_size:
            chunks.append(Chunk(text=text, page=page.page_number, chunk_index=global_index))
            global_index += 1
            continue

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Chunk(text=chunk_text, page=page.page_number, chunk_index=global_index)
                )
                global_index += 1

            if end == text_len:
                break

            # Move forward, leaving `chunk_overlap` chars of context
            # so a sentence split mid-chunk boundary isn't fully lost.
            start = end - chunk_overlap

    return chunks
