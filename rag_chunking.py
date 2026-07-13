"""
Text chunking strategies for RAG.

Supported strategies:
- auto: split by paragraphs, then further split by token length.
- header: split by Markdown/Word-style headers (#, ##).
- separator: split by a custom delimiter.
"""

import re
from typing import List


def split_text(
    text: str,
    strategy: str = "auto",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separator: str = "\n",
) -> List[str]:
    """Split text into chunks according to the selected strategy."""
    text = text.strip()
    if not text:
        return []

    if strategy == "header":
        chunks = _split_by_header(text)
    elif strategy == "separator":
        chunks = [c.strip() for c in text.split(separator) if c.strip()]
    else:
        # auto: split by paragraphs first
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

    # Further split any chunk that exceeds chunk_size by character length.
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(_split_by_window(chunk, chunk_size, chunk_overlap))

    return [c for c in final_chunks if c]


def _split_by_header(text: str) -> List[str]:
    """Split Markdown-like text by headers (# or ##)."""
    # Match lines starting with # or ##
    pattern = re.compile(r"(?=^#{1,2}\s+)", re.MULTILINE)
    parts = pattern.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_window(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Sliding window split by character count."""
    chunks = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start += step
    return chunks
