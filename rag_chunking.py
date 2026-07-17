"""Text chunking strategies for the RAG evidence pipeline.

The default auto strategy prefers document structure before a size window:
paragraphs, headings, numbered rules and FAQ entries remain evidence units.
Only an overlong evidence unit is split again near a sentence boundary.
"""

import re
from typing import List


_SECTION_START = re.compile(
    r"^\s*(?:#{1,6}\s+|第[一二三四五六七八九十百0-9]+[章节条款]|"
    r"[一二三四五六七八九十]+[、．.]|(?:Q|问|问题)\s*\d+\s*[：:]|"
    r"\d+[、.)）])"
)


def split_text(
    text: str,
    strategy: str = "auto",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separator: str = "\n",
) -> List[str]:
    """Split text into retrieval evidence units.

    Auto preserves section/rule boundaries first. A short regulation with
    three numbered rules therefore produces three inspectable citations rather
    than one opaque text block. A short atomic paragraph remains one chunk.
    """
    text = text.strip()
    if not text:
        return []

    if strategy == "header":
        chunks = _split_by_header(text)
    elif strategy == "separator":
        chunks = [c.strip() for c in text.split(separator) if c.strip()]
    else:
        chunks = _split_auto_units(text)

    final_chunks: List[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(_split_by_window(chunk, chunk_size, chunk_overlap))

    return [c for c in final_chunks if c]


def _split_auto_units(text: str) -> List[str]:
    """Keep headings and numbered rules with their following explanation."""
    units: List[str] = []
    for paragraph in [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        structured_starts = sum(bool(_SECTION_START.match(line)) for line in lines)
        if len(lines) <= 1 or structured_starts < 2:
            units.append(paragraph)
            continue

        current: List[str] = []
        for line in lines:
            if _SECTION_START.match(line) and current:
                units.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            units.append("\n".join(current))
    return units


def _split_by_header(text: str) -> List[str]:
    pattern = re.compile(r"(?=^#{1,6}\s+)", re.MULTILINE)
    parts = pattern.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_window(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Use a size window only as a last resort, favouring sentence endings."""
    chunks: List[str] = []
    start = 0
    minimum = max(1, int(chunk_size * 0.55))
    while start < len(text):
        hard_end = min(len(text), start + chunk_size)
        end = hard_end
        if hard_end < len(text):
            window = text[start + minimum:hard_end]
            boundary = max(window.rfind(mark) for mark in "。！？；\n")
            if boundary >= 0:
                end = start + minimum + boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - max(0, chunk_overlap), start + 1)
    return chunks
