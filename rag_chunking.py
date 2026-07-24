"""Structure-aware text chunking for the RAG evidence pipeline.

Headings are useful retrieval metadata, but are not factual evidence by
themselves.  The default strategy therefore carries a heading forward into the
next factual unit instead of emitting a title-only chunk.
"""

import re
from typing import List


_SECTION_START = re.compile(
    r"^\s*(?:#{1,6}\s+|第[一二三四五六七八九十百0-9]+[章节条款]|"
    r"[一二三四五六七八九十]+[、．.]|(?:Q|问|问题)\s*\d+\s*[：:]|"
    r"\d+[、.)）])"
)
_HEADING_ENDINGS = (
    "范围",
    "原则",
    "说明",
    "流程",
    "标准",
    "承诺",
    "职责",
    "责任",
    "要求",
    "管理",
    "服务",
    "附则",
    "总则",
    "预案",
    "机制",
)
_FACT_MARKERS = re.compile(
    r"[。！？；：:]|"
    r"\d+(?:\.\d+)?\s*(?:分钟|小时|天|工作日|元|万元|%|次|年|月|日)|"
    r"(?:应当|应该|必须|不得|严禁|可以|需要|负责|提供|完成|到场|上门|受理|登记|响应)"
)
_TABLE_DIVIDER = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")
_CRITICAL_VALUE = re.compile(
    r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万两半]+)\s*"
    r"(?:分钟|小时|工作日|天|元|万元|%|次|年|个月|月|日)"
)
_FAQ_QUESTION = re.compile(
    r"^\s*(?:Q|问|问题)\s*\d+\s*[：:].+[？?]?\s*$",
    re.IGNORECASE,
)


def split_text(
    text: str,
    strategy: str = "auto",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separator: str = "\n",
) -> List[str]:
    """Split text into factual evidence units without standalone headings."""
    text = text.strip()
    if not text:
        return []

    if strategy == "header":
        chunks = _split_by_header(text)
    elif strategy == "separator":
        chunks = [c.strip() for c in text.split(separator) if c.strip()]
    else:
        chunks = _split_auto_units(text)

    chunks = _merge_headings_with_facts(chunks)
    final_chunks: List[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        elif _contains_markdown_table(chunk):
            final_chunks.extend(_split_markdown_table(chunk, chunk_size))
        else:
            final_chunks.extend(_split_by_window(chunk, chunk_size, chunk_overlap))

    return [c for c in final_chunks if c and not _is_heading_only(c)]


def _is_heading_only(text: str) -> bool:
    """Return True only when a unit has structure but no factual payload."""
    value = (text or "").strip()
    if not value or "\n" in value:
        return False
    plain = re.sub(r"^#{1,6}\s*", "", value).strip()
    if not plain:
        return False
    if (
        len(plain) <= 40
        and plain.endswith(("：", ":"))
        and not re.search(r"\d", plain)
        and not re.search(r"[。！？；]", plain)
    ):
        return True
    if re.fullmatch(r"第[一二三四五六七八九十百0-9]+[章节篇部].{0,30}", plain):
        return True
    if _FACT_MARKERS.search(plain):
        return False
    if value.startswith("#"):
        return True
    if plain in {"总则", "附则"}:
        return True
    if _SECTION_START.match(value) and len(plain) <= 36:
        return plain.endswith(_HEADING_ENDINGS)
    return len(plain) <= 24 and plain.endswith(_HEADING_ENDINGS)


def _merge_headings_with_facts(units: List[str]) -> List[str]:
    """Carry one or more headings into the next factual evidence unit."""
    merged: List[str] = []
    pending: List[str] = []
    for index, raw in enumerate(units):
        unit = (raw or "").strip()
        if not unit:
            continue
        leading_title = bool(
            index == 0
            and len(units) > 1
            and "\n" not in unit
            and len(unit) <= 48
            and not re.search(r"[。！？；：:]", unit)
            and not _CRITICAL_VALUE.search(unit)
        )
        if leading_title or _is_heading_only(unit):
            pending.append(unit)
            continue
        if "\n" not in unit and _FAQ_QUESTION.fullmatch(unit):
            pending.append(unit)
            continue
        if pending:
            unit = "\n".join(pending + [unit])
            pending = []
        merged.append(unit)
    # A trailing heading has no factual meaning and is intentionally omitted.
    return merged


def _split_auto_units(text: str) -> List[str]:
    """Keep numbered rules and FAQ question-answer pairs as evidence units."""
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


def _contains_markdown_table(text: str) -> bool:
    lines = [line.rstrip() for line in (text or "").splitlines()]
    return any(
        index > 0
        and "|" in lines[index - 1]
        and bool(_TABLE_DIVIDER.fullmatch(lines[index]))
        for index in range(len(lines))
    )


def _split_markdown_table(text: str, chunk_size: int) -> List[str]:
    """Split a long Markdown table by rows while repeating its header."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    divider_index = next(
        (
            index
            for index, line in enumerate(lines)
            if index > 0
            and "|" in lines[index - 1]
            and _TABLE_DIVIDER.fullmatch(line)
        ),
        -1,
    )
    if divider_index < 1:
        return _split_by_window(text, chunk_size, 0)

    prefix = lines[: divider_index - 1]
    header = lines[divider_index - 1 : divider_index + 1]
    rows = lines[divider_index + 1 :]
    if not rows:
        return ["\n".join(prefix + header)]

    chunks: List[str] = []
    current = prefix + header
    for row in rows:
        candidate = "\n".join(current + [row])
        if len(candidate) > chunk_size and len(current) > len(prefix) + len(header):
            chunks.append("\n".join(current))
            current = header + [row]
        else:
            current.append(row)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _split_by_window(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Use a size window only as a last resort, favouring sentence endings."""
    chunks: List[str] = []
    start = 0
    minimum = max(1, int(chunk_size * 0.55))
    while start < len(text):
        hard_end = min(len(text), start + chunk_size)
        end = hard_end
        if hard_end < len(text):
            window = text[start + minimum : hard_end]
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
