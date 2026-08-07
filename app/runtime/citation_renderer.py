"""Build and render citations from one immutable EvidenceSet."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.runtime.contracts import (
    Citation,
    EvidenceItem,
    EvidenceSet,
    content_hash,
    stable_id,
)


# Match every evidence marker shape, including malformed/model-invented IDs.
# Validation happens against the immutable EvidenceSet below; restricting the
# regex itself allowed unknown values containing spaces to leak into the UI.
EVIDENCE_MARKER = re.compile(r"\[\[evidence:([^\]\r\n]+)\]\]")
UNSTRUCTURED_MARKER = re.compile(r"\[\[([^\]\r\n]+)\]\]")
LEGACY_MARKER = re.compile(r"【引用\s*(\d+)】|\[(\d+)\]")
_GENERIC_BIGRAMS = {
    "可以",
    "需要",
    "建议",
    "相关",
    "情况",
    "根据",
    "进行",
    "服务",
    "信息",
    "数据",
}
_CRITICAL_VALUE = re.compile(
    r"(?P<number>\d+(?:\.\d+)?|[一二三四五六七八九十百千万两半]+)\s*"
    r"(?P<unit>个\s*工作日|个\s*种植舱|工作日|种植舱|分钟|小时|天|份|元|万元|%|次|年|个月|月|日)"
)
_PHONE_VALUE = re.compile(r"(?<!\d)(?:0\d{2,3}[- ]?\d{7,8}|1[3-9]\d{9})(?!\d)")
_CALCULATION_INPUT_VALUE = re.compile(
    r"(?:连续|持续)\s*"
    r"(?P<number>\d+(?:\.\d+)?|[一二三四五六七八九十百千万两半]+)\s*"
    r"(?P<unit>分钟|小时|天|个月|月|年)\s*"
    r"(?:需要|共需|总共|合计|计算)"
)
_CALCULATION_ENTITY_INPUT = re.compile(
    r"(?P<number>\d+(?:\.\d+)?|[一二三四五六七八九十百千万两半]+)\s*"
    r"(?P<unit>个\s*种植舱|种植舱)"
)
_CALCULATION_QUESTION = re.compile(r"多少|计算|共需|总共|合计|总需求")
_TOOL_RESULT_NUMBER = re.compile(
    r"['\"]result['\"]\s*:\s*(?P<number>[-+]?\d+(?:\.\d+)?)"
)


def _semantic_bigrams(text: str) -> Set[str]:
    chars = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", (text or "").lower())
    return {
        chars[index] + chars[index + 1]
        for index in range(len(chars) - 1)
        if chars[index] + chars[index + 1] not in _GENERIC_BIGRAMS
    }


def _critical_values(text: str) -> Set[str]:
    """Extract policy-sensitive time, money, percentage and count values."""
    values: Set[str] = set()
    for match in _CRITICAL_VALUE.finditer(text or ""):
        number = re.sub(r"\s+", "", match.group("number")).lower()
        unit = re.sub(r"\s+", "", match.group("unit")).lower()
        # “3个工作日”与“3工作日”是同一时效表达，但仍必须与
        # “5个工作日”保持不同，避免引用中的数字被模型悄悄改写。
        if unit == "个工作日":
            unit = "工作日"
        values.add(f"{number}{unit}")
    for match in _PHONE_VALUE.finditer(text or ""):
        digits = re.sub(r"\D", "", match.group(0))
        values.add(f"phone:{digits}")
    return values


def _calculation_input_values(text: str) -> Set[str]:
    """Extract duration operands supplied by the user for a calculation."""

    values: Set[str] = set()
    for match in _CALCULATION_INPUT_VALUE.finditer(text or ""):
        number = re.sub(r"\s+", "", match.group("number")).lower()
        unit = re.sub(r"\s+", "", match.group("unit")).lower()
        values.add(f"{number}{unit}")
    if _CALCULATION_QUESTION.search(text or ""):
        for match in _CALCULATION_ENTITY_INPUT.finditer(text or ""):
            number = re.sub(r"\s+", "", match.group("number")).lower()
            unit = re.sub(r"\s+", "", match.group("unit")).lower()
            values.add(f"{number}{unit}")
    return values


def _normalize_number(number: str) -> str:
    normalized = str(number or "").strip().lstrip("+")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _successful_tool_result_numbers(tool_invocations: Iterable[Any]) -> Set[str]:
    """Return numeric results from successful Tool receipts, never arguments."""

    values: Set[str] = set()
    for invocation in tool_invocations or []:
        if isinstance(invocation, dict):
            invocation_status = invocation.get("invocation_status")
            business_status = invocation.get("business_status")
            result_summary = invocation.get("result_summary")
        else:
            invocation_status = getattr(invocation, "invocation_status", None)
            business_status = getattr(invocation, "business_status", None)
            result_summary = getattr(invocation, "result_summary", None)
        if invocation_status != "success" or business_status != "success":
            continue
        for match in _TOOL_RESULT_NUMBER.finditer(str(result_summary or "")):
            values.add(_normalize_number(match.group("number")))
    return values


def _tool_supports_critical_value(value: str, tool_numbers: Set[str]) -> bool:
    match = re.match(r"(?P<number>[-+]?\d+(?:\.\d+)?)", value or "")
    return bool(
        match and _normalize_number(match.group("number")) in tool_numbers
    )


def _supporting_excerpt(content: str, values: Set[str], limit: int = 400) -> str:
    """Return a short immutable excerpt that contains the supporting values."""

    lines = [line.strip() for line in re.split(r"[\r\n]+", content or "") if line.strip()]
    matched = [line for line in lines if _critical_values(line) & values]
    excerpt = " ".join(matched) if matched else ""
    return excerpt[:limit]


def build_skill_evidence(
    answer: str,
    skill_sources: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build answer-linked evidence from Skills activated in this exact Snapshot."""

    answer_values = _critical_values(answer or "")
    result: List[Dict[str, Any]] = []
    for source in skill_sources or []:
        content = str(source.get("content_snapshot") or "")
        supported = answer_values & _critical_values(content)
        if not supported:
            continue
        result.append(
            {
                "skill_id": source.get("skill_id"),
                "name": source.get("name"),
                "version": source.get("version"),
                "snapshot_id": source.get("snapshot_id"),
                "content_hash": source.get("content_hash"),
                "supported_values": sorted(supported),
                "supporting_excerpt": _supporting_excerpt(content, supported),
            }
        )
    return result


def _is_structural_content(content: str, title: str = "") -> bool:
    raw = (content or "").strip()
    if (
        "\n" not in raw
        and len(raw) <= 40
        and raw.endswith(("：", ":"))
        and not re.search(r"\d", raw)
        and not re.search(r"[。！？；]", raw)
    ):
        return True
    normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", raw).lower()
    title_normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", title or "").lower()
    if not normalized:
        return True
    if title_normalized and normalized == title_normalized:
        return True
    return bool(
        "\n" not in raw
        and re.fullmatch(
            r"第[0-9一二三四五六七八九十百]+[章节篇部].{0,24}|[总附]则",
            normalized,
        )
    )


def _strip_markdown_display(text: str) -> str:
    """Normalize presentation markup without changing the factual words."""

    visible = EVIDENCE_MARKER.sub("", text or "")
    visible = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", visible)
    visible = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", visible)
    visible = re.sub(r"^[\s#>*+-]+", "", visible)
    visible = re.sub(r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*", "", visible)
    visible = re.sub(r"[*_`~]+", "", visible)
    visible = re.sub(r"\s+", " ", visible).strip(" |\t\r\n")
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", visible):
        return ""
    return visible


def _table_cells(line: str) -> Optional[List[str]]:
    """Return Markdown table cells, or ``None`` for an ordinary line."""

    stripped = (line or "").strip()
    if "|" not in stripped:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    return cells if len(cells) >= 2 else None


def _is_table_separator(cells: Optional[List[str]]) -> bool:
    if not cells:
        return False
    return all(bool(re.fullmatch(r":?-{3,}:?", cell.replace(" ", ""))) for cell in cells)


def _normalize_table_claim(line: str) -> str:
    cells = _table_cells(line) or []
    factual_cells = [
        _strip_markdown_display(cell)
        for cell in cells
        if _strip_markdown_display(cell)
    ]
    if len(factual_cells) == 2:
        return f"{factual_cells[0]}为{factual_cells[1]}"
    return "；".join(factual_cells)


def _sentence_spans(line: str) -> List[Tuple[int, int]]:
    """Split one non-table line into sentence-sized display spans."""

    spans: List[Tuple[int, int]] = []
    start = 0
    for boundary in re.finditer(r"[。！？；.!?;]+", line or ""):
        spans.append((start, boundary.end()))
        start = boundary.end()
    if start < len(line):
        spans.append((start, len(line)))
    return spans or [(0, len(line))]


def _is_display_heading(raw: str, normalized: str) -> bool:
    # A heading can itself contain a factual claim (and even critical values),
    # so Markdown heading syntax must never exempt it from evidence checks.
    return not normalized


def _build_atomic_claims(answer: str) -> Tuple[List[Dict[str, Any]], Set[int]]:
    """Map Markdown citation markers to normalized atomic factual claims.

    Markdown remains the rendering surface only.  Tables become row facts,
    ordinary prose becomes sentence facts, and a citation-only display line is
    attached to the immediately preceding still-unbound fact.  A citation-only
    line without such a fact is ignored rather than invented into a claim.
    """

    lines = (answer or "").splitlines(keepends=True)
    claims: List[Dict[str, Any]] = []
    ignored_markers: Set[int] = set()
    offset = 0

    for line_index, raw_with_ending in enumerate(lines):
        line = raw_with_ending.rstrip("\r\n")
        next_line = (
            lines[line_index + 1].rstrip("\r\n")
            if line_index + 1 < len(lines)
            else ""
        )
        cells = _table_cells(line)
        next_cells = _table_cells(next_line)
        line_markers = list(EVIDENCE_MARKER.finditer(line))

        if cells is not None:
            if _is_table_separator(cells) or _is_table_separator(next_cells):
                offset += len(raw_with_ending)
                continue
            claim_text = _normalize_table_claim(line)
            if claim_text:
                claims.append(
                    {
                        "text": claim_text,
                        "marker_starts": [offset + item.start() for item in line_markers],
                        "line_index": line_index,
                        "kind": "table_row",
                    }
                )
            offset += len(raw_with_ending)
            continue

        for segment_start, segment_end in _sentence_spans(line):
            segment = line[segment_start:segment_end]
            segment_markers = list(EVIDENCE_MARKER.finditer(segment))
            marker_starts = [
                offset + segment_start + item.start() for item in segment_markers
            ]
            claim_text = _strip_markdown_display(segment)

            if marker_starts and not claim_text:
                previous = claims[-1] if claims else None
                if (
                    previous
                    and not previous["marker_starts"]
                    and line_index - int(previous["line_index"]) <= 1
                ):
                    previous["marker_starts"].extend(marker_starts)
                else:
                    ignored_markers.update(marker_starts)
                continue

            if claim_text and not _is_display_heading(segment, claim_text):
                claims.append(
                    {
                        "text": claim_text,
                        "marker_starts": marker_starts,
                        "line_index": line_index,
                        "kind": "sentence",
                    }
                )

        offset += len(raw_with_ending)

    return claims, ignored_markers


def _citation_is_supported(context: str, evidence: EvidenceItem) -> bool:
    context_terms = _semantic_bigrams(context)
    evidence_terms = _semantic_bigrams(evidence.content_snapshot)
    overlap = context_terms & evidence_terms
    denominator = max(1, min(len(context_terms), len(evidence_terms)))
    return len(overlap) >= 2 and len(overlap) / denominator >= 0.12


def build_evidence_set(
    query: str,
    results: Iterable[Dict[str, Any]],
    knowledge_versions: Optional[Dict[int, Dict[str, Any]]] = None,
    allowed_document_ids: Optional[Set[int]] = None,
    retrieval_status: str = "completed",
) -> EvidenceSet:
    versions = knowledge_versions or {}
    items: List[EvidenceItem] = []
    seen: Set[Tuple[str, int, str]] = set()
    for result in results:
        raw_doc_id = result.get("doc_id", result.get("document_id"))
        try:
            doc_id_int = int(raw_doc_id)
        except (TypeError, ValueError):
            continue
        if allowed_document_ids is not None and doc_id_int not in allowed_document_ids:
            continue
        content = str(result.get("content") or result.get("chunk_text") or "")
        if not content:
            continue
        title = str(result.get("doc_title") or result.get("title") or "")
        if _is_structural_content(content, title):
            continue
        chunk_index = int(result.get("chunk_index") or 0)
        chunk_digest = str(result.get("chunk_hash") or content_hash(content))
        dedupe_key = (str(doc_id_int), chunk_index, chunk_digest)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        version = versions.get(doc_id_int) or {}
        document_hash = str(
            result.get("document_hash")
            or version.get("document_hash")
            or content_hash({"document_id": doc_id_int, "title": result.get("doc_title")})
        )
        document_version = str(
            result.get("document_version")
            or version.get("document_version")
            or document_hash[:16]
        )
        chunk_id = str(
            result.get("chunk_id")
            or f"doc-{doc_id_int}-v-{document_version}-chunk-{chunk_index}"
        )
        evidence_id = stable_id(
            "ev",
            {
                "document_id": doc_id_int,
                "document_version": document_version,
                "chunk_id": chunk_id,
                "chunk_hash": chunk_digest,
            },
        )
        sources = result.get("retrieval_sources") or [result.get("source") or "unknown"]
        retrieval_mode = "+".join(sorted({str(item) for item in sources if item}))
        items.append(
            EvidenceItem(
                evidence_id=evidence_id,
                knowledge_id=str(result.get("knowledge_id") or doc_id_int),
                knowledge_version=str(result.get("knowledge_version") or document_version),
                document_id=str(doc_id_int),
                document_version=document_version,
                document_hash=document_hash,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                chunk_hash=chunk_digest,
                content_snapshot=content,
                retrieval_score=(
                    float(result["score"]) if result.get("score") is not None else None
                ),
                retrieval_mode=retrieval_mode or "unknown",
                title=title,
            )
        )
    return EvidenceSet(items=items, query=query, retrieval_status=retrieval_status)


def prompt_evidence_allowlist(evidence: EvidenceSet) -> str:
    if not evidence.items:
        return (
            "本次没有可引用的检索证据。不得生成引用标记。"
            "对于制度、时效、收费、权限、流程和安全责任，不得给出确定性结论，"
            "不得将“未检索到”说成“文档没有规定”，也不得补充行业经验、数字或步骤。"
        )
    lines = [
        "只能使用以下完整 evidence_id 引用，格式为 [[evidence:ev_xxx]]；"
        "不得省略 `ev_` 前缀、不得自行编造 ID。凡使用证据中的事实，"
        "必须在对应句末放置标记；用户明确要求引用时至少引用一条匹配证据。",
        "制度、时效、收费、权限、流程和安全责任只能依据下列证据回答；"
        "数字、时间、金额、条件和规则必须由同一句末的引用直接支持。"
        "证据未覆盖的部分必须明确说无法确认，不得用行业常识补全：",
    ]
    for item in evidence.items:
        lines.append(
            f"- {item.evidence_id} | {item.title} | chunk={item.chunk_id}\n"
            f"  {item.content_snapshot}"
        )
    return "\n".join(lines)


def render_citations(
    answer: str,
    evidence: EvidenceSet,
    tool_invocations: Optional[Iterable[Any]] = None,
    skill_sources: Optional[Iterable[Dict[str, Any]]] = None,
    action_receipts: Optional[Iterable[Any]] = None,
) -> Tuple[str, List[Citation], List[Dict[str, Any]]]:
    """Validate model markers, render indices and return UI-safe snapshots."""
    by_id = evidence.by_id()
    ordered_ids: List[str] = []
    violations: List[Dict[str, Any]] = []
    # User-provided facts and evidence activated in this exact run are legal
    # support. Skill support is deliberately not rendered as a fake RAG citation.
    grounded_critical_values: Set[str] = set(_critical_values(evidence.query))
    for skill in skill_sources or []:
        grounded_critical_values.update(
            _critical_values(str(skill.get("content_snapshot") or ""))
        )
    for receipt in action_receipts or []:
        if isinstance(receipt, dict):
            status = receipt.get("status")
            result = receipt.get("result")
        else:
            status = getattr(receipt, "status", None)
            result = getattr(receipt, "result", None)
        if status == "committed":
            grounded_critical_values.update(_critical_values(str(result or "")))
    calculation_inputs = _calculation_input_values(evidence.query)
    tool_result_numbers = _successful_tool_result_numbers(tool_invocations or [])

    def normalize_evidence_id(raw_evidence_id: str) -> str:
        evidence_id = raw_evidence_id.strip()
        if evidence_id not in by_id and f"ev_{evidence_id}" in by_id:
            return f"ev_{evidence_id}"
        return evidence_id

    # Compatibility for V1.7 prompts that still emit positional markers. Turn
    # them into IDs first; otherwise a newly rendered [1] could be mistaken for
    # the first retrieval candidate even when it came from a different ID.
    def replace_legacy(match: re.Match) -> str:
        index = int(match.group(1) or match.group(2))
        if index < 1 or index > len(evidence.items):
            violations.append(
                {"code": "invalid_positional_citation", "index": index}
            )
            return ""
        evidence_id = evidence.items[index - 1].evidence_id
        return f"[[evidence:{evidence_id}]]"

    normalized = LEGACY_MARKER.sub(replace_legacy, answer or "")
    claims, ignored_marker_starts = _build_atomic_claims(normalized)
    marker_decisions: Dict[int, Tuple[str, Optional[str]]] = {}

    def resolved_evidence_id(raw_evidence_id: str) -> str:
        evidence_id = normalize_evidence_id(raw_evidence_id)
        # Some providers preserve the stable hash but omit the readable
        # ``ev_`` namespace prefix. Accept only the unambiguous prefixed form.
        if evidence_id not in by_id and f"ev_{evidence_id}" in by_id:
            evidence_id = f"ev_{evidence_id}"
        return evidence_id

    all_markers = {
        match.start(): match for match in EVIDENCE_MARKER.finditer(normalized)
    }
    for marker_start in ignored_marker_starts:
        marker_decisions[marker_start] = ("ignored_display", None)

    for claim in claims:
        marker_starts = list(claim.get("marker_starts") or [])
        if not marker_starts:
            continue
        claim_text = str(claim.get("text") or "")
        valid_markers: List[Tuple[int, str]] = []
        for marker_start in marker_starts:
            marker = all_markers.get(marker_start)
            if marker is None:
                continue
            evidence_id = resolved_evidence_id(marker.group(1))
            if evidence_id not in by_id:
                violations.append(
                    {"code": "invalid_evidence_id", "evidence_id": evidence_id}
                )
                marker_decisions[marker_start] = ("rejected", None)
                continue
            valid_markers.append((marker_start, evidence_id))

        if not valid_markers:
            continue
        supporting_ids = {
            evidence_id
            for _, evidence_id in valid_markers
            if _citation_is_supported(claim_text, by_id[evidence_id])
        }
        if not supporting_ids:
            for marker_start, evidence_id in valid_markers:
                violations.append(
                    {
                        "code": "unsupported_evidence_citation",
                        "evidence_id": evidence_id,
                        "claim_context": claim_text[:240],
                    }
                )
                marker_decisions[marker_start] = ("rejected", None)
            continue

        context_values = _critical_values(claim_text)
        supporting_evidence_values: Set[str] = set()
        for evidence_id in supporting_ids:
            supporting_evidence_values.update(
                _critical_values(by_id[evidence_id].content_snapshot)
            )
        unsupported_values = sorted(
            value
            for value in context_values - supporting_evidence_values - calculation_inputs
            if not _tool_supports_critical_value(value, tool_result_numbers)
        )
        if unsupported_values:
            violations.append(
                {
                    "code": "unsupported_critical_value",
                    "evidence_id": sorted(supporting_ids)[0],
                    "evidence_ids": sorted(supporting_ids),
                    "values": unsupported_values,
                    "claim_context": claim_text[:240],
                }
            )
            for marker_start, _ in valid_markers:
                marker_decisions[marker_start] = ("rejected", None)
            continue

        grounded_critical_values.update(context_values & supporting_evidence_values)
        for marker_start, evidence_id in valid_markers:
            marker_decisions[marker_start] = ("accepted", evidence_id)

    def replace_id(match: re.Match) -> str:
        decision, evidence_id = marker_decisions.get(
            match.start(), ("ignored_display", None)
        )
        if decision != "accepted" or not evidence_id:
            return ""
        if evidence_id not in ordered_ids:
            ordered_ids.append(evidence_id)
        return f"【引用{ordered_ids.index(evidence_id) + 1}】"

    rendered = EVIDENCE_MARKER.sub(replace_id, normalized)
    # If a citation-only display line could not be attached to an immediately
    # preceding fact, remove its now-empty arrow rather than leaving UI debris.
    rendered = re.sub(r"(?m)^[ \t]*(?:>\s*)?[—–-]\s*$\r?\n?", "", rendered)

    def remove_unstructured_marker(match: re.Match) -> str:
        violations.append(
            {
                "code": "unstructured_reference_marker",
                "marker": match.group(1).strip()[:160],
            }
        )
        return ""

    rendered = UNSTRUCTURED_MARKER.sub(remove_unstructured_marker, rendered)
    ungrounded_values = sorted(
        value
        for value in (
            _critical_values(answer or "")
            - grounded_critical_values
            - calculation_inputs
        )
        if not _tool_supports_critical_value(value, tool_result_numbers)
    )
    if ungrounded_values:
        violations.append(
            {
                "code": "ungrounded_critical_value",
                "values": ungrounded_values,
            }
        )
    citations: List[Citation] = []
    for index, evidence_id in enumerate(ordered_ids, start=1):
        item = by_id[evidence_id]
        citations.append(
            Citation(
                index=index,
                evidence_id=evidence_id,
                label=f"[{index}] {item.title}",
                title=item.title,
                document_id=item.document_id,
                document_version=item.document_version,
                chunk_id=item.chunk_id,
                chunk_index=item.chunk_index,
                content_snapshot=item.content_snapshot,
                retrieval_score=item.retrieval_score,
                retrieval_mode=item.retrieval_mode,
            )
        )
    return rendered, citations, violations
