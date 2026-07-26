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


def _citation_context(answer: str, match: re.Match) -> str:
    """Return one sentence/list item, not the surrounding Markdown section."""

    line_start = answer.rfind("\n", 0, match.start()) + 1
    line_end = answer.find("\n", match.end())
    if line_end < 0:
        line_end = len(answer)
    line = answer[line_start:line_end]
    marker_start = match.start() - line_start
    marker_end = match.end() - line_start

    visible_line = EVIDENCE_MARKER.sub("", line)
    visible_line = re.sub(r"^[\s#>*+-]+", "", visible_line)
    visible_line = re.sub(r"[*_`\s]+", " ", visible_line).strip()
    source_label = re.fullmatch(
        r"(?:依据|引用来源|参考文档)(?:[：:].{0,80})?",
        visible_line,
    )
    if source_label:
        next_start = line_end
        while next_start < len(answer) and answer[next_start] in "\r\n":
            next_start += 1
        next_end = answer.find("\n", next_start)
        if next_end < 0:
            next_end = len(answer)
        next_line = answer[next_start:next_end].strip()
        return "\n".join(part for part in (line.strip(), next_line) if part)

    prefix = line[:marker_start].rstrip()
    boundary_search_end = len(prefix.rstrip("。！？；.!?; "))
    sentence_start = max(
        (prefix.rfind(token, 0, boundary_search_end) for token in "。！？；.!?;"),
        default=-1,
    ) + 1
    sentence_ends = [
        position + 1
        for token in "。！？；.!?;"
        if (position := line.find(token, marker_end)) >= 0
    ]
    sentence_end = min(sentence_ends) if sentence_ends else len(line)
    return line[sentence_start:sentence_end].strip()


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
) -> Tuple[str, List[Citation], List[Dict[str, Any]]]:
    """Validate model markers, render indices and return UI-safe snapshots."""
    by_id = evidence.by_id()
    ordered_ids: List[str] = []
    violations: List[Dict[str, Any]] = []
    grounded_critical_values: Set[str] = set()
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

    def replace_id(match: re.Match) -> str:
        evidence_id = normalize_evidence_id(match.group(1))
        # Some providers preserve the stable hash but omit the readable
        # ``ev_`` namespace prefix.  Accept that one unambiguous formatting
        # variation, then immediately normalize it back to the immutable ID.
        # A suffix that matches zero or multiple EvidenceItems stays invalid.
        if evidence_id not in by_id:
            suffix_matches = [
                candidate
                for candidate in by_id
                if candidate == f"ev_{evidence_id}"
            ]
            if len(suffix_matches) == 1:
                evidence_id = suffix_matches[0]
        if evidence_id not in by_id:
            violations.append(
                {"code": "invalid_evidence_id", "evidence_id": evidence_id}
            )
            return ""
        context = _citation_context(normalized, match)
        if not _citation_is_supported(context, by_id[evidence_id]):
            violations.append(
                {
                    "code": "unsupported_evidence_citation",
                    "evidence_id": evidence_id,
                    "claim_context": context[:240],
                }
            )
            return ""
        context_values = _critical_values(context)
        attached_evidence_values: Set[str] = set()
        for attached_match in EVIDENCE_MARKER.finditer(context):
            attached_id = normalize_evidence_id(attached_match.group(1))
            if attached_id in by_id:
                attached_evidence_values.update(
                    _critical_values(by_id[attached_id].content_snapshot)
                )
        unsupported_values = sorted(
            value
            for value in context_values - attached_evidence_values - calculation_inputs
            if not _tool_supports_critical_value(value, tool_result_numbers)
        )
        if unsupported_values:
            violations.append(
                {
                    "code": "unsupported_critical_value",
                    "evidence_id": evidence_id,
                    "values": unsupported_values,
                    "claim_context": context[:240],
                }
            )
            return ""
        grounded_critical_values.update(context_values & attached_evidence_values)
        if evidence_id not in ordered_ids:
            ordered_ids.append(evidence_id)
        return f"【引用{ordered_ids.index(evidence_id) + 1}】"

    rendered = EVIDENCE_MARKER.sub(replace_id, normalized)

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
