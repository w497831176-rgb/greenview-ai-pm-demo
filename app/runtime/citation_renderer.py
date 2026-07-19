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


def _semantic_bigrams(text: str) -> Set[str]:
    chars = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", (text or "").lower())
    return {
        chars[index] + chars[index + 1]
        for index in range(len(chars) - 1)
        if chars[index] + chars[index + 1] not in _GENERIC_BIGRAMS
    }


def _citation_context(answer: str, match: re.Match) -> str:
    line_start = answer.rfind("\n", 0, match.start()) + 1
    # Models often place a marker on a short "依据" line and put the actual
    # supported facts in the immediately following Markdown bullets. Validate
    # the whole local paragraph/section, not only the marker's physical line.
    max_end = min(len(answer), match.end() + 800)
    section_end = answer.find("\n\n", match.end())
    if section_end < 0 or section_end > max_end:
        section_end = max_end
    first_section = answer[line_start:section_end].strip()
    # Models often emit a short source-label line, then a blank line, then the
    # actual supported Markdown list.  Treat that label plus the immediately
    # following paragraph/list as one citation context.
    if (
        section_end < max_end
        and any(marker in first_section for marker in ("引用来源", "依据", "参考文档"))
        and len(first_section) <= 120
    ):
        next_start = section_end + 2
        next_end = answer.find("\n\n", next_start)
        if next_end < 0 or next_end > max_end:
            next_end = max_end
        section_end = next_end
    return answer[line_start:section_end].strip()


def _citation_is_supported(context: str, evidence: EvidenceItem) -> bool:
    overlap = _semantic_bigrams(context) & _semantic_bigrams(
        evidence.content_snapshot
    )
    return len(overlap) >= 2


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
                title=str(result.get("doc_title") or result.get("title") or ""),
            )
        )
    return EvidenceSet(items=items, query=query, retrieval_status=retrieval_status)


def prompt_evidence_allowlist(evidence: EvidenceSet) -> str:
    if not evidence.items:
        return "本次没有可引用的检索证据。不得生成引用标记。"
    lines = [
        "只能使用以下完整 evidence_id 引用，格式为 [[evidence:ev_xxx]]；"
        "不得省略 `ev_` 前缀、不得自行编造 ID。凡使用证据中的事实，"
        "必须在对应句末放置标记；用户明确要求引用时至少引用一条匹配证据："
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
) -> Tuple[str, List[Citation], List[Dict[str, Any]]]:
    """Validate model markers, render indices and return UI-safe snapshots."""
    by_id = evidence.by_id()
    ordered_ids: List[str] = []
    violations: List[Dict[str, Any]] = []

    def replace_id(match: re.Match) -> str:
        evidence_id = match.group(1).strip()
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
        context = _citation_context(answer or "", match)
        if not _citation_is_supported(context, by_id[evidence_id]):
            violations.append(
                {
                    "code": "unsupported_evidence_citation",
                    "evidence_id": evidence_id,
                    "claim_context": context[:240],
                }
            )
            return ""
        if evidence_id not in ordered_ids:
            ordered_ids.append(evidence_id)
        return f"【引用{ordered_ids.index(evidence_id) + 1}】"

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
