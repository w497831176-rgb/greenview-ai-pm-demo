"""Render one answer from the immutable evidence bundle for this run."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.runtime.contracts import (
    ActionReceipt,
    Citation,
    EvidenceItem,
    EvidenceSet,
    RunEvidenceBundle,
    ToolEvidence,
    ToolEvidenceFact,
    ToolEvidenceLink,
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
    r"(?P<unit>个\s*工作日|个\s*种植舱|工作日|种植舱|分钟|小时|天|份|元|万元|%|℃|°C|摄氏度|级|次|年|个月|月|日)"
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
_TOOL_FACT_LABELS: Dict[str, Tuple[str, ...]] = {
    "business_status": ("status", "\u72b6\u6001", "\u8fdb\u5ea6"),
    "business_identifier": ("id", "\u7f16\u53f7", "\u5355\u53f7"),
    "weather_condition": (
        "condition",
        "\u5929\u6c14",
        "\u5929\u51b5",
        "\u72b6\u51b5",
    ),
    "wind_condition": ("wind", "\u98ce\u5411", "\u98ce\u529b", "\u98ce"),
    "location": ("city", "location", "\u57ce\u5e02", "\u5730\u70b9"),
}
_TOOL_FACT_UNCERTAINTY = re.compile(
    r"\u672a\u77e5|\u65e0\u6cd5\u6838\u5b9e|\u672a\u63d0\u4f9b|\u4ee5.+\u4e3a\u51c6|unknown|unavailable",
    re.I,
)
_TOOL_CLAUSE_SPLIT = re.compile(r"[\r\n\u3002\uff01\uff1f\uff1b;]+")
_TOOL_SUBCLAUSE_SPLIT = re.compile(
    r"(?:[\r\n\u3002\uff01\uff1f\uff1b;\uff0c,\u3001\uff1a\u2014\u2013]"
    r"|(?<!evidence):|\u5e76\u4e14|\u800c\u4e14|\u540c\u65f6|\u4ee5\u53ca|"
    r"\u53e6\u5916|\u6b64\u5916|\u4f46\u662f|\u4e0d\u8fc7|\u4e14|[()\uff08\uff09])+"
)
_GENERIC_IDENTIFIER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9]*[-_]"
    r"[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*(?![A-Za-z0-9_-])"
)
_TOOL_FIELD_TOKEN_LABELS: Dict[str, Tuple[str, ...]] = {
    "humidity": ("\u6e7f\u5ea6", "humidity"),
    "temperature": ("\u6e29\u5ea6", "\u6c14\u6e29", "temperature"),
    "temp": ("\u6e29\u5ea6", "\u6c14\u6e29", "temp"),
    "rain": ("\u964d\u96e8", "\u4e0b\u96e8", "rain"),
    "precipitation": ("\u964d\u6c34", "\u964d\u96e8", "precipitation"),
    "probability": ("\u6982\u7387", "\u53ef\u80fd\u6027", "probability"),
    "chance": ("\u6982\u7387", "\u53ef\u80fd\u6027", "chance"),
    "wind": ("\u98ce\u529b", "\u98ce\u901f", "\u98ce\u5411", "wind"),
    "amount": ("\u91d1\u989d", "\u6570\u989d", "amount"),
    "cost": ("\u8d39\u7528", "\u6210\u672c", "cost"),
    "price": ("\u4ef7\u683c", "\u5355\u4ef7", "price"),
    "fee": ("\u8d39\u7528", "\u6536\u8d39", "fee"),
    "count": ("\u6570\u91cf", "\u4e2a\u6570", "count"),
    "quantity": ("\u6570\u91cf", "quantity"),
    "duration": ("\u65f6\u957f", "duration"),
}
_TOOL_FIELD_UNIT_TOKENS = {
    "pct",
    "percent",
    "percentage",
    "c",
    "celsius",
    "minute",
    "minutes",
    "hour",
    "hours",
    "day",
    "days",
    "level",
    "grade",
}
_SELF_DESCRIBING_TOOL_UNITS = {"\u2103", "\u5143"}
_WEATHER_CONDITION_TOKEN = re.compile(
    r"\u6674\u6717|\u6674|\u591a\u4e91|\u9634\u5929|\u9634|"
    r"\u5c0f\u96e8|\u4e2d\u96e8|\u5927\u96e8|\u66b4\u96e8|\u96e8|"
    r"\u5c0f\u96ea|\u4e2d\u96ea|\u5927\u96ea|\u66b4\u96ea|\u96ea|"
    r"\u96fe|\u973e|\u96f7\u9635\u96e8|clear|cloudy|overcast|rain|snow|fog",
    re.I,
)
_IMPLICIT_STATUS_TOKEN = re.compile(
    r"\u5f85[\u4e00-\u9fff]{1,6}|\u5df2[\u4e00-\u9fff]{1,6}|"
    r"\u672a[\u4e00-\u9fff]{1,6}|\u5904\u7406\u4e2d|\u8fdb\u884c\u4e2d|"
    r"pending|open|closed|completed|cancelled|canceled|processing",
    re.I,
)
_WIND_DIRECTION_TOKEN = re.compile(
    r"(?:\u4e1c\u5317|\u4e1c\u5357|\u897f\u5317|\u897f\u5357|"
    r"\u504f\u4e1c|\u504f\u897f|\u504f\u5357|\u504f\u5317|"
    r"\u65e0\u6301\u7eed|\u65cb\u8f6c|\u4e1c|\u897f|\u5357|\u5317)\u98ce"
)
_VIOLATION_DETAILS = {
    "invalid_positional_citation": (
        "The answer referenced a positional citation outside this run's evidence."
    ),
    "invalid_evidence_id": (
        "The answer referenced an evidence ID outside the frozen run bundle."
    ),
    "unsupported_evidence_citation": (
        "The cited RAG snapshot did not support the associated claim."
    ),
    "unsupported_tool_evidence_marker": (
        "The referenced ToolEvidence did not support the associated claim."
    ),
    "unsupported_critical_value": (
        "The cited RAG evidence did not support every critical value in the claim."
    ),
    "unstructured_reference_marker": (
        "The answer emitted a reference marker that is not a governed evidence ID."
    ),
    "ungrounded_critical_value": (
        "The answer contained a critical value unsupported by frozen run evidence."
    ),
    "unsupported_tool_fact": (
        "The answer changed a categorical fact from frozen ToolEvidence."
    ),
}
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
        elif unit in {"°c", "摄氏度"}:
            unit = "℃"
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


def _canonical_unit(unit: Optional[str]) -> Optional[str]:
    normalized = re.sub(r"\s+", "", str(unit or ""))
    aliases = {
        "percent": "%",
        "percentage": "%",
        "pct": "%",
        "celsius": "℃",
        "°c": "℃",
        "°C": "℃",
        "摄氏度": "℃",
        "minutes": "分钟",
        "minute": "分钟",
        "mins": "分钟",
        "seconds": "秒",
        "hours": "小时",
        "days": "天",
        "yuan": "元",
        "cny": "元",
        "level": "级",
    }
    return aliases.get(normalized, normalized or None)


def _fact_critical_values(fact: ToolEvidenceFact) -> Set[str]:
    """Return exact value+unit facts; a number alone never gains a business unit."""

    values = _critical_values(str(fact.display_value or ""))
    values.update(_critical_values(str(fact.value or "")))
    unit = _canonical_unit(fact.unit)
    if unit and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", fact.normalized_value):
        values.add(f"{_normalize_number(fact.normalized_value)}{unit}")
    return values


def _tool_support_index(
    bundle: RunEvidenceBundle,
) -> Tuple[Dict[str, List[Tuple[ToolEvidence, ToolEvidenceFact]]], Dict[str, List[Tuple[ToolEvidence, ToolEvidenceFact]]]]:
    exact: Dict[str, List[Tuple[ToolEvidence, ToolEvidenceFact]]] = {}
    calculated: Dict[str, List[Tuple[ToolEvidence, ToolEvidenceFact]]] = {}
    for evidence in bundle.tool_evidence:
        for fact in evidence.facts:
            for value in _fact_critical_values(fact):
                exact.setdefault(value, []).append((evidence, fact))
            if (
                fact.semantic_type == "calculated_result"
                and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", fact.normalized_value)
            ):
                calculated.setdefault(
                    _normalize_number(fact.normalized_value), []
                ).append((evidence, fact))
    return exact, calculated


def _tool_supports_critical_value(
    value: str,
    exact: Dict[str, List[Tuple[ToolEvidence, ToolEvidenceFact]]],
    calculated: Dict[str, List[Tuple[ToolEvidence, ToolEvidenceFact]]],
) -> bool:
    if value in exact:
        return True
    match = re.match(r"(?P<number>[-+]?\d+(?:\.\d+)?)", value or "")
    return bool(
        match and _normalize_number(match.group("number")) in calculated
    )


def _tool_links_for_answer(
    answer: str,
    bundle: RunEvidenceBundle,
) -> List[ToolEvidenceLink]:
    tool_segments = [
        segment
        for scope in _tool_claim_scopes(answer, bundle)
        for segment in scope
    ]
    linked: Dict[Tuple[str, str], Dict[str, Set[str]]] = {}

    def add_link(
        evidence: ToolEvidence,
        fact: ToolEvidenceFact,
        claim_value: str,
    ) -> None:
        key = (evidence.evidence_id, evidence.invocation_id)
        slot = linked.setdefault(key, {"fact_ids": set(), "claim_values": set()})
        slot["fact_ids"].add(fact.fact_id)
        slot["claim_values"].add(claim_value)

    groups = _tool_fact_groups(bundle)
    for segment in tool_segments:
        for claim_value in _critical_values(segment):
            for evidence, fact in _scoped_numeric_support_pairs(
                segment, bundle, claim_value
            ):
                add_link(evidence, fact, claim_value)

    # Exact strings such as weather condition and business identifiers are
    # linkable only from the selector-scoped JSON parent group.
    for segment in tool_segments:
        for group in _select_tool_groups(segment, groups):
            evidence = group["evidence"]
            for fact in group["facts"]:
                display = str(fact.display_value or fact.value or "").strip()
                if (
                    len(display) >= 2
                    and not re.fullmatch(
                        r"(?:true|false|null|success|ok|\u6210\u529f)",
                        display,
                        re.I,
                    )
                    and display in segment
                ):
                    add_link(evidence, fact, display)

    return [
        ToolEvidenceLink(
            evidence_id=evidence_id,
            invocation_id=invocation_id,
            fact_ids=sorted(payload["fact_ids"]),
            claim_values=sorted(payload["claim_values"]),
        )
        for (evidence_id, invocation_id), payload in sorted(linked.items())
    ]


def _unsupported_tool_fact_claims(
    answer: str,
    bundle: RunEvidenceBundle,
) -> List[Dict[str, Any]]:
    """Reject explicit categorical Tool claims not present in frozen facts.

    Labels are semantic, not server/tool specific. A clause is first scoped by
    an exact location or business identifier when available, then checked
    against exact categorical values from the same invocation.
    """

    groups: List[Dict[str, Any]] = []
    for evidence in bundle.tool_evidence:
        by_parent: Dict[str, List[ToolEvidenceFact]] = {}
        for fact in evidence.facts:
            parent = re.sub(
                r"(?:\.[A-Za-z_][A-Za-z0-9_]*|\[[^\]]+\])$",
                "",
                fact.json_path,
            )
            by_parent.setdefault(parent or "$", []).append(fact)
        groups.extend(
            {
                "evidence": evidence,
                "parent": parent,
                "facts": facts,
            }
            for parent, facts in by_parent.items()
        )

    def fact_value(fact: ToolEvidenceFact) -> str:
        return str(fact.display_value or fact.value or "").strip()

    all_identifier_values = sorted(
        {
            fact_value(fact)
            for group in groups
            for fact in group["facts"]
            if fact.semantic_type == "business_identifier" and fact_value(fact)
        },
        key=len,
        reverse=True,
    )

    violations: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    def add_violation(
        group: Dict[str, Any],
        semantic_type: str,
        value: str,
        expected: Iterable[str],
        context: str,
    ) -> None:
        evidence = group["evidence"]
        key = (evidence.evidence_id, group["parent"], semantic_type, value)
        if key in seen:
            return
        seen.add(key)
        violations.append(
            {
                "code": "unsupported_tool_fact",
                "evidence_id": evidence.evidence_id,
                "evidence_ids": [evidence.evidence_id],
                "json_path": group["parent"],
                "semantic_type": semantic_type,
                "values": [value],
                "expected_values": sorted(set(expected)),
                "claim_context": context[:240],
            }
        )

    def local_context(clause: str, group: Dict[str, Any]) -> str:
        own_ids = [
            fact_value(fact)
            for fact in group["facts"]
            if fact.semantic_type == "business_identifier"
            and fact_value(fact) in clause
        ]
        if not own_ids:
            return clause
        identifier_occurrences = [
            (clause.find(value), value)
            for value in all_identifier_values
            if clause.find(value) >= 0
        ]
        # A single structured object mentioned in one subclause owns both the
        # text before and after its identifier. This covers natural Chinese
        # order such as "状态已完成的工单是 WO-100" without borrowing facts
        # from a neighbouring row.
        if len(identifier_occurrences) <= 1:
            return clause
        start = min(clause.find(value) for value in own_ids)
        later = [
            clause.find(value, start + 1)
            for value in all_identifier_values
            if clause.find(value, start + 1) > start
        ]
        end = min(later) if later else len(clause)
        return clause[start:end]

    def identifier_family_tokens(
        group: Dict[str, Any], context: str
    ) -> Set[str]:
        tokens: Set[str] = set()
        for fact in group["facts"]:
            if fact.semantic_type != "business_identifier":
                continue
            expected_id = fact_value(fact)
            prefix_match = re.match(
                r"(?P<prefix>[A-Za-z][A-Za-z_-]*)(?=\d)", expected_id
            )
            if prefix_match is None:
                continue
            tokens.update(
                re.findall(
                    rf"(?<![A-Za-z0-9_-]){re.escape(prefix_match.group('prefix'))}"
                    rf"[A-Za-z0-9_-]+",
                    context,
                    flags=re.I,
                )
            )
        return tokens

    sentence_tool_segments = _tool_claim_scopes(answer, bundle)
    clauses = [
        clause
        for segments in sentence_tool_segments
        for clause in segments
    ]
    for clause in clauses:
        selector_groups = [
            group
            for group in groups
            if any(
                fact.semantic_type in {"business_identifier", "location"}
                and fact_value(fact)
                and fact_value(fact) in clause
                for fact in group["facts"]
            )
        ]
        family_groups = [
            group for group in groups if identifier_family_tokens(group, clause)
        ]
        exact_groups = [
            group
            for group in groups
            if any(
                len(fact_value(fact)) >= 2 and fact_value(fact) in clause
                for fact in group["facts"]
            )
        ]
        has_condition_claim = bool(_WEATHER_CONDITION_TOKEN.search(clause))
        has_status_claim = bool(
            re.search(r"\u72b6\u6001|\u8fdb\u5ea6|status", clause, re.I)
            or _IMPLICIT_STATUS_TOKEN.search(clause)
        )
        has_wind_claim = bool(
            _WIND_DIRECTION_TOKEN.search(clause)
            or re.search(r"\u98ce\u5411|\u98ce\u529b|wind", clause, re.I)
        )
        inferred_groups: List[Dict[str, Any]] = []
        if has_condition_claim or has_status_claim or has_wind_claim:
            inferred_groups = [
                group
                for group in groups
                if any(
                    (has_condition_claim and fact.semantic_type == "weather_condition")
                    or (has_status_claim and fact.semantic_type == "business_status")
                    or (has_wind_claim and fact.semantic_type == "wind_condition")
                    for fact in group["facts"]
                )
            ]
        candidates = selector_groups or family_groups or exact_groups or inferred_groups
        # A RAG/Skill-only clause containing words such as "status" is not a
        # Tool claim unless it also carries an exact fact from one Tool object.
        for group in candidates:
            context = local_context(clause, group)
            group_values = {
                fact_value(fact)
                for fact in group["facts"]
                if fact_value(fact)
            }
            for semantic_type, labels in _TOOL_FACT_LABELS.items():
                if semantic_type == "wind_condition":
                    continue
                semantic_facts = [
                    fact
                    for fact in group["facts"]
                    if fact.semantic_type == semantic_type
                ]
                if not semantic_facts:
                    continue
                expected = sorted(
                    {fact_value(fact) for fact in semantic_facts if fact_value(fact)}
                )
                label_pattern = "|".join(
                    (
                        rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])"
                        if re.fullmatch(r"[A-Za-z0-9_]+", label)
                        else re.escape(label)
                    )
                    for label in labels
                )
                pattern = re.compile(
                    rf"(?:{label_pattern})\s*(?:\u4e3a|\u662f|[:\uff1a])?\s*"
                    rf"(?P<claim>[^\uff0c,\u3002\uff1b;\uff01!\uff1f?\u3001]{{1,32}})",
                    re.I,
                )
                for match in pattern.finditer(context):
                    claim = str(match.group("claim") or "").strip()
                    if not claim or _TOOL_FACT_UNCERTAINTY.search(claim):
                        continue
                    if any(value in claim for value in expected):
                        continue
                    # A broad label such as "weather" may be followed by a
                    # different exact fact (for example temperature).
                    if any(value and value in claim for value in group_values):
                        continue
                    add_violation(
                        group,
                        semantic_type,
                        claim,
                        expected,
                        context,
                    )

            condition_values = {
                fact_value(fact)
                for fact in group["facts"]
                if fact.semantic_type == "weather_condition" and fact_value(fact)
            }
            if condition_values:
                condition_tokens = set(_WEATHER_CONDITION_TOKEN.findall(context))
                unsupported = sorted(
                    token
                    for token in condition_tokens
                    if not any(
                        token in expected_value or expected_value in token
                        for expected_value in condition_values
                    )
                )
                for token in unsupported:
                    add_violation(
                        group,
                        "weather_condition",
                        token,
                        condition_values,
                        context,
                    )

            direction_values: Set[str] = set()
            for fact in group["facts"]:
                if fact.semantic_type != "wind_condition" or not fact_value(fact):
                    continue
                extracted = set(_WIND_DIRECTION_TOKEN.findall(fact_value(fact)))
                if extracted:
                    direction_values.update(extracted)
                elif "direction" in fact.field_name.lower():
                    direction_values.add(fact_value(fact))
            if direction_values:
                direction_tokens = set(_WIND_DIRECTION_TOKEN.findall(context))
                unsupported = sorted(
                    token
                    for token in direction_tokens
                    if not any(
                        token in expected_value or expected_value in token
                        for expected_value in direction_values
                    )
                )
                for token in unsupported:
                    add_violation(
                        group,
                        "wind_direction",
                        token,
                        direction_values,
                        context,
                    )

            identifier_values = {
                fact_value(fact)
                for fact in group["facts"]
                if fact.semantic_type == "business_identifier" and fact_value(fact)
            }
            family = identifier_family_tokens(group, context)
            for token in sorted(family - identifier_values):
                add_violation(
                    group,
                    "business_identifier",
                    token,
                    identifier_values,
                    context,
                )

            status_values = {
                fact_value(fact)
                for fact in group["facts"]
                if fact.semantic_type == "business_status" and fact_value(fact)
            }
            if status_values and identifier_values:
                for identifier in identifier_values:
                    if identifier not in context:
                        continue
                    tail = context.split(identifier, 1)[1][:40]
                    status_match = _IMPLICIT_STATUS_TOKEN.search(tail)
                    if (
                        status_match is not None
                        and not any(value in tail for value in status_values)
                    ):
                        add_violation(
                            group,
                            "business_status",
                            status_match.group(0),
                            status_values,
                            context,
                        )

    # Validate identifier/status relationships independently from global
    # substring links.  A correct status from one result row must not make a
    # different (or invented) identifier look grounded.  The association is
    # deliberately based on punctuation-delimited local subclauses rather
    # than tool/server names or domain-specific IDs.
    relation_groups = [
        group
        for group in groups
        if any(
            fact.semantic_type == "business_identifier"
            for fact in group["facts"]
        )
        and any(
            fact.semantic_type == "business_status"
            for fact in group["facts"]
        )
    ]
    all_expected_identifiers = {
        fact_value(fact)
        for group in relation_groups
        for fact in group["facts"]
        if fact.semantic_type == "business_identifier" and fact_value(fact)
    }
    all_expected_statuses = {
        fact_value(fact)
        for group in relation_groups
        for fact in group["facts"]
        if fact.semantic_type == "business_status" and fact_value(fact)
    }
    for tool_segments in sentence_tool_segments:
        if not tool_segments:
            continue
        searchable_sentence = "\uff0c".join(tool_segments)
        sentence_has_status = bool(
            re.search(r"\u72b6\u6001|\u8fdb\u5ea6|status", searchable_sentence, re.I)
            or _IMPLICIT_STATUS_TOKEN.search(searchable_sentence)
            or any(status in searchable_sentence for status in all_expected_statuses)
        )
        if sentence_has_status and relation_groups:
            unknown_identifiers = sorted(
                set(_GENERIC_IDENTIFIER_TOKEN.findall(searchable_sentence))
                - all_expected_identifiers
            )
            for identifier in unknown_identifiers:
                likely_groups = [
                    group
                    for group in relation_groups
                    if any(
                        fact.semantic_type == "business_status"
                        and fact_value(fact)
                        and fact_value(fact) in searchable_sentence
                        for fact in group["facts"]
                    )
                ] or relation_groups
                group = likely_groups[0]
                add_violation(
                    group,
                    "business_identifier",
                    identifier,
                    all_expected_identifiers,
                    searchable_sentence,
                )

        for segment in tool_segments:
            if not segment:
                continue
            segment_has_status = bool(
                re.search(r"\u72b6\u6001|\u8fdb\u5ea6|status", segment, re.I)
                or _IMPLICIT_STATUS_TOKEN.search(segment)
                or any(status in segment for status in all_expected_statuses)
            )
            if not segment_has_status:
                continue
            for group in relation_groups:
                identifiers = {
                    fact_value(fact)
                    for fact in group["facts"]
                    if fact.semantic_type == "business_identifier"
                    and fact_value(fact)
                }
                mentioned = sorted(
                    identifier for identifier in identifiers if identifier in segment
                )
                if not mentioned:
                    continue
                statuses = {
                    fact_value(fact)
                    for fact in group["facts"]
                    if fact.semantic_type == "business_status" and fact_value(fact)
                }
                if statuses and not any(status in segment for status in statuses):
                    add_violation(
                        group,
                        "business_status",
                        segment,
                        statuses,
                        segment,
                    )
    return violations


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


def _tool_fact_groups(bundle: RunEvidenceBundle) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for evidence in bundle.tool_evidence:
        by_parent: Dict[str, List[ToolEvidenceFact]] = {}
        for fact in evidence.facts:
            parent = re.sub(
                r"(?:\.[A-Za-z_][A-Za-z0-9_]*|\[[^\]]+\])$",
                "",
                fact.json_path,
            )
            by_parent.setdefault(parent or "$", []).append(fact)
        groups.extend(
            {
                "evidence": evidence,
                "parent": parent,
                "facts": facts,
            }
            for parent, facts in by_parent.items()
        )
    return groups


def _tool_fact_domain_labels(fact: ToolEvidenceFact) -> Set[str]:
    """Return generic schema semantics, excluding unit-only field suffixes."""

    normalized = re.sub(r"[^a-z0-9]+", "_", fact.field_name.lower()).strip("_")
    labels: Set[str] = set()
    for token in (item for item in normalized.split("_") if item):
        if token in _TOOL_FIELD_UNIT_TOKENS:
            continue
        labels.update(_TOOL_FIELD_TOKEN_LABELS.get(token, ()))
    return labels


def _rag_subclaim_is_supported(
    subclaim: str,
    evidence_items: Iterable[EvidenceItem],
) -> bool:
    """Use the citation contract to distinguish local RAG and Tool clauses."""

    claim = _strip_markdown_display(subclaim)
    if not claim:
        return False
    claim_values = _critical_values(claim)
    claim_identifiers = set(_GENERIC_IDENTIFIER_TOKEN.findall(claim))
    supporting = [
        evidence
        for evidence in evidence_items
        if _citation_is_supported(claim, evidence)
    ]
    if not supporting:
        return False
    supported_values: Set[str] = set()
    supported_identifiers: Set[str] = set()
    for evidence in supporting:
        supported_values.update(_critical_values(evidence.content_snapshot))
        supported_identifiers.update(
            _GENERIC_IDENTIFIER_TOKEN.findall(evidence.content_snapshot)
        )
    return claim_values.issubset(supported_values) and claim_identifiers.issubset(
        supported_identifiers
    )


def _atomic_evidence_scopes(
    answer: str,
    bundle: RunEvidenceBundle,
) -> List[Dict[str, Any]]:
    """Project each citation-level atomic claim into local evidence scopes.

    A sentence-final RAG marker governs the atomic sentence, but only local
    subclauses actually supported by its cited snapshots are excluded from
    Tool validation. Unsupported sibling subclauses remain fail-closed.
    """

    claims, _ = _build_atomic_claims(answer or "")
    all_markers = {
        match.start(): match for match in EVIDENCE_MARKER.finditer(answer or "")
    }
    by_id = bundle.retrieved_rag_candidates.by_id()
    scopes: List[Dict[str, Any]] = []
    for claim in claims:
        cited: List[EvidenceItem] = []
        for marker_start in claim.get("marker_starts") or []:
            marker = all_markers.get(int(marker_start))
            if marker is None:
                continue
            evidence_id = marker.group(1).strip()
            if evidence_id not in by_id and f"ev_{evidence_id}" in by_id:
                evidence_id = f"ev_{evidence_id}"
            if evidence_id in by_id:
                cited.append(by_id[evidence_id])
        segments: List[Dict[str, Any]] = []
        for segment in _TOOL_SUBCLAUSE_SPLIT.split(str(claim.get("text") or "")):
            segment = segment.strip()
            if not segment:
                continue
            segments.append(
                {
                    "text": segment,
                    "rag_supported": bool(
                        cited and _rag_subclaim_is_supported(segment, cited)
                    ),
                }
            )
        scopes.append(
            {
                "claim": claim,
                "cited": cited,
                "segments": segments,
            }
        )
    return scopes


def _tool_claim_scopes(
    answer: str,
    bundle: RunEvidenceBundle,
) -> List[List[str]]:
    """Return Tool-eligible subclauses from the shared atomic scope map."""

    return [
        tool_segments
        for scope in _atomic_evidence_scopes(answer, bundle)
        if (
            tool_segments := [
                str(segment["text"])
                for segment in scope["segments"]
                if not segment["rag_supported"]
            ]
        )
    ]


def _select_tool_groups(
    segment: str,
    groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    selected = [
        group
        for group in groups
        if any(
            fact.semantic_type in {"business_identifier", "location"}
            and str(fact.display_value or fact.value or "").strip()
            and str(fact.display_value or fact.value or "").strip() in segment
            for fact in group["facts"]
        )
    ]
    return selected or groups


def _recognized_tool_domain_labels(segment: str) -> Set[str]:
    return {
        label
        for labels in _TOOL_FIELD_TOKEN_LABELS.values()
        for label in labels
        if label and label in segment
    }


def _selector_scoped_numeric_facts(
    segment: str,
    bundle: RunEvidenceBundle,
) -> List[Tuple[ToolEvidence, ToolEvidenceFact]]:
    groups = _select_tool_groups(segment, _tool_fact_groups(bundle))
    return [
        (group["evidence"], fact)
        for group in groups
        for fact in group["facts"]
        if _fact_critical_values(fact)
        or re.fullmatch(r"[-+]?\d+(?:\.\d+)?", fact.normalized_value)
    ]


def _fact_has_local_tool_context(
    segment: str,
    bundle: RunEvidenceBundle,
    evidence: ToolEvidence,
    fact: ToolEvidenceFact,
) -> bool:
    if fact.semantic_type == "calculated_result":
        return True
    if _canonical_unit(fact.unit) in _SELF_DESCRIBING_TOOL_UNITS:
        return True
    for group in _tool_fact_groups(bundle):
        if group["evidence"].evidence_id != evidence.evidence_id:
            continue
        if not any(item.fact_id == fact.fact_id for item in group["facts"]):
            continue
        for sibling in group["facts"]:
            display = str(sibling.display_value or sibling.value or "").strip()
            if not display:
                continue
            if sibling.fact_id == fact.fact_id:
                non_numeric = _CRITICAL_VALUE.sub("", display).strip()
                if non_numeric and non_numeric in segment:
                    return True
                continue
            if (
                sibling.semantic_type
                in {
                    "weather_condition",
                    "wind_condition",
                    "business_status",
                }
                and display in segment
            ):
                return True
    return False


def _scoped_numeric_facts(
    segment: str,
    bundle: RunEvidenceBundle,
) -> List[Tuple[ToolEvidence, ToolEvidenceFact]]:
    facts = _selector_scoped_numeric_facts(segment, bundle)
    recognized_labels = _recognized_tool_domain_labels(segment)
    if not recognized_labels:
        return [
            pair
            for pair in facts
            if _fact_has_local_tool_context(segment, bundle, pair[0], pair[1])
        ]
    field_scoped = [
        pair
        for pair in facts
        if _tool_fact_domain_labels(pair[1]) & recognized_labels
    ]
    # A recognized field claim such as "humidity" must never fall back to an
    # unrelated same-unit field merely because the expected field is absent.
    return field_scoped


def _scoped_numeric_support_pairs(
    segment: str,
    bundle: RunEvidenceBundle,
    claim_value: str,
) -> List[Tuple[ToolEvidence, ToolEvidenceFact]]:
    pairs: List[Tuple[ToolEvidence, ToolEvidenceFact]] = []
    match = re.match(r"(?P<number>[-+]?\d+(?:\.\d+)?)", claim_value or "")
    claim_number = _normalize_number(match.group("number")) if match else None
    for evidence, fact in _scoped_numeric_facts(segment, bundle):
        if claim_value in _fact_critical_values(fact):
            pairs.append((evidence, fact))
            continue
        if (
            fact.semantic_type == "calculated_result"
            and claim_number is not None
            and _normalize_number(fact.normalized_value) == claim_number
        ):
            pairs.append((evidence, fact))
    return pairs


def _unsupported_tool_numeric_claims(
    answer: str,
    bundle: RunEvidenceBundle,
) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for scope in _tool_claim_scopes(answer, bundle):
        for segment in scope:
            selector_facts = _selector_scoped_numeric_facts(segment, bundle)
            numeric_facts = _scoped_numeric_facts(segment, bundle)
            domain_scoped = bool(_recognized_tool_domain_labels(segment))
            selector_scoped = any(
                fact.semantic_type in {"business_identifier", "location"}
                and str(fact.display_value or fact.value or "").strip() in segment
                for group in _tool_fact_groups(bundle)
                for fact in group["facts"]
                if str(fact.display_value or fact.value or "").strip()
            )
            if not (domain_scoped or selector_scoped):
                continue
            for claim_value in sorted(_critical_values(segment)):
                if _scoped_numeric_support_pairs(segment, bundle, claim_value):
                    continue
                expected = sorted(
                    {
                        value
                        for _, fact in numeric_facts
                        for value in _fact_critical_values(fact)
                    }
                )
                if not expected and domain_scoped:
                    expected = sorted(
                        {
                            value
                            for _, fact in selector_facts
                            for value in _fact_critical_values(fact)
                        }
                    )
                if not expected and not selector_facts:
                    continue
                key = (segment, claim_value)
                if key in seen:
                    continue
                seen.add(key)
                violations.append(
                    {
                        "code": "unsupported_tool_fact",
                        "semantic_type": "structured_numeric_result",
                        "values": [claim_value],
                        "expected_values": expected,
                        "claim_context": segment[:240],
                    }
                )
    return violations


def _atomic_tool_supported_values(
    answer: str,
    bundle: RunEvidenceBundle,
) -> List[Set[str]]:
    result: List[Set[str]] = []
    for scope in _atomic_evidence_scopes(answer, bundle):
        supported: Set[str] = set()
        for segment_info in scope["segments"]:
            if segment_info["rag_supported"]:
                continue
            segment = str(segment_info["text"])
            for claim_value in _critical_values(segment):
                if _scoped_numeric_support_pairs(segment, bundle, claim_value):
                    supported.add(claim_value)
        result.append(supported)
    return result


def _scoped_tool_supported_values(
    answer: str,
    bundle: RunEvidenceBundle,
) -> Set[str]:
    return set().union(*_atomic_tool_supported_values(answer, bundle))


def _source_text_supports_value(
    segment: str,
    source_text: str,
    value: str,
) -> bool:
    if value not in _critical_values(source_text):
        return False
    segment_terms = _semantic_bigrams(segment)
    source_terms = _semantic_bigrams(source_text)
    overlap = segment_terms & source_terms
    denominator = max(1, min(len(segment_terms), len(source_terms)))
    return len(overlap) >= 2 and len(overlap) / denominator >= 0.12


def _critical_value_occurrence_violations(
    answer: str,
    bundle: RunEvidenceBundle,
    calculation_inputs: Set[str],
) -> List[Dict[str, Any]]:
    """Validate each critical value in its own atomic/local fact scope."""

    skill_texts = [
        str(item.get("content_snapshot") or "")
        for item in bundle.skill_evidence
        if str(item.get("content_snapshot") or "")
    ]
    receipt_texts = [
        str(receipt.result or "")
        for receipt in bundle.committed_receipts
        if receipt.result is not None
    ]
    violations: List[Dict[str, Any]] = []
    for scope in _atomic_evidence_scopes(answer, bundle):
        cited = list(scope["cited"])
        for segment_info in scope["segments"]:
            segment = str(segment_info["text"])
            values = _critical_values(segment)
            if not values:
                continue
            supporting_rag = [
                evidence
                for evidence in cited
                if _citation_is_supported(segment, evidence)
            ]
            unsupported: List[str] = []
            for value in sorted(values):
                if value in calculation_inputs:
                    continue
                if any(
                    value in _critical_values(evidence.content_snapshot)
                    for evidence in supporting_rag
                ):
                    continue
                if _scoped_numeric_support_pairs(segment, bundle, value):
                    continue
                if any(
                    _source_text_supports_value(segment, source_text, value)
                    for source_text in [*skill_texts, *receipt_texts]
                ):
                    continue
                unsupported.append(value)
            if unsupported:
                violations.append(
                    {
                        "code": "ungrounded_critical_value",
                        "values": unsupported,
                        "claim_context": segment[:240],
                    }
                )
    return violations


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
                subquery=(str(result.get("subquery")) if result.get("subquery") else None),
                named_document_scope=(
                    dict(result.get("named_document_scope") or {})
                    if isinstance(result.get("named_document_scope"), dict)
                    else None
                ),
                retrieval_path=(
                    str(result.get("retrieval_path"))
                    if result.get("retrieval_path")
                    else None
                ),
                retrieval_matches=list(result.get("retrieval_matches") or []),
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


def build_run_evidence_bundle(
    evidence: EvidenceSet,
    *,
    skill_sources: Optional[Iterable[Dict[str, Any]]] = None,
    tool_invocations: Optional[Iterable[Any]] = None,
    action_receipts: Optional[Iterable[Any]] = None,
) -> RunEvidenceBundle:
    """Freeze every admissible source before answer validation or projection."""

    tool_evidence: List[ToolEvidence] = []
    seen_tools: Set[str] = set()
    for invocation in tool_invocations or []:
        raw = (
            invocation.get("tool_evidence")
            if isinstance(invocation, dict)
            else getattr(invocation, "tool_evidence", None)
        )
        if not raw:
            continue
        item = raw if isinstance(raw, ToolEvidence) else ToolEvidence.model_validate(raw)
        if item.evidence_id in seen_tools:
            continue
        seen_tools.add(item.evidence_id)
        tool_evidence.append(item)

    committed_receipts: List[ActionReceipt] = []
    for receipt in action_receipts or []:
        item = (
            receipt
            if isinstance(receipt, ActionReceipt)
            else ActionReceipt.model_validate(receipt)
        )
        if item.may_claim_success:
            committed_receipts.append(item)

    return RunEvidenceBundle(
        retrieved_rag_candidates=evidence,
        skill_evidence=[dict(item) for item in (skill_sources or [])],
        tool_evidence=tool_evidence,
        committed_receipts=committed_receipts,
    )


def prompt_run_evidence_bundle(bundle: RunEvidenceBundle) -> str:
    """Render the Agent-visible view from the same immutable evidence contract."""

    blocks = [prompt_evidence_allowlist(bundle.retrieved_rag_candidates)]
    if bundle.tool_evidence:
        lines = [
            "[只读Tool证据] 只能使用以下成功结果事实；参数不是结果证据。",
            "引用Tool事实时不得改变数值、单位、ID或业务状态：",
        ]
        for item in bundle.tool_evidence:
            lines.append(
                f"- {item.evidence_id} | {item.server_name}/{item.tool_name} | "
                f"invocation={item.invocation_id}"
            )
            for fact in item.facts:
                lines.append(
                    f"  {fact.json_path} = {fact.display_value}"
                )
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def render_bundle_citations(
    answer: str,
    bundle: RunEvidenceBundle,
) -> Tuple[str, List[Citation], List[Dict[str, Any]], RunEvidenceBundle]:
    """Validate and link an answer using only the frozen run evidence bundle."""

    evidence = bundle.retrieved_rag_candidates
    by_id = evidence.by_id()
    tool_by_id = {item.evidence_id: item for item in bundle.tool_evidence}
    ordered_ids: List[str] = []
    violations: List[Dict[str, Any]] = []
    # Only governed runtime evidence can support business facts. Values in the
    # user query remain calculation inputs at most; request arguments cannot
    # become Provider/Tool result evidence.
    calculation_inputs = _calculation_input_values(evidence.query)

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
    atomic_tool_values = _atomic_tool_supported_values(normalized, bundle)
    claims, ignored_marker_starts = _build_atomic_claims(normalized)
    marker_decisions: Dict[int, Tuple[str, Optional[str]]] = {}
    marker_tool_links: List[ToolEvidenceLink] = []

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

    invalid_marker_starts: Set[int] = set()
    for marker_start, marker in all_markers.items():
        evidence_id = resolved_evidence_id(marker.group(1))
        if evidence_id in by_id or evidence_id in tool_by_id:
            continue
        violations.append(
            {"code": "invalid_evidence_id", "evidence_id": evidence_id}
        )
        invalid_marker_starts.add(marker_start)
        marker_decisions[marker_start] = ("rejected", None)

    for marker_start in ignored_marker_starts:
        marker = all_markers.get(marker_start)
        if marker is None or marker_start in invalid_marker_starts:
            continue
        evidence_id = resolved_evidence_id(marker.group(1))
        if evidence_id not in tool_by_id:
            continue
        violations.append(
            {
                "code": "unsupported_tool_evidence_marker",
                "evidence_id": evidence_id,
                "claim_context": "",
            }
        )
        marker_decisions[marker_start] = ("rejected", None)

    for claim_index, claim in enumerate(claims):
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
            if marker_start in invalid_marker_starts:
                continue
            if evidence_id in tool_by_id:
                restricted_bundle = bundle.model_copy(
                    update={"tool_evidence": [tool_by_id[evidence_id]]}
                )
                scoped_links = [
                    link
                    for link in _tool_links_for_answer(
                        claim_text,
                        restricted_bundle,
                    )
                    if link.evidence_id == evidence_id
                ]
                if not scoped_links:
                    violations.append(
                        {
                            "code": "unsupported_tool_evidence_marker",
                            "evidence_id": evidence_id,
                            "claim_context": claim_text[:240],
                        }
                    )
                    marker_decisions[marker_start] = ("rejected", None)
                    continue
                marker_tool_links.extend(scoped_links)
                marker_decisions[marker_start] = ("accepted_tool", evidence_id)
                continue
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
        for marker_start, evidence_id in valid_markers:
            if evidence_id in supporting_ids:
                continue
            violations.append(
                {
                    "code": "unsupported_evidence_citation",
                    "evidence_id": evidence_id,
                    "claim_context": claim_text[:240],
                }
            )
            marker_decisions[marker_start] = ("rejected", None)
        if not supporting_ids:
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
            if value not in (
                atomic_tool_values[claim_index]
                if claim_index < len(atomic_tool_values)
                else set()
            )
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

        for marker_start, evidence_id in valid_markers:
            if evidence_id in supporting_ids:
                marker_decisions[marker_start] = ("accepted", evidence_id)

    def replace_id(match: re.Match) -> str:
        decision, evidence_id = marker_decisions.get(
            match.start(), ("ignored_display", None)
        )
        if decision == "accepted_tool":
            return ""
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
    violations.extend(
        _critical_value_occurrence_violations(
            normalized,
            bundle,
            calculation_inputs,
        )
    )
    violations.extend(_unsupported_tool_fact_claims(normalized, bundle))
    violations.extend(_unsupported_tool_numeric_claims(normalized, bundle))
    violations = [
        {
            **item,
            "detail": str(
                item.get("detail")
                or _VIOLATION_DETAILS.get(str(item.get("code") or ""))
                or "The answer violated the frozen run evidence contract."
            ),
        }
        for item in violations
    ]
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
    merged_tool_links: Dict[Tuple[str, str], Dict[str, Set[str]]] = {}
    for link in [
        *_tool_links_for_answer(normalized, bundle),
        *marker_tool_links,
    ]:
        key = (link.evidence_id, link.invocation_id)
        slot = merged_tool_links.setdefault(
            key,
            {"fact_ids": set(), "claim_values": set()},
        )
        slot["fact_ids"].update(link.fact_ids)
        slot["claim_values"].update(link.claim_values)
    tool_links = [
        ToolEvidenceLink(
            evidence_id=evidence_id,
            invocation_id=invocation_id,
            fact_ids=sorted(payload["fact_ids"]),
            claim_values=sorted(payload["claim_values"]),
        )
        for (evidence_id, invocation_id), payload in sorted(
            merged_tool_links.items()
        )
    ]
    validated_ids = [item.evidence_id for item in citations]
    delivered_ids = [
        *validated_ids,
        *[link.evidence_id for link in tool_links],
    ]
    withheld = [
        {
            "code": str(item.get("code") or "evidence_contract_violation"),
            "values": list(item.get("values") or []),
            "detail": str(item.get("detail") or ""),
            "claim_context": str(item.get("claim_context") or ""),
        }
        for item in violations
    ]
    final_bundle = bundle.model_copy(
        update={
            "validated_rag_evidence_ids": validated_ids,
            "delivered_evidence_ids": delivered_ids if not violations else [],
            "tool_evidence_links": tool_links if not violations else [],
            "withheld": withheld,
            "violations": list(violations),
        }
    )
    return rendered, citations, violations, final_bundle


def render_citations(
    answer: str,
    evidence: EvidenceSet | RunEvidenceBundle,
    tool_invocations: Optional[Iterable[Any]] = None,
    skill_sources: Optional[Iterable[Dict[str, Any]]] = None,
    action_receipts: Optional[Iterable[Any]] = None,
) -> Tuple[str, List[Citation], List[Dict[str, Any]]]:
    """Compatibility adapter; canonical validation consumes a bundle only."""

    bundle = (
        evidence
        if isinstance(evidence, RunEvidenceBundle)
        else build_run_evidence_bundle(
            evidence,
            skill_sources=skill_sources,
            tool_invocations=tool_invocations,
            action_receipts=action_receipts,
        )
    )
    rendered, citations, violations, _ = render_bundle_citations(answer, bundle)
    return rendered, citations, violations
