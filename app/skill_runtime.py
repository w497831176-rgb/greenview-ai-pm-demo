"""Deterministic Skill contracts used by the owner-chat runtime.

A Skill is not an extra free-form prompt.  This module gives each Skill a
small, inspectable contract: when it may run, when it must not run, its
priority/conflict policy and the evidence needed to explain that decision.
The functions deliberately make no model call so they can be used by the
management console and deterministic acceptance tests.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_PRIORITY = 50


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        return []
    values: List[str] = []
    for item in raw_values:
        values.extend(re.split(r"[,，、；;。.!！？?｜|/\\n]+", str(item)))
    return [item.strip() for item in values if item.strip()]


def _as_int(value: Any, default: int = DEFAULT_PRIORITY) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return default


def _normalise(text: str) -> str:
    text = (text or "").lower().strip()
    for stop_word in ("用户", "业主", "提到", "说到", "询问", "问题", "关于", "相关", "的", "时", "如果", "当", "要", "等"):
        text = text.replace(stop_word, "")
    for source, canonical in (("孩子", "儿童"), ("小孩", "儿童"), ("娃", "儿童")):
        text = text.replace(source, canonical)
    return text


def _bigrams(text: str) -> set[str]:
    chars = [char for char in text if char.strip()]
    if len(chars) < 2:
        return set(chars)
    return {chars[index] + chars[index + 1] for index in range(len(chars) - 1)}


def _term_matches(term: str, message: str) -> bool:
    term = _normalise(term)
    message = _normalise(message)
    if len(term) < 2 or not message:
        return False
    if term in message:
        return True
    # Acceptance markers, ticket ids and product codes must never use fuzzy
    # character-bigram matching.  For example ``NO_SKILL_V152`` and
    # ``DEMO_SKILL_V152`` share many bigrams but mean the opposite policy.
    if re.search(r"[a-z0-9_]{4,}", term):
        return False
    term_bigrams, message_bigrams = _bigrams(term), _bigrams(message)
    if not term_bigrams or not message_bigrams:
        return False
    overlap = term_bigrams & message_bigrams
    jaccard = len(overlap) / len(term_bigrams | message_bigrams)
    term_coverage = len(overlap) / len(term_bigrams)
    return jaccard >= 0.45 or (len(overlap) >= 2 and term_coverage >= 0.40)


def skill_contract(skill: Dict[str, Any]) -> Dict[str, Any]:
    """Return a safe, backwards-compatible runtime contract for one Skill."""
    original = skill.get("skill_metadata")
    metadata = dict(original) if isinstance(original, dict) else {}
    positive = _as_list(metadata.get("positive_triggers")) or _as_list(skill.get("trigger_condition"))
    negative = _as_list(metadata.get("negative_triggers"))
    version = str(metadata.get("version") or "legacy-1.0.0")
    return {
        "contract_version": str(metadata.get("contract_version") or ("legacy" if not metadata else "1.0")),
        "version": version,
        "positive_triggers": positive,
        "negative_triggers": negative,
        "priority": _as_int(metadata.get("priority")),
        "conflict_group": str(metadata.get("conflict_group") or "").strip(),
        "composable": bool(metadata.get("composable", False)),
        "always_on": bool(metadata.get("always_on", False)),
        "input_contract": str(metadata.get("input_contract") or ""),
        "output_contract": str(metadata.get("output_contract") or ""),
        "tool_hints": _as_list(metadata.get("tool_hints")),
        "owner": str(metadata.get("owner") or "平台管理员"),
        "legacy_mode": not bool(metadata),
    }


def canonical_metadata(skill: Dict[str, Any], incoming: Any = None) -> Dict[str, Any]:
    """Keep imported fields, while making the Skill contract explicit."""
    current = skill.get("skill_metadata") if isinstance(skill.get("skill_metadata"), dict) else {}
    supplied = incoming if isinstance(incoming, dict) else {}
    metadata = {**current, **supplied}
    positive = _as_list(metadata.get("positive_triggers")) or _as_list(skill.get("trigger_condition"))
    metadata.update({
        "contract_version": str(metadata.get("contract_version") or "1.0"),
        "version": str(metadata.get("version") or "1.0.0"),
        "positive_triggers": positive,
        "negative_triggers": _as_list(metadata.get("negative_triggers")),
        "priority": _as_int(metadata.get("priority")),
        "conflict_group": str(metadata.get("conflict_group") or "").strip(),
        "composable": bool(metadata.get("composable", False)),
        "always_on": bool(metadata.get("always_on", False)),
        "input_contract": str(metadata.get("input_contract") or ""),
        "output_contract": str(metadata.get("output_contract") or ""),
        "tool_hints": _as_list(metadata.get("tool_hints")),
        "owner": str(metadata.get("owner") or "平台管理员"),
    })
    return metadata


def next_patch_version(version: str) -> str:
    """Return the next patch version; imported/non-semver data gets 1.0.1."""
    matched = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", (version or "").strip())
    if not matched:
        return "1.0.1"
    major, minor, patch = (int(item) for item in matched.groups())
    return f"{major}.{minor}.{patch + 1}"


def evaluate_skill_match(skill: Dict[str, Any], message: str) -> Dict[str, Any]:
    """Explain whether one Skill can be selected for one owner message."""
    contract = skill_contract(skill)
    matched_negative = [term for term in contract["negative_triggers"] if _term_matches(term, message)]
    if matched_negative:
        return {
            "skill_id": skill.get("id"), "name": skill.get("name", ""), "matched": False,
            "outcome": "negative_excluded", "match_reason": f"命中负向触发：{'、'.join(matched_negative)}",
            "matched_positive": [], "matched_negative": matched_negative, "contract": contract,
        }
    if contract["always_on"]:
        return {
            "skill_id": skill.get("id"), "name": skill.get("name", ""), "matched": True,
            "outcome": "always_on", "match_reason": "默认注入已开启", "matched_positive": [],
            "matched_negative": [], "contract": contract,
        }
    matched_positive = [term for term in contract["positive_triggers"] if _term_matches(term, message)]
    if matched_positive:
        return {
            "skill_id": skill.get("id"), "name": skill.get("name", ""), "matched": True,
            "outcome": "positive_matched", "match_reason": f"命中正向触发：{'、'.join(matched_positive)}",
            "matched_positive": matched_positive, "matched_negative": [], "contract": contract,
        }
    if not contract["positive_triggers"]:
        reason = "未配置正向触发，且未开启默认注入；不会隐式注入"
    else:
        reason = "未命中正向触发"
    return {
        "skill_id": skill.get("id"), "name": skill.get("name", ""), "matched": False,
        "outcome": "not_matched", "match_reason": reason, "matched_positive": [],
        "matched_negative": [], "contract": contract,
    }


def select_skills(skills: Iterable[Dict[str, Any]], message: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply trigger and conflict policy, returning selected Skills and evidence."""
    evaluated = [evaluate_skill_match(skill, message) for skill in skills if skill and skill.get("enabled")]
    ordered = sorted(
        evaluated,
        key=lambda item: (-item["contract"]["priority"], int(item.get("skill_id") or 0)),
    )
    selected: List[Dict[str, Any]] = []
    occupied_groups: set[str] = set()
    for item in ordered:
        if not item["matched"]:
            item["selected"] = False
            continue
        group = item["contract"]["conflict_group"]
        if group and group in occupied_groups and not item["contract"]["composable"]:
            item.update({"selected": False, "outcome": "conflict_skipped", "match_reason": f"与冲突组“{group}”中更高优先级 Skill 冲突"})
            continue
        item["selected"] = True
        selected.append(item)
        if group and not item["contract"]["composable"]:
            occupied_groups.add(group)
    return selected, ordered


def activation_evidence(decision: Dict[str, Any], injected_chars: int) -> Dict[str, Any]:
    contract = decision["contract"]
    return {
        "id": decision.get("skill_id"),
        "name": decision.get("name", ""),
        "version": contract["version"],
        "contract_version": contract["contract_version"],
        "priority": contract["priority"],
        "match_reason": decision["match_reason"],
        "matched_positive": decision.get("matched_positive", []),
        "negative_triggers": contract["negative_triggers"],
        "tool_hints": contract["tool_hints"],
        "injected_chars": injected_chars,
        "authority_note": "Skill 仅提供业务方法；实际工具权限仍由当前 Agent 的 MCP 绑定决定。",
    }
