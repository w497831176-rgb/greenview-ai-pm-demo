"""
Badcase schema and state machine helpers (pure, no heavy imports).

This module owns the canonical contract between the backend and the frontend
for Badcase detail pages. It is intentionally free of FastAPI / Agno imports so
it can be unit-tested without installing the full project dependency tree.
"""

import json
from typing import Any, Dict, List, Optional, Set

VALID_CATEGORIES = {
    "knowledge_gap",   # 知识库缺口
    "skill_prompt",    # Skill/Prompt 问题
    "mcp_capability",  # MCP/能力缺口
    "routing",         # 路由问题
    "response_quality", # 模型回答质量
    "other",
    "pending",
}

VALID_STATUSES = {"pending", "classified", "fixing", "verifying", "closed", "rejected"}

CATEGORY_LABELS = {
    "knowledge_gap": "知识库缺口",
    "skill_prompt": "Skill/Prompt 问题",
    "mcp_capability": "MCP/能力缺口",
    "routing": "路由问题",
    "response_quality": "模型回答质量",
    "other": "其他",
    "pending": "待分类",
}

STATUS_LABELS = {
    "pending": "待分类",
    "classified": "已分类",
    "fixing": "修复中",
    "verifying": "验证中",
    "closed": "已关闭",
    "rejected": "已驳回",
}

# Canonical state machine transitions.
# Each key is the source status; the value is the set of statuses reachable
# through normal lifecycle actions (excluding the admin-only transition fallback).
STATUS_TRANSITIONS: Dict[str, Set[str]] = {
    "pending": {"classified", "rejected"},
    "classified": {"fixing", "rejected"},
    "fixing": {"verifying", "rejected"},
    "verifying": {"closed", "fixing", "rejected"},
    "closed": set(),
    "rejected": set(),
}

# Actions exposed to operators in the UI, mapped to the status they require.
ACTION_STATUS_REQUIREMENTS: Dict[str, Set[str]] = {
    "classify": {"pending"},
    "darwin-fix": {"classified"},
    "extract-knowledge": {"classified"},
    "edit-draft": {"fixing"},
    "review-draft": {"fixing"},
    "apply-draft": {"fixing"},  # generic UI action for applying any approved draft
    "accept-capability-gap": {"fixing"},  # backward-compatible alias
    "retest": {"verifying"},
    "verify-pass": {"verifying"},
    "verify-fail": {"verifying"},
    "reject": {"pending", "classified", "fixing", "verifying"},
    "transition": {"pending", "classified", "fixing", "verifying", "closed", "rejected"},
    # check-tools is intentionally omitted from the UI action list; it remains a
    # backend diagnostic helper available from pending/classified.
}

# Draft status transitions.
# knowledge/skill_prompt: draft -> under_review -> approved -> published
# capability_gap:       draft -> under_review -> approved -> accepted
# All types may also go from draft/under_review -> rejected.
DRAFT_STATUS_TRANSITIONS: Dict[str, Dict[str, Set[str]]] = {
    "knowledge": {
        "draft": {"under_review", "rejected"},
        "under_review": {"draft", "approved", "rejected"},
        "approved": {"published"},
        "published": set(),
        "rejected": set(),
    },
    "skill_prompt": {
        "draft": {"under_review", "rejected"},
        "under_review": {"draft", "approved", "rejected"},
        "approved": {"published"},
        "published": set(),
        "rejected": set(),
    },
    "capability_gap": {
        "draft": {"under_review", "rejected"},
        "under_review": {"draft", "approved", "rejected"},
        "approved": {"accepted"},
        "accepted": set(),
        "rejected": set(),
    },
}

DRAFT_TYPE_ALIASES = {
    "knowledge": "knowledge",
    "knowledge_draft": "knowledge",
    "skill_prompt": "skill_prompt",
    "skill_prompt_draft": "skill_prompt",
    "capability_gap": "capability_gap",
    "capability_gap_draft": "capability_gap",
}

# Categories for which "extract knowledge" is the primary repair path.
KNOWLEDGE_CATEGORIES = {"knowledge_gap"}
# Categories for which "skill/prompt draft" is the primary repair path.
SKILL_CATEGORIES = {"skill_prompt", "routing", "response_quality"}
# Categories for which "capability gap" is the primary repair path.
CAPABILITY_CATEGORIES = {"mcp_capability"}


def is_terminal_status(status: str) -> bool:
    return status in {"closed", "rejected"}


def validate_status_transition(from_status: str, to_status: str) -> None:
    """Raise ValueError if the transition is not allowed by the lifecycle."""
    if from_status not in VALID_STATUSES:
        raise ValueError(f"invalid source status: {from_status}")
    if to_status not in VALID_STATUSES:
        raise ValueError(f"invalid target status: {to_status}")
    if to_status not in STATUS_TRANSITIONS.get(from_status, set()):
        raise ValueError(f"cannot transition from '{from_status}' to '{to_status}'")


def allowed_target_statuses(from_status: str) -> List[str]:
    return sorted(STATUS_TRANSITIONS.get(from_status, set()))


def allowed_actions(status: str) -> List[str]:
    """Return the actions that are legal from the given status.

    Terminal statuses intentionally expose no actions to the frontend.
    """
    if status not in VALID_STATUSES or is_terminal_status(status):
        return []
    return sorted([action for action, required in ACTION_STATUS_REQUIREMENTS.items() if status in required])


def effective_allowed_actions(badcase: Dict[str, Any]) -> List[str]:
    """Return frontend-visible actions, considering runtime evidence.

    - Terminal statuses: empty.
    - verifying without retest_response: hide verify-pass.
    """
    status = badcase.get("status", "pending")
    actions = allowed_actions(status)
    if status == "verifying" and not badcase.get("retest_response"):
        actions = [a for a in actions if a != "verify-pass"]
    return actions


def _normalize_draft_type(draft_type: str) -> str:
    return DRAFT_TYPE_ALIASES.get(draft_type, draft_type)


def validate_draft_status_transition(draft_type: str, from_status: str, to_status: str) -> None:
    """Raise ValueError if the draft status transition is illegal."""
    normalized = _normalize_draft_type(draft_type)
    transitions = DRAFT_STATUS_TRANSITIONS.get(normalized)
    if not transitions:
        raise ValueError(f"unknown draft type: {draft_type}")
    if from_status not in transitions:
        raise ValueError(f"invalid source status for {normalized} draft: {from_status}")
    if to_status not in transitions.get(from_status, set()):
        raise ValueError(f"cannot transition {normalized} draft from '{from_status}' to '{to_status}'")


def is_draft_terminal(draft_type: str, status: str) -> bool:
    normalized = _normalize_draft_type(draft_type)
    transitions = DRAFT_STATUS_TRANSITIONS.get(normalized)
    if not transitions:
        return False
    return status in transitions and not transitions[status]


def is_draft_editable(draft_type: str, status: str) -> bool:
    """A draft is editable unless it has reached a terminal status."""
    return not is_draft_terminal(draft_type, status)


def require_status(case_status: str, action: str, allowed: Set[str]) -> None:
    """Raise ValueError with a friendly message when an action is invoked from a wrong status."""
    if case_status not in allowed:
        raise ValueError(
            f"cannot execute '{action}' from status '{case_status}'; "
            f"allowed statuses are: {sorted(allowed)}"
        )


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except Exception:
            return {}
    return value or {}


def _format_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Parse action_detail JSON and add display helpers for the frontend."""
    formatted = dict(action)
    detail = formatted.get("action_detail")
    if isinstance(detail, str) and detail:
        try:
            formatted["action_detail_parsed"] = json.loads(detail)
        except Exception:
            formatted["action_detail_parsed"] = {"raw": detail}
    else:
        formatted["action_detail_parsed"] = detail or {}
    formatted["status_before_label"] = STATUS_LABELS.get(formatted.get("status_before", ""), "-")
    formatted["status_after_label"] = STATUS_LABELS.get(formatted.get("status_after", ""), "-")
    return formatted


def _enrich_badcase(badcase: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Add frontend-compatible aliases to a badcase record.

    The returned schema is the single source of truth for the Badcase detail page.
    """
    if not badcase:
        return None

    enriched = dict(badcase)

    # Core identity and labels.
    enriched["query"] = enriched.get("original_query") or enriched.get("title") or "-"
    enriched["category_label"] = CATEGORY_LABELS.get(enriched.get("category", ""), enriched.get("category", "-"))
    enriched["status_label"] = STATUS_LABELS.get(enriched.get("status", ""), enriched.get("status", "-"))
    enriched["source"] = enriched.get("source") or "auto"
    enriched["source_label"] = "人工反馈" if enriched.get("source") == "manual" else "自动发现"

    # Structured context objects (parsed JSON).
    enriched["context"] = _parse_json_field(enriched.get("context_json"))
    enriched["retest_context"] = _parse_json_field(enriched.get("retest_context_json"))

    # Darwin analysis.
    enriched["darwin_analysis"] = enriched.get("darwin_analysis") or ""
    enriched["darwin_analysis_parsed"] = _parse_json_field(enriched.get("darwin_analysis"))

    # Analysis evidence fallback.
    enriched["analysis_evidence"] = (
        enriched.get("evidence")
        or enriched.get("root_cause")
        or enriched.get("description")
        or "暂无分析"
    )

    # Action history formatting.
    actions = enriched.get("actions") or []
    enriched["actions"] = [_format_action(a) for a in actions]

    # Allowed actions for the current status (frontend button guidance).
    enriched["allowed_actions"] = effective_allowed_actions(enriched)
    enriched["is_terminal"] = is_terminal_status(enriched.get("status", "pending"))

    # Ensure retest_response is exposed directly.
    enriched["retest_response"] = enriched.get("retest_response") or ""
    enriched["retest_trace_id"] = enriched.get("retest_trace_id") or ""

    # Clean up legacy / confusing aliases.
    enriched.pop("retest_result", None)

    return enriched


def repair_path_for_category(category: str) -> str:
    """Return the primary repair path label for a given category."""
    if category in KNOWLEDGE_CATEGORIES:
        return "knowledge"
    if category in SKILL_CATEGORIES:
        return "skill_prompt"
    if category in CAPABILITY_CATEGORIES:
        return "capability_gap"
    return "darwin"
