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

# Operational lifecycle: a Badcase is not closed at "prompt changed".  It
# remains distinguishable between triage, investigation, fix, verification,
# release observation and final closure.  Compatibility states (pending /
# classified) are intentionally retained for existing data and UI entrypoints.
VALID_STATUSES = {
    "pending", "classified", "investigating", "fixing", "verifying", "released",
    "closed", "rejected", "duplicate", "accepted_limitation",
}

ROOT_CAUSE_DOMAINS = {
    "routing",
    "knowledge_rag",
    "model_instruction",
    "tool_mcp",
    "authority_safety",
    "human_collaboration",
    "ux",
    "external_dependency",
    "unknown",
}

ROOT_CAUSE_DOMAIN_LABELS = {
    "routing": "意图/路由",
    "knowledge_rag": "知识/RAG",
    "model_instruction": "模型/指令/Skill",
    "tool_mcp": "Tool/MCP",
    "authority_safety": "权限/安全",
    "human_collaboration": "人机协同",
    "ux": "产品体验",
    "external_dependency": "外部依赖/能力边界",
    "unknown": "待归因",
}

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
    "investigating": "归因中",
    "fixing": "修复中",
    "verifying": "验证中",
    "released": "已发布观察",
    "closed": "已关闭",
    "rejected": "已驳回",
    "duplicate": "重复案例",
    "accepted_limitation": "已接受限制",
}

ACTION_LABELS = {
    "classify": "分类",
    "darwin-fix": "Darwin 深度分析",
    "extract-knowledge": "生成修复草稿",
    "generate-repair-drafts": "生成修复草稿",
    "edit-draft": "编辑草稿",
    "review-draft": "审核草稿",
    "apply-draft": "应用草稿",
    "accept-capability-gap": "接受能力缺口",
    "retest": "真实复测",
    "verify-pass": "验证通过",
    "verify-fail": "验证不通过",
    "close": "观察后关闭",
    "accept-limitation": "记录已知限制",
    "mark-duplicate": "关联重复案例",
    "reject": "驳回",
    "transition": "状态跳转",
}

# Canonical state machine transitions.
# Each key is the source status; the value is the set of statuses reachable
# through normal lifecycle actions (excluding the admin-only transition fallback).
STATUS_TRANSITIONS: Dict[str, Set[str]] = {
    "pending": {"classified", "rejected", "duplicate", "accepted_limitation"},
    "classified": {"investigating", "fixing", "rejected", "duplicate", "accepted_limitation"},
    "investigating": {"fixing", "rejected", "duplicate", "accepted_limitation"},
    "fixing": {"verifying", "rejected"},
    "verifying": {"released", "fixing", "rejected"},
    "released": {"closed", "fixing"},
    "closed": set(),
    "rejected": set(),
    "duplicate": set(),
    "accepted_limitation": set(),
}

# Actions exposed to operators in the UI, mapped to the status they require.
ACTION_STATUS_REQUIREMENTS: Dict[str, Set[str]] = {
    "classify": {"pending"},
    "darwin-fix": {"classified", "investigating"},
    "extract-knowledge": {"classified"},  # knowledge-gap repair draft
    "edit-draft": {"fixing"},
    "review-draft": {"fixing"},
    "apply-draft": {"fixing"},  # generic UI action for applying any approved draft
    "accept-capability-gap": {"fixing"},  # backward-compatible alias
    "retest": {"fixing", "verifying"},
    "verify-pass": {"verifying"},
    "verify-fail": {"verifying", "released"},  # post-release observation can reveal a regression
    "close": {"released"},
    "accept-limitation": {"pending", "classified", "investigating", "fixing"},
    "mark-duplicate": {"pending", "classified", "investigating", "fixing"},
    "reject": {"pending", "classified", "investigating", "fixing", "verifying"},
    "transition": {"pending", "classified", "investigating", "fixing", "verifying", "released", "closed", "rejected", "duplicate", "accepted_limitation"},
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
    return status in {"closed", "rejected", "duplicate", "accepted_limitation"}


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


def _has_post_apply_retest(badcase: Dict[str, Any]) -> bool:
    """Return True iff a retest was performed after the most recent draft apply."""
    last_applied_at = badcase.get("last_applied_at")
    last_retest_at = badcase.get("last_retest_at")
    if not last_retest_at:
        return False
    if not last_applied_at:
        # No apply recorded yet; any retest is acceptable.
        return bool(badcase.get("retest_response"))
    return last_retest_at >= last_applied_at


def effective_allowed_actions(badcase: Dict[str, Any]) -> List[str]:
    """Return frontend-visible actions, considering runtime evidence and category.

    - Terminal statuses: empty.
    - classified/investigating: only show category-relevant draft actions.
    - verifying: hide verify-pass until a retest has been run after the latest apply.
    """
    status = badcase.get("status", "pending")
    actions = allowed_actions(status)
    category = badcase.get("category", "pending")

    if status in {"classified", "investigating"}:
        # Only knowledge-gap cases expose the explicit knowledge draft action;
        # other categories rely on Darwin-fix to generate the right draft type.
        if category not in KNOWLEDGE_CATEGORIES and category != "pending":
            actions = [a for a in actions if a != "extract-knowledge"]

    if status == "verifying" and not _has_post_apply_retest(badcase):
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
    enriched["root_cause_domain"] = enriched.get("root_cause_domain") or "unknown"
    enriched["root_cause_domain_label"] = ROOT_CAUSE_DOMAIN_LABELS.get(
        enriched["root_cause_domain"], enriched["root_cause_domain"]
    )
    secondary_domains = _parse_json_field(enriched.get("secondary_root_cause_domains"))
    enriched["secondary_root_cause_domains"] = secondary_domains if isinstance(secondary_domains, list) else []
    enriched["source"] = enriched.get("source") or "auto"
    source = enriched.get("source")
    if source == "manual":
        enriched["source_label"] = "人工反馈"
    elif source == "user_feedback":
        enriched["source_label"] = "用户反馈"
    elif source == "evaluation":
        enriched["source_label"] = "评估失败"
    elif source in {"mcp_failure", "tool_failure"}:
        enriched["source_label"] = "工具监测"
    elif source == "handoff":
        enriched["source_label"] = "人工协同复盘"
    else:
        enriched["source_label"] = "自动发现"

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
