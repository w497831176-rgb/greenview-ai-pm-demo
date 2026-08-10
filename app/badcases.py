"""
Badcase Closed-Loop API
=======================

Implements the quality-operations lifecycle:
    pending -> classified -> investigating -> fixing -> verifying
    -> released (operator observation) -> closed

Rejected, duplicate and accepted-limitation are retained as explicit terminal
outcomes rather than silently deleting uncomfortable evidence.

Supports automatic classification, knowledge extraction, Darwin skill
optimization, model switch retry, and verification.
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from app.badcase_schema import (
    CATEGORY_LABELS,
    ROOT_CAUSE_DOMAINS,
    VALID_CATEGORIES,
    VALID_STATUSES,
    _enrich_badcase as _schema_enrich_badcase,
    _has_post_apply_retest,
    allowed_actions,
    effective_allowed_actions,
    is_draft_editable,
    is_draft_terminal,
    is_terminal_status,
    repair_path_for_category,
    require_status,
    validate_draft_status_transition,
    validate_status_transition,
    user_status_label,
)
from app.observability import _background_budget_gate
from app.runtime.cost_ledger import build_cost_entry, cost_entry_usage_payload
from app.runtime.darwin_evidence import (
    persist_darwin_operation,
    start_darwin_operation,
)
from app.runtime.provider_accounting import merge_non_null, provider_accounting_scope
from app.runtime.provider_evidence import provider_evidence_from_run
from app.settings import MODEL_ID, build_model
from app.skill_runtime import canonical_metadata, next_patch_version
from app.utils.cost_utils import build_price_snapshot, compute_cost_cny
import skill_storage

_BUDGET_BLOCKED_DETAIL = "预算已达上限，Darwin/AI 分类等 Pro/额外评估操作被阻止，请联系管理员调整预算或等待次日刷新"
from db.property_db import (
    add_badcase_action,
    create_badcase as db_create_badcase,
    create_capability_gap_draft as db_create_capability_gap_draft,
    create_knowledge_doc as db_create_knowledge_doc,
    create_knowledge_draft as db_create_knowledge_draft,
    create_skill as db_create_skill,
    create_skill_version,
    create_skill_prompt_draft as db_create_skill_prompt_draft,
    delete_badcase as db_delete_badcase,
    delete_knowledge_doc as db_delete_knowledge_doc,
    get_agent_by_agent_id,
    get_agent_knowledge_bindings,
    get_agent_tools,
    get_badcase as db_get_badcase,
    get_chat_trace,
    get_capability_gap_draft as db_get_capability_gap_draft,
    get_chat_message,
    get_enabled_price_for_model,
    get_evaluation_case,
    get_knowledge_draft as db_get_knowledge_draft,
    get_skill,
    get_skill_by_name,
    get_current_runtime_release,
    get_agent_skills,
    get_skill_prompt_draft as db_get_skill_prompt_draft,
    list_badcase_actions,
    list_badcases as db_list_badcases,
    list_badcases_page as db_list_badcases_page,
    list_capability_gap_drafts as db_list_capability_gap_drafts,
    list_evaluation_runs,
    list_knowledge_drafts as db_list_knowledge_drafts,
    list_skill_prompt_drafts as db_list_skill_prompt_drafts,
    list_skills,
    now_cn,
    get_provider_attempts_for_trace,
    set_agent_skills,
    update_badcase as db_update_badcase,
    update_capability_gap_draft as db_update_capability_gap_draft,
    update_knowledge_draft as db_update_knowledge_draft,
    update_skill as db_update_skill,
    update_skill_prompt_draft as db_update_skill_prompt_draft,
)

router = APIRouter(tags=["badcases"])


def _validated_retest_trace(badcase: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the live complete Trace bound to this retest context, if any."""
    try:
        context = json.loads(badcase.get("retest_context_json") or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        context = {}
    trace_id = str(badcase.get("retest_trace_id") or "").strip()
    trace = get_chat_trace(trace_id) if trace_id else None
    if not (
        trace
        and trace.get("status") == "complete"
        and str(trace.get("session_id") or "").strip()
        == str(context.get("session_id") or "").strip()
    ):
        return None
    return trace


def _enrich_badcase(badcase: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Add schema presentation plus a live Trace gate for visible actions."""
    live_trace = _validated_retest_trace(badcase or {})
    payload = dict(badcase or {})
    payload["retest_trace_live_verified"] = bool(live_trace)
    return _schema_enrich_badcase(payload)


def _enforce_background_budget(strategy: str) -> Dict[str, Any]:
    """Fail closed before an optional paid background model operation."""
    budget_gate = _background_budget_gate(strategy)
    if not budget_gate.get("allowed"):
        raise HTTPException(
            status_code=int(budget_gate["http_status"]),
            detail=budget_gate["detail"],
        )
    return budget_gate


def _budget_threshold_enabled(value: Any) -> bool:
    try:
        return value is not None and float(value) > 0
    except (TypeError, ValueError):
        return False


def _darwin_budget_gate() -> Dict[str, Any]:
    """Keep reconciliation attention advisory when no Darwin budget is enabled."""
    budget_gate = _background_budget_gate("darwin")
    daily_enabled = _budget_threshold_enabled(
        budget_gate.get("daily_threshold_cny")
    )
    monthly_enabled = _budget_threshold_enabled(
        budget_gate.get("monthly_threshold_cny")
    )
    if (
        budget_gate.get("allowed") is False
        and budget_gate.get("reason_code") == "budget_reconciliation_attention"
        and not daily_enabled
        and not monthly_enabled
    ):
        logger.warning(
            "Darwin budget reconciliation needs attention, but daily/monthly "
            "budgets are disabled; continuing as an advisory warning"
        )
        return {
            **budget_gate,
            "allowed": True,
            "alert_level": "warning",
            "http_status": None,
            "detail": None,
            "warning_code": "budget_reconciliation_attention",
        }
    return budget_gate


def _complete_darwin_suggestion(value: Dict[str, Any]) -> bool:
    category = str(value.get("recommended_category") or "").strip()
    root_cause = str(value.get("root_cause_hypothesis") or "").strip()
    repair_path = str(value.get("repair_path_suggestion") or "").strip()
    suggested_actions = value.get("suggested_actions")
    return bool(
        category in VALID_CATEGORIES
        and category != "pending"
        and root_cause
        and repair_path
        and isinstance(suggested_actions, list)
        and any(str(item).strip() for item in suggested_actions)
    )


class BadcaseCreate(BaseModel):
    title: str
    description: str = ""
    category: str = "other"
    status: str = "pending"
    evidence: str = ""
    source_message_id: Optional[int] = None
    session_id: Optional[str] = None
    source: str = "manual"
    original_query: Optional[str] = None
    ai_response: Optional[str] = None
    feedback_reason: Optional[str] = None
    priority: str = "medium"
    symptom: Optional[str] = None
    expected_behavior: Optional[str] = None
    actual_behavior: Optional[str] = None
    root_cause_domain: Optional[str] = None
    secondary_root_cause_domains: List[str] = []
    impact_scope: Optional[str] = None
    owner: Optional[str] = None
    linked_evaluation_case_id: Optional[int] = None
    linked_evaluation_run_id: Optional[int] = None


class BadcaseUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    # Status is intentionally not a free-form edit.  State changes must use a
    # lifecycle action so Trace / audit / verification evidence remain intact.
    status: Optional[str] = None
    evidence: Optional[str] = None
    root_cause: Optional[str] = None
    fix_plan: Optional[str] = None
    rejected_reason: Optional[str] = None
    priority: Optional[str] = None
    symptom: Optional[str] = None
    expected_behavior: Optional[str] = None
    actual_behavior: Optional[str] = None
    root_cause_domain: Optional[str] = None
    secondary_root_cause_domains: Optional[List[str]] = None
    impact_scope: Optional[str] = None
    owner: Optional[str] = None


class ClassifyRequest(BaseModel):
    auto: bool = True
    category: Optional[str] = None
    reason: str = ""
    root_cause_domain: Optional[str] = None


class ExtractKnowledgeRequest(BaseModel):
    auto: bool = True
    title: Optional[str] = None
    content: Optional[str] = None
    category: str = "未分类"


class DarwinFixRequest(BaseModel):
    prompt: Optional[str] = None


class SwitchModelRetryRequest(BaseModel):
    model_id: Optional[str] = None
    user_message: Optional[str] = None


class VerifyRequest(BaseModel):
    passed: bool = True
    note: str = ""
    verification_evidence: str = ""


class CloseReleaseRequest(BaseModel):
    observation_note: str = ""


class DuplicateRequest(BaseModel):
    primary_badcase_id: int
    note: str = ""


class AcceptLimitationRequest(BaseModel):
    reason: str
    alternative_path: str = ""


class RejectRequest(BaseModel):
    rejected_reason: str = ""
    review_result: str = ""


class SystemObservationRequest(BaseModel):
    reason: str = ""


class TransitionRequest(BaseModel):
    status: str = "verifying"
    note: str = ""


class AgentConfigApplyEvidenceRequest(BaseModel):
    agent_id: str
    before_description: str
    before_instructions: str
    after_description: str
    after_instructions: str
    skill_ids_before: List[int] = Field(default_factory=list)
    skill_ids_after: List[int] = Field(default_factory=list)
    knowledge_doc_ids_before: List[int] = Field(default_factory=list)
    knowledge_doc_ids_after: List[int] = Field(default_factory=list)
    mcp_tools_before: List[str] = Field(default_factory=list)
    mcp_tools_after: List[str] = Field(default_factory=list)
    review_note: str = ""


class RuntimeReleaseEvidenceRequest(BaseModel):
    release_id: str
    version: int
    parent_release_id: Optional[str] = None
    note: str = ""


class PublishSkillDraftRequest(BaseModel):
    target_skill_id: Optional[int] = None
    target_agent_id: Optional[str] = None


class AcceptGapRequest(BaseModel):
    note: str = ""


class ReviewDraftRequest(BaseModel):
    status: str = "approved"  # under_review | approved | rejected
    note: str = ""


class EditKnowledgeDraftRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None


class EditSkillDraftRequest(BaseModel):
    title: Optional[str] = None
    skill_name: Optional[str] = None
    prompt_content: Optional[str] = None
    trigger_keywords: Optional[str] = None


class EditCapabilityGapDraftRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    gap_type: Optional[str] = None
    suggested_action: Optional[str] = None


def _record_action(badcase_id: int, action_type: str, detail: Any, before: str, after: str, created_by: str = "system"):
    """Record a badcase lifecycle action."""
    return add_badcase_action(
        badcase_id=badcase_id,
        action_type=action_type,
        action_detail=json.dumps(detail, ensure_ascii=False) if not isinstance(detail, str) else detail,
        status_before=before,
        status_after=after,
        created_by=created_by,
    )


def _require_case_status(case: Dict[str, Any], action: str, allowed: Set[str]) -> None:
    """Enforce the authoritative state machine and raise HTTP 400 if violated."""
    try:
        require_status(case["status"], action, allowed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _load_case(case_id: int) -> Dict[str, Any]:
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="badcase not found")
    return case


def _load_draft(draft_id: int, case_id: int, getter, draft_name: str = "draft") -> Dict[str, Any]:
    draft = getter(draft_id)
    if not draft or draft.get("badcase_id") != case_id:
        raise HTTPException(status_code=404, detail=f"{draft_name} not found")
    return draft


def _attach_drafts(case: Dict[str, Any]) -> Dict[str, Any]:
    """Attach draft lists to a case dict before enrichment."""
    case_id = case["id"]
    case["actions"] = list_badcase_actions(case_id)
    case["knowledge_drafts"] = [d for d in db_list_knowledge_drafts() if d.get("badcase_id") == case_id]
    case["skill_prompt_drafts"] = db_list_skill_prompt_drafts(badcase_id=case_id)
    case["capability_gap_drafts"] = db_list_capability_gap_drafts(badcase_id=case_id)
    evaluation_case_id = case.get("linked_evaluation_case_id")
    case["linked_evaluation_case"] = get_evaluation_case(evaluation_case_id) if evaluation_case_id else None
    runs = list_evaluation_runs(limit=50)
    case["linked_evaluation_runs"] = [
        run for run in runs
        if run.get("badcase_id") == case_id or run.get("id") == case.get("linked_evaluation_run_id")
    ]
    return case


def _draft_snapshot(draft: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """Return a snapshot of the draft fields that matter for audit history."""
    return {field: draft.get(field) for field in fields if draft.get(field) is not None}


_KNOWLEDGE_DRAFT_SNAPSHOT_FIELDS = ["id", "title", "content", "category", "status"]
_SKILL_DRAFT_SNAPSHOT_FIELDS = ["id", "title", "skill_name", "prompt_content", "trigger_keywords", "status"]
_CAPABILITY_GAP_DRAFT_SNAPSHOT_FIELDS = ["id", "title", "description", "gap_type", "suggested_action", "status"]


def _require_draft_transition(draft_type: str, draft: Dict[str, Any], new_status: str) -> None:
    """Enforce strict draft status transitions and raise HTTP 400 on violation."""
    try:
        validate_draft_status_transition(draft_type, draft.get("status", "draft"), new_status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _extract_usage(usage_obj: Any) -> Dict[str, Any]:
    if isinstance(usage_obj, dict):
        return {
            "input_tokens": usage_obj.get("input_tokens") or usage_obj.get("prompt_tokens"),
            "output_tokens": usage_obj.get("output_tokens") or usage_obj.get("completion_tokens"),
            "reasoning_tokens": usage_obj.get("reasoning_tokens"),
            "cached_tokens": usage_obj.get("cached_tokens") or usage_obj.get("prompt_cache_hit_tokens"),
            "total_tokens": usage_obj.get("total_tokens"),
        }
    return {
        "input_tokens": getattr(usage_obj, "input_tokens", None) or getattr(usage_obj, "prompt_tokens", None),
        "output_tokens": getattr(usage_obj, "output_tokens", None) or getattr(usage_obj, "completion_tokens", None),
        "reasoning_tokens": getattr(usage_obj, "reasoning_tokens", None),
        "cached_tokens": getattr(usage_obj, "cached_tokens", None) or getattr(usage_obj, "prompt_cache_hit_tokens", None),
        "total_tokens": getattr(usage_obj, "total_tokens", None),
    }


def _merge_provider_evidence(usage: Dict[str, Any], value: Any) -> None:
    evidence = provider_evidence_from_run(value)
    merge_non_null(usage, evidence.get("usage") or {})
    merge_non_null(
        usage,
        {
            "provider_response_model": evidence.get("provider_response_model"),
            "provider_request_id": evidence.get("provider_request_id"),
        },
    )


async def _collect_response(generator) -> Tuple[str, Dict[str, Any]]:
    """Collect text and usage from an Agno async generator or a single response."""
    response = ""
    usage = {}
    try:
        if isinstance(generator, str):
            return generator, usage
        if hasattr(generator, "__aiter__"):
            async for chunk in generator:
                if hasattr(chunk, "content") and chunk.content:
                    response += str(chunk.content)
                elif hasattr(chunk, "delta") and chunk.delta:
                    response += str(chunk.delta)
                elif isinstance(chunk, str):
                    response += chunk
                _merge_provider_evidence(usage, chunk)
                if hasattr(chunk, "usage") and chunk.usage:
                    usage.update(_extract_usage(chunk.usage))
            return response.strip(), usage
        result = await generator
        _merge_provider_evidence(usage, result)
        if hasattr(result, "content"):
            if hasattr(result, "usage") and result.usage:
                usage.update(_extract_usage(result.usage))
            return str(result.content).strip(), usage
        if isinstance(result, str):
            return result.strip(), usage
        return "", usage
    except Exception:
        raise


async def _llm_generate(
    prompt: str,
    model: Optional[Any] = None,
    model_id: Optional[str] = None,
    *,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    stage: str = "badcase_ai",
    model_selection_reason: str = "Badcase AI operation",
    explicit_retry: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Generate text inside the central per-Provider-request ledger boundary."""
    from agno.agent import Agent

    selected_model = model or build_model(model_id or MODEL_ID)
    resolved_model_id = str(model_id or getattr(selected_model, "id", None) or MODEL_ID)
    operation_trace_id = trace_id or f"badcase-ai-{uuid.uuid4().hex[:16]}"
    with provider_accounting_scope(
        trace_id=operation_trace_id,
        session_id=session_id,
        stage=stage,
        model_selection_reason=model_selection_reason,
        price_snapshot=get_enabled_price_for_model(resolved_model_id),
        model_policy_version="badcase_operation",
        explicit_retry=explicit_retry,
    ) as accounting:
        agent = Agent(model=selected_model, markdown=False)
        content, usage = await _collect_response(agent.arun(prompt, stream=False))
    active_model = selected_model
    usage["thinking_enabled"] = getattr(active_model, "use_thinking", None)
    usage["provider_attempts"] = list(accounting.attempts)
    usage["trace_id"] = operation_trace_id
    return content, usage


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a text block."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _focused_badcase_model_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Keep authoritative failure evidence visible to AI diagnosis.

    Retrieval chunks can be very large and previously pushed evaluation and
    contract failures beyond a blunt prompt slice.  The diagnosis stages need
    the failure evidence itself, while the full immutable context remains
    available on the Badcase and Trace detail pages.
    """
    focused_keys = (
        "trace_id",
        "session_id",
        "message_id",
        "feedback_type",
        "route_decision",
        "route_intent",
        "route_reason",
        "current_agent",
        "activated_skills",
        "citations",
        "tool_invocations",
        "tool_calls",
        "mcp_calls",
        "evaluation_results",
        "contract_violations",
    )
    focused = {
        key: context.get(key)
        for key in focused_keys
        if context.get(key) not in (None, "", [], {})
    }
    retrieval = context.get("retrieval_evidence") or []
    if retrieval:
        focused["retrieval_evidence_summary"] = [
            {
                "evidence_id": item.get("evidence_id"),
                "document_id": item.get("document_id"),
                "chunk_id": item.get("chunk_id"),
                "title": item.get("title"),
            }
            for item in retrieval[:8]
        ]
        focused["retrieval_evidence_count"] = len(retrieval)
    return focused


def _find_darwin_skill() -> Optional[Dict[str, Any]]:
    """Find the Darwin optimization skill by name."""
    for name in ("达尔文", "darwin", "Darwin"):
        skill = get_skill_by_name(name)
        if skill:
            return skill
    return None


@router.get("")
async def list_badcases(
    status: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
    has_trace: Optional[bool] = None,
    has_retest: Optional[bool] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
    root_cause_domain: Optional[str] = None,
    priority: Optional[str] = None,
    search: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    view_scope: str = "current",
    user_status: Optional[str] = None,
):
    """Return a bounded, lightweight Badcase workbench page."""
    if status and status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail=f"invalid status: {status}")
    if category and category not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"invalid category: {category}")
    if root_cause_domain and root_cause_domain not in ROOT_CAUSE_DOMAINS:
        raise HTTPException(status_code=400, detail=f"invalid root_cause_domain: {root_cause_domain}")
    if priority and priority not in {"low", "medium", "high"}:
        raise HTTPException(status_code=400, detail=f"invalid priority: {priority}")
    if page < 1:
        raise HTTPException(status_code=400, detail="page must be >= 1")
    if page_size not in {20, 50}:
        raise HTTPException(status_code=400, detail="page_size must be 20 or 50")
    if view_scope not in {"current", "history", "all"}:
        raise HTTPException(status_code=400, detail="view_scope must be current, history or all")
    if user_status and user_status not in {"review", "processing", "verifying", "ended"}:
        raise HTTPException(status_code=400, detail="invalid user_status")
    result = db_list_badcases_page(
        page=page,
        page_size=page_size,
        status=status,
        category=category,
        source=source,
        root_cause_domain=root_cause_domain,
        priority=priority,
        search=search,
        has_trace=has_trace,
        has_retest=has_retest,
        created_after=created_after,
        created_before=created_before,
        view_scope=view_scope,
        user_status=user_status,
    )
    for item in result["items"]:
        item["category_label"] = CATEGORY_LABELS.get(item.get("category"), "待分类")
        item["user_status_label"] = user_status_label(str(item.get("status") or "pending"))
        item["terminal_outcome_label"] = (
            "自动误抓" if item.get("is_auto_false_positive")
            else "重复问题" if item.get("status") == "duplicate"
            else "已知限制，暂不处理" if item.get("status") == "accepted_limitation"
            else "已解决" if item.get("status") == "closed"
            else ""
        )
        item["record_layer_label"] = (
            "系统观察" if item.get("is_system_observation")
            else "历史待核验" if item.get("is_history_insufficient")
            else "当前问题"
        )
    # Keep aliases for older callers, but both aliases now contain lightweight
    # list rows only; long evidence remains detail-only.
    return {**result, "badcases": result["items"], "count": result["total"]}


@router.get("/{case_id}")
async def get_badcase(case_id: int):
    """Get a single badcase with actions and drafts."""
    case = _load_case(case_id)
    _attach_drafts(case)
    return {"badcase": _enrich_badcase(case)}


@router.post("")
async def create_badcase(request: BadcaseCreate):
    """Create an operator-reported suspected Badcase in the review queue."""
    if request.category not in VALID_CATEGORIES:
        request.category = "other"
    # This public endpoint is the manual-entry path.  Neither callers nor AI
    # may skip the human review queue by supplying a later lifecycle status.
    request.status = "pending"
    request.source = "manual"
    case = db_create_badcase(
        title=request.title,
        description=request.description,
        category=request.category,
        status=request.status,
        evidence=request.evidence,
        source_message_id=request.source_message_id,
        session_id=request.session_id,
        source=request.source,
        original_query=request.original_query,
        ai_response=request.ai_response,
        feedback_reason=request.feedback_reason,
        priority=request.priority,
        symptom=request.symptom,
        expected_behavior=request.expected_behavior,
        actual_behavior=request.actual_behavior,
        root_cause_domain=request.root_cause_domain or "unknown",
        secondary_root_cause_domains=json.dumps(request.secondary_root_cause_domains, ensure_ascii=False),
        impact_scope=request.impact_scope,
        owner=request.owner,
        linked_evaluation_case_id=request.linked_evaluation_case_id,
        linked_evaluation_run_id=request.linked_evaluation_run_id,
    )
    return {"badcase": _enrich_badcase(case)}


@router.put("/{case_id}")
async def update_badcase(case_id: int, request: BadcaseUpdate):
    """Update badcase fields."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")
    if request.status is not None and request.status != case.get("status"):
        raise HTTPException(
            status_code=400,
            detail="状态必须通过分类、归因、验证、发布观察、关闭等生命周期操作变更，不能直接编辑",
        )
    if request.root_cause_domain and request.root_cause_domain not in ROOT_CAUSE_DOMAINS:
        raise HTTPException(status_code=400, detail=f"invalid root_cause_domain: {request.root_cause_domain}")
    updated = db_update_badcase(
        case_id=case_id,
        title=request.title,
        description=request.description,
        category=request.category,
        status=request.status,
        evidence=request.evidence,
        root_cause=request.root_cause,
        fix_plan=request.fix_plan,
        rejected_reason=request.rejected_reason,
        priority=request.priority,
        symptom=request.symptom,
        expected_behavior=request.expected_behavior,
        actual_behavior=request.actual_behavior,
        root_cause_domain=request.root_cause_domain,
        secondary_root_cause_domains=(
            json.dumps(request.secondary_root_cause_domains, ensure_ascii=False)
            if request.secondary_root_cause_domains is not None else None
        ),
        impact_scope=request.impact_scope,
        owner=request.owner,
    )
    if updated:
        _record_action(
            case_id, "update", request.dict(exclude_unset=True),
            case["status"], updated["status"], "user"
        )
    return {"badcase": _enrich_badcase(updated)}


@router.delete("/{case_id}")
async def delete_badcase(case_id: int):
    """Delete a badcase."""
    deleted = db_delete_badcase(case_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="not found")
    return {"ok": True, "deleted_id": case_id}


@router.post("/{case_id}/classify")
async def classify_badcase(case_id: int, request: ClassifyRequest = ClassifyRequest()):
    """Classify a badcase into one of the operational categories."""
    case = _load_case(case_id)
    _require_case_status(case, "classify", {"pending"})

    context = case.get("context_json") or ""
    if isinstance(context, str) and context:
        try:
            context_obj = json.loads(context)
        except Exception:
            context_obj = {}
    else:
        context_obj = context or {}
    focused_context = _focused_badcase_model_context(context_obj)

    classify_trace_id = uuid.uuid4().hex[:16]
    model_id = "deepseek-v4-flash"
    raw = ""
    parsed: Dict[str, Any] = {}
    usage: Dict[str, Optional[int]] = {}
    status = "success"
    error_summary = None
    start = time.time()

    # AI classification is an extra evaluation step; enforce the daily budget
    # only when the caller asks for automatic (LLM-based) classification.
    if request.auto:
        _enforce_background_budget("badcase_classify")

    if request.auto:
        prompt = (
            "你是一名 AI 运营问题分类专家。请根据下面的 Badcase 信息，从以下类别中选择一个最贴切的，"
            "给出根因假设、修复路径建议、优先级，并严格输出 JSON：\n"
            "- knowledge_gap：知识库内容缺失、错误或未命中\n"
            "- skill_prompt：Skill 触发条件或 Prompt 指令缺陷\n"
            "- mcp_capability：MCP/外部工具/系统能力缺失或调用失败\n"
            "- routing：意图路由错误\n"
            "- response_quality：模型回复质量差、格式错误、未遵循指令\n"
            "- other：其他\n\n"
            f"标题：{case['title']}\n"
            f"描述：{case.get('description', '')}\n"
            f"反馈原因：{case.get('feedback_reason', '')}\n"
            f"原问题：{case.get('original_query', '')}\n"
            f"原回答：{case.get('ai_response', '')[:500]}\n"
            f"关键证据：{json.dumps(focused_context, ensure_ascii=False)[:4000]}\n\n"
            "证据优先规则：若来源是 runtime_contract、evaluation 或 tool_failure，"
            "必须以 contract_violations、evaluation_results、失败 Tool 证据为事实，"
            "不得只根据已经过安全渲染的最终回答反推根因。"
            "unstructured_reference_marker 表示模型原始输出尝试生成 EvidenceSet 外标记；"
            "最终回答没有该标记通常是 CitationRenderer 正确拦截，不是漏答。\n"
            "输出字段：suggested_category, root_cause_hypothesis, repair_path_suggestion, priority。"
            "repair_path_suggestion 应从 knowledge、skill_prompt、mcp_capability、ops_only 中选择。"
            "另输出 root_cause_domain，只能是 routing、knowledge_rag、model_instruction、tool_mcp、"
            "authority_safety、human_collaboration、ux、external_dependency、unknown。"
        )
        try:
            raw, usage = await _llm_generate(
                prompt,
                model_id=model_id,
                trace_id=classify_trace_id,
                session_id=case.get("session_id") or f"badcase:{case_id}",
                stage="badcase_classify",
                model_selection_reason="Badcase classification uses Flash",
            )
            parsed = _extract_json(raw) or {}
        except Exception as e:
            logger.exception("AI classification failed")
            status = "failed"
            error_summary = f"{type(e).__name__}: provider operation failed"
            raise HTTPException(
                status_code=502,
                detail="AI suggestion failed; Badcase status was not changed",
            ) from e

        category = parsed.get("suggested_category", parsed.get("category"))
        reason = parsed.get("root_cause_hypothesis", parsed.get("reason"))
        if category not in (VALID_CATEGORIES - {"pending"}) or not str(reason or "").strip():
            raise HTTPException(
                status_code=502,
                detail="AI suggestion returned an invalid structure; Badcase status was not changed",
            )
        repair_path = parsed.get("repair_path_suggestion", repair_path_for_category(category))
        priority = parsed.get("priority", "medium")
        if category not in VALID_CATEGORIES:
            category = "other"
        if priority not in ("high", "medium", "low"):
            priority = "medium"
        root_cause_domain = parsed.get("root_cause_domain", "unknown")
        if root_cause_domain not in ROOT_CAUSE_DOMAINS:
            root_cause_domain = "unknown"
    else:
        category = request.category or "other"
        reason = request.reason
        repair_path = repair_path_for_category(category)
        priority = "medium"
        if category not in (VALID_CATEGORIES - {"pending"}):
            raise HTTPException(status_code=400, detail=f"invalid category: {category}")
        root_cause_domain = request.root_cause_domain or "unknown"
        if root_cause_domain not in ROOT_CAUSE_DOMAINS:
            raise HTTPException(status_code=400, detail=f"invalid root_cause_domain: {root_cause_domain}")

    if request.auto:
        suggestion = {
            "category": category,
            "reason": reason,
            "repair_path_suggestion": repair_path,
            "priority": priority,
            "root_cause_domain": root_cause_domain,
            "classify_trace_id": classify_trace_id,
        }
        _record_action(
            case_id,
            "ai-suggestion",
            suggestion,
            case["status"],
            case["status"],
            "ai_flash",
        )
        return {
            "badcase": _enrich_badcase(_load_case(case_id)),
            "suggestion": suggestion,
            "suggested_category": category,
            "root_cause_hypothesis": reason,
            "repair_path_suggestion": repair_path,
            "priority": priority,
            "root_cause_domain": root_cause_domain,
            "status_changed": False,
        }

    # Only an explicit operator decision adopts the category and enters the
    # processing group.  Historical internal states remain readable.
    new_status = "fixing"
    updated = db_update_badcase(
        case_id,
        category=category,
        status=new_status,
        root_cause=reason,
        fix_plan=repair_path,
        priority=priority,
        root_cause_domain=root_cause_domain,
    )
    _record_action(
        case_id,
        "classify",
        {
            "category": category,
            "reason": reason,
            "repair_path_suggestion": repair_path,
            "priority": priority,
            "root_cause_domain": root_cause_domain,
            "raw": raw if request.auto else None,
            "classify_trace_id": classify_trace_id,
        },
        case["status"],
        new_status,
        "ai_flash" if request.auto else "operator",
    )
    return {
        "badcase": _enrich_badcase(updated),
        "suggested_category": category,
        "root_cause_hypothesis": reason,
        "repair_path_suggestion": repair_path,
        "priority": priority,
        "root_cause_domain": root_cause_domain,
    }


@router.post("/{case_id}/extract-knowledge")
async def extract_knowledge(case_id: int, request: ExtractKnowledgeRequest = ExtractKnowledgeRequest()):
    """Retired AI draft path; historical drafts remain readable."""
    raise HTTPException(
        status_code=410,
        detail="AI draft creation is retired; request a suggestion and let an operator act manually",
    )

    # Kept below only for historical source compatibility; unreachable by API.
    case = _load_case(case_id)
    _require_case_status(case, "extract-knowledge", {"classified"})
    if case.get("category") not in ("knowledge_gap", "pending"):
        raise HTTPException(
            status_code=400,
            detail=f"extract-knowledge is only for knowledge_gap category, got {case.get('category')}"
        )

    title = request.title or case["title"]
    provider_trace_id: Optional[str] = None
    needs_model = request.auto or not str(request.content or "").strip()
    if needs_model:
        _enforce_background_budget("badcase_extract_knowledge")
        prompt = (
            "请根据以下 Badcase 信息，总结成一段可直接写入知识库的知识条目。"
            "回答应包含：问题现象、正确结论、给业主的标准话术。\n\n"
            f"标题：{case['title']}\n"
            f"描述：{case.get('description', '')}\n"
            f"证据：{case.get('evidence', '')}\n\n"
            "直接输出知识条目内容，不要添加解释。"
        )
        provider_trace_id = f"badcase-knowledge-{case_id}-{uuid.uuid4().hex[:10]}"
        content, extract_usage = await _llm_generate(
            prompt,
            trace_id=provider_trace_id,
            session_id=case.get("session_id") or f"badcase:{case_id}",
            stage="badcase_extract_knowledge",
            model_selection_reason="AI suggestion for a manually reviewed knowledge draft",
        )
    else:
        content = request.content

    draft = db_create_knowledge_draft(
        badcase_id=case_id,
        title=title,
        content=content,
        category=request.category,
        status="draft",
    )

    # Move to fixing state if currently classified.
    if case["status"] == "classified":
        updated = db_update_badcase(case_id, status="fixing", fix_plan="extracted to knowledge draft")
        _record_action(
            case_id,
            "extract-knowledge",
            {
                "draft_id": draft["id"],
                "provider_trace_id": provider_trace_id,
                "provider_local_attempt_id": (
                    extract_usage.get("local_attempt_id") if provider_trace_id else None
                ),
            },
            case["status"],
            "fixing",
        )
        case = updated or case

    return {
        "badcase": _enrich_badcase(case),
        "knowledge_draft": draft,
        "provider_trace_id": provider_trace_id,
    }


@router.post("/{case_id}/publish-draft/{draft_id}")
async def publish_knowledge_draft(case_id: int, draft_id: int):
    """Backward-compatible alias: apply an approved knowledge draft."""
    case = _load_case(case_id)
    _require_case_status(case, "publish-draft", {"fixing"})
    draft = _load_draft(draft_id, case_id, db_get_knowledge_draft, "knowledge draft")
    if draft.get("status") != "approved":
        raise HTTPException(status_code=400, detail="draft must be approved before applying")

    updated, doc = await _apply_knowledge_draft(case_id, draft_id, draft, case, "publish-knowledge")
    _attach_drafts(updated)
    return {"badcase": _enrich_badcase(updated), "knowledge_doc": doc}


@router.post("/{case_id}/publish-skill-draft/{draft_id}")
async def publish_skill_prompt_draft_endpoint(
    case_id: int, draft_id: int, request: PublishSkillDraftRequest = PublishSkillDraftRequest()
):
    """Backward-compatible alias: apply an approved skill/prompt draft to an agent."""
    case = _load_case(case_id)
    _require_case_status(case, "publish-skill-draft", {"fixing"})
    draft = _load_draft(draft_id, case_id, db_get_skill_prompt_draft, "skill/prompt draft")
    if draft.get("status") != "approved":
        raise HTTPException(status_code=400, detail="draft must be approved before applying")

    updated = await _apply_skill_prompt_draft(
        case_id, draft_id, draft, case, "publish-skill-prompt", request.target_agent_id
    )
    _attach_drafts(updated)
    return {"badcase": _enrich_badcase(updated)}


@router.post("/{case_id}/accept-capability-gap/{draft_id}")
async def accept_capability_gap_endpoint(
    case_id: int, draft_id: int, request: AcceptGapRequest = AcceptGapRequest()
):
    """Backward-compatible alias: accept an approved capability gap as backlog."""
    case = _load_case(case_id)
    _require_case_status(case, "accept-capability-gap", {"fixing"})
    draft = _load_draft(draft_id, case_id, db_get_capability_gap_draft, "capability gap draft")
    if draft.get("status") != "approved":
        raise HTTPException(status_code=400, detail="draft must be approved before applying")

    await _apply_capability_gap_draft(
        case_id, draft_id, draft, case, "accept-capability-gap", request.note
    )
    _attach_drafts(case)
    return {
        "badcase": _enrich_badcase(case),
        "note": "能力缺口已记录为产品待办，未自动创建工具；Badcase 仍保持修复中",
    }


# -----------------------------------------------------------------------------
# Draft review / edit / apply endpoints
# -----------------------------------------------------------------------------


def _move_to_verifying_after_apply(case: Dict[str, Any], case_id: int, action_type: str, detail: Any) -> Dict[str, Any]:
    """Move badcase from fixing to verifying after a draft has been applied.

    Clears any pre-apply retest evidence so that verify-pass can only be
    granted after a post-apply retest.
    """
    before = case["status"]
    new_status = "verifying"
    applied_at = now_cn()
    updated = db_update_badcase(
        case_id,
        status=new_status,
        fix_plan=f"{action_type} applied",
        retest_response="",
        retest_context_json="",
        retest_trace_id="",
        last_applied_at=applied_at,
    )
    _record_action(case_id, action_type, detail, before, new_status)
    return updated or case


async def _apply_knowledge_draft(
    case_id: int, draft_id: int, draft: Dict[str, Any], case: Dict[str, Any], action_type: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create a knowledge doc, reindex it, and only then publish the draft.

    If reindex fails, delete the orphan doc and keep the draft approved/case fixing.
    """
    import rag_indexer

    doc = db_create_knowledge_doc(
        title=draft["title"],
        content=draft["content"],
        category=draft.get("category", "未分类"),
    )
    try:
        rag_indexer.reindex_document(doc["id"])
    except Exception as exc:
        logger.exception("reindex after %s failed", action_type)
        # Clean up the orphan document so we don't leave an unindexed doc behind.
        try:
            db_delete_knowledge_doc(doc["id"])
        except Exception:
            logger.exception("failed to delete orphan knowledge doc %s", doc["id"])
        raise HTTPException(
            status_code=500,
            detail=f"知识库索引失败，应用未生效：{exc}",
        )

    db_update_knowledge_draft(draft_id, status="published", knowledge_doc_id=doc["id"])
    detail = {
        "doc_id": doc["id"],
        "draft_id": draft_id,
        "draft_snapshot": _draft_snapshot(draft, _KNOWLEDGE_DRAFT_SNAPSHOT_FIELDS),
        "index_result": "success",
    }
    updated = _move_to_verifying_after_apply(case, case_id, action_type, detail)
    return updated, doc


async def _apply_skill_prompt_draft(
    case_id: int,
    draft_id: int,
    draft: Dict[str, Any],
    case: Dict[str, Any],
    action_type: str,
    target_agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or update a formal Skill from an approved draft and bind it to an Agent.

    The badcase only moves to verifying once the agent binding succeeds.
    """
    if draft.get("status") != "approved":
        raise HTTPException(status_code=400, detail="draft must be approved before applying")
    if not target_agent_id:
        # Preserve the v1.3.4 first-batch contract: applying without a target agent
        # is intentionally blocked until an agent is selected.
        raise HTTPException(
            status_code=409,
            detail="待选择目标 Agent：请提供 target_agent_id 以建立 agent_skills 绑定",
        )

    agent = get_agent_by_agent_id(target_agent_id)
    if not agent:
        raise HTTPException(status_code=400, detail=f"target agent not found: {target_agent_id}")

    skill_name = draft.get("skill_name") or draft.get("title") or "未命名 Skill"
    description = draft.get("title") or skill_name
    instructions = draft.get("prompt_content") or ""
    trigger_condition = draft.get("trigger_keywords") or ""

    def governed_metadata(source_skill: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        metadata = canonical_metadata(
            {
                "trigger_condition": trigger_condition,
                "skill_metadata": (source_skill or {}).get("skill_metadata") or {},
            },
            (source_skill or {}).get("skill_metadata") or {},
        )
        metadata["positive_triggers"] = metadata.get("positive_triggers") or [
            item.strip() for item in re.split(r"[,，、；;｜|\\/]+", trigger_condition) if item.strip()
        ]
        metadata["version"] = next_patch_version(str(metadata.get("version") or "legacy-1.0.0")) if source_skill else "1.0.0"
        return metadata

    # Create or update the formal Skill.
    skill_id = draft.get("skill_id")
    existing_skill = get_skill(skill_id) if skill_id else None
    try:
        if existing_skill:
            metadata = governed_metadata(existing_skill)
            skill = db_update_skill(
                skill_id=existing_skill["id"],
                name=skill_name,
                description=description,
                instructions=instructions,
                category=existing_skill.get("category") or "skill_prompt",
                enabled=existing_skill.get("enabled", True),
                trigger_condition=trigger_condition,
                skill_metadata=metadata,
                storage_path=existing_skill.get("storage_path", ""),
                model_id=existing_skill.get("model_id"),
            )
        else:
            existing_by_name = get_skill_by_name(skill_name)
            if existing_by_name:
                metadata = governed_metadata(existing_by_name)
                skill = db_update_skill(
                    skill_id=existing_by_name["id"],
                    name=skill_name,
                    description=description,
                    instructions=instructions,
                    category=existing_by_name.get("category") or "skill_prompt",
                    enabled=existing_by_name.get("enabled", True),
                    trigger_condition=trigger_condition,
                    skill_metadata=metadata,
                    storage_path=existing_by_name.get("storage_path", ""),
                    model_id=existing_by_name.get("model_id"),
                )
            else:
                metadata = governed_metadata(None)
                skill = db_create_skill(
                    name=skill_name,
                    description=description,
                    instructions=instructions,
                    category="skill_prompt",
                    enabled=True,
                    trigger_condition=trigger_condition,
                    skill_metadata=metadata,
                )
    except Exception as exc:
        logger.exception("failed to create/update formal skill from draft")
        raise HTTPException(status_code=500, detail=f"Skill 持久化失败：{exc}")

    if not skill:
        raise HTTPException(status_code=500, detail="Skill 创建/更新后未返回有效记录")

    skill_id = skill["id"]
    metadata = skill.get("skill_metadata") or {}
    skill_storage.write_skill_md(skill_id, metadata, skill.get("instructions") or "")
    skill_storage.write_skill_revision(skill_id, str(metadata.get("version") or "legacy-1.0.0"), metadata, skill.get("instructions") or "")
    create_skill_version(skill_id, str(metadata.get("version") or "legacy-1.0.0"), "从 Badcase 审核草稿应用")

    # Bind the skill to the target agent, preserving existing bindings.
    before_skill_ids = get_agent_skills(target_agent_id)
    try:
        new_skill_ids = list(dict.fromkeys(before_skill_ids + [skill_id]))
        set_agent_skills(target_agent_id, new_skill_ids)
    except Exception as exc:
        logger.exception("failed to bind skill to agent")
        raise HTTPException(status_code=500, detail=f"Skill 绑定到 Agent 失败：{exc}")

    after_skill_ids = get_agent_skills(target_agent_id)
    now = datetime.now(timezone.utc).isoformat()

    db_update_skill_prompt_draft(
        draft_id,
        status="published",
        skill_id=skill_id,
        published_at=now,
        published_by="operator",
    )

    detail = {
        "draft_id": draft_id,
        "draft_snapshot": _draft_snapshot(draft, _SKILL_DRAFT_SNAPSHOT_FIELDS),
        "skill_id": skill_id,
        "agent_id": target_agent_id,
        "agent_name": agent.get("name"),
        "version": skill.get("updated_at") or now,
        "timestamp": now,
        "agent_skills_before": before_skill_ids,
        "agent_skills_after": after_skill_ids,
    }
    updated = _move_to_verifying_after_apply(case, case_id, action_type, detail)
    return updated


async def _apply_capability_gap_draft(
    case_id: int, draft_id: int, draft: Dict[str, Any], case: Dict[str, Any], action_type: str, note: str = ""
) -> Dict[str, Any]:
    """Accept a capability gap as backlog only; do not move case to verifying."""
    now = datetime.now(timezone.utc).isoformat()
    db_update_capability_gap_draft(
        draft_id,
        status="accepted",
        accepted_at=now,
        accepted_by="operator",
    )
    detail = {
        "draft_id": draft_id,
        "draft_snapshot": _draft_snapshot(draft, _CAPABILITY_GAP_DRAFT_SNAPSHOT_FIELDS),
        "note": note or "待建设能力",
        "status": "accepted",
        "message": "能力缺口已记录为产品待办，未自动创建真实工具",
    }
    _record_action(case_id, action_type, detail, case["status"], case["status"])
    return case


@router.put("/{case_id}/knowledge-drafts/{draft_id}")
async def edit_knowledge_draft(
    case_id: int, draft_id: int, request: EditKnowledgeDraftRequest = EditKnowledgeDraftRequest()
):
    """Edit a knowledge draft (fixing status only). Approved drafts reset to draft."""
    case = _load_case(case_id)
    _require_case_status(case, "edit-knowledge-draft", {"fixing"})
    draft = _load_draft(draft_id, case_id, db_get_knowledge_draft, "knowledge draft")
    if not is_draft_editable("knowledge", draft.get("status", "draft")):
        raise HTTPException(status_code=400, detail="cannot edit terminal draft")

    before = _draft_snapshot(draft, _KNOWLEDGE_DRAFT_SNAPSHOT_FIELDS)
    new_status = "draft" if draft.get("status") == "approved" else draft.get("status")
    updated = db_update_knowledge_draft(
        draft_id,
        title=request.title,
        content=request.content,
        category=request.category,
        status=new_status,
    )
    after = _draft_snapshot(updated or draft, _KNOWLEDGE_DRAFT_SNAPSHOT_FIELDS)
    _record_action(
        case_id,
        "edit-knowledge-draft",
        {"draft_id": draft_id, "before": before, "after": after},
        case["status"],
        case["status"],
    )
    _attach_drafts(case)
    return {"badcase": _enrich_badcase(case), "knowledge_draft": updated}


@router.post("/{case_id}/knowledge-drafts/{draft_id}/review")
async def review_knowledge_draft(
    case_id: int, draft_id: int, request: ReviewDraftRequest = ReviewDraftRequest()
):
    """Review a knowledge draft with strict status transitions."""
    case = _load_case(case_id)
    _require_case_status(case, "review-knowledge-draft", {"fixing"})
    draft = _load_draft(draft_id, case_id, db_get_knowledge_draft, "knowledge draft")
    _require_draft_transition("knowledge", draft, request.status)

    before = _draft_snapshot(draft, _KNOWLEDGE_DRAFT_SNAPSHOT_FIELDS)
    updated = db_update_knowledge_draft(draft_id, status=request.status)
    after = _draft_snapshot(updated or draft, _KNOWLEDGE_DRAFT_SNAPSHOT_FIELDS)
    _record_action(
        case_id,
        "review-knowledge-draft",
        {"draft_id": draft_id, "before": before, "after": after, "note": request.note},
        case["status"],
        case["status"],
    )
    _attach_drafts(case)
    return {"badcase": _enrich_badcase(case), "knowledge_draft": updated}


@router.post("/{case_id}/knowledge-drafts/{draft_id}/apply")
async def apply_knowledge_draft(case_id: int, draft_id: int):
    """Apply an approved knowledge draft to the official knowledge base and reindex."""
    case = _load_case(case_id)
    _require_case_status(case, "apply-knowledge-draft", {"fixing"})
    draft = _load_draft(draft_id, case_id, db_get_knowledge_draft, "knowledge draft")
    if draft.get("status") != "approved":
        raise HTTPException(status_code=400, detail="draft must be approved before applying")

    updated, doc = await _apply_knowledge_draft(case_id, draft_id, draft, case, "apply-knowledge-draft")
    _attach_drafts(updated)
    return {"badcase": _enrich_badcase(updated), "knowledge_doc": doc}


@router.put("/{case_id}/skill-prompt-drafts/{draft_id}")
async def edit_skill_prompt_draft(
    case_id: int, draft_id: int, request: EditSkillDraftRequest = EditSkillDraftRequest()
):
    """Edit a skill/prompt draft (fixing status only). Approved drafts reset to draft."""
    case = _load_case(case_id)
    _require_case_status(case, "edit-skill-prompt-draft", {"fixing"})
    draft = _load_draft(draft_id, case_id, db_get_skill_prompt_draft, "skill/prompt draft")
    if not is_draft_editable("skill_prompt", draft.get("status", "draft")):
        raise HTTPException(status_code=400, detail="cannot edit terminal draft")

    before = _draft_snapshot(draft, _SKILL_DRAFT_SNAPSHOT_FIELDS)
    new_status = "draft" if draft.get("status") == "approved" else draft.get("status")
    updated = db_update_skill_prompt_draft(
        draft_id,
        title=request.title,
        skill_name=request.skill_name,
        prompt_content=request.prompt_content,
        trigger_keywords=request.trigger_keywords,
        status=new_status,
    )
    after = _draft_snapshot(updated or draft, _SKILL_DRAFT_SNAPSHOT_FIELDS)
    _record_action(
        case_id,
        "edit-skill-prompt-draft",
        {"draft_id": draft_id, "before": before, "after": after},
        case["status"],
        case["status"],
    )
    _attach_drafts(case)
    return {"badcase": _enrich_badcase(case), "skill_prompt_draft": updated}


@router.post("/{case_id}/skill-prompt-drafts/{draft_id}/review")
async def review_skill_prompt_draft(
    case_id: int, draft_id: int, request: ReviewDraftRequest = ReviewDraftRequest()
):
    """Review a skill/prompt draft with strict status transitions."""
    case = _load_case(case_id)
    _require_case_status(case, "review-skill-prompt-draft", {"fixing"})
    draft = _load_draft(draft_id, case_id, db_get_skill_prompt_draft, "skill/prompt draft")
    _require_draft_transition("skill_prompt", draft, request.status)

    before = _draft_snapshot(draft, _SKILL_DRAFT_SNAPSHOT_FIELDS)
    updated = db_update_skill_prompt_draft(draft_id, status=request.status)
    after = _draft_snapshot(updated or draft, _SKILL_DRAFT_SNAPSHOT_FIELDS)
    _record_action(
        case_id,
        "review-skill-prompt-draft",
        {"draft_id": draft_id, "before": before, "after": after, "note": request.note},
        case["status"],
        case["status"],
    )
    _attach_drafts(case)
    return {"badcase": _enrich_badcase(case), "skill_prompt_draft": updated}


@router.post("/{case_id}/skill-prompt-drafts/{draft_id}/apply")
async def apply_skill_prompt_draft(
    case_id: int, draft_id: int, request: PublishSkillDraftRequest = PublishSkillDraftRequest()
):
    """Apply an approved skill/prompt draft to a target agent."""
    case = _load_case(case_id)
    _require_case_status(case, "apply-skill-prompt-draft", {"fixing"})
    draft = _load_draft(draft_id, case_id, db_get_skill_prompt_draft, "skill/prompt draft")
    if draft.get("status") != "approved":
        raise HTTPException(status_code=400, detail="draft must be approved before applying")

    updated = await _apply_skill_prompt_draft(
        case_id, draft_id, draft, case, "apply-skill-prompt-draft", request.target_agent_id
    )
    _attach_drafts(updated)
    return {"badcase": _enrich_badcase(updated)}


@router.put("/{case_id}/capability-gap-drafts/{draft_id}")
async def edit_capability_gap_draft(
    case_id: int, draft_id: int, request: EditCapabilityGapDraftRequest = EditCapabilityGapDraftRequest()
):
    """Edit a capability gap draft (fixing status only). Approved drafts reset to draft."""
    case = _load_case(case_id)
    _require_case_status(case, "edit-capability-gap-draft", {"fixing"})
    draft = _load_draft(draft_id, case_id, db_get_capability_gap_draft, "capability gap draft")
    if not is_draft_editable("capability_gap", draft.get("status", "draft")):
        raise HTTPException(status_code=400, detail="cannot edit terminal draft")

    before = _draft_snapshot(draft, _CAPABILITY_GAP_DRAFT_SNAPSHOT_FIELDS)
    new_status = "draft" if draft.get("status") == "approved" else draft.get("status")
    updated = db_update_capability_gap_draft(
        draft_id,
        title=request.title,
        description=request.description,
        gap_type=request.gap_type,
        suggested_action=request.suggested_action,
        status=new_status,
    )
    after = _draft_snapshot(updated or draft, _CAPABILITY_GAP_DRAFT_SNAPSHOT_FIELDS)
    _record_action(
        case_id,
        "edit-capability-gap-draft",
        {"draft_id": draft_id, "before": before, "after": after},
        case["status"],
        case["status"],
    )
    _attach_drafts(case)
    return {"badcase": _enrich_badcase(case), "capability_gap_draft": updated}


@router.post("/{case_id}/capability-gap-drafts/{draft_id}/review")
async def review_capability_gap_draft(
    case_id: int, draft_id: int, request: ReviewDraftRequest = ReviewDraftRequest()
):
    """Review a capability gap draft with strict status transitions."""
    case = _load_case(case_id)
    _require_case_status(case, "review-capability-gap-draft", {"fixing"})
    draft = _load_draft(draft_id, case_id, db_get_capability_gap_draft, "capability gap draft")
    _require_draft_transition("capability_gap", draft, request.status)

    before = _draft_snapshot(draft, _CAPABILITY_GAP_DRAFT_SNAPSHOT_FIELDS)
    updated = db_update_capability_gap_draft(draft_id, status=request.status)
    after = _draft_snapshot(updated or draft, _CAPABILITY_GAP_DRAFT_SNAPSHOT_FIELDS)
    _record_action(
        case_id,
        "review-capability-gap-draft",
        {"draft_id": draft_id, "before": before, "after": after, "note": request.note},
        case["status"],
        case["status"],
    )
    _attach_drafts(case)
    return {"badcase": _enrich_badcase(case), "capability_gap_draft": updated}


@router.post("/{case_id}/capability-gap-drafts/{draft_id}/apply")
async def apply_capability_gap_draft(
    case_id: int, draft_id: int, request: AcceptGapRequest = AcceptGapRequest()
):
    """Apply an approved capability gap draft as a product backlog item (no real tool created)."""
    case = _load_case(case_id)
    _require_case_status(case, "apply-capability-gap-draft", {"fixing"})
    draft = _load_draft(draft_id, case_id, db_get_capability_gap_draft, "capability gap draft")
    if draft.get("status") != "approved":
        raise HTTPException(status_code=400, detail="draft must be approved before applying")

    await _apply_capability_gap_draft(
        case_id, draft_id, draft, case, "apply-capability-gap-draft", request.note
    )
    _attach_drafts(case)
    return {
        "badcase": _enrich_badcase(case),
        "note": "能力缺口已记录为产品待办，未自动创建工具；Badcase 仍保持修复中",
    }


def _build_price_snapshot(model_id: str) -> Optional[Dict[str, Any]]:
    price = get_enabled_price_for_model(model_id)
    return build_price_snapshot(price)


def _calculate_cost(model_id: str, usage: Dict[str, Optional[int]]) -> tuple:
    snapshot = _build_price_snapshot(model_id)
    cost, _status = compute_cost_cny(snapshot, usage)
    return cost, snapshot


def _cost_evidence_for_call(
    stage: str,
    requested_model: str,
    usage: Dict[str, Any],
    status: str,
):
    provider_response_model = usage.get("provider_response_model")
    price_model = provider_response_model or requested_model
    cost = build_cost_entry(
        stage=stage,
        provider="deepseek",
        requested_model=requested_model,
        response_model=None,
        provider_response_model=provider_response_model,
        thinking_enabled=usage.get("thinking_enabled"),
        model_policy_version="v1.8.2-s5-badcase",
        provider_usage=usage or None,
        price_row=get_enabled_price_for_model(price_model),
        provider_succeeded=status == "success",
    )
    normalized = cost_entry_usage_payload(
        cost,
        provider_request_id=usage.get("provider_request_id"),
    )
    return cost, normalized


@router.post("/{case_id}/darwin-fix")
async def darwin_fix(case_id: int, request: DarwinFixRequest = DarwinFixRequest()):
    """Generate an operator-reviewed Darwin suggestion without lifecycle changes."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")
    if case["status"] not in {"pending", "classified", "investigating", "fixing"}:
        raise HTTPException(status_code=400, detail=f"Darwin suggestion is unavailable for status {case['status']}")

    context = case.get("context_json") or ""
    if isinstance(context, str) and context:
        try:
            context_obj = json.loads(context)
        except Exception:
            context_obj = {}
    else:
        context_obj = context or {}
    focused_context = _focused_badcase_model_context(context_obj)

    darwin = _find_darwin_skill()
    darwin_instructions = darwin.get("instructions", "") if darwin else ""
    darwin_name = darwin.get("name", "达尔文") if darwin else "达尔文"

    prompt = (
        f"你是已安装的 Darwin（达尔文）优化 Skill：{darwin_name}。\n"
        f"{darwin_instructions}\n\n"
        "请对以下 Badcase 做深度运营分析。注意：你不能自动修改代码、不能自动创建真实 MCP 工具、不能声称已完成业务操作。"
        "你只能输出分析结论和人工可审核的草稿。\n\n"
        f"标题：{case['title']}\n"
        f"分类：{case.get('category', 'other')}\n"
        f"描述：{case.get('description', '')}\n"
        f"反馈原因：{case.get('feedback_reason', '')}\n"
        f"原问题：{case.get('original_query', '')}\n"
        f"原回答：{case.get('ai_response', '')[:600]}\n"
        f"关键证据：{json.dumps(focused_context, ensure_ascii=False)[:5000]}\n\n"
        "证据优先规则：contract_violations、evaluation_results 和 Tool 分阶段失败证据"
        "高于经过安全渲染的最终回答；不得把 CitationRenderer 正确剥离未授权标记"
        "误判为模型未满足用户格式要求。\n\n"
        "请严格输出 JSON（不要 Markdown 代码块）：\n"
        "{\n"
        '  "phenomenon_impact": "<问题现象与业务影响>",\n'
        '  "root_cause_hypothesis": "<根因假设>",\n'
        '  "root_cause_domain": "<routing|knowledge_rag|model_instruction|tool_mcp|authority_safety|human_collaboration|ux|external_dependency|unknown>",\n'
        '  "evidence_uncertainties": "<证据与不确定性>",\n'
        '  "repair_path_suggestion": "<建议修复路径：knowledge|skill_prompt|mcp_capability|ops_only>",\n'
        '  "recommended_category": "<推荐分类>",\n'
        '  "expected_impact": "<预期影响>",\n'
        '  "risks": "<风险说明>",\n'
        '  "suggested_actions": ["<建议动作1>", "<建议动作2>"],\n'
        '  "drafts": [\n'
        '    {"type": "knowledge", "title": "<知识库草稿标题>", "content": "<正文>", "target_doc_title": "<目标文档名，可选>"},\n'
        '    {"type": "skill_prompt", "title": "<Skill草稿标题>", "skill_name": "<Skill名称>", "prompt_content": "<Prompt内容>", "trigger_keywords": "<触发关键词>"},\n'
        '    {"type": "capability_gap", "title": "<能力缺口标题>", "description": "<缺口描述>", "gap_type": "mcp_write|integration|data", "suggested_action": "<建议>"}\n'
        "  ]\n"
        "}\n"
    )
    if request.prompt:
        prompt = f"{request.prompt}\n\n{prompt}"

    darwin_trace_id = uuid.uuid4().hex[:16]
    model_id = "deepseek-v4-pro"
    operation_started_at = now_cn()
    start_darwin_operation(
        trace_id=darwin_trace_id,
        badcase_id=case_id,
        started_at=operation_started_at,
    )
    start = time.time()
    status = "success"
    error_summary = None
    usage = {}

    # Darwin uses Pro and is an extra evaluation step; enforce the daily budget.
    budget_gate = _darwin_budget_gate()
    if not budget_gate.get("allowed"):
        blocked_reason = budget_gate.get("reason") or _BUDGET_BLOCKED_DETAIL
        persist_darwin_operation(
            trace_id=darwin_trace_id,
            badcase_id=case_id,
            model_call=None,
            operation_status="failed",
            started_at=operation_started_at,
            completed_at=now_cn(),
            status_before=case["status"],
            status_after=case["status"],
            error_summary=blocked_reason,
        )
        raise HTTPException(
            status_code=int(budget_gate["http_status"]),
            detail=budget_gate["detail"],
        )

    try:
        analysis_text, usage = await _llm_generate(
            prompt,
            model_id=model_id,
            trace_id=darwin_trace_id,
            session_id=f"badcase-darwin:{case_id}:{darwin_trace_id}",
            stage="darwin",
            model_selection_reason="仅低频Darwin深度分析，优先复杂分析质量，成本和耗时更高。",
        )
    except Exception as e:
        analysis_text = ""
        status = "failed"
        error_summary = f"{type(e).__name__}: provider operation failed"
    provider_attempts = get_provider_attempts_for_trace(darwin_trace_id)
    model_call = provider_attempts[-1] if provider_attempts else None

    if status != "success":
        persist_darwin_operation(
            trace_id=darwin_trace_id,
            badcase_id=case_id,
            model_call=model_call,
            operation_status="failed",
            started_at=operation_started_at,
            completed_at=now_cn(),
            status_before=case["status"],
            status_after=case["status"],
            error_summary=error_summary or "Darwin Provider call failed",
        )
        raise HTTPException(status_code=502, detail="Darwin Provider call failed")

    analysis_obj = _extract_json(analysis_text) or {}
    if not _complete_darwin_suggestion(analysis_obj):
        persist_darwin_operation(
            trace_id=darwin_trace_id,
            badcase_id=case_id,
            model_call=model_call,
            operation_status="failed",
            started_at=operation_started_at,
            completed_at=now_cn(),
            status_before=case["status"],
            status_after=case["status"],
            error_summary="Darwin returned an invalid structured suggestion",
        )
        raise HTTPException(
            status_code=502,
            detail="Darwin returned an invalid structured suggestion; Badcase status was not changed",
        )

    # Ensure required keys exist.
    analysis_obj.setdefault("phenomenon_impact", "")
    analysis_obj.setdefault("root_cause_hypothesis", analysis_obj.get("root_cause", ""))
    analysis_obj.setdefault("root_cause_domain", "unknown")
    if analysis_obj.get("root_cause_domain") not in ROOT_CAUSE_DOMAINS:
        analysis_obj["root_cause_domain"] = "unknown"
    analysis_obj.setdefault("evidence_uncertainties", "")
    analysis_obj.setdefault("repair_path_suggestion", repair_path_for_category(case.get("category", "other")))
    analysis_obj.setdefault("recommended_category", case.get("category", "other"))
    analysis_obj.setdefault("suggested_actions", [])
    analysis_obj.setdefault("expected_impact", "")
    analysis_obj.setdefault("risks", "")
    analysis_obj.setdefault("drafts", [])

    # Enrich with runtime metadata.
    analysis_obj["model"] = model_id
    analysis_obj["trace_id"] = darwin_trace_id
    direct_cost = model_call.get("estimated_cost_cny") if model_call else None
    cost_source = model_call.get("cost_source") if model_call else None
    # Keep the historical response key for UI compatibility, but make the
    # actual accounting semantics explicit alongside it.
    analysis_obj["token_cost_estimate"] = direct_cost
    analysis_obj["calculated_direct_cost"] = direct_cost
    analysis_obj["cost_source"] = cost_source
    analysis_obj["cost_disclaimer"] = "platform_price_snapshot_not_provider_final_bill"

    # Keep a backward-compatible root_cause alias for downstream consumers.
    analysis_obj.setdefault("root_cause", analysis_obj["root_cause_hypothesis"])

    before = case["status"]
    updated = db_update_badcase(
        case_id,
        darwin_analysis=json.dumps(analysis_obj, ensure_ascii=False),
        darwin_trace_id=darwin_trace_id,
    )
    _record_action(
        case_id,
        "ai-suggestion",
        {
            "suggestion_type": "darwin",
            "model_id": model_id,
            "darwin_trace_id": darwin_trace_id,
            "analysis_keys": list(analysis_obj.keys()),
        },
        before,
        before,
        "ai_expert",
    )
    persist_darwin_operation(
        trace_id=darwin_trace_id,
        badcase_id=case_id,
        model_call=model_call,
        operation_status="complete",
        started_at=operation_started_at,
        completed_at=now_cn(),
        drafts=[],
        status_before=before,
        status_after=before,
    )
    return {
        "badcase": _enrich_badcase(updated),
        "analysis": analysis_obj,
        "drafts": [],
        "status_changed": False,
        "model_id": model_id,
        "darwin_skill_found": bool(darwin),
        "darwin_trace_id": darwin_trace_id,
        "usage_source": model_call.get("usage_source") if model_call else "unavailable",
        "total_tokens": model_call.get("total_tokens") if model_call else None,
        "estimated_cost_cny": model_call.get("estimated_cost_cny") if model_call else None,
        "calculated_direct_cost": model_call.get("estimated_cost_cny") if model_call else None,
        "cost_source": model_call.get("cost_source") if model_call else None,
        "cost_disclaimer": "platform_price_snapshot_not_provider_final_bill",
        "budget_warning_code": budget_gate.get("warning_code"),
    }

    # Legacy draft generation is deliberately unreachable. Historical draft
    # records and endpoints remain readable, but AI analysis no longer creates
    # or applies any repair object.
    created_drafts: List[Dict[str, Any]] = []
    for draft in analysis_obj.get("drafts", []) or []:
        draft_type = draft.get("type")
        try:
            if draft_type == "knowledge":
                created = db_create_knowledge_draft(
                    badcase_id=case_id,
                    title=draft.get("title", "未命名知识草稿"),
                    content=draft.get("content", ""),
                    category=draft.get("target_doc_title") or "未分类",
                )
                created_drafts.append({"type": "knowledge", "draft": created})
            elif draft_type == "skill_prompt":
                existing = get_skill_by_name(draft.get("skill_name", ""))
                created = db_create_skill_prompt_draft(
                    badcase_id=case_id,
                    skill_id=existing.get("id") if existing else None,
                    skill_name=draft.get("skill_name", ""),
                    title=draft.get("title", "未命名 Skill 草稿"),
                    prompt_content=draft.get("prompt_content", ""),
                    trigger_keywords=draft.get("trigger_keywords", ""),
                )
                created_drafts.append({"type": "skill_prompt", "draft": created})
            elif draft_type == "capability_gap":
                created = db_create_capability_gap_draft(
                    badcase_id=case_id,
                    title=draft.get("title", "未命名能力缺口"),
                    description=draft.get("description", ""),
                    gap_type=draft.get("gap_type", "other"),
                    suggested_action=draft.get("suggested_action", ""),
                )
                created_drafts.append({"type": "capability_gap", "draft": created})
        except Exception as exc:
            logger.exception("failed to create draft from Darwin output")
            created_drafts.append({"type": draft_type, "error": str(exc)})

    # Fallback: ensure the classified category is represented as a draft so
    # the operations loop can proceed without faking a successful model output.
    has_knowledge = any(d.get("type") == "knowledge" for d in created_drafts)
    has_capability = any(d.get("type") == "capability_gap" for d in created_drafts)
    if case.get("category") == "knowledge_gap" and not has_knowledge:
        try:
            created = db_create_knowledge_draft(
                badcase_id=case_id,
                title=f"补充：{case.get('title', '知识库缺口')[:40]}",
                content=case.get("description", ""),
                category="未分类",
            )
            created_drafts.append({"type": "knowledge", "draft": created})
        except Exception:
            logger.exception("failed to create fallback knowledge draft")
    if case.get("category") == "mcp_capability" and not has_capability:
        try:
            created = db_create_capability_gap_draft(
                badcase_id=case_id,
                title="MCP/能力缺口草稿",
                description=analysis_obj.get("root_cause") or case.get("description", ""),
                gap_type="mcp_write",
                suggested_action="待产品评估后补充对应 MCP 写操作或系统集成能力，当前不可自动完成业务操作。",
            )
            created_drafts.append({"type": "capability_gap", "draft": created})
        except Exception:
            logger.exception("failed to create fallback capability gap draft")

    before = case["status"]
    new_status = "fixing"
    updated = db_update_badcase(
        case_id,
        root_cause=analysis_obj.get("root_cause_hypothesis", case.get("root_cause")),
        root_cause_domain=analysis_obj.get("root_cause_domain", case.get("root_cause_domain") or "unknown"),
        fix_plan=json.dumps(analysis_obj.get("suggested_actions", []), ensure_ascii=False),
        darwin_analysis=json.dumps(analysis_obj, ensure_ascii=False),
        darwin_trace_id=darwin_trace_id,
        status=new_status,
    )
    _record_action(
        case_id,
        "darwin-fix",
        {
            "model_id": model_id,
            "darwin_trace_id": darwin_trace_id,
            "drafts_created": len(created_drafts),
            "analysis_keys": list(analysis_obj.keys()),
        },
        before,
        new_status,
        "ai_expert",
    )
    persist_darwin_operation(
        trace_id=darwin_trace_id,
        badcase_id=case_id,
        model_call=model_call,
        operation_status="complete",
        started_at=operation_started_at,
        completed_at=now_cn(),
        drafts=created_drafts,
        status_before=before,
        status_after=new_status,
    )
    return {
        "badcase": _enrich_badcase(updated),
        "analysis": analysis_obj,
        "drafts": created_drafts,
        "model_id": model_id,
        "darwin_skill_found": bool(darwin),
        "darwin_trace_id": darwin_trace_id,
        "usage_source": model_call.get("usage_source") if model_call else "unavailable",
        "total_tokens": model_call.get("total_tokens") if model_call else None,
        "estimated_cost_cny": model_call.get("estimated_cost_cny") if model_call else None,
        "calculated_direct_cost": model_call.get("estimated_cost_cny") if model_call else None,
        "cost_source": model_call.get("cost_source") if model_call else None,
        "cost_disclaimer": "platform_price_snapshot_not_provider_final_bill",
    }


@router.post("/{case_id}/retry")
async def switch_model_retry_alias(case_id: int, request: SwitchModelRetryRequest = SwitchModelRetryRequest()):
    """Frontend alias for /switch-model-retry."""
    return await switch_model_retry(case_id, request)


@router.post("/{case_id}/switch-model-retry")
async def switch_model_retry(case_id: int, request: SwitchModelRetryRequest = SwitchModelRetryRequest()):
    """Retry the user message with an alternative model."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")

    user_message = request.user_message
    if not user_message and case.get("source_message_id"):
        msg = get_chat_message(case["source_message_id"])
        if msg:
            user_message = msg.get("content", "")
    if not user_message:
        user_message = case.get("title") or ""
        if case.get("description"):
            user_message = f"{user_message}\n{case['description']}".strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="user_message or source_message_id required")

    _enforce_background_budget("badcase_switch_model_retry")

    # Prefer explicit model_id; otherwise retry with the runtime default Flash.
    model_id = request.model_id or "deepseek-v4-flash"

    alt_model = build_model(model_id)
    prompt = (
        "你是YIAI物业物业客服助手。请专业、简洁地回答业主问题。"
        "当问题超出物业维修、收费或客服范围时，主动提出转人工。\n\n"
        f"业主问题：{user_message}"
    )
    retry_text, _ = await _llm_generate(
        prompt,
        model=alt_model,
        model_id=model_id,
        trace_id=f"badcase-retry-{case_id}-{uuid.uuid4().hex[:10]}",
        session_id=case.get("session_id") or f"badcase:{case_id}",
        stage="badcase_switch_model_retry",
        model_selection_reason="operator-triggered Badcase model retry",
        explicit_retry=True,
    )

    before = case["status"]
    _record_action(
        case_id,
        "ai-suggestion",
        {"suggestion_type": "model-retry", "model_id": model_id, "response": retry_text},
        before,
        before,
        "ai",
    )
    return {
        "badcase": _enrich_badcase(_load_case(case_id)),
        "model_id": model_id,
        "retry_response": retry_text,
        "status_changed": False,
    }


@router.post("/{case_id}/verify")
async def verify_badcase(case_id: int, request: VerifyRequest = VerifyRequest()):
    """Verify a fix using real retest evidence, then enter release observation.

    Passing verification is deliberately not final closure.  The demo makes the
    operator record a release/observation note before the case can be closed,
    rather than pretending it has production monitoring.
    """
    case = _load_case(case_id)

    if request.passed:
        _require_case_status(case, "verify-pass", {"verifying"})
        live_retest_trace = _validated_retest_trace(case)
        case = {**case, "retest_trace_live_verified": bool(live_retest_trace)}
        if not _has_post_apply_retest(case):
            raise HTTPException(
                status_code=400,
                detail="请先在当前修复应用后完成一次真实复测",
            )
        if not live_retest_trace:
            raise HTTPException(
                status_code=400,
                detail="复测 Trace 不存在、未完成或与本次复测会话不一致",
            )
        if not (request.verification_evidence or request.note).strip():
            raise HTTPException(status_code=400, detail="请记录原案例/同类或边界验证结论后再发布观察")
        new_status = "released"
        updated = db_update_badcase(
            case_id,
            status=new_status,
            verified_by="operator",
            release_note=(request.verification_evidence or request.note).strip(),
            released_at=now_cn(),
        )
    else:
        # A release-observation regression is a real signal to return to
        # fixing; it must not require a fake second verification state.
        _require_case_status(case, "verify-fail", {"verifying", "released"})
        if not request.note or not request.note.strip():
            raise HTTPException(status_code=400, detail="verification failure note required")
        new_status = "fixing"
        updated = db_update_badcase(case_id, status=new_status, fix_plan=request.note.strip() or "verification failed")

    _record_action(
        case_id,
        "verify",
        {"passed": request.passed, "note": request.note, "verification_evidence": request.verification_evidence},
        case["status"],
        new_status,
    )
    return {"badcase": _enrich_badcase(updated)}


@router.post("/{case_id}/transition")
async def transition_badcase(case_id: int, request: TransitionRequest = TransitionRequest()):
    """Manually transition a badcase to another valid state (state machine enforced)."""
    case = _load_case(case_id)
    if request.status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail=f"invalid status: {request.status}")
    if is_terminal_status(case["status"]) and case["status"] != request.status:
        raise HTTPException(status_code=400, detail="cannot transition out of terminal status")
    try:
        validate_status_transition(case["status"], request.status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    updated = db_update_badcase(case_id, status=request.status)
    _record_action(case_id, "transition", {"note": request.note}, case["status"], request.status, "user")
    return {"badcase": _enrich_badcase(updated)}


@router.post("/{case_id}/record-agent-config-apply")
async def record_agent_config_apply(case_id: int, request: AgentConfigApplyEvidenceRequest):
    """Record a human-reviewed Agent configuration apply as Badcase evidence.

    The Agent itself is still edited through the existing Agent management API.
    This endpoint verifies the live result and records the before/after evidence;
    it never calls a model and never changes Skill/RAG/MCP bindings.
    """
    case = _load_case(case_id)
    _require_case_status(case, "record-agent-config-apply", {"fixing"})
    agent = get_agent_by_agent_id(request.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="target agent not found")
    if (agent.get("description") or "") != request.after_description:
        raise HTTPException(status_code=409, detail="live agent description does not match reviewed after value")
    if (agent.get("instructions") or "") != request.after_instructions:
        raise HTTPException(status_code=409, detail="live agent instructions do not match reviewed after value")

    live_skill_ids = sorted(int(value) for value in get_agent_skills(request.agent_id))
    live_knowledge_ids = sorted(int(value) for value in (get_agent_knowledge_bindings(request.agent_id) or []))
    live_mcp_tools = sorted(
        str(item.get("tool_name")) for item in get_agent_tools(request.agent_id)
        if item.get("tool_name")
    )
    before_bindings = (
        sorted(request.skill_ids_before),
        sorted(request.knowledge_doc_ids_before),
        sorted(request.mcp_tools_before),
    )
    after_bindings = (
        sorted(request.skill_ids_after),
        sorted(request.knowledge_doc_ids_after),
        sorted(request.mcp_tools_after),
    )
    if before_bindings != after_bindings:
        raise HTTPException(status_code=409, detail="S7 routing fix must not change Skill/RAG/MCP bindings")
    if after_bindings != (live_skill_ids, live_knowledge_ids, live_mcp_tools):
        raise HTTPException(status_code=409, detail="live Agent bindings do not match reviewed evidence")
    if not request.review_note.strip():
        raise HTTPException(status_code=400, detail="human review note required")

    detail = {
        "target_type": "agent_route_config",
        "agent_id": request.agent_id,
        "agent_row_id": agent.get("id"),
        "before": {
            "description": request.before_description,
            "instructions": request.before_instructions,
        },
        "after": {
            "description": request.after_description,
            "instructions": request.after_instructions,
        },
        "bindings_unchanged": {
            "skill_ids": live_skill_ids,
            "knowledge_doc_ids": live_knowledge_ids,
            "mcp_tools": live_mcp_tools,
        },
        "human_reviewed": True,
        "review_note": request.review_note.strip(),
        "auto_applied_darwin_draft": False,
    }
    updated = _move_to_verifying_after_apply(
        case, case_id, "apply-agent-config", detail
    )
    return {"badcase": _enrich_badcase(updated), "evidence": detail}


@router.post("/{case_id}/record-runtime-release")
async def record_runtime_release(case_id: int, request: RuntimeReleaseEvidenceRequest):
    """Link the actually published RuntimeRelease to a verifying Badcase."""
    case = _load_case(case_id)
    _require_case_status(case, "record-runtime-release", {"verifying"})
    current = get_current_runtime_release()
    if not current:
        raise HTTPException(status_code=503, detail="no published RuntimeRelease")
    if current.get("release_id") != request.release_id or int(current.get("version") or 0) != request.version:
        raise HTTPException(status_code=409, detail="current RuntimeRelease does not match supplied evidence")
    actual_parent = current.get("parent_release_id")
    if request.parent_release_id is not None and actual_parent != request.parent_release_id:
        raise HTTPException(status_code=409, detail="RuntimeRelease parent does not match supplied evidence")
    detail = {
        "release_id": current.get("release_id"),
        "version": current.get("version"),
        "parent_release_id": actual_parent,
        "config_hash": current.get("config_hash"),
        "note": request.note.strip(),
        "effective_on": "new_session",
    }
    release_label = f"v{current.get('version')} / {current.get('release_id')}"
    updated = db_update_badcase(case_id, release_version=release_label)
    _record_action(case_id, "runtime-release", detail, case["status"], case["status"], "operator")
    return {"badcase": _enrich_badcase(updated), "release": detail}


@router.get("/{case_id}/actions")
async def list_actions(case_id: int):
    """List lifecycle actions for a badcase."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")
    actions = list_badcase_actions(case_id)
    return {"actions": actions, "count": len(actions)}


@router.post("/{case_id}/darwin-optimize")
async def darwin_optimize_alias(case_id: int, request: DarwinFixRequest = DarwinFixRequest()):
    """Alias for /darwin-fix to match test-case naming."""
    return await darwin_fix(case_id, request)


@router.post("/{case_id}/darwin")
async def darwin_alias_frontend(case_id: int, request: DarwinFixRequest = DarwinFixRequest()):
    """Frontend alias for /darwin-fix."""
    return await darwin_fix(case_id, request)


@router.post("/{case_id}/close")
async def close_badcase(case_id: int, request: CloseReleaseRequest = CloseReleaseRequest()):
    """Close an observed release; no claim of automatic online monitoring."""
    case = _load_case(case_id)
    _require_case_status(case, "close", {"released"})
    if not request.observation_note.strip():
        raise HTTPException(status_code=400, detail="请记录发布后的观察或人工确认结论")
    updated = db_update_badcase(
        case_id,
        status="closed",
        observed_at=now_cn(),
        release_note=request.observation_note.strip(),
    )
    _record_action(
        case_id, "close", {"observation_note": request.observation_note.strip()},
        case["status"], "closed", "operator",
    )
    return {"badcase": _enrich_badcase(updated)}


@router.post("/{case_id}/reject")
async def reject_badcase(case_id: int, request: RejectRequest = RejectRequest()):
    """Reject a badcase with a required reason (only from non-terminal states)."""
    case = _load_case(case_id)
    _require_case_status(case, "reject", {"pending", "classified", "investigating", "fixing", "verifying"})
    if not request.rejected_reason or not request.rejected_reason.strip():
        raise HTTPException(status_code=400, detail="rejected_reason required")

    new_status = "rejected"
    updated = db_update_badcase(case_id, status=new_status, rejected_reason=request.rejected_reason.strip())
    action_type = "mark-auto-false-positive" if request.review_result == "automatic_false_positive" else "reject"
    _record_action(
        case_id,
        action_type,
        {"reason": request.rejected_reason.strip(), "review_result": request.review_result},
        case["status"],
        new_status,
        "operator",
    )
    return {"badcase": _enrich_badcase(updated)}


@router.post("/{case_id}/system-observation")
async def mark_system_observation(case_id: int, request: SystemObservationRequest):
    """Keep a historical record but remove it from the current-problem view."""
    case = _load_case(case_id)
    if not request.reason.strip():
        raise HTTPException(status_code=400, detail="reason required")
    _record_action(
        case_id,
        "system-observation",
        {"reason": request.reason.strip()},
        case["status"],
        case["status"],
        "operator",
    )
    return {"badcase": _enrich_badcase(_load_case(case_id))}


@router.post("/{case_id}/accept-limitation")
async def accept_limitation(case_id: int, request: AcceptLimitationRequest):
    """Close a case as an explicit current product boundary, not a fake fix."""
    case = _load_case(case_id)
    _require_case_status(case, "accept-limitation", {"pending", "classified", "investigating", "fixing"})
    if not request.reason.strip():
        raise HTTPException(status_code=400, detail="accepted limitation reason required")
    detail = {
        "reason": request.reason.strip(),
        "alternative_path": request.alternative_path.strip(),
        "message": "该能力边界已记录；系统不会把未实现能力伪装成已修复。",
    }
    updated = db_update_badcase(
        case_id,
        status="accepted_limitation",
        accepted_limitation_reason=json.dumps(detail, ensure_ascii=False),
    )
    _record_action(case_id, "accept-limitation", detail, case["status"], "accepted_limitation", "operator")
    return {"badcase": _enrich_badcase(updated)}


@router.post("/{case_id}/duplicate")
async def mark_duplicate(case_id: int, request: DuplicateRequest):
    """Preserve duplicate evidence while linking to the canonical owner case."""
    case = _load_case(case_id)
    _require_case_status(case, "mark-duplicate", {"pending", "classified", "investigating", "fixing"})
    if request.primary_badcase_id == case_id:
        raise HTTPException(status_code=400, detail="primary_badcase_id cannot equal current case")
    primary = _load_case(request.primary_badcase_id)
    detail = {
        "primary_badcase_id": primary["id"],
        "primary_title": primary.get("title"),
        "note": request.note.strip(),
    }
    updated = db_update_badcase(case_id, status="duplicate", duplicate_of_id=primary["id"])
    _record_action(case_id, "mark-duplicate", detail, case["status"], "duplicate", "operator")
    return {"badcase": _enrich_badcase(updated)}


class RetestRuntimeError(RuntimeError):
    """Controlled retest failure carrying only non-sensitive runtime evidence."""

    def __init__(
        self,
        reason_code: str,
        *,
        answer: str = "",
        done: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.answer = answer
        self.done = dict(done or {})
        self.trace_id = str(trace_id or self.done.get("trace_id") or "").strip() or None


async def _consume_chat_stream(message: str, session_id: str, user_id: str = "retest") -> Dict[str, Any]:
    """Run a real chat stream and require one truthful successful terminal event."""
    # Lazy import to avoid circular dependency between routers.
    from app.chat import stream_chat_response

    final_answer = ""
    done_payload: Dict[str, Any] = {}
    done_received = False
    done_count = 0
    noncomplete_done_received = False
    error_event_received = False
    evidence_trace_id: Optional[str] = None
    event_name = ""
    async for chunk in stream_chat_response(message, session_id, user_id):
        for line in chunk.splitlines():
            if not line.strip():
                event_name = ""
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            try:
                payload = json.loads(data)
            except Exception:
                if event_name == "error":
                    error_event_received = True
                continue
            if not isinstance(payload, dict):
                if event_name == "error":
                    error_event_received = True
                continue

            payload_trace_id = str(payload.get("trace_id") or "").strip()
            if payload_trace_id:
                evidence_trace_id = payload_trace_id
            content = payload.get("content")
            if event_name == "delta" and content:
                final_answer += str(content)
            elif event_name in {"final", "done"}:
                terminal_answer = content or payload.get("answer")
                if terminal_answer:
                    final_answer = str(terminal_answer)
            if event_name == "error":
                error_event_received = True
            elif event_name == "done":
                done_received = True
                done_count += 1
                if payload.get("status") != "complete":
                    noncomplete_done_received = True
                done_payload = payload

    if error_event_received:
        raise RetestRuntimeError(
            "retest_runtime_error_event",
            answer=final_answer,
            done=done_payload,
            trace_id=evidence_trace_id,
        )
    if not done_received:
        raise RetestRuntimeError(
            "retest_done_missing",
            answer=final_answer,
            trace_id=evidence_trace_id,
        )
    if noncomplete_done_received or done_payload.get("status") != "complete":
        raise RetestRuntimeError(
            "retest_done_not_complete",
            answer=final_answer,
            done=done_payload,
            trace_id=evidence_trace_id,
        )
    if done_count != 1:
        raise RetestRuntimeError(
            "retest_multiple_done_events",
            answer=final_answer,
            done=done_payload,
            trace_id=evidence_trace_id,
        )
    if not str(done_payload.get("trace_id") or "").strip():
        raise RetestRuntimeError(
            "retest_trace_missing",
            answer=final_answer,
            done=done_payload,
            trace_id=evidence_trace_id,
        )
    if not final_answer.strip():
        raise RetestRuntimeError(
            "retest_answer_missing",
            done=done_payload,
            trace_id=evidence_trace_id,
        )
    return {"answer": final_answer, "done": done_payload}


@router.post("/{case_id}/retest")
async def retest_badcase(case_id: int, request: SwitchModelRetryRequest = SwitchModelRetryRequest()):
    """Retest the badcase user message through the real chat runtime.

    Allowed from both `fixing` (pre-apply diagnosis) and `verifying` (post-apply
    validation). The status does not change; a post-apply retest keeps the case
    in `verifying` so the operator can proceed to verify-pass.
    """
    case = _load_case(case_id)
    _require_case_status(case, "retest", {"fixing", "verifying"})

    user_message = request.user_message or case.get("original_query")
    if not user_message and case.get("source_message_id"):
        msg = get_chat_message(case["source_message_id"])
        if msg:
            user_message = msg.get("content", "")
    if not user_message:
        user_message = case.get("title") or ""
        if case.get("description"):
            user_message = f"{user_message}\n{case['description']}".strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="user_message or original_query required")

    _enforce_background_budget("badcase_retest")

    retest_session_id = f"retest-{uuid.uuid4().hex[:12]}"
    retest_started_at = now_cn()
    case_anchor = (case.get("status"), case.get("last_applied_at"))
    try:
        result = await _consume_chat_stream(user_message, retest_session_id, user_id="retest")
        answer = str(result.get("answer") or "").strip()
        done = result.get("done") or {}
        retest_trace_id = str(done.get("trace_id") or "").strip()
        trace = get_chat_trace(retest_trace_id) if retest_trace_id else None
        if not trace:
            raise RetestRuntimeError(
                "retest_trace_not_persisted",
                answer=answer,
                done=done,
                trace_id=retest_trace_id,
            )
        if trace.get("status") != "complete":
            raise RetestRuntimeError(
                "retest_trace_not_complete",
                answer=answer,
                done=done,
                trace_id=retest_trace_id,
            )
        if trace.get("session_id") != retest_session_id:
            raise RetestRuntimeError(
                "retest_trace_session_mismatch",
                answer=answer,
                done=done,
                trace_id=retest_trace_id,
            )
        current_case = _load_case(case_id)
        current_anchor = (
            current_case.get("status"),
            current_case.get("last_applied_at"),
        )
        if current_anchor != case_anchor:
            raise RetestRuntimeError(
                "retest_case_changed_during_run",
                answer=answer,
                done=done,
                trace_id=retest_trace_id,
            )
        case = current_case
    except RetestRuntimeError as exc:
        logger.warning("retest rejected: %s", exc.reason_code)
        current_case = db_get_badcase(case_id) or case
        current_status = current_case.get("status") or case["status"]
        _record_action(
            case_id,
            "retest",
            {
                "retest_session_id": retest_session_id,
                "retest_trace_id": exc.trace_id,
                "result": "failed",
                "reason_code": exc.reason_code,
                "run_status": exc.done.get("status") or "failed",
            },
            current_status,
            current_status,
        )
        raise HTTPException(
            status_code=502,
            detail=f"retest failed ({exc.reason_code}); see retained Trace evidence",
        )
    except Exception as e:
        logger.exception("retest chat stream failed")
        # Keep the case in fixing on retest error.
        _record_action(
            case_id,
            "retest",
            {
                "retest_session_id": retest_session_id,
                "error": f"{type(e).__name__}: retest runtime failed",
            },
            case["status"],
            case["status"],
        )
        raise HTTPException(status_code=502, detail="retest failed; see retained Trace evidence")

    token_detail = done.get("token_detail") or {}
    model_id = done.get("model_id") or MODEL_ID
    total_tokens = token_detail.get("total_tokens") or done.get("token_count")

    # Compute retest cost and price snapshot before building the context.
    usage = {
        "input_tokens": token_detail.get("input_tokens") if token_detail else None,
        "output_tokens": token_detail.get("output_tokens") if token_detail else None,
        "reasoning_tokens": token_detail.get("reasoning_tokens") if token_detail else None,
        "cached_tokens": token_detail.get("cached_tokens") if token_detail else None,
        "total_tokens": total_tokens,
    }
    retest_cost: Optional[float] = None
    retest_price: Optional[Dict[str, Any]] = None
    try:
        retest_cost, retest_price = _calculate_cost(model_id, usage)
    except Exception:
        pass

    retest_context = {
        "record_kind": "logical_aggregate",
        "include_in_provider_aggregate": False,
        "provider_evidence_source": "referenced_underlying_trace_attempts",
        "run_status": "complete",
        "retest_started_at": retest_started_at,
        "session_id": retest_session_id,
        "trace_persisted": True,
        "trace_status": trace.get("status"),
        "trace_session_id": trace.get("session_id"),
        "route_intent": done.get("route_intent"),
        "current_agent": done.get("current_agent"),
        "activated_skills": done.get("activated_skills"),
        "rag_citations": done.get("citations"),
        "tool_calls": done.get("tool_calls"),
        "mcp_tool_calls": done.get("mcp_calls"),
        "model_id": model_id,
        "trace_id": retest_trace_id,
        "token_count": done.get("token_count"),
        "total_tokens": total_tokens,
        "token_detail": token_detail,
        "usage_source": done.get("usage_source"),
        "estimated_cost_cny": retest_cost,
        "price_snapshot": retest_price,
        "auto_badcase_id": done.get("auto_badcase_id"),
    }

    before = case["status"]
    # Retest does not change the status. In fixing it is a pre-apply diagnosis;
    # in verifying it is the post-apply validation required before verify-pass.
    new_status = before
    retest_at = now_cn()
    updated = db_update_badcase(
        case_id,
        retest_response=answer,
        retest_context_json=json.dumps(retest_context, ensure_ascii=False, default=str),
        retest_trace_id=retest_trace_id,
        last_retest_at=retest_at,
    )
    _record_action(
        case_id,
        "retest",
        {
            "retest_session_id": retest_session_id,
            "retest_trace_id": retest_trace_id,
            "model_id": model_id,
            "answer_preview": answer[:200],
        },
        before,
        new_status,
    )
    return {
        "badcase": _enrich_badcase(updated),
        "retest_response": answer,
        "retest_context": retest_context,
    }


@router.post("/{case_id}/check-tools")
async def check_tools_badcase(case_id: int):
    """Analyze whether the badcase is caused by missing or misconfigured tools."""
    case = _load_case(case_id)
    _require_case_status(case, "check-tools", {"pending", "classified"})

    _enforce_background_budget("badcase_check_tools")

    from db.property_db import list_skills, list_mcp_servers

    enabled_skills = [s for s in list_skills() if s.get("enabled")]
    enabled_servers = [s for s in list_mcp_servers() if s.get("enabled")]
    skill_names = [s.get("name", "") for s in enabled_skills]
    tool_descriptions = []
    for server in enabled_servers:
        for tool in server.get("tools", []):
            tool_descriptions.append(f"- {server.get('name', '')}:{tool.get('name', '')} ({tool.get('description', '')})")

    prompt = (
        "你是一名 AI 工具配置审计专家。请根据以下 Badcase 信息，分析该问题是否由工具/Skill 缺失或配置错误导致。"
        "如果可能，请指出应该启用哪个 Skill 或 MCP 工具，并给出具体建议。\n\n"
        f"标题：{case['title']}\n"
        f"描述：{case.get('description', '')}\n"
        f"证据：{case.get('evidence', '')}\n\n"
        f"当前已启用 Skills：{', '.join(skill_names)}\n"
        f"当前已启用 MCP 工具：\n{chr(10).join(tool_descriptions)}\n\n"
        "请直接输出分析结论与建议，不要添加解释。"
    )
    analysis, _ = await _llm_generate(
        prompt,
        trace_id=f"badcase-tools-{case_id}-{uuid.uuid4().hex[:10]}",
        session_id=case.get("session_id") or f"badcase:{case_id}",
        stage="badcase_check_tools",
        model_selection_reason="AI suggestion for operator-reviewed tool diagnosis",
    )

    before = case["status"]
    _record_action(
        case_id,
        "ai-suggestion",
        {"suggestion_type": "tool-check", "analysis": analysis},
        before,
        before,
        "ai",
    )
    return {
        "badcase": _enrich_badcase(_load_case(case_id)),
        "analysis": analysis,
        "status_changed": False,
    }
