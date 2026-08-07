"""V1.8 RuntimeCoordinator: one authority over the three runtime paths."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from app.runtime.agent_factory import build_agent_from_snapshot, vertical_agent_cards
from app.runtime.badcase_capture import capture_runtime_badcase
from app.runtime.citation_renderer import (
    build_skill_evidence,
    build_evidence_set,
    prompt_evidence_allowlist,
    render_citations,
)
from app.runtime.contracts import (
    ActionProposal,
    ActionReceipt,
    AnswerContract,
    ApprovalEvent,
    CapabilityDecision,
    HandoffExecutionContract,
    HandoffKind,
    LaneDecision,
    LaneDecisionSource,
    ResponseMode,
    RiskLevel,
    RouteDecision,
    RunState,
    RunStatus,
    RuntimeLane,
    RuntimePath,
    ToolEffect,
    ToolInvocation,
    content_hash,
)
from app.runtime.cost_ledger import build_cost_entry
from app.runtime.evidence_ledger import EvidenceLedger
from app.runtime.mcp_executor import (
    build_model_native_read_tools,
    preinvoke_read_tools,
)
from app.runtime.snapshot_resolver import resolve_snapshot
from app.runtime.tool_planner import plan_tools, unique_write_plan
from app.runtime.provider_accounting import merge_non_null, provider_accounting_scope
from app.runtime.provider_evidence import provider_evidence_from_run
from app.settings import MODEL_ID, USE_THINKING, build_model
from app.work_order_workflow import (
    _is_draft_follow_up,
    action_gateway,
    advance_work_order_workflow,
    is_cancel_request,
    is_confirmation,
    is_explicit_work_order_request,
)
from db.property_db import (
    create_chat_trace,
    ensure_chat_session,
    get_chat_session,
    get_action_proposal,
    get_action_receipt_by_idempotency_key,
    get_latest_action_proposal,
    get_pending_action_proposal,
    get_work_order_draft,
    list_chat_messages,
    list_action_approvals,
    now_cn,
    record_mcp_call_audit,
    record_trace_event,
    request_handoff,
    resume_handoff_after_owner_message,
    save_chat_message,
    update_chat_trace,
)


PROVIDER_FAILURE_MARKERS = (
    "insufficient balance",
    "insufficient_quota",
    "quota exceeded",
    "rate limit exceeded",
    "invalid api key",
    "incorrect api key",
    "authentication failed",
    "authentication fails",
)
PROVIDER_FAILURE_PUBLIC_MESSAGE = "模型服务暂时不可用，请稍后重试或检查模型账户状态。"
RUNTIME_FAILURE_PUBLIC_MESSAGE = "系统暂时无法完成本次请求，请稍后重试；如问题持续，可选择转人工核实。"
KNOWLEDGE_INSUFFICIENT_RESPONSE = (
    "当前知识依据不足，无法确认具体结论。我不能将“未检索到”表述为"
    "“文档没有规定”，也不会补充知识库之外的行业经验、时间范围或处理步骤。"
    "你可以补充问题，或选择转人工核实。"
)
NON_PROPERTY_SAFETY_RESPONSE = (
    "这超出物业 AI 的专业处理范围。请先确保自身安全；如存在人身危险或"
    "正在发生的违法暴力行为，请立即联系 110、120 等当地紧急服务或身边"
    "可信人员。我不会把这类问题表述成物业已经接管。"
)
OUT_OF_SCOPE_RESPONSE = (
    "这个问题不属于当前物业服务能力范围，且当前没有匹配的非物业处理角色。"
    "我可以继续协助物业报修、服务咨询或工单查询。"
)
PROPERTY_AGENT_UNAVAILABLE_RESPONSE = (
    "我能确认这是物业相关问题，但当前没有可靠选中同域处理角色。"
    "我不会跨域调用其他Agent，也不会编造物业流程；请补充具体服务对象或诉求后再试。"
)


def build_lane_agent_unavailable_decision(*, property_lane: bool) -> CapabilityDecision:
    """Return the canonical capability state when no same-domain Agent is selected.

    Skill/RAG/Tool may be skipped. Write and Handoff deliberately use their
    own contract vocabulary instead of reusing the generic ``skipped`` state.
    """

    return CapabilityDecision(
        selected_agent_id=None,
        skill={"status": "skipped", "reason_code": "no_lane_agent"},
        rag={"status": "skipped", "reason_code": "no_lane_agent"},
        tool={"status": "skipped", "reason_code": "no_lane_agent"},
        write={
            "status": "not_required",
            "reason_code": "no_lane_agent" if property_lane else "isolated_general",
        },
        handoff={
            "status": "available" if property_lane else "not_required",
            "reason_code": "owner_can_request" if property_lane else "isolated_general",
        },
    )


class ProviderFailureError(RuntimeError):
    """A Provider failure returned as model text instead of an exception."""


class InternalControlPayloadLeakError(RuntimeError):
    """A control-plane JSON payload reached the user-answer boundary."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _sse(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {_json(payload)}\n\n"


async def _static_response_stream(content: str) -> AsyncIterator[Any]:
    """Yield one deterministic answer without invoking a model Provider."""
    yield SimpleNamespace(content=content, event="RunContent", metrics={})


def _provider_failure_reason(text: str) -> Optional[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not normalized or len(normalized) > 800:
        return None
    for marker in PROVIDER_FAILURE_MARKERS:
        if marker in normalized:
            return marker
    return None


def _provider_failure_prefix(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not normalized:
        return False
    return any(marker.startswith(normalized) for marker in PROVIDER_FAILURE_MARKERS)


def _is_structured_realtime_query(
    answer_contract: AnswerContract,
    read_tool_plans: List[Any],
) -> bool:
    """A realtime read is declared by the semantic contract, not message words."""

    return (
        answer_contract.response_mode == ResponseMode.REALTIME_READ
        and bool(read_tool_plans)
    )


def _decision(status: str, reason: str, **details: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": status, "reason": reason}
    payload.update({key: value for key, value in details.items() if value is not None})
    return payload


def _extract_tool_calls(value: Any) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    candidate = getattr(value, "run_response", None) or value
    raw_calls = getattr(candidate, "tool_calls", None) or getattr(candidate, "tools", None) or []
    for raw in raw_calls:
        if isinstance(raw, dict):
            name = raw.get("tool_name") or raw.get("name") or raw.get("tool") or ""
            arguments = raw.get("arguments") or raw.get("args") or {}
        else:
            name = getattr(raw, "tool", None) or getattr(raw, "name", None) or ""
            arguments = getattr(raw, "arguments", None) or getattr(raw, "args", None) or {}
        if hasattr(arguments, "model_dump"):
            arguments = arguments.model_dump()
        elif not isinstance(arguments, dict):
            arguments = {"value": str(arguments)}
        item = {"tool_name": str(name), "arguments": arguments}
        if item not in calls:
            calls.append(item)
    return calls


def _metrics_dict(value: Any) -> Dict[str, Optional[int]]:
    metrics = getattr(value, "metrics", None)
    if not metrics:
        return {}
    result: Dict[str, Optional[int]] = {}
    for source, target in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("reasoning_tokens", "reasoning_tokens"),
        ("cached_tokens", "cached_tokens"),
        ("cached_input_tokens", "cached_input_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        raw = (
            metrics.get(source)
            if isinstance(metrics, dict)
            else getattr(metrics, source, None)
        )
        if raw is not None:
            try:
                result[target] = int(raw)
            except (TypeError, ValueError):
                pass
    return result


def _estimate_tokens(text: str) -> Optional[int]:
    if not text:
        return None
    # Deliberately labelled local estimate; never used as provider usage or
    # multiplied by a price to fabricate an actual amount.
    return max(1, len(text) // 4)


def _aggregate_cost_field(entries: List[Any], field: str) -> Optional[int]:
    """Sum one Provider-usage field only when every request reported it."""

    if not entries:
        return None
    values = [getattr(item, field, None) for item in entries]
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def _aggregate_usage_source(sources: List[str], *, model_invoked: bool) -> str:
    if not model_invoked:
        return "not_applicable"
    if not sources or "unavailable" in sources:
        return "unavailable"
    if len(set(sources)) == 1:
        return sources[0]
    return "mixed"


def _claims_business_success(text: str) -> bool:
    normalized = text or ""
    return bool(
        re.search(
            r"(?:已|已经|正式).{0,6}(?:创建|提交|写入|更新|操作).{0,8}(?:成功|完成)|"
            r"(?:创建|提交|写入|更新|操作).{0,8}(?:成功|已完成)|"
            r"(?:资源\s*ID|工单号)[：:]\s*(?:WO|TICKET|ORDER)[-_][A-Za-z0-9_-]+",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _lane_candidates(
    cards: List[Dict[str, Any]], lane: RuntimeLane
) -> List[Dict[str, Any]]:
    if lane == RuntimeLane.SAFETY_HANDOFF:
        return []
    scope = "property" if lane == RuntimeLane.PROPERTY_GOVERNED else "isolated_general"
    return [card for card in cards if _agent_domain_scope(card) == scope]


def _visible_chat_history(
    session_id: str,
    *,
    current_trace_id: Optional[str] = None,
    rounds: int = 5,
) -> List[Dict[str, str]]:
    """Return only successful user-visible messages; control-plane runs are excluded."""

    visible: List[Dict[str, str]] = []
    for item in list_chat_messages(session_id):
        role = str(item.get("role") or "").lower()
        status = str(item.get("status") or "success").lower()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "owner", "assistant"} or not content:
            continue
        if status not in {"success", "complete", "completed"}:
            continue
        if current_trace_id and str(item.get("trace_id") or "") == current_trace_id:
            continue
        if _is_internal_control_payload(content):
            continue
        visible.append(
            {"role": "user" if role in {"user", "owner"} else "assistant", "content": content}
        )
    return visible[-max(0, int(rounds)) * 2 :]


def _render_visible_history_context(
    history: List[Dict[str, str]],
    current_message: str,
    *,
    boundary: str,
) -> str:
    lines = [f"[执行边界] {boundary}"]
    if history:
        lines.append("[最近成功可见对话]")
        for item in history:
            label = "用户" if item.get("role") == "user" else "助手"
            lines.append(f"{label}：{item.get('content', '')}")
    lines.extend(["[本轮用户问题]", current_message])
    return "\n".join(lines)


def _is_internal_control_payload(text: str) -> bool:
    """Reject a whole response that is Router/Lane control JSON, not user prose."""

    raw = str(text or "").strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw).strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        return False
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    control_keys = {
        "target_agent_id",
        "selected_agent_id",
        "lane",
        "reason_code",
        "business_intent",
        "allowed_domain",
        "response_mode",
        "evidence_required",
    }
    return bool(control_keys.intersection(payload))


def _lane_explanation(decision: LaneDecision) -> str:
    labels = {
        RuntimeLane.SAFETY_HANDOFF: "人工协同",
        RuntimeLane.PROPERTY_GOVERNED: "物业受控回答",
        RuntimeLane.ISOLATED_GENERAL: "隔离通用回答",
    }
    return (
        f"进入{labels[decision.lane]}路径；识别任务为"
        f"{decision.business_intent or '未命名意图'}。{decision.reason or 'Router未返回判断理由。'}"
    )


def _agent_domain_scope(agent: Optional[Dict[str, Any]]) -> str:
    value = str((agent or {}).get("domain_scope") or "property")
    return value if value in {"property", "isolated_general"} else "property"


def _knowledge_evidence_decision(
    answer_contract: AnswerContract,
    evidence_count: int,
    structured_realtime_query: bool,
    allowed_document_ids: Optional[set[int]] = None,
    *,
    domain_scope: str = "property",
    skill_evidence_count: int = 0,
    tool_evidence_count: int = 0,
) -> Dict[str, Any]:
    """Return the lightweight evidence gate decision used by runtime and Trace."""

    count = max(0, int(evidence_count or 0))
    skill_count = max(0, int(skill_evidence_count or 0))
    tool_count = max(0, int(tool_evidence_count or 0))
    accepted_count = count + skill_count + tool_count
    required = bool(answer_contract.evidence_required)
    if answer_contract.response_mode == ResponseMode.REALTIME_READ:
        blocked = bool(required and tool_count == 0)
    else:
        blocked = bool(required and accepted_count == 0)
    return {
        "required": required,
        "blocked": blocked,
        "evidence_count": count,
        "skill_evidence_count": skill_count,
        "tool_evidence_count": tool_count,
        "accepted_evidence_count": accepted_count,
        "domain_scope": domain_scope,
        "allowed_document_ids": sorted(allowed_document_ids or set()),
        "evidence_decision": (
            "rejected_insufficient"
            if blocked
            else ("accepted" if accepted_count else "not_required")
        ),
        "reason": (
            "no_accepted_evidence"
            if blocked
            else (
                "accepted_direct_evidence"
                if accepted_count
                else "knowledge_not_required"
            )
        ),
        "model_invoked": not blocked,
    }


def _requires_rag_citation(
    answer_contract: AnswerContract,
    *,
    evidence_count: int,
    linked_skill_evidence_count: int,
    successful_tool_evidence_count: int,
) -> bool:
    """Require a RAG citation only when RAG is the remaining answer evidence.

    A successful read Tool is first-class evidence for a realtime business
    result. Incidental retrieval candidates must not force a Tool-only answer
    to cite an unrelated knowledge chunk. Pure RAG answers remain fail-closed.
    """

    return bool(
        int(evidence_count or 0) > 0
        and int(linked_skill_evidence_count or 0) == 0
        and int(successful_tool_evidence_count or 0) == 0
        and answer_contract.response_mode == ResponseMode.GROUNDED_ANSWER
    )


def _effective_lane_decision(
    decision: LaneDecision,
    *,
    handoff_status: str = "none",
    handoff_reason_code: str = "",
    handoff_queue: str = "",
) -> Tuple[LaneDecision, Optional[str]]:
    """Enforce the product invariant before Lane SSE or evidence persistence.

    The Router remains the only semantic classifier. This function consumes
    only its structured result (plus an already-persisted Handoff state); it
    never inspects user text. The optional return value preserves the Router's
    reported lane for audit when normalization was required.
    """

    active_handoff = str(handoff_status or "none") in {
        "requested",
        "active",
        "waiting_user",
    }
    persisted_safety = active_handoff and (
        str(handoff_reason_code or "").strip() == "safety_risk"
        or str(handoff_queue or "").strip() == "emergency"
    )
    user_requested = (
        str(decision.business_intent or "").strip()
        == "user_requested_handoff"
    )
    if decision.lane == RuntimeLane.SAFETY_HANDOFF:
        if persisted_safety and user_requested:
            return (
                LaneDecision(
                    lane=RuntimeLane.SAFETY_HANDOFF,
                    business_intent="safety_risk",
                    reason="会话已有现实安全风险人工协同，本轮继续由紧急队列处理。",
                    decision_source=decision.decision_source,
                ),
                decision.lane.value,
            )
        return decision, None
    if not (user_requested or active_handoff):
        return decision, None
    effective_intent = (
        "safety_risk" if persisted_safety else "user_requested_handoff"
    )
    return (
        LaneDecision(
            lane=RuntimeLane.SAFETY_HANDOFF,
            business_intent=effective_intent,
            reason=(
                (
                    "会话已有现实安全风险人工协同，本轮继续由紧急队列处理。"
                    if persisted_safety
                    else "会话已有普通人工协同，本轮继续由工作人员处理。"
                )
                if active_handoff and not user_requested
                else (
                    decision.reason
                    or "本轮已确认需要由工作人员接手。"
                )
            ),
            decision_source=decision.decision_source,
        ),
        decision.lane.value,
    )


def _handoff_contract_for(
    decision: LaneDecision,
) -> HandoffExecutionContract:
    if decision.lane != RuntimeLane.SAFETY_HANDOFF:
        raise ValueError("Handoff execution contract requires effective A lane")
    if str(decision.business_intent or "").strip() == "user_requested_handoff":
        return HandoffExecutionContract(
            kind=HandoffKind.USER_REQUESTED,
            reason_code="user_requested",
            queue="property_service",
            safety_override=False,
            response_mode=ResponseMode.HUMAN_HANDOFF,
        )
    return HandoffExecutionContract(
        kind=HandoffKind.SAFETY_RISK,
        reason_code="safety_risk",
        queue="emergency",
        safety_override=True,
        response_mode=ResponseMode.EMERGENCY_HANDOFF,
    )


def _answer_contract_for(
    decision: LaneDecision,
    runtime_path: RuntimePath = RuntimePath.CONSULTATION,
) -> AnswerContract:
    """Compile fixed downstream permissions from A/B/C and workflow state."""

    common_forbidden = [
        "unsupported_property_fact",
        "unverified_execution_success",
        "internal_control_payload",
    ]
    if decision.lane == RuntimeLane.SAFETY_HANDOFF:
        handoff_contract = _handoff_contract_for(decision)
        return AnswerContract(
            response_mode=handoff_contract.response_mode,
            evidence_required=False,
            skill_policy="skipped",
            rag_policy="skipped",
            tool_policy="skipped",
            write_policy="forbidden",
            handoff_policy="required",
            forbidden_claims=common_forbidden,
            decision_reason=(
                "业主明确要求工作人员接手，立即发起普通人工协同。"
                if handoff_contract.kind == HandoffKind.USER_REQUESTED
                else "现实安全风险优先，语义判断后立即发起紧急人工协同。"
            ),
        )
    if decision.lane == RuntimeLane.ISOLATED_GENERAL:
        return AnswerContract(
            response_mode=ResponseMode.SAFE_GENERAL,
            evidence_required=False,
            skill_policy="skipped",
            rag_policy="skipped",
            tool_policy="skipped",
            write_policy="forbidden",
            handoff_policy="skipped",
            forbidden_claims=common_forbidden
            + ["property_official_fact", "harmful_instructions"],
            decision_reason="C只允许隔离通用回答；物业能力全部跳过，危险或越权内容由回答安全边界拒绝。",
        )
    if runtime_path == RuntimePath.CONTROLLED_ACTION:
        return AnswerContract(
            response_mode=ResponseMode.CONTROLLED_WRITE,
            evidence_required=True,
            evidence_requirements=["proposal", "permission", "owner_confirmation", "receipt"],
            skill_policy="skipped",
            rag_policy="skipped",
            tool_policy="selected",
            write_policy="allowed_after_confirmation",
            handoff_policy="optional",
            forbidden_claims=common_forbidden + ["write_success_without_receipt"],
            decision_reason="已存在的受控业务状态继续原流程，并保持Proposal、权限、确认、ActionGateway和Receipt边界。",
        )
    return AnswerContract(
        response_mode=ResponseMode.GROUNDED_ANSWER,
        evidence_required=True,
        evidence_requirements=["activated_skill_or_adopted_rag_or_successful_tool"],
        skill_policy="selected",
        rag_policy="selected",
        tool_policy="selected",
        write_policy="forbidden",
        handoff_policy="optional",
        forbidden_claims=common_forbidden
        + [
            "property_process_or_timing_without_evidence",
            "realtime_value_without_successful_tool",
        ],
        decision_reason="B只允许物业Agent；物业事实依赖Skill/RAG，实时状态依赖成功Tool，写入另走受控流程。",
    )


def _append_runtime_evidence_summary(
    answer: str,
    message: str,
    tool_calls: List[Dict[str, Any]],
    tool_invocations: List[ToolInvocation],
    citations: List[Any],
) -> str:
    """Append the requested one-line summary from runtime evidence, not prose."""

    compact = re.sub(r"\s+", "", message or "")
    if not any(
        marker in compact
        for marker in ("一行汇总", "一行总结", "汇总实际调用", "汇总调用的工具")
    ):
        return answer

    tool_names: List[str] = []
    if any(
        call.get("tool_name") == "get_skill_instructions"
        for call in tool_calls
    ):
        tool_names.append("get_skill_instructions")
    for invocation in tool_invocations:
        name = f"{invocation.server_name}/{invocation.tool_name}"
        if name not in tool_names:
            tool_names.append(name)

    document_titles: List[str] = []
    for citation in citations:
        title = str(getattr(citation, "title", "") or "").strip()
        if title and title not in document_titles:
            document_titles.append(title)

    footer = (
        "运行时核验汇总：本次实际调用工具："
        + ("、".join(tool_names) if tool_names else "无")
        + "；实际引用知识库文档："
        + ("、".join(document_titles) if document_titles else "无")
        + "。"
    )
    return (answer or "").rstrip() + "\n\n" + footer


def _action_receipt_from_payload(payload: Dict[str, Any]) -> ActionReceipt:
    """Project a database/API row onto the immutable public Receipt contract."""

    field_names = set(getattr(ActionReceipt, "model_fields", {}).keys())
    projected = {
        key: value
        for key, value in (payload or {}).items()
        if key in field_names
    }
    return ActionReceipt.model_validate(projected)


def _records_new_mcp_invocation(action_type: str, phase: Any) -> bool:
    """A Receipt replay is evidence reuse, not another MCP invocation."""

    return (
        str(action_type or "").startswith("mcp.")
        and str(phase or "") != "idempotent_replay"
    )


def _record_citation_violations(
    ledger: EvidenceLedger,
    violations: List[Dict[str, Any]],
) -> None:
    for violation in violations:
        code = str(violation.get("code") or "citation_violation")
        metadata = {
            key: value
            for key, value in violation.items()
            if key not in {"code", "detail"}
        }
        ledger.violation(
            code,
            str(
                violation.get("detail")
                or "Model citation was not present in the immutable EvidenceSet."
            ),
            **metadata,
        )


def _price_for_snapshot(snapshot_config: Dict[str, Any], model_id: str) -> Optional[Dict[str, Any]]:
    candidates = [
        item
        for item in snapshot_config.get("price_snapshots") or []
        if item.get("model_id") == model_id and item.get("enabled", True)
    ]
    if candidates:
        candidates.sort(key=lambda item: str(item.get("effective_date") or ""), reverse=True)
        return candidates[0]
    return None


def _model_config_for_snapshot(
    snapshot_config: Dict[str, Any],
    model_id: str,
) -> Dict[str, Any]:
    policy = snapshot_config.get("model_policy") or {}
    for item in [policy.get("default"), *(policy.get("available") or [])]:
        if isinstance(item, dict) and item.get("model_id") == model_id:
            return item
    return {}


def _build_model_from_snapshot(
    snapshot_config: Dict[str, Any],
    model_id: str,
) -> Any:
    config = _model_config_for_snapshot(snapshot_config, model_id)
    params = config.get("model_params") or {}
    overrides: Dict[str, Any] = {}
    if config.get("base_url"):
        overrides["base_url"] = config["base_url"]
    if "use_thinking" in params:
        overrides["use_thinking"] = bool(params["use_thinking"])
    return build_model(model_id, **overrides)


def _model_provider(snapshot_config: Dict[str, Any], model_id: str) -> str:
    return str(
        _model_config_for_snapshot(snapshot_config, model_id).get("provider")
        or "unknown"
    )


def _thinking_for_snapshot(snapshot_config: Dict[str, Any], model_id: str) -> bool:
    params = _model_config_for_snapshot(snapshot_config, model_id).get("model_params") or {}
    return bool(params.get("use_thinking", USE_THINKING))


def _results_from_snapshot(
    query: str,
    live_results: List[Dict[str, Any]],
    knowledge_versions: Dict[int, Dict[str, Any]],
    allowed_document_ids: set[int],
    top_k: int,
    context_threshold: float = 0.2,
) -> Tuple[List[Dict[str, Any]], bool]:
    import rag_retrieval

    published_chunks: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for doc_id in allowed_document_ids:
        document = knowledge_versions.get(doc_id) or {}
        for chunk in document.get("chunk_snapshots") or []:
            published_chunks[(doc_id, int(chunk.get("chunk_index") or 0))] = {
                **chunk,
                "document": document,
            }

    verified: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int]] = set()
    for result in live_results:
        try:
            doc_id = int(result.get("doc_id", result.get("document_id")))
            chunk_index = int(result.get("chunk_index") or 0)
        except (TypeError, ValueError):
            continue
        snapshot_chunk = published_chunks.get((doc_id, chunk_index))
        content = str(result.get("content") or result.get("chunk_text") or "")
        if (
            not snapshot_chunk
            or content_hash(content) != snapshot_chunk.get("chunk_hash")
        ):
            continue
        document = snapshot_chunk["document"]
        if rag_retrieval._is_structural_chunk(
            snapshot_chunk.get("content") or content,
            document.get("title") or "",
        ):
            continue
        verified.append(
            {
                **result,
                "doc_id": doc_id,
                "doc_title": document.get("title") or result.get("doc_title"),
                "content": snapshot_chunk.get("content") or content,
                "chunk_hash": snapshot_chunk.get("chunk_hash"),
                "document_hash": document.get("document_hash"),
                "document_version": document.get("document_version"),
            }
        )
        seen.add((doc_id, chunk_index))
    if verified:
        return verified[:top_k], False

    fallback: List[Dict[str, Any]] = []
    for (doc_id, chunk_index), snapshot_chunk in published_chunks.items():
        document = snapshot_chunk["document"]
        content = str(snapshot_chunk.get("content") or "")
        if rag_retrieval._is_structural_chunk(
            content,
            document.get("title") or "",
        ):
            continue
        query_values = rag_retrieval._required_evidence_values(query)
        if query_values and not query_values.issubset(
            rag_retrieval._critical_values(content)
        ):
            continue
        context_score = rag_retrieval._context_relevance_score(query, content)
        if context_score < context_threshold:
            continue
        fallback.append(
            {
                "doc_id": doc_id,
                "doc_title": document.get("title") or "",
                "chunk_index": chunk_index,
                "content": content,
                "chunk_hash": snapshot_chunk.get("chunk_hash"),
                "document_hash": document.get("document_hash"),
                "document_version": document.get("document_version"),
                "score": round(context_score, 6),
                "context_score": round(context_score, 6),
                "evidence_status": "accepted",
                "evidence_reason": "accepted_snapshot_lexical",
                "retrieval_sources": ["runtime_release_snapshot_lexical"],
            }
        )
    fallback.sort(key=lambda item: float(item.get("context_score") or 0), reverse=True)
    return fallback[:top_k], True


class RuntimeCoordinator:
    """Resolve, authorize and record every owner chat run."""

    async def stream(
        self,
        message: str,
        session_id: str,
        user_id: str,
    ) -> AsyncIterator[str]:
        trace_id = uuid.uuid4().hex[:16]
        run_id = f"run_{uuid.uuid4().hex}"
        started = time.time()
        ensure_chat_session(session_id)
        snapshot = resolve_snapshot(session_id)
        path = self._select_path(session_id, message, snapshot.config)
        state = RunState(
            run_id=run_id,
            trace_id=trace_id,
            session_id=session_id,
            snapshot_id=snapshot.snapshot_id,
            path=path,
            status=RunStatus.RUNNING,
            next_step="resolve_snapshot",
        )
        create_chat_trace(
            trace_id=trace_id,
            session_id=session_id,
            user_message=message,
            risk_level="L2" if path == RuntimePath.CONTROLLED_ACTION else "L0",
            version_snapshot=snapshot.snapshot_hash,
        )
        save_chat_message(
            session_id=session_id,
            role="user",
            content=message,
            trace_id=trace_id,
            status="success",
            usage_source="not_applicable",
        )
        ledger = EvidenceLedger(
            trace_id=trace_id,
            session_id=session_id,
            config_snapshot={
                "snapshot_id": snapshot.snapshot_id,
                "release_id": snapshot.release_id,
                "snapshot_hash": snapshot.snapshot_hash,
            },
            release_id=snapshot.release_id,
            config_hash=snapshot.snapshot_hash,
            runtime_path=path.value,
        )
        record_trace_event(
            trace_id,
            "resolve_snapshot",
            "success",
            output_summary=f"{snapshot.release_id}/{snapshot.snapshot_hash[:12]}",
            metadata={
                "release_id": snapshot.release_id,
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_hash": snapshot.snapshot_hash,
            },
        )
        yield _sse(
            "start",
            {
                "session_id": session_id,
                "trace_id": trace_id,
                "run_id": run_id,
                "release_id": snapshot.release_id,
                "snapshot_id": snapshot.snapshot_id,
                "runtime_path": path.value,
            },
        )
        try:
            await self._resolve_semantic_lane(
                message,
                session_id,
                user_id,
                trace_id,
                snapshot,
                state,
                ledger,
            )
            lane_payload = state.lane_decision.model_dump(mode="json")
            lane_payload["explanation"] = _lane_explanation(state.lane_decision)
            yield _sse(
                "lane",
                {
                    "trace_id": trace_id,
                    **lane_payload,
                    "answer_contract": state.answer_contract.model_dump(mode="json"),
                },
            )

            if state.lane_decision.lane == RuntimeLane.SAFETY_HANDOFF:
                async for event in self._stream_a_handoff(
                    message, session_id, trace_id, snapshot, state, ledger, started
                ):
                    yield event
                return

            if state.answer_contract.response_mode == ResponseMode.CONTROLLED_WRITE:
                path = RuntimePath.CONTROLLED_ACTION
                state.path = path
                ledger.runtime_path = path.value

            if path == RuntimePath.CONTROLLED_ACTION:
                state.capability_decision = CapabilityDecision(
                    selected_agent_id=None,
                    skill={"status": "skipped", "reason_code": "controlled_action"},
                    rag={"status": "skipped", "reason_code": "controlled_action"},
                    tool={"status": "skipped", "reason_code": "controlled_action"},
                    write={"status": "required", "reason_code": "controlled_action"},
                    handoff={"status": "available", "reason_code": "owner_can_request"},
                )
                async for event in self._stream_controlled_action(
                    message, session_id, trace_id, snapshot, state, ledger, started
                ):
                    yield event
                return

            async for event in self._stream_consultation(
                message,
                session_id,
                user_id,
                trace_id,
                snapshot,
                state,
                ledger,
                started,
            ):
                yield event
        except asyncio.CancelledError:
            # A reverse proxy timeout or closed browser used to leave a Trace
            # permanently "in_progress".  Preserve a terminal transport fact
            # even though no SSE event can be delivered to the disconnected
            # client.
            state.status = RunStatus.FAILED
            state.next_step = None
            ledger.violation(
                "client_stream_cancelled",
                "SSE client disconnected before the governed run completed.",
            )
            ledger.capture_state(state)
            ledger.persist("failed")
            update_chat_trace(trace_id, status="failed")
            record_trace_event(
                trace_id,
                "client_stream_cancelled",
                "failed",
                latency_ms=int((time.time() - started) * 1000),
                output_summary="SSE client disconnected before completion",
            )
            raise
        except Exception as exc:
            auto_badcase = None
            failure_code = (
                "provider_failure"
                if isinstance(exc, ProviderFailureError)
                else (
                    "internal_control_payload_leak"
                    if isinstance(exc, InternalControlPayloadLeakError)
                    else "runtime_failure"
                )
            )
            public_error = (
                PROVIDER_FAILURE_PUBLIC_MESSAGE
                if failure_code == "provider_failure"
                else RUNTIME_FAILURE_PUBLIC_MESSAGE
            )
            state.status = RunStatus.FAILED
            state.next_step = None
            ledger.violation(failure_code, str(exc))
            ledger.capture_state(state)
            ledger.persist("failed")
            try:
                auto_badcase = capture_runtime_badcase(
                    ledger=ledger.contract,
                    original_query=message,
                    ai_response="",
                    runtime_error=str(exc),
                    runtime_error_type=failure_code,
                )
                if auto_badcase:
                    ledger.append(
                        "badcase_links",
                        {
                            "badcase_id": auto_badcase.get("id"),
                            "source": auto_badcase.get("source"),
                            "trigger": failure_code,
                        },
                    )
                    ledger.persist("failed")
            except Exception:
                pass
            update_chat_trace(trace_id, status="failed")
            record_trace_event(
                trace_id,
                failure_code,
                "failed",
                latency_ms=int((time.time() - started) * 1000),
                output_summary=str(exc)[:240],
            )
            yield _sse(
                "error",
                {
                    "error": public_error,
                    "error_code": failure_code,
                    "status": "failed",
                    "trace_id": trace_id,
                    "release_id": snapshot.release_id,
                    "snapshot_id": snapshot.snapshot_id,
                    "auto_badcase_id": (
                        auto_badcase.get("id") if auto_badcase else None
                    ),
                },
            )
            yield _sse(
                "done",
                {
                    "status": "failed",
                    "error_code": failure_code,
                    "trace_id": trace_id,
                    "release_id": snapshot.release_id,
                    "snapshot_id": snapshot.snapshot_id,
                    "auto_badcase_id": (
                        auto_badcase.get("id") if auto_badcase else None
                    ),
                },
            )

    @staticmethod
    def _select_path(
        session_id: str,
        message: str,
        snapshot_config: Dict[str, Any],
    ) -> RuntimePath:
        # Persisted state may continue deterministically. New natural-language
        # requests are not promoted to a write path by words or regexes here;
        # the semantic LaneDecision/AnswerContract owns that decision.
        pending = get_pending_action_proposal(session_id)
        draft = get_work_order_draft(session_id)
        if pending and (is_confirmation(message) or is_cancel_request(message)):
            return RuntimePath.CONTROLLED_ACTION
        if draft and _is_draft_follow_up(message, draft):
            return RuntimePath.CONTROLLED_ACTION
        if RuntimeCoordinator._latest_committed_dynamic_action(session_id, message):
            return RuntimePath.CONTROLLED_ACTION
        return RuntimePath.CONSULTATION

    async def _resolve_semantic_lane(
        self,
        message: str,
        session_id: str,
        user_id: str,
        trace_id: str,
        snapshot: Any,
        state: RunState,
        ledger: EvidenceLedger,
    ) -> None:
        """Resolve one strict semantic decision and account for its Provider call."""

        cards = vertical_agent_cards(snapshot.config)
        if state.path == RuntimePath.CONTROLLED_ACTION:
            pending = get_pending_action_proposal(session_id) or {}
            payload = pending.get("payload") or {}
            requested_agent_id = str(payload.get("agent_id") or "")
            property_cards = _lane_candidates(cards, RuntimeLane.PROPERTY_GOVERNED)
            selected_card = next(
                (item for item in property_cards if str(item.get("agent_id")) == requested_agent_id),
                None,
            )
            if selected_card is None:
                selected_card = next(
                    (item for item in property_cards if str(item.get("agent_id")) == "maintenance"),
                    property_cards[0] if property_cards else None,
                )
            if selected_card is None:
                raise RuntimeError("structured action state has no Published property Agent")
            reported_decision = LaneDecision(
                lane=RuntimeLane.PROPERTY_GOVERNED,
                business_intent="continue_controlled_action",
                reason="会话中存在未完成的受控业务状态，继续原状态机。",
                decision_source=LaneDecisionSource.STRUCTURED_STATE,
            )
        else:
            from agents.router import classify_lane_decision

            router_config = next(
                (
                    item
                    for item in snapshot.config.get("agents") or []
                    if item.get("agent_id") == "router"
                    or item.get("category") in {"router", "orchestration"}
                ),
                {},
            )
            default_model = (snapshot.config.get("model_policy") or {}).get("default") or {}
            router_model_id = str(
                router_config.get("model_id")
                or default_model.get("model_id")
                or MODEL_ID
            )
            if router_model_id.lower() != "deepseek-v4-flash":
                raise RuntimeError("semantic Router must use deepseek-v4-flash")
            visible_history = _visible_chat_history(
                session_id,
                current_trace_id=trace_id,
                rounds=5,
            )
            router_started = time.time()
            router_thinking = _thinking_for_snapshot(snapshot.config, router_model_id)
            with provider_accounting_scope(
                trace_id=trace_id,
                session_id=session_id,
                stage="router",
                model_selection_reason="semantic LaneDecision from Published Snapshot",
                price_snapshot=_price_for_snapshot(snapshot.config, router_model_id),
                model_policy_version=str(
                    (snapshot.config.get("model_policy") or {}).get("version") or "v1.8"
                ),
            ):
                result = await classify_lane_decision(
                    message,
                    vertical_agents=cards,
                    user_id=user_id,
                    session_id=f"{session_id}::semantic_router::{trace_id}",
                    model=_build_model_from_snapshot(snapshot.config, router_model_id),
                    visible_history=visible_history,
                )
            evidence = result.get("provider_evidence") or {}
            usage = evidence.get("usage") or result.get("metrics") or {}
            response_model = evidence.get("provider_response_model")
            provider_status = str(result.get("provider_status") or "failed")
            billing_model = response_model or router_model_id
            thinking = router_thinking
            cost = build_cost_entry(
                stage="router",
                provider=_model_provider(snapshot.config, router_model_id),
                requested_model=router_model_id,
                response_model=None,
                provider_response_model=response_model,
                thinking_enabled=thinking,
                model_policy_version=str(
                    (snapshot.config.get("model_policy") or {}).get("version") or "v1.8"
                ),
                provider_usage=usage if usage else None,
                price_row=_price_for_snapshot(snapshot.config, billing_model),
                local_estimate_tokens=_estimate_tokens(message),
                provider_succeeded=provider_status == "success",
            )
            state.cost_entries.append(cost)
            state.model_calls.append(
                {
                    "stage": "router",
                    "model_id": router_model_id,
                    "requested_model": router_model_id,
                    "provider_response_model": response_model,
                    "provider_request_id": evidence.get("provider_request_id"),
                    "thinking_enabled": thinking,
                    "latency_ms": int((time.time() - router_started) * 1000),
                    "usage": usage,
                    "usage_source": cost.usage_source.value,
                    "status": provider_status,
                }
            )
            if result.get("decision") is None:
                if provider_status != "success":
                    raise ProviderFailureError("semantic Router Provider call failed")
                raise RuntimeError("semantic LaneDecision schema validation failed")
            reported_decision = result["decision"]

        current_session = get_chat_session(session_id) or {}
        effective_decision, normalized_from_lane = _effective_lane_decision(
            reported_decision,
            handoff_status=str(
                current_session.get("handoff_status") or "none"
            ),
            handoff_reason_code=str(
                current_session.get("handoff_reason_code") or ""
            ),
            handoff_queue=str(current_session.get("handoff_queue") or ""),
        )
        state.lane_decision = effective_decision
        if normalized_from_lane:
            ledger.append(
                "system_observations",
                {
                    "type": "effective_lane_invariant",
                    "router_reported_lane": normalized_from_lane,
                    "router_reported_business_intent": reported_decision.business_intent,
                    "effective_lane": RuntimeLane.SAFETY_HANDOFF.value,
                    "business_intent": state.lane_decision.business_intent,
                    "reason_code": (
                        "safety_risk"
                        if state.lane_decision.business_intent == "safety_risk"
                        else "user_requested"
                    ),
                },
            )

        if (
            state.lane_decision.lane == RuntimeLane.PROPERTY_GOVERNED
            and self._is_work_order_action_context(session_id, message)
        ):
            state.path = RuntimePath.CONTROLLED_ACTION
        state.answer_contract = _answer_contract_for(state.lane_decision, state.path)
        lane_payload = state.lane_decision.model_dump(mode="json")
        lane_payload["explanation"] = _lane_explanation(state.lane_decision)
        ledger.set("lane_decision", lane_payload)
        ledger.set("answer_contract", state.answer_contract.model_dump(mode="json"))
        record_trace_event(
            trace_id,
            "lane_decision",
            "success",
            output_summary=lane_payload["explanation"],
            metadata={
                **lane_payload,
                "answer_contract": state.answer_contract.model_dump(mode="json"),
            },
        )

    async def _select_agent_after_lane(
        self,
        message: str,
        session_id: str,
        user_id: str,
        trace_id: str,
        snapshot: Any,
        state: RunState,
        cards: List[Dict[str, Any]],
        visible_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Select a same-domain Agent without changing or invalidating the Lane."""

        from agents.router import select_lane_agent

        default_model = (snapshot.config.get("model_policy") or {}).get("default") or {}
        selector_model_id = str(default_model.get("model_id") or MODEL_ID)
        if selector_model_id.lower() != "deepseek-v4-flash":
            return {
                "selected_agent_id": None,
                "reason": "Agent选择器未使用允许的Flash模型。",
                "selection_source": "selector_policy_blocked",
                "provider_status": "not_applicable",
                "candidate_count": len(cards),
            }
        started = time.time()
        selector_thinking = _thinking_for_snapshot(snapshot.config, selector_model_id)
        with provider_accounting_scope(
            trace_id=trace_id,
            session_id=session_id,
            stage="agent_selector",
            model_selection_reason="same-domain Agent selection after fixed Lane",
            price_snapshot=_price_for_snapshot(snapshot.config, selector_model_id),
            model_policy_version=str(
                (snapshot.config.get("model_policy") or {}).get("version") or "v1.8"
            ),
        ):
            selection = await select_lane_agent(
                message,
                lane=state.lane_decision.lane,
                vertical_agents=cards,
                user_id=user_id,
                session_id=f"{session_id}::agent_selector::{trace_id}",
                model=_build_model_from_snapshot(snapshot.config, selector_model_id),
                visible_history=visible_history,
            )
        provider_status = str(selection.get("provider_status") or "not_applicable")
        if provider_status != "not_applicable":
            evidence = selection.get("provider_evidence") or {}
            usage = evidence.get("usage") or selection.get("metrics") or {}
            response_model = evidence.get("provider_response_model")
            billing_model = response_model or selector_model_id
            thinking = selector_thinking
            cost = build_cost_entry(
                stage="agent_selector",
                provider=_model_provider(snapshot.config, selector_model_id),
                requested_model=selector_model_id,
                response_model=None,
                provider_response_model=response_model,
                thinking_enabled=thinking,
                model_policy_version=str(
                    (snapshot.config.get("model_policy") or {}).get("version") or "v1.8"
                ),
                provider_usage=usage if usage else None,
                price_row=_price_for_snapshot(snapshot.config, billing_model),
                local_estimate_tokens=_estimate_tokens(message),
                provider_succeeded=provider_status == "success",
            )
            state.cost_entries.append(cost)
            state.model_calls.append(
                {
                    "stage": "agent_selector",
                    "model_id": selector_model_id,
                    "requested_model": selector_model_id,
                    "provider_response_model": response_model,
                    "provider_request_id": evidence.get("provider_request_id"),
                    "thinking_enabled": thinking,
                    "latency_ms": int((time.time() - started) * 1000),
                    "usage": usage,
                    "usage_source": cost.usage_source.value,
                    "status": provider_status,
                }
            )
        selected = selection.get("selected_agent_id")
        record_trace_event(
            trace_id,
            "agent_selection",
            "success" if selected else "skipped",
            latency_ms=int((time.time() - started) * 1000),
            output_summary=(
                f"selected={selected}"
                if selected
                else f"no Agent selected: {selection.get('selection_source')}"
            ),
            metadata={
                "lane": state.lane_decision.lane.value,
                "selected_agent_id": selected,
                "selection_source": selection.get("selection_source"),
                "candidate_count": selection.get("candidate_count"),
                "validation_error": selection.get("validation_error"),
            },
        )
        return selection

    async def _stream_contract_static_answer(
        self,
        session_id: str,
        trace_id: str,
        snapshot: Any,
        state: RunState,
        ledger: EvidenceLedger,
        started: float,
    ) -> AsyncIterator[str]:
        """Complete a static safety refusal without a business Agent."""

        if state.answer_contract is None or state.lane_decision is None:
            raise RuntimeError("static answer started without contracts")
        content = "这个请求可能造成现实伤害，我不能提供危险配方、比例或实施步骤。我可以改为说明安全处置与风险预防原则。"
        reason = "unsafe_request_refused"
        state.capability_decision = CapabilityDecision(
            selected_agent_id=None,
            skill={"status": "skipped", "reason_code": reason},
            rag={"status": "skipped", "reason_code": reason},
            tool={"status": "skipped", "reason_code": reason},
            write={"status": "not_required", "reason_code": reason},
            handoff={"status": "not_required", "reason_code": reason},
        )
        state.status = RunStatus.COMPLETED
        state.next_step = None
        ledger.capture_state(state)
        ledger.append(
            "evaluation_results",
            {"case": "answer_contract", "passed": True, "response_mode": state.answer_contract.response_mode.value},
        )
        ledger.persist("complete")
        agent_name = "语义澄清" if clarify else "安全回答边界"
        agent_id = "semantic_clarifier" if clarify else "safe_refusal"
        router_tokens = sum(int(item.total_tokens or 0) for item in state.cost_entries)
        saved = save_chat_message(
            session_id=session_id,
            role="assistant",
            content=content,
            token_count=0,
            round_token_count=router_tokens,
            trace_id=trace_id,
            current_agent=agent_name,
            current_agent_id=agent_id,
            status="success",
            usage_source="not_applicable",
        )
        update_chat_trace(
            trace_id,
            intent=state.lane_decision.business_intent,
            agent_name=agent_name,
            agent_id=agent_id,
            status="complete",
        )
        record_trace_event(
            trace_id,
            "final_response",
            "success",
            latency_ms=int((time.time() - started) * 1000),
            output_summary=content[:240],
            metadata={
                "lane_decision": state.lane_decision.model_dump(mode="json"),
                "answer_contract": state.answer_contract.model_dump(mode="json"),
                "capability_decision": state.capability_decision.model_dump(mode="json"),
                "vertical_model_invoked": False,
            },
        )
        yield _sse("delta", {"content": content})
        yield _sse("final", {"content": content, "current_agent": agent_name, "current_agent_id": agent_id})
        yield _sse(
            "done",
            {
                "status": "complete",
                "message_id": saved.get("id"),
                "trace_id": trace_id,
                "runtime_path": RuntimePath.CONSULTATION.value,
                "release_id": snapshot.release_id,
                "snapshot_id": snapshot.snapshot_id,
                "current_agent": agent_name,
                "current_agent_id": agent_id,
                "lane_decision": state.lane_decision.model_dump(mode="json"),
                "answer_contract": state.answer_contract.model_dump(mode="json"),
                "capability_decision": state.capability_decision.model_dump(mode="json"),
                "cost_entries": [item.model_dump(mode="json") for item in state.cost_entries],
                "vertical_provider_request_count": 0,
                "round_token_count": router_tokens,
            },
        )

    async def _stream_a_handoff(
        self,
        message: str,
        session_id: str,
        trace_id: str,
        snapshot: Any,
        state: RunState,
        ledger: EvidenceLedger,
        started: float,
    ) -> AsyncIterator[str]:
        """Execute either A-lane Handoff subtype from one immutable contract."""

        if state.lane_decision is None or state.answer_contract is None:
            raise RuntimeError("A handoff started without contracts")
        handoff_contract = _handoff_contract_for(state.lane_decision)
        current_handoff = get_chat_session(session_id) or {}
        current_handoff_status = str(
            current_handoff.get("handoff_status") or "none"
        )
        if handoff_contract.kind == HandoffKind.USER_REQUESTED:
            handoff_result = await self._maybe_handoff(
                message,
                session_id,
                trace_id,
                snapshot.release_id,
                decision=state.lane_decision,
            )
            if handoff_result is None:
                raise RuntimeError("A user-requested Handoff was not persisted")
            reply, handoff_state, handoff_policy = handoff_result
        else:
            persisted_safety = (
                str(current_handoff.get("handoff_reason_code") or "")
                == "safety_risk"
                or str(current_handoff.get("handoff_queue") or "")
                == "emergency"
            )
            active_handoff = current_handoff_status in {
                "requested",
                "active",
                "waiting_user",
            }
            handoff = current_handoff
            if not active_handoff or not persisted_safety:
                handoff = request_handoff(
                    session_id,
                    state.lane_decision.reason or "检测到明确、现实的安全风险。",
                    risk_level="L3",
                    reason_code=handoff_contract.reason_code,
                    queue=handoff_contract.queue,
                    handoff_package={
                        "trace_id": trace_id,
                        "release_id": snapshot.release_id,
                        "trigger_message": message,
                        "semantic_lane": state.lane_decision.model_dump(mode="json"),
                        "handoff_kind": handoff_contract.kind.value,
                        "safety_override": handoff_contract.safety_override,
                    },
                )
            if current_handoff_status == "waiting_user":
                handoff = resume_handoff_after_owner_message(session_id)
            handoff_state = str(
                handoff.get("handoff_status") or current_handoff_status or "requested"
            )
            handoff_policy = {
                "level": "L3",
                "reason_code": handoff_contract.reason_code,
                "queue": handoff_contract.queue,
                "safety_override": handoff_contract.safety_override,
                "matched_signals": ["semantic_safety_risk"],
            }
            reply = (
                "请立即远离危险源，不要触碰设备或积水，并提醒周围人员避开；如存在火灾、"
                "触电、燃气或人身危险，请立即联系119、120、110等当地紧急渠道。系统已发起"
                "安全人工协同，但物业协同不能替代现实紧急救援。"
            )
        decision_summary = {
            "agent": _decision("skipped", "handoff_preempted"),
            "skill": _decision("skipped", "handoff_preempted"),
            "rag": _decision("skipped", "handoff_preempted"),
            "tool": _decision("skipped", "handoff_preempted"),
            "write": _decision("skipped", "handoff_preempted"),
            "handoff": _decision(
                "selected",
                handoff_contract.reason_code,
                queue=handoff_contract.queue,
                safety_override=handoff_contract.safety_override,
            ),
        }
        state.capability_decision = CapabilityDecision(
            selected_agent_id=None,
            skill={"status": "skipped", "reason_code": "handoff_preempted"},
            rag={"status": "skipped", "reason_code": "handoff_preempted"},
            tool={"status": "skipped", "reason_code": "handoff_preempted"},
            write={"status": "not_required", "reason_code": "handoff_preempted"},
            handoff={
                "status": "required",
                "reason_code": handoff_contract.reason_code,
                "details": {
                    "queue": handoff_contract.queue,
                    "safety_override": handoff_contract.safety_override,
                    "handler": "human_copilot",
                    "matched_signals": handoff_policy.get("matched_signals") or [],
                },
            },
        )
        state.status = RunStatus.COMPLETED
        state.next_step = None
        ledger.capture_state(state)
        ledger.append(
            "handoff_events",
            {
                "status": handoff_state,
                "reason_code": handoff_contract.reason_code,
                "handoff_kind": handoff_contract.kind.value,
                "queue": handoff_contract.queue,
                "safety_override": handoff_contract.safety_override,
                "handler": "human_copilot",
                "router_model_invoked": any(
                    item.get("stage") == "router" for item in state.model_calls
                ),
                "vertical_model_invoked": False,
            },
        )
        ledger.append(
            "evaluation_results",
            {
                "case": "a_handoff_contract",
                "passed": True,
                "decision_summary": decision_summary,
            },
        )
        ledger.persist("complete")
        router_tokens = sum(int(item.total_tokens or 0) for item in state.cost_entries)
        saved = save_chat_message(
            session_id=session_id,
            role="assistant",
            content=reply,
            trace_id=trace_id,
            current_agent="人工协同控制器",
            current_agent_id="human_copilot",
            status="success",
            token_count=0,
            round_token_count=router_tokens,
            usage_source="not_applicable",
        )
        update_chat_trace(
            trace_id,
            intent=state.lane_decision.business_intent or handoff_contract.reason_code,
            agent_name="人工协同控制器",
            agent_id="human_copilot",
            status="complete",
        )
        record_trace_event(
            trace_id,
            "a_handoff",
            "success",
            latency_ms=int((time.time() - started) * 1000),
            output_summary=(
                f"A Handoff created: {handoff_contract.reason_code}; "
                "business capabilities skipped"
            ),
            metadata={
                "reason_code": handoff_contract.reason_code,
                "queue": handoff_contract.queue,
                "safety_override": handoff_contract.safety_override,
                "answer_contract": state.answer_contract.model_dump(mode="json"),
                "capability_decision": state.capability_decision.model_dump(mode="json"),
            },
        )
        yield _sse("delta", {"content": reply})
        yield _sse("final", {"content": reply, "current_agent": "人工协同控制器", "current_agent_id": "human_copilot"})
        yield _sse(
            "done",
            {
                "status": "complete",
                "message_id": saved.get("id"),
                "trace_id": trace_id,
                "handoff": True,
                "handoff_state": handoff_state,
                "handoff_reason": handoff_contract.reason_code,
                "handoff_queue": handoff_contract.queue,
                "safety_override": handoff_contract.safety_override,
                "handler": "human_copilot",
                "lane_decision": state.lane_decision.model_dump(mode="json"),
                "answer_contract": state.answer_contract.model_dump(mode="json"),
                "capability_decision": state.capability_decision.model_dump(mode="json"),
                "decision_summary": decision_summary,
                "release_id": snapshot.release_id,
                "snapshot_id": snapshot.snapshot_id,
                "cost_entries": [item.model_dump(mode="json") for item in state.cost_entries],
                "vertical_provider_request_count": 0,
                "round_token_count": router_tokens,
            },
        )

    async def _stream_external_safety_boundary(
        self,
        session_id: str,
        trace_id: str,
        snapshot: Any,
        state: RunState,
        ledger: EvidenceLedger,
        started: float,
    ) -> AsyncIterator[str]:
        """End a non-property high-risk request safely without a model or handoff."""

        decision_summary = {
            "agent": _decision("skipped", "external_safety_boundary"),
            "skill": _decision("skipped", "external_safety_boundary"),
            "rag": _decision("skipped", "external_safety_boundary"),
            "tool": _decision("skipped", "external_safety_boundary"),
            "handoff": _decision(
                "skipped",
                "use_public_emergency_service",
                domain_scope="isolated_general",
            ),
        }
        state.status = RunStatus.COMPLETED
        state.next_step = None
        ledger.capture_state(state)
        ledger.append(
            "evaluation_results",
            {
                "case": "external_safety_boundary",
                "passed": True,
                "model_invoked": False,
                "handoff_created": False,
            },
        )
        ledger.append(
            "evaluation_results",
            {
                "case": "capability_decision",
                "passed": True,
                "decision_summary": decision_summary,
            },
        )
        ledger.persist("complete")
        saved = save_chat_message(
            session_id=session_id,
            role="assistant",
            content=NON_PROPERTY_SAFETY_RESPONSE,
            token_count=0,
            round_token_count=0,
            token_detail={
                "input_tokens": None,
                "output_tokens": None,
                "reasoning_tokens": None,
                "cached_tokens": None,
                "total_tokens": None,
                "local_estimate_tokens": None,
                "model_invoked": False,
            },
            current_agent="安全边界",
            current_agent_id="external_safety_boundary",
            trace_id=trace_id,
            status="success",
            thinking_enabled=False,
            usage_source="not_applicable",
        )
        update_chat_trace(
            trace_id,
            intent="external_safety_boundary",
            agent_name="安全边界",
            agent_id="external_safety_boundary",
            status="complete",
        )
        record_trace_event(
            trace_id,
            "external_safety_boundary",
            "success",
            latency_ms=int((time.time() - started) * 1000),
            output_summary="non-property high-risk request ended without model",
            metadata={
                "domain_scope": "isolated_general",
                "model_invoked": False,
                "decision_summary": decision_summary,
            },
        )
        yield _sse(
            "final",
            {
                "content": NON_PROPERTY_SAFETY_RESPONSE,
                "current_agent": "安全边界",
                "current_agent_id": "external_safety_boundary",
            },
        )
        yield _sse(
            "done",
            {
                "status": "complete",
                "message_id": saved.get("id"),
                "trace_id": trace_id,
                "runtime_path": RuntimePath.CONSULTATION.value,
                "release_id": snapshot.release_id,
                "snapshot_id": snapshot.snapshot_id,
                "current_agent": "安全边界",
                "current_agent_id": "external_safety_boundary",
                "domain_scope": "isolated_general",
                "decision_summary": decision_summary,
                "cost_entries": [],
                "usage_source": "not_applicable",
                "token_count": 0,
                "round_token_count": 0,
            },
        )

    async def _stream_unconfigured_lane_boundary(
        self,
        session_id: str,
        trace_id: str,
        snapshot: Any,
        state: RunState,
        ledger: EvidenceLedger,
        started: float,
    ) -> AsyncIterator[str]:
        """Complete truthfully when a valid Lane has no usable same-domain Agent."""

        property_lane = (
            state.lane_decision is not None
            and state.lane_decision.lane == RuntimeLane.PROPERTY_GOVERNED
        )
        content = (
            PROPERTY_AGENT_UNAVAILABLE_RESPONSE
            if property_lane
            else OUT_OF_SCOPE_RESPONSE
        )
        boundary_id = (
            "property_agent_unavailable"
            if property_lane
            else "isolated_general_boundary"
        )
        boundary_name = "物业角色边界" if property_lane else "通用边界"
        domain_scope = "property" if property_lane else "isolated_general"

        state.capability_decision = build_lane_agent_unavailable_decision(
            property_lane=property_lane
        )
        state.status = RunStatus.COMPLETED
        state.next_step = None
        ledger.capture_state(state)
        ledger.append(
            "evaluation_results",
            {
                "case": "lane_valid_agent_unavailable_boundary",
                "passed": True,
                "lane": state.lane_decision.lane.value if state.lane_decision else None,
                "boundary_model_invoked": False,
                "provider_request_count": len(state.model_calls),
            },
        )
        ledger.persist("complete")
        saved = save_chat_message(
            session_id=session_id,
            role="assistant",
            content=content,
            trace_id=trace_id,
            current_agent=boundary_name,
            current_agent_id=boundary_id,
            status="success",
            usage_source="not_applicable",
        )
        update_chat_trace(
            trace_id,
            intent=boundary_id,
            agent_name=boundary_name,
            agent_id=boundary_id,
            status="complete",
        )
        record_trace_event(
            trace_id,
            "lane_agent_boundary",
            "success",
            latency_ms=int((time.time() - started) * 1000),
            output_summary="Lane remained valid; no same-domain Agent was selected",
            metadata={
                "lane_decision": (
                    state.lane_decision.model_dump(mode="json")
                    if state.lane_decision
                    else None
                ),
                "boundary_model_invoked": False,
                "provider_request_count": len(state.model_calls),
            },
        )
        yield _sse(
            "final",
            {
                "content": content,
                "current_agent": boundary_name,
                "current_agent_id": boundary_id,
                "domain_scope": domain_scope,
            },
        )
        yield _sse(
            "done",
            {
                "status": "complete",
                "message_id": saved.get("id"),
                "trace_id": trace_id,
                "runtime_path": RuntimePath.CONSULTATION.value,
                "release_id": snapshot.release_id,
                "snapshot_id": snapshot.snapshot_id,
                "current_agent": boundary_name,
                "current_agent_id": boundary_id,
                "domain_scope": domain_scope,
                "lane_decision": (
                    state.lane_decision.model_dump(mode="json")
                    if state.lane_decision
                    else None
                ),
                "capability_decision": state.capability_decision.model_dump(
                    mode="json"
                ),
                "cost_entries": [
                    item.model_dump(mode="json") for item in state.cost_entries
                ],
                "usage_source": (
                    state.cost_entries[-1].usage_source.value
                    if state.cost_entries
                    else "not_applicable"
                ),
                "token_count": sum(item.total_tokens or 0 for item in state.cost_entries),
                "round_token_count": sum(
                    item.total_tokens or 0 for item in state.cost_entries
                ),
            },
        )

    @staticmethod
    def _is_work_order_action_context(session_id: str, message: str) -> bool:
        pending = get_pending_action_proposal(session_id)
        draft = get_work_order_draft(session_id)
        latest = (
            get_latest_action_proposal(session_id, "work_order.create")
            if is_confirmation(message)
            else None
        )
        return bool(
            (draft and _is_draft_follow_up(message, draft))
            or (
                pending
                and pending.get("action_type") == "work_order.create"
                and (is_confirmation(message) or is_cancel_request(message))
            )
            or is_explicit_work_order_request(message)
            or latest
        )

    @staticmethod
    def _latest_committed_dynamic_action(
        session_id: str,
        message: str,
    ) -> Optional[Dict[str, Any]]:
        if not is_confirmation(message):
            return None
        latest = get_latest_action_proposal(session_id)
        if (
            latest
            and str(latest.get("action_type") or "").startswith("mcp.")
            and latest.get("status") == "committed"
        ):
            return latest
        return None

    @staticmethod
    def _match_write_tool(
        snapshot_config: Dict[str, Any],
        message: str,
    ) -> Optional[Dict[str, Any]]:
        plan = unique_write_plan(snapshot_config, message)
        if not plan:
            return None
        agent = next(
            (
                item
                for item in snapshot_config.get("agents") or []
                if item.get("agent_id") == plan.agent_id
            ),
            {},
        )
        return {
            **plan.model_dump(mode="json"),
            "agent_name": str(agent.get("name") or plan.agent_id),
        }

    async def _maybe_handoff(
        self,
        message: str,
        session_id: str,
        trace_id: str,
        release_id: str,
        decision: Optional[LaneDecision] = None,
    ) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        requested_by_user = bool(
            decision
            and str(decision.business_intent or "").strip()
            == "user_requested_handoff"
        )
        policy: Dict[str, Any] = {
            "should_request_handoff": requested_by_user,
            "reason": (decision.reason if decision else ""),
            "level": "L3",
            "reason_code": "user_requested",
            "queue": "property_service",
            "safety_override": False,
            "matched_signals": ["semantic_user_requested_handoff"] if requested_by_user else [],
        }
        current = get_chat_session(session_id) or {}
        current_status = str(current.get("handoff_status") or "none")
        if current_status == "waiting_user":
            resumed = resume_handoff_after_owner_message(session_id)
            return (
                "已将补充信息同步给接管工作人员，人工处理已恢复。",
                str(resumed.get("handoff_status") or "active"),
                {
                    **policy,
                    "reason_code": str(
                        current.get("handoff_reason_code") or "user_requested"
                    ),
                    "queue": str(
                        current.get("handoff_queue") or "property_service"
                    ),
                    "matched_signals": ["waiting_user"],
                },
            )
        if current_status in {"requested", "active"}:
            return (
                (
                    "人工协同已在等待领取。"
                    if current_status == "requested"
                    else "工作人员已领取，当前正在人工协同处理中。"
                ),
                current_status,
                {
                    **policy,
                    "reason_code": str(
                        current.get("handoff_reason_code") or "user_requested"
                    ),
                    "queue": str(
                        current.get("handoff_queue") or "property_service"
                    ),
                    "matched_signals": [current_status],
                },
            )
        if policy.get("should_request_handoff"):
            session = request_handoff(
                session_id,
                str(policy.get("reason") or "需要人工协同"),
                risk_level=str(policy.get("level") or "L3"),
                reason_code=str(policy.get("reason_code") or "user_requested"),
                queue=policy.get("queue"),
                handoff_package={
                    "trace_id": trace_id,
                    "release_id": release_id,
                    "trigger_message": message,
                    "policy": policy,
                    "handoff_kind": HandoffKind.USER_REQUESTED.value,
                    "safety_override": False,
                },
            )
            status = str(session.get("handoff_status") or "requested")
            if status == "active":
                reply = "工作人员已领取，当前正在人工协同处理中。"
            else:
                reply = "已发起人工协同：等待工作人员领取。"
            return (
                reply,
                status,
                policy,
            )
        return None

    async def _stream_controlled_action(
        self,
        message: str,
        session_id: str,
        trace_id: str,
        snapshot: Any,
        state: RunState,
        ledger: EvidenceLedger,
        started: float,
    ) -> AsyncIterator[str]:
        state.next_step = "collect_or_resume_action"
        use_work_order = self._is_work_order_action_context(
            session_id,
            message,
        )
        if use_work_order:
            result = advance_work_order_workflow(
                session_id,
                message,
                trace_id=trace_id,
                release_id=snapshot.release_id,
            )
        else:
            result = await self._advance_dynamic_mcp_action(
                message=message,
                session_id=session_id,
                trace_id=trace_id,
                snapshot=snapshot,
            )
        if result is None:
            raise RuntimeError("controlled action path produced no workflow result")
        selected_agent_id = str(result.get("agent_id") or "maintenance")
        selected_agent_name = str(result.get("agent_name") or "维修 Agent")
        action_type = str(result.get("action_type") or "work_order.create")
        route = RouteDecision(
            candidates=[selected_agent_id],
            selected_agent_id=selected_agent_id,
            reason=str(result.get("route_reason") or "受控维修工单流程"),
            confidence=1.0,
            required_capability_types=["action", "hitl"],
        )
        state.route_decision = route
        state.selected_agent = {
            "agent_id": selected_agent_id,
            "name": selected_agent_name,
        }
        proposal_id = result.get("proposal_id")
        proposal_row: Optional[Dict[str, Any]] = None
        if proposal_id:
            proposal_row = get_action_proposal(str(proposal_id))
            if proposal_row:
                state.pending_actions.append(
                    ActionProposal(
                        proposal_id=proposal_row["proposal_id"],
                        session_id=proposal_row["session_id"],
                        trace_id=proposal_row.get("trace_id"),
                        release_id=proposal_row.get("release_id"),
                        action_type=proposal_row["action_type"],
                        risk_level=proposal_row["risk_level"],
                        payload=proposal_row.get("payload") or {},
                        parameter_hash=content_hash(proposal_row.get("payload") or {}),
                        idempotency_key=proposal_row["idempotency_key"],
                        status=proposal_row["status"],
                    )
                )
                for approval in list_action_approvals(str(proposal_id)):
                    state.approval_events.append(
                        ApprovalEvent(
                            proposal_id=str(proposal_id),
                            decision=str(approval["decision"]),
                            actor=str(approval["actor"]),
                            parameter_hash=content_hash(proposal_row.get("payload") or {}),
                            comment=approval.get("comment"),
                            decided_at=str(approval["decided_at"]),
                        )
                    )
        receipt_data = result.get("receipt")
        if receipt_data:
            receipt = _action_receipt_from_payload(receipt_data)
            state.action_receipts.append(receipt)
            receipt_result = receipt.result or {}
            if _records_new_mcp_invocation(
                action_type,
                result.get("action"),
            ):
                proposal_payload = (proposal_row or {}).get("payload") or {}
                invocation = ToolInvocation(
                    plan_id=proposal_payload.get("plan_id"),
                    server_name=str(
                        receipt_result.get("server_name")
                        or proposal_payload.get("server_name")
                        or ""
                    ),
                    tool_name=str(
                        receipt_result.get("tool_name")
                        or proposal_payload.get("tool_name")
                        or ""
                    ),
                    effect=ToolEffect(
                        str(receipt_result.get("effect") or "create")
                    ),
                    arguments=(
                        receipt_result.get("arguments")
                        or proposal_payload.get("arguments")
                        or {}
                    ),
                    planner_source=proposal_payload.get("planner_source"),
                    match_reason=proposal_payload.get("match_reason"),
                    discovery_status="success",
                    transport_status=(
                        "success" if receipt.may_claim_success else "failed"
                    ),
                    invocation_status=(
                        "success" if receipt.may_claim_success else "failed"
                    ),
                    business_status=str(
                        receipt_result.get("business_status")
                        or ("success" if receipt.may_claim_success else "unknown")
                    ),
                    latency_ms=receipt_result.get("latency_ms"),
                    result_summary=receipt_result.get("result_summary"),
                    error_summary=receipt.error_summary,
                    receipt_id=receipt.receipt_id,
                )
                state.tool_invocations.append(invocation)
                record_mcp_call_audit(
                    trace_id=trace_id,
                    server_name=invocation.server_name,
                    tool_name=invocation.tool_name,
                    arguments=invocation.arguments,
                    status=(
                        invocation.business_status
                        if invocation.invocation_status == "success"
                        else invocation.invocation_status
                    ),
                    result_summary=invocation.result_summary,
                    error_summary=invocation.error_summary,
                    latency_ms=invocation.latency_ms,
                    invocation_mode="confirmed_action",
                )
        if result.get("action") in {
            "awaiting_confirmation",
            "awaiting_parameters",
            "draft_updated",
            "confirmation_blocked",
        }:
            state.status = RunStatus.PAUSED
            state.next_step = "await_user_confirmation"
        elif result.get("action") == "failed":
            state.status = RunStatus.FAILED
            state.next_step = "retry_or_handoff"
        else:
            state.status = RunStatus.COMPLETED
            state.next_step = None

        reply = str(result.get("reply") or "")
        tool_call = {
            "tool_name": "action_gateway",
            "arguments": {
                "action_type": action_type,
                "proposal_id": proposal_id,
                "phase": result.get("action"),
            },
            "status": (
                "committed"
                if state.action_receipts
                and state.action_receipts[-1].may_claim_success
                else result.get("action")
            ),
            "receipt_id": (
                state.action_receipts[-1].receipt_id if state.action_receipts else None
            ),
            "resource_id": (
                state.action_receipts[-1].resource_id if state.action_receipts else None
            ),
        }
        controlled_mcp_payload = []
        for invocation in state.tool_invocations:
            payload = invocation.model_dump(mode="json")
            payload["status"] = (
                invocation.business_status
                if invocation.invocation_status == "success"
                else invocation.invocation_status
            )
            payload["invocation_mode"] = "confirmed_action"
            controlled_mcp_payload.append(payload)
        ledger.capture_state(state)
        ledger.append(
            "evaluation_results",
            {
                "case": "action_receipt_contract",
                "passed": (
                    not _claims_business_success(reply)
                    or bool(
                        state.action_receipts
                        and state.action_receipts[-1].may_claim_success
                    )
                ),
            },
        )
        ledger.persist(
            "paused"
            if state.status == RunStatus.PAUSED
            else ("failed" if state.status == RunStatus.FAILED else "complete")
        )
        update_chat_trace(
            trace_id,
            intent=selected_agent_id,
            agent_name=selected_agent_name,
            agent_id=selected_agent_id,
            status=(
                "failed" if state.status == RunStatus.FAILED else "complete"
            ),
        )
        record_trace_event(
            trace_id,
            "action_gateway",
            "failed" if state.status == RunStatus.FAILED else "success",
            latency_ms=int((time.time() - started) * 1000),
            output_summary=reply[:240],
            metadata={
                "proposal_id": proposal_id,
                "receipt_id": tool_call.get("receipt_id"),
                "resource_id": tool_call.get("resource_id"),
                "workflow_status": state.status.value,
            },
        )
        saved = save_chat_message(
            session_id=session_id,
            role="assistant",
            content=reply,
            token_count=0,
            round_token_count=0,
            token_detail={
                "input_tokens": None,
                "output_tokens": None,
                "cached_tokens": None,
                "reasoning_tokens": None,
                "total_tokens": None,
            },
            citations=[],
            activated_skills=[],
            route_intent=selected_agent_id,
            route_reason=route.reason,
            current_agent=selected_agent_name,
            current_agent_id=selected_agent_id,
            tool_calls=[tool_call],
            model_id=None,
            thinking_enabled=False,
            model_selection_reason="controlled_action_workflow",
            trace_id=trace_id,
            status=state.status.value,
            latency_ms=int((time.time() - started) * 1000),
            mcp_calls=controlled_mcp_payload or None,
            usage_source="not_applicable",
        )
        yield _sse(
            "route",
            {
                "intent": selected_agent_id,
                "reason": route.reason,
                "current_agent": selected_agent_name,
                "current_agent_id": selected_agent_id,
                "trace_id": trace_id,
            },
        )
        yield _sse("delta", {"content": reply})
        yield _sse("tool_calls", {"tool_calls": [tool_call]})
        yield _sse(
            "done",
            {
                "status": state.status.value,
                "message_id": saved.get("id"),
                "trace_id": trace_id,
                "runtime_path": RuntimePath.CONTROLLED_ACTION.value,
                "release_id": snapshot.release_id,
                "snapshot_id": snapshot.snapshot_id,
                "proposal_id": proposal_id,
                "action_receipts": [
                    item.model_dump(mode="json") for item in state.action_receipts
                ],
                "tool_calls": [tool_call],
                "mcp_calls": controlled_mcp_payload,
                "citations": [],
                "activated_skills": [],
                "usage_source": "not_applicable",
            },
        )

    async def _advance_dynamic_mcp_action(
        self,
        message: str,
        session_id: str,
        trace_id: str,
        snapshot: Any,
    ) -> Dict[str, Any]:
        pending = get_pending_action_proposal(session_id)
        if pending and str(pending.get("action_type") or "").startswith("mcp."):
            payload = pending.get("payload") or {}
            base = {
                "handled": True,
                "proposal_id": pending["proposal_id"],
                "action_type": pending["action_type"],
                "agent_id": payload.get("agent_id"),
                "agent_name": payload.get("agent_name") or payload.get("agent_id"),
                "route_reason": "发布快照中的写 MCP 进入受控确认路径。",
            }
            if is_cancel_request(message):
                action_gateway.reject(
                    pending["proposal_id"],
                    actor=f"owner:{session_id}",
                    comment="用户拒绝动态 MCP 写操作",
                )
                return {
                    **base,
                    "action": "rejected",
                    "reply": "已拒绝本次待确认操作；MCP 未执行，业务数据未写入。",
                }
            if not is_confirmation(message):
                return {
                    **base,
                    "action": "awaiting_confirmation",
                    "reply": (
                        f"操作 {payload.get('server_name')}/{payload.get('tool_name')} "
                        "仍在等待确认。请回复“确认提交”，或回复“拒绝”。"
                    ),
                }
            proposal = action_gateway.approve(
                pending["proposal_id"],
                actor=f"owner:{session_id}",
                comment="用户明确确认动态 MCP 写操作",
            )
            receipt = await action_gateway.execute_async(proposal.proposal_id)
            if not receipt.may_claim_success:
                return {
                    **base,
                    "action": "failed",
                    "reply": (
                        "操作未提交成功：后端没有签发包含真实资源 ID 的 committed "
                        "Receipt。不会把 MCP 调用失败包装成业务成功。"
                    ),
                    "receipt": receipt.model_dump(mode="json"),
                    "error_summary": receipt.error_summary,
                }
            return {
                **base,
                "action": "committed",
                "reply": (
                    f"操作已真实提交成功，资源 ID：{receipt.resource_id}。"
                    f"Receipt：{receipt.receipt_id}。"
                ),
                "receipt": receipt.model_dump(mode="json"),
            }

        replay = self._latest_committed_dynamic_action(session_id, message)
        if replay:
            receipt = (
                get_action_receipt_by_idempotency_key(
                    str(replay.get("idempotency_key") or "")
                )
                or {}
            )
            payload = replay.get("payload") or {}
            return {
                "handled": True,
                "proposal_id": replay.get("proposal_id"),
                "action_type": replay.get("action_type"),
                "agent_id": payload.get("agent_id"),
                "agent_name": payload.get("agent_name") or payload.get("agent_id"),
                "route_reason": "已提交动态 MCP 操作的幂等 Receipt 回放。",
                "action": "idempotent_replay",
                "reply": (
                    "该操作已提交成功，资源 ID："
                    f"{receipt.get('resource_id')}。本次重复确认未再次调用 MCP。"
                ),
                "receipt": receipt,
            }

        match = self._match_write_tool(snapshot.config, message)
        if not match:
            return {
                "handled": True,
                "action": "failed",
                "action_type": "mcp.unknown",
                "agent_id": "runtime_governor",
                "agent_name": "运行时治理器",
                "route_reason": "写工具未能唯一匹配，默认拒绝。",
                "reply": "未能在当前发布快照中唯一匹配允许的写工具，操作已拒绝。",
            }

        arguments = dict(match.get("arguments") or {})
        missing = list(match.get("missing_required") or [])
        schema_errors = list(match.get("schema_errors") or [])
        if missing or schema_errors:
            return {
                "handled": True,
                "action": "awaiting_parameters",
                "action_type": f"mcp.{match['server_name']}.{match['tool_name']}",
                "agent_id": match["agent_id"],
                "agent_name": match["agent_name"],
                "route_reason": "发布快照 ToolPlan 已匹配写工具，但参数不完整。",
                "reply": (
                    "该写操作尚未生成 Proposal，参数不符合已发布 Schema："
                    + "；".join(
                        schema_errors
                        or ["缺少参数 " + "、".join(missing)]
                    )
                    + "。请补充自然语言信息或附带 JSON 参数重新提交。"
                ),
            }
        action_type = f"mcp.{match['server_name']}.{match['tool_name']}"
        proposal = action_gateway.propose(
            session_id=session_id,
            action_type=action_type,
            payload={
                "agent_id": match["agent_id"],
                "agent_name": match["agent_name"],
                "server_name": match["server_name"],
                "tool_name": match["tool_name"],
                "arguments": arguments,
                "plan_id": match.get("plan_id"),
                "planner_source": match.get("planner_source"),
                "match_reason": match.get("match_reason"),
            },
            trace_id=trace_id,
            release_id=snapshot.release_id,
            risk_level=RiskLevel.L2,
        )
        return {
            "handled": True,
            "action": "awaiting_confirmation",
            "proposal_id": proposal.proposal_id,
            "action_type": action_type,
            "agent_id": match["agent_id"],
            "agent_name": match["agent_name"],
            "route_reason": (
                "发布快照 ToolPlan 命中写 MCP 并生成 Proposal；"
                + str(match.get("match_reason") or "")
            ),
            "reply": (
                f"已生成待确认 Proposal，尚未执行 {match['server_name']}/"
                f"{match['tool_name']}。\n\n参数：{_json(arguments)}\n\n"
                "确认无误请回复“确认提交”；取消请回复“拒绝”。"
            ),
        }

    async def _stream_consultation(
        self,
        message: str,
        session_id: str,
        user_id: str,
        trace_id: str,
        snapshot: Any,
        state: RunState,
        ledger: EvidenceLedger,
        started: float,
    ) -> AsyncIterator[str]:
        state.next_step = "route"
        if state.lane_decision is None:
            raise RuntimeError("consultation started without LaneDecision")
        if state.answer_contract is None:
            raise RuntimeError("consultation started without AnswerContract")
        visible_history = _visible_chat_history(
            session_id,
            current_trace_id=trace_id,
            rounds=5,
        )
        all_cards = vertical_agent_cards(snapshot.config)
        cards = _lane_candidates(all_cards, state.lane_decision.lane)
        if not cards:
            async for event in self._stream_unconfigured_lane_boundary(
                session_id, trace_id, snapshot, state, ledger, started
            ):
                yield event
            return

        candidates = [str(item["agent_id"]) for item in cards]
        selection = await self._select_agent_after_lane(
            message,
            session_id,
            user_id,
            trace_id,
            snapshot,
            state,
            all_cards,
            visible_history,
        )
        selected = str(selection.get("selected_agent_id") or "")
        if selected not in candidates:
            async for event in self._stream_unconfigured_lane_boundary(
                session_id, trace_id, snapshot, state, ledger, started
            ):
                yield event
            return
        selected_card = next(
            item for item in cards if str(item.get("agent_id")) == selected
        )
        property_query = state.lane_decision.lane == RuntimeLane.PROPERTY_GOVERNED
        domain_scope = _agent_domain_scope(selected_card)
        out_of_scope_without_agent = False
        route = RouteDecision(
            candidates=candidates,
            selected_agent_id=selected,
            reason=str(selection.get("reason") or state.lane_decision.reason or "同域Agent选择完成。"),
            confidence=None,
            required_capability_types=["agent", "skill", "rag", "readonly_tool"],
        )
        state.route_decision = route
        state.selected_agent = next(
            item for item in snapshot.config["agents"] if item.get("agent_id") == selected
        )
        yield _sse(
            "route",
            {
                "intent": selected,
                "reason": route.reason,
                "current_agent": state.selected_agent.get("name"),
                "current_agent_id": selected,
                "domain_scope": domain_scope,
                "lane": state.lane_decision.lane.value,
                "business_intent": state.lane_decision.business_intent,
                "trace_id": trace_id,
            },
        )

        router_cost = next(
            (item for item in state.cost_entries if item.stage == "router"),
            None,
        )
        if router_cost is None:
            raise RuntimeError("consultation has no accounted semantic Router request")
        handoff_policy = {"reason_code": "semantic_no_handoff"}
        read_tool_plans = (
            plan_tools(
                snapshot.config,
                selected,
                message,
                RuntimePath.CONSULTATION,
                effects=[ToolEffect.READ],
                execution_modes=["auto_preinvoke", "model_native"],
            )
            if property_query
            and state.answer_contract.tool_policy == "selected"
            else []
        )
        structured_realtime_query = _is_structured_realtime_query(
            state.answer_contract,
            read_tool_plans,
        )
        direct_knowledge_required = bool(
            state.answer_contract.evidence_required
            and state.answer_contract.response_mode == ResponseMode.GROUNDED_ANSWER
        )
        state.next_step = "retrieve"
        retrieval_started = time.time()
        allowed_doc_ids = (
            {
                int(item) for item in state.selected_agent.get("knowledge_doc_ids") or []
            }
            if property_query and state.answer_contract.rag_policy == "selected"
            else set()
        )
        knowledge_versions = {
            int(item["knowledge_doc_id"]): item
            for item in snapshot.config.get("knowledge") or []
        }
        results: List[Dict[str, Any]] = []
        retrieval: Dict[str, Any] = {}
        used_snapshot_fallback = False
        retrieval_status = (
            "skipped_structured_realtime_query"
            if structured_realtime_query
            else (
                "skipped_no_bound_knowledge"
                if direct_knowledge_required and not allowed_doc_ids
                else "not_requested"
            )
        )
        if property_query and not structured_realtime_query:
            yield _sse(
                "progress",
                {"trace_id": trace_id, "stage": "rag.retrieve", "status": "running"},
            )
        if property_query and allowed_doc_ids and not structured_realtime_query:
            try:
                import rag_retrieval

                retrieval = await asyncio.to_thread(
                    rag_retrieval.advanced_search,
                    message,
                    snapshot.config.get("retrieval_policy") or {},
                    allowed_document_ids=sorted(allowed_doc_ids),
                )
                results = list((retrieval or {}).get("results") or [])
                results, used_snapshot_fallback = _results_from_snapshot(
                    message,
                    results,
                    knowledge_versions,
                    allowed_doc_ids,
                    int(
                        (snapshot.config.get("retrieval_policy") or {}).get("top_k")
                        or 5
                    ),
                    float(
                        (snapshot.config.get("retrieval_policy") or {}).get(
                            "context_threshold"
                        )
                        or 0.2
                    ),
                )
                retrieval_status = (
                    "completed_snapshot_fallback"
                    if used_snapshot_fallback
                    else "completed"
                )
            except Exception as exc:
                results, _ = _results_from_snapshot(
                    message,
                    [],
                    knowledge_versions,
                    allowed_doc_ids,
                    int(
                        (snapshot.config.get("retrieval_policy") or {}).get("top_k")
                        or 5
                    ),
                    float(
                        (snapshot.config.get("retrieval_policy") or {}).get(
                            "context_threshold"
                        )
                        or 0.2
                    ),
                )
                retrieval_status = (
                    "completed_snapshot_fallback" if results else "failed"
                )
                ledger.violation(
                    "live_retrieval_failed",
                    str(exc),
                    snapshot_fallback_count=len(results),
                )
        evidence = build_evidence_set(
            message,
            results,
            knowledge_versions=knowledge_versions,
            allowed_document_ids=allowed_doc_ids,
            retrieval_status=retrieval_status,
        )
        state.retrieval_evidence = evidence
        record_trace_event(
            trace_id,
            "retrieval",
            (
                "failed"
                if retrieval_status == "failed"
                else (
                    "skipped"
                    if retrieval_status
                    in {
                        "skipped_structured_realtime_query",
                        "skipped_no_bound_knowledge",
                    }
                    else "success"
                )
            ),
            latency_ms=int((time.time() - retrieval_started) * 1000),
            output_summary=f"{len(evidence.items)} evidence items",
            metadata={
                "snapshot_id": snapshot.snapshot_id,
                "allowed_document_ids": sorted(allowed_doc_ids),
                "evidence_ids": [item.evidence_id for item in evidence.items],
                "evidence": [
                    {
                        "evidence_id": item.evidence_id,
                        "document_id": item.document_id,
                        "chunk_index": item.chunk_index,
                        "retrieval_score": item.retrieval_score,
                        "retrieval_mode": item.retrieval_mode,
                    }
                    for item in evidence.items
                ],
                "retrieval_status": retrieval_status,
                "snapshot_fallback": used_snapshot_fallback,
                "evidence_policy": (retrieval or {}).get("evidence_policy"),
                "filter_summary": (retrieval or {}).get("filter_summary"),
                "direct_knowledge_required": direct_knowledge_required,
                "decision_reason": (
                    "structured_realtime_query"
                    if structured_realtime_query
                    else (
                        "knowledge_evidence_required"
                        if direct_knowledge_required
                        else "knowledge_evidence_not_required"
                    )
                ),
            },
        )

        state.next_step = "readonly_mcp"
        if read_tool_plans:
            yield _sse(
                "progress",
                {"trace_id": trace_id, "stage": "mcp.invoke", "status": "running"},
            )
        if property_query and read_tool_plans:
            mcp_context, invocations = await preinvoke_read_tools(
                snapshot.config, selected, message
            )
        else:
            mcp_context, invocations = "", []
        preinvoked_tools = {
            (invocation.server_name, invocation.tool_name)
            for invocation in invocations
            if invocation.tool_name != "discovery"
            and invocation.invocation_status == "success"
        }
        model_native_toolkits = (
            build_model_native_read_tools(
                snapshot.config,
                selected,
                message,
                excluded_tools=preinvoked_tools,
            )
            if property_query and read_tool_plans
            else []
        )
        state.tool_invocations = list(invocations)
        for invocation in invocations:
            record_trace_event(
                trace_id,
                f"mcp.{invocation.server_name}.{invocation.tool_name}",
                (
                    "success"
                    if invocation.invocation_status == "success"
                    and invocation.business_status == "success"
                    else "failed"
                ),
                output_summary=invocation.result_summary or invocation.error_summary,
                metadata=invocation.model_dump(mode="json"),
            )
            audit_status = (
                invocation.business_status
                if invocation.invocation_status == "success"
                else (
                    invocation.transport_status
                    if invocation.transport_status in {"timeout", "failed"}
                    else invocation.invocation_status
                )
            )
            record_mcp_call_audit(
                trace_id=trace_id,
                server_name=invocation.server_name,
                tool_name=invocation.tool_name,
                arguments=invocation.arguments,
                status=audit_status,
                result_summary=invocation.result_summary,
                error_summary=invocation.error_summary,
                latency_ms=invocation.latency_ms,
                invocation_mode="policy_preinvoke",
            )

        evidence_prompt = prompt_evidence_allowlist(evidence) if property_query else ""
        answer_boundary = (
            "\n[回答边界] 这是隔离通用回答。不得调用或声称使用物业Skill、RAG、MCP/Tool、ActionGateway，"
            "不得把一般建议表述为物业官方事实。"
            if not property_query
            else "\n[回答边界] 物业具体事实必须严格来自下方合法Evidence；没有依据不得补充流程、地点、时限或能力。"
        )
        build = build_agent_from_snapshot(
            snapshot,
            selected,
            message,
            tools=model_native_toolkits,
            evidence_prompt=evidence_prompt + mcp_context + answer_boundary,
            enable_skills=(
                property_query
                and not structured_realtime_query
                and state.answer_contract.skill_policy == "selected"
            ),
        )
        state.activated_skills = build.activated_skills
        for call in build.skill_tool_calls:
            record_trace_event(
                trace_id,
                f"skill.{call['skill_id']}.get_skill_instructions",
                "success",
                output_summary=(
                    f"loaded Skill {call['skill_id']} "
                    f"version={call['skill_version']}"
                ),
                metadata=call,
            )
        successful_tool_evidence = [
            invocation
            for invocation in invocations
            if invocation.invocation_status == "success"
            and invocation.business_status == "success"
        ]
        knowledge_gate = _knowledge_evidence_decision(
            state.answer_contract,
            len(evidence.items),
            structured_realtime_query,
            allowed_doc_ids,
            domain_scope=domain_scope,
            skill_evidence_count=len(build.skill_evidence_sources),
            tool_evidence_count=len(successful_tool_evidence),
        )
        direct_knowledge_required = bool(knowledge_gate["required"])
        knowledge_evidence_blocked = bool(knowledge_gate["blocked"])
        handoff_reason_code = str(
            handoff_policy.get("reason_code") or "ai_direct"
        )
        decision_summary = {
            "agent": _decision(
                "selected",
                "matched_intent",
                agent_id=selected,
                domain_scope=domain_scope,
                property_query=property_query,
                answer_channel=(
                    "property_governed"
                    if domain_scope == "property"
                    else "isolated_general"
                ),
            ),
            "skill": _decision(
                "skipped" if not build.activated_skills else "selected",
                (
                    "structured_realtime_query"
                    if structured_realtime_query
                    else (
                        "matched_intent"
                        if build.activated_skills
                        else "no_match"
                    )
                ),
                skill_ids=[
                    item.skill_id for item in build.activated_skills
                ],
            ),
            "rag": _decision(
                (
                    "skipped"
                    if structured_realtime_query or not allowed_doc_ids
                    else "selected"
                ),
                (
                    "structured_realtime_query"
                    if structured_realtime_query
                    else (
                        "knowledge_evidence_required"
                        if allowed_doc_ids
                        else "no_bound_knowledge"
                    )
                ),
                retrieval_status=retrieval_status,
                evidence_count=len(evidence.items),
                skill_evidence_count=len(build.skill_evidence_sources),
                tool_evidence_count=len(successful_tool_evidence),
                direct_knowledge_required=direct_knowledge_required,
                evidence_decision=knowledge_gate["evidence_decision"],
            ),
            "tool": _decision(
                "selected" if read_tool_plans else "skipped",
                (
                    "exact_workorder_lookup"
                    if structured_realtime_query
                    else ("matched_intent" if read_tool_plans else "not_required")
                ),
                tools=[
                    f"{plan.server_name}/{plan.tool_name}"
                    for plan in read_tool_plans
                ],
            ),
            "handoff": _decision(
                "skipped",
                (
                    "negated_by_user"
                    if handoff_reason_code == "negated_by_user"
                    else "no_handoff_intent"
                ),
                policy_reason=handoff_reason_code,
            ),
        }
        state.capability_decision = CapabilityDecision(
            selected_agent_id=selected,
            skill={
                "status": "selected" if build.activated_skills else "skipped",
                "reason_code": "matched_intent" if build.activated_skills else "no_match",
                "details": {
                    "skill_ids": [item.skill_id for item in build.activated_skills]
                },
            },
            rag={
                "status": (
                    "selected"
                    if allowed_doc_ids and not structured_realtime_query
                    else "skipped"
                ),
                "reason_code": (
                    "knowledge_evidence_required"
                    if allowed_doc_ids and not structured_realtime_query
                    else (
                        "structured_realtime_query"
                        if structured_realtime_query
                        else "no_bound_knowledge"
                    )
                ),
                "details": {
                    "retrieval_status": retrieval_status,
                    "evidence_count": len(evidence.items),
                },
            },
            tool={
                "status": "selected" if read_tool_plans else "skipped",
                "reason_code": (
                    "exact_workorder_lookup"
                    if structured_realtime_query
                    else ("matched_intent" if read_tool_plans else "not_required")
                ),
                "details": {
                    "tools": [
                        f"{plan.server_name}/{plan.tool_name}"
                        for plan in read_tool_plans
                    ]
                },
            },
            write={"status": "not_required", "reason_code": "consultation_path"},
            handoff={
                "status": "available",
                "reason_code": (
                    "negated_by_user"
                    if handoff_reason_code == "negated_by_user"
                    else "owner_can_request"
                ),
            },
        )
        ledger.append(
            "evaluation_results",
            {
                "case": "capability_decision",
                "passed": True,
                "decision_summary": decision_summary,
            },
        )
        record_trace_event(
            trace_id,
            "capability_decision",
            "success",
            output_summary=(
                f"agent={selected}; "
                f"skill={decision_summary['skill']['status']}; "
                f"rag={decision_summary['rag']['status']}; "
                f"tool={decision_summary['tool']['status']}; "
                "handoff=skipped"
            ),
            metadata={"decision_summary": decision_summary},
        )
        state.next_step = "answer"
        model_invoked = bool(
            knowledge_gate["model_invoked"] and not out_of_scope_without_agent
        )
        if knowledge_evidence_blocked or out_of_scope_without_agent:
            record_trace_event(
                trace_id,
                "evidence_gate",
                "success",
                output_summary=(
                    "no matching isolated Agent; vertical model skipped"
                    if out_of_scope_without_agent
                    else "knowledge evidence insufficient; vertical model skipped"
                ),
                metadata={
                    "direct_knowledge_required": direct_knowledge_required,
                    "decision": (
                        "out_of_scope_no_matching_agent"
                        if out_of_scope_without_agent
                        else knowledge_gate["evidence_decision"]
                    ),
                    "reason": (
                        "no_matching_isolated_general_agent"
                        if out_of_scope_without_agent
                        else knowledge_gate["reason"]
                    ),
                    "retrieval_status": retrieval_status,
                    "evidence_count": 0,
                    "allowed_document_ids": knowledge_gate[
                        "allowed_document_ids"
                    ],
                    "model_invoked": model_invoked,
                },
            )
            yield _sse(
                "progress",
                {
                    "trace_id": trace_id,
                    "stage": "evidence.gate",
                    "status": "completed",
                },
            )
        else:
            yield _sse(
                "progress",
                {"trace_id": trace_id, "stage": "model.invoke", "status": "running"},
            )
        agent_started = time.time()
        contextual_message = _render_visible_history_context(
            visible_history,
            message,
            boundary=(
                "本轮是只读咨询路径。只回答本轮用户问题；不得输出 Router、LaneDecision、"
                "CapabilityDecision 或其他控制 JSON；不得创建工单、草稿或待确认 Action。"
            ),
        )
        full_content = ""
        provisional_buffer = ""
        tool_calls: List[Dict[str, Any]] = list(build.skill_tool_calls)
        final_metrics: Dict[str, Optional[int]] = {}
        final_provider_evidence: Dict[str, Any] = {
            "provider_response_model": None,
            "provider_request_id": None,
            "usage": {},
        }
        last_progress_at = time.time()
        model_id = str(
            state.selected_agent.get("model_id")
            or ((snapshot.config.get("model_policy") or {}).get("default") or {}).get(
                "model_id"
            )
            or MODEL_ID
        )
        accounting_context = (
            provider_accounting_scope(
                trace_id=trace_id,
                session_id=session_id,
                stage="vertical_agent",
                model_selection_reason=f"agent model from snapshot:{selected}",
                price_snapshot=_price_for_snapshot(snapshot.config, model_id),
                model_policy_version=str(
                    (snapshot.config.get("model_policy") or {}).get("version") or "v1.8"
                ),
            )
            if model_invoked
            else nullcontext(None)
        )
        with accounting_context as provider_scope:
            response_stream = (
                build.agent.arun(
                    contextual_message,
                    user_id=user_id,
                    session_id=f"{session_id}::vertical::{selected}::{trace_id}",
                    stream=True,
                    stream_events=True,
                )
                if model_invoked
                else _static_response_stream(
                    OUT_OF_SCOPE_RESPONSE
                    if out_of_scope_without_agent
                    else KNOWLEDGE_INSUFFICIENT_RESPONSE
                )
            )
            async for chunk in response_stream:
                content = getattr(chunk, "content", None) or getattr(chunk, "delta", None)
                event_name = str(getattr(chunk, "event", "") or "").lower()
                # RunCompleted carries the full answer again together with the
                # final metrics.  Capture its metrics but do not append its content
                # a second time.
                if content and "completed" not in event_name:
                    content_delta = str(content)
                    full_content += content_delta
                    provisional_buffer += content_delta
                    # Buffer until the complete answer passes control-payload and
                    # evidence validation. Internal JSON must never leak as a delta.
                for call in _extract_tool_calls(chunk):
                    if call not in tool_calls:
                        tool_calls.append(call)
                metrics = _metrics_dict(chunk)
                if metrics:
                    final_metrics.update(metrics)
                chunk_evidence = provider_evidence_from_run(chunk)
                merge_non_null(final_provider_evidence, chunk_evidence)
                if time.time() - last_progress_at >= 12:
                    yield _sse(
                        "progress",
                        {
                            "trace_id": trace_id,
                            "stage": "model.invoke",
                            "status": "running",
                        },
                    )
                    last_progress_at = time.time()

        provider_requests = list(provider_scope.attempts) if provider_scope else []

        provider_failure_reason = _provider_failure_reason(full_content)
        if _is_internal_control_payload(full_content):
            ledger.violation(
                "internal_control_payload_leak",
                "A business Agent returned control-plane JSON at the user-answer boundary.",
                selected_agent_id=selected,
                lane=state.lane_decision.lane.value,
            )
            record_trace_event(
                trace_id,
                "internal_control_payload_leak",
                "blocked",
                output_summary="control-plane JSON blocked before user delivery",
                metadata={
                    "selected_agent_id": selected,
                    "lane": state.lane_decision.lane.value,
                },
            )
            raise InternalControlPayloadLeakError(
                "business Agent returned an internal control payload"
            )
        model_native_invocations = []
        for toolkit in model_native_toolkits:
            model_native_invocations.extend(
                list(getattr(toolkit, "recorded_invocations", []) or [])
            )
            if hasattr(toolkit, "close"):
                try:
                    await asyncio.wait_for(toolkit.close(), timeout=3)
                except Exception:
                    pass
        state.tool_invocations.extend(model_native_invocations)
        for invocation in model_native_invocations:
            record_trace_event(
                trace_id,
                f"mcp.{invocation.server_name}.{invocation.tool_name}",
                (
                    "success"
                    if invocation.invocation_status == "success"
                    and invocation.business_status == "success"
                    else "failed"
                ),
                latency_ms=invocation.latency_ms,
                output_summary=invocation.result_summary or invocation.error_summary,
                metadata=invocation.model_dump(mode="json"),
            )
            record_mcp_call_audit(
                trace_id=trace_id,
                server_name=invocation.server_name,
                tool_name=invocation.tool_name,
                arguments=invocation.arguments,
                status=(
                    invocation.business_status
                    if invocation.invocation_status == "success"
                    else invocation.invocation_status
                ),
                result_summary=invocation.result_summary,
                error_summary=invocation.error_summary,
                latency_ms=invocation.latency_ms,
                invocation_mode="model_native",
            )

        if provider_failure_reason:
            raise ProviderFailureError(
                f"model Provider returned failure text: {provider_failure_reason}"
            )

        loaded_skill_tool = any(
            call.get("tool_name") == "get_skill_instructions" for call in tool_calls
        )
        if build.activated_skills and not loaded_skill_tool:
            ledger.violation(
                "skill_selected_not_loaded",
                "Skill trigger matched, but Agno get_skill_instructions was not observed.",
                selected_skill_ids=[
                    item.skill_id for item in build.activated_skills
                ],
            )

        rendered, citations, citation_violations = render_citations(
            full_content,
            evidence,
            tool_invocations=state.tool_invocations,
            skill_sources=build.skill_evidence_sources,
            action_receipts=state.action_receipts,
        )
        linked_skill_evidence = build_skill_evidence(
            full_content,
            build.skill_evidence_sources,
        )
        citation_required = _requires_rag_citation(
            state.answer_contract,
            evidence_count=len(evidence.items),
            linked_skill_evidence_count=len(linked_skill_evidence),
            successful_tool_evidence_count=len(successful_tool_evidence),
        )
        if citation_required and not citations:
            citation_violations.append(
                {
                    "code": "required_citation_missing",
                    "detail": (
                        "The user explicitly requested RAG citations, but no "
                        "validated EvidenceItem was linked in the answer."
                    ),
                }
            )
        answer_has_governed_evidence = bool(
            citations or linked_skill_evidence or successful_tool_evidence
        )
        knowledge_grounding_failed = bool(
            direct_knowledge_required
            and (citation_violations or not answer_has_governed_evidence)
        )
        if knowledge_grounding_failed:
            rendered = KNOWLEDGE_INSUFFICIENT_RESPONSE
            citations = []
        state.citations = citations
        _record_citation_violations(ledger, citation_violations)
        rendered = _append_runtime_evidence_summary(
            rendered,
            message,
            tool_calls,
            state.tool_invocations,
            citations,
        )

        vertical_cost_entries = []
        vertical_usage_sources: List[str] = []
        vertical_usage_source = "not_applicable"
        if model_invoked:
            vertical_thinking = _thinking_for_snapshot(snapshot.config, model_id)
            for index, request_evidence in enumerate(provider_requests, start=1):
                sequence = int(
                    request_evidence.get("provider_request_sequence") or index
                )
                request_id = request_evidence.get("provider_request_id")
                vertical_usage = dict(request_evidence.get("usage") or {})
                vertical_response_model = request_evidence.get(
                    "provider_response_model"
                )
                vertical_billing_model = vertical_response_model or model_id
                vertical_cost = build_cost_entry(
                    stage="vertical_agent",
                    provider=_model_provider(snapshot.config, model_id),
                    requested_model=model_id,
                    response_model=None,
                    provider_response_model=vertical_response_model,
                    thinking_enabled=vertical_thinking,
                    model_policy_version=str(
                        (snapshot.config.get("model_policy") or {}).get("version")
                        or "v1.8"
                    ),
                    provider_usage=vertical_usage if vertical_usage else None,
                    price_row=_price_for_snapshot(
                        snapshot.config,
                        vertical_billing_model,
                    ),
                    local_estimate_tokens=None,
                )
                vertical_usage_source = vertical_cost.usage_source.value
                vertical_cost_entries.append(vertical_cost)
                vertical_usage_sources.append(vertical_usage_source)
                state.cost_entries.append(vertical_cost)
                state.model_calls.append(
                    {
                        "stage": "vertical_agent",
                        "model_id": model_id,
                        "requested_model": model_id,
                        "provider_response_model": vertical_response_model,
                        "provider_request_id": request_id,
                        "provider_request_sequence": sequence,
                        "thinking_enabled": vertical_thinking,
                        "latency_ms": None,
                        "usage": vertical_usage,
                        "usage_source": vertical_usage_source,
                        "status": request_evidence.get("status") or "success",
                    }
                )

        state.status = RunStatus.COMPLETED
        state.next_step = None
        ledger.capture_state(state)
        ledger.set(
            "skill_evidence",
            linked_skill_evidence,
        )
        ledger.append(
            "evaluation_results",
            {
                "case": "consultation_no_write",
                "passed": not state.pending_actions and not state.action_receipts,
            },
        )
        ledger.append(
            "evaluation_results",
            {
                "case": "citation_allowlist",
                "passed": not citation_violations,
                "violations": citation_violations,
            },
        )
        ledger.append(
            "evaluation_results",
            {
                "case": "required_rag_citation",
                "passed": (not citation_required) or bool(citations),
                "required": citation_required,
                "citation_count": len(citations),
            },
        )
        ledger.append(
            "evaluation_results",
            {
                "case": "knowledge_evidence_gate",
                "passed": (
                    not direct_knowledge_required
                    or bool(citations)
                    or bool(linked_skill_evidence)
                    or rendered.startswith("当前知识依据不足")
                ),
                "required": direct_knowledge_required,
                "evidence_count": len(evidence.items),
                "skill_evidence_count": len(build.skill_evidence_sources),
                "tool_evidence_count": len(successful_tool_evidence),
                "domain_scope": domain_scope,
                "model_invoked": model_invoked,
                "decision": (
                    "rejected_insufficient"
                    if rendered.startswith("当前知识依据不足")
                    else "answered_with_evidence"
                ),
            },
        )
        ledger.persist("complete")
        update_chat_trace(
            trace_id,
            intent=selected,
            agent_name=str(state.selected_agent.get("name") or selected),
            agent_id=selected,
            status="complete",
        )
        record_trace_event(
            trace_id,
            "final_response",
            "success",
            latency_ms=int((time.time() - started) * 1000),
            output_summary=rendered[:240],
            metadata={
                "snapshot_id": snapshot.snapshot_id,
                "activated_skill_ids": [
                    item.skill_id for item in state.activated_skills
                ],
                "skill_evidence": ledger.contract.skill_evidence,
                "evidence_ids": [item.evidence_id for item in evidence.items],
                "citation_evidence_ids": [
                    item.evidence_id for item in citations
                ],
                "mcp_invocation_ids": [
                    item.invocation_id for item in state.tool_invocations
                ],
                "decision_summary": decision_summary,
                "evidence_decision": (
                    "rejected_insufficient"
                    if rendered.startswith("当前知识依据不足")
                    else "answered_with_evidence"
                ),
                "citation_violations": citation_violations,
                "model_invoked": model_invoked,
                "domain_scope": domain_scope,
                "lane_decision": state.lane_decision.model_dump(mode="json"),
                "answer_contract": state.answer_contract.model_dump(mode="json"),
                "capability_decision": state.capability_decision.model_dump(mode="json"),
            },
        )

        citations_payload = []
        for item in citations:
            payload = item.model_dump(mode="json")
            payload.update(
                {
                    "doc_id": item.document_id,
                    "doc_title": item.title,
                    "content": item.content_snapshot,
                    "used_in_answer": True,
                }
            )
            citations_payload.append(payload)
        skills_payload = [
            item.model_dump(mode="json") for item in state.activated_skills
        ]
        skill_evidence_by_id = {
            int(item["skill_id"]): item
            for item in ledger.contract.skill_evidence
            if item.get("skill_id") is not None
        }
        for item in skills_payload:
            evidence_item = skill_evidence_by_id.get(int(item.get("skill_id") or 0))
            if evidence_item:
                item["evidence"] = evidence_item
        mcp_payload = []
        for item in state.tool_invocations:
            payload = item.model_dump(mode="json")
            payload["status"] = (
                item.business_status
                if item.invocation_status == "success"
                else (
                    item.transport_status
                    if item.transport_status in {"timeout", "failed"}
                    else item.invocation_status
                )
            )
            payload["invocation_mode"] = (
                "model_native"
                if item in model_native_invocations
                else "policy_preinvoke"
            )
            mcp_payload.append(payload)
        vertical_usage_source = _aggregate_usage_source(
            vertical_usage_sources,
            model_invoked=model_invoked,
        )
        vertical_total_tokens = _aggregate_cost_field(
            vertical_cost_entries,
            "total_tokens",
        )
        token_count = vertical_total_tokens or 0
        vertical_token_detail = (
            {
                "input_tokens": _aggregate_cost_field(vertical_cost_entries, "input_tokens"),
                "output_tokens": _aggregate_cost_field(vertical_cost_entries, "output_tokens"),
                "reasoning_tokens": _aggregate_cost_field(vertical_cost_entries, "reasoning_tokens"),
                "cached_tokens": _aggregate_cost_field(vertical_cost_entries, "cached_input_tokens"),
                "total_tokens": vertical_total_tokens,
                "local_estimate_tokens": None,
                "provider_request_count": len(vertical_cost_entries),
                "model_invoked": True,
            }
            if model_invoked
            else {
                "input_tokens": None,
                "output_tokens": None,
                "reasoning_tokens": None,
                "cached_tokens": None,
                "total_tokens": None,
                "local_estimate_tokens": None,
                "provider_request_count": 0,
                "model_invoked": False,
            }
        )
        saved = save_chat_message(
            session_id=session_id,
            role="assistant",
            content=rendered,
            token_count=token_count,
            round_token_count=(router_cost.total_tokens or 0) + token_count,
            token_detail=vertical_token_detail,
            citations=citations_payload,
            activated_skills=skills_payload,
            route_intent=selected,
            route_reason=route.reason,
            current_agent=str(state.selected_agent.get("name") or selected),
            current_agent_id=selected,
            tool_calls=tool_calls or None,
            model_id=model_id if model_invoked else None,
            thinking_enabled=(
                _thinking_for_snapshot(snapshot.config, model_id)
                if model_invoked
                else False
            ),
            model_selection_reason=f"published snapshot:{snapshot.release_id}",
            trace_id=trace_id,
            status="success",
            latency_ms=int((time.time() - agent_started) * 1000),
            mcp_calls=mcp_payload or None,
            usage_source=vertical_usage_source,
        )
        auto_badcase = capture_runtime_badcase(
            ledger=ledger.contract,
            original_query=message,
            ai_response=rendered,
            source_message_id=saved.get("id"),
            delivery_context={
                "normal_completed": True,
                "safe_rejection": rendered.startswith("当前知识依据不足"),
                "renderer_intercepted": knowledge_grounding_failed,
            },
        )
        if ledger.contract.system_observations:
            ledger.persist("complete")
        if auto_badcase:
            ledger.append(
                "badcase_links",
                {
                    "badcase_id": auto_badcase.get("id"),
                    "source": auto_badcase.get("source"),
                    "trigger": "runtime_evidence",
                },
            )
            ledger.persist("complete")

        # Replace provisional stream text with the citation-validated answer.
        # This preserves one authoritative EvidenceSet for final text,
        # citations, clickable snapshots and Trace retrieval evidence.
        yield _sse(
            "final",
            {
                "content": rendered,
                "citations": citations_payload,
                "current_agent": state.selected_agent.get("name"),
                "current_agent_id": selected,
                "domain_scope": domain_scope,
            },
        )
        if tool_calls:
            yield _sse("tool_calls", {"tool_calls": tool_calls})
        yield _sse(
            "done",
            {
                "status": "complete",
                "message_id": saved.get("id"),
                "trace_id": trace_id,
                "runtime_path": RuntimePath.CONSULTATION.value,
                "release_id": snapshot.release_id,
                "snapshot_id": snapshot.snapshot_id,
                "current_agent": state.selected_agent.get("name"),
                "current_agent_id": selected,
                "route_intent": selected,
                "route_reason": route.reason,
                "content": rendered,
                "citations": citations_payload,
                "activated_skills": skills_payload,
                "tool_calls": tool_calls,
                "mcp_calls": mcp_payload,
                "decision_summary": decision_summary,
                "lane_decision": state.lane_decision.model_dump(mode="json"),
                "answer_contract": state.answer_contract.model_dump(mode="json"),
                "capability_decision": state.capability_decision.model_dump(mode="json"),
                "cost_entries": [
                    item.model_dump(mode="json") for item in state.cost_entries
                ],
                "usage_source": vertical_usage_source,
                "token_count": token_count,
                "vertical_provider_request_count": len(vertical_cost_entries),
                "round_token_count": (router_cost.total_tokens or 0) + token_count,
                "auto_badcase_id": (
                    auto_badcase.get("id") if auto_badcase else None
                ),
            },
        )
