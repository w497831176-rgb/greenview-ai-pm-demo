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

from app.runtime.agent_factory import (
    build_agent_from_snapshot,
    resolve_model_used_skills,
    router_agent_cards,
    vertical_agent_cards,
)
from app.runtime.badcase_capture import capture_runtime_badcase
from app.runtime.citation_renderer import (
    build_skill_evidence,
    build_evidence_set,
    prompt_evidence_allowlist,
    render_citations,
    render_rag_citations,
)
from app.runtime.contracts import (
    ActionProposal,
    ActionReceipt,
    AgentTurnResult,
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
    WORK_ORDER_CREATE_INTENT,
    _is_draft_follow_up,
    action_gateway,
    advance_work_order_workflow,
    apply_structured_proposal_request,
    is_cancel_request,
    is_confirmation,
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


class RouterContractInvalidError(RuntimeError):
    """The sole Router returned an invalid structured decision."""


class AgentContractInvalidError(RuntimeError):
    """The frozen B/C Agent returned an invalid structured envelope."""


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
    def field(raw: Any, name: str) -> Any:
        return raw.get(name) if isinstance(raw, dict) else getattr(raw, name, None)

    def arguments_from(raw: Any) -> Dict[str, Any]:
        nested = field(raw, "function")
        arguments = (
            field(raw, "tool_args")
            or field(raw, "arguments")
            or field(raw, "args")
            or (field(nested, "arguments") if nested is not None else None)
            or {}
        )
        if hasattr(arguments, "model_dump"):
            arguments = arguments.model_dump()
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
            arguments = parsed if isinstance(parsed, dict) else {"value": arguments}
        elif not isinstance(arguments, dict):
            arguments = {"value": str(arguments)}
        return arguments

    calls: List[Dict[str, Any]] = []
    candidate = getattr(value, "run_response", None) or value
    raw_calls = getattr(candidate, "tool_calls", None) or getattr(candidate, "tools", None) or []
    for raw in raw_calls:
        nested = field(raw, "function")
        name = (
            field(raw, "tool_name")
            or field(raw, "name")
            or field(raw, "tool")
            or (field(nested, "name") if nested is not None else None)
            or ""
        )
        error = field(raw, "tool_call_error") or field(raw, "error")
        status_value = field(raw, "status")
        if hasattr(status_value, "value"):
            status_value = status_value.value
        raw_status = str(status_value or "").strip().lower().rsplit(".", 1)[-1]
        if error or raw_status in {"failed", "error", "cancelled", "canceled"}:
            status = "failed"
        elif raw_status in {"success", "completed", "complete"} or not raw_status:
            status = "success"
        else:
            status = raw_status
        result = field(raw, "result")
        item = {
            "tool_name": str(name),
            "arguments": arguments_from(raw),
            "status": status,
        }
        tool_call_id = (
            field(raw, "tool_call_id")
            or field(raw, "call_id")
            or field(raw, "id")
        )
        if tool_call_id:
            item["tool_call_id"] = str(tool_call_id)
        if error:
            item["error_summary"] = str(error)
        if result is not None:
            item["result_summary"] = str(result)[:500]
        if item not in calls:
            calls.append(item)
    return calls


async def _completed_agent_run_output(agent: Any, session_id: str) -> Any:
    """Load the completed output for this exact frozen-Agent run, if stored.

    Agno's streaming events do not consistently carry the final structured
    content or the accumulated ToolExecution list.  The completed RunOutput is
    the authoritative artifact of the same run; reading it does not invoke a
    Provider or introduce another Agent decision.
    """

    async_getter = getattr(agent, "aget_last_run_output", None)
    if callable(async_getter):
        return await async_getter(session_id=session_id)
    sync_getter = getattr(agent, "get_last_run_output", None)
    if callable(sync_getter):
        return await asyncio.to_thread(sync_getter, session_id=session_id)
    return None


def _completed_agent_run_content(run_output: Any) -> str:
    """Serialize a completed RunOutput without leaking Pydantic repr syntax."""

    content_getter = getattr(run_output, "get_content_as_string", None)
    if callable(content_getter):
        try:
            return str(content_getter(exclude_none=False) or "")
        except TypeError:
            return str(content_getter() or "")
    content = getattr(run_output, "content", None)
    model_dumper = getattr(content, "model_dump_json", None)
    if callable(model_dumper):
        return str(model_dumper(exclude_none=False))
    if isinstance(content, (dict, list)):
        return _json(content)
    return str(content or "")


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
    if lane == RuntimeLane.HANDOFF:
        return []
    scope = "property" if lane == RuntimeLane.PROPERTY_GOVERNED else "isolated_general"
    return [card for card in cards if _agent_domain_scope(card) == scope]


def _visible_chat_history(
    session_id: str,
    *,
    current_trace_id: Optional[str] = None,
    rounds: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Return all visible bubbles verbatim, chronologically and timestamped.

    The compatibility parameters never remove, truncate, reprioritise, or
    summarise a bubble. The current user bubble is simply the final item.
    """

    visible: List[Dict[str, str]] = []
    for item in list_chat_messages(session_id):
        role = str(item.get("role") or "").lower()
        status = str(item.get("status") or "success").lower()
        content = str(item.get("content") or "")
        if role not in {"user", "owner", "assistant"} or not content:
            continue
        if status not in {"success", "complete", "completed"}:
            continue
        visible.append(
            {
                "role": "user" if role in {"user", "owner"} else "assistant",
                "content": content,
                "timestamp": str(item.get("created_at") or item.get("timestamp") or ""),
            }
        )
    return visible


def _render_visible_history_context(
    history: List[Dict[str, str]],
    current_message: str,
    *,
    boundary: str,
) -> str:
    lines = [f"[execution boundary] {boundary}", "[complete visible session]"]
    for item in history:
        lines.append(
            json.dumps(
                {
                    "role": item.get("role"),
                    "content": item.get("content", ""),
                    "timestamp": item.get("timestamp", ""),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def _build_retrieval_queries(history: List[Dict[str, str]]) -> List[str]:
    """Build generic current-plus-context query segments without history dilution.

    Router still receives every visible message with its timestamp. Retrieval
    deliberately searches the current user bubble and each prior user context
    as separate semantic segments. Assistant output is not treated as source
    evidence, and unrelated values in one old bubble cannot veto another
    segment's valid evidence.
    """

    user_messages = [
        str(item.get("content") or "").strip()
        for item in history
        if item.get("role") == "user" and str(item.get("content") or "").strip()
    ]
    if not user_messages:
        raise ValueError("retrieval requires visible user context")
    current = user_messages[-1]
    queries = [current]
    for prior in reversed(user_messages[:-1]):
        contextual = f"{prior}\n{current}"
        if contextual not in queries:
            queries.append(contextual)
    return queries


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


def _parse_agent_turn_result(raw: str) -> AgentTurnResult:
    """Parse one whole strict envelope; never repair or pass through prose."""

    text = str(raw or "").strip()
    if not text:
        raise AgentContractInvalidError("selected Agent returned an empty answer")
    structured_text = text
    lines = structured_text.splitlines()
    if (
        len(lines) >= 3
        and lines[0].strip().lower() in {"```", "```json"}
        and lines[-1].strip() == "```"
    ):
        structured_text = "\n".join(lines[1:-1]).strip()
    if not (structured_text.startswith("{") and structured_text.endswith("}")):
        raise AgentContractInvalidError(
            "selected Agent response was not one complete JSON object"
        )
    try:
        payload = json.loads(structured_text)
    except json.JSONDecodeError as exc:
        raise AgentContractInvalidError(
            "selected Agent response was not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise AgentContractInvalidError("selected Agent envelope must be an object")
    try:
        return AgentTurnResult.model_validate(payload)
    except Exception as exc:
        raise AgentContractInvalidError(
            f"selected Agent envelope schema invalid: {type(exc).__name__}"
        ) from exc


def _validate_agent_capability_usage(
    agent_turn: AgentTurnResult,
    *,
    activated_skills: List[Any],
    evidence: Any,
    mcp_invocations: List[Any],
    tool_calls: List[Dict[str, Any]],
) -> None:
    """Fail closed when the Agent's usage declaration differs from runtime facts."""

    declared = agent_turn.capability_usage
    actual_skill_ids = sorted({int(item.skill_id) for item in activated_skills})
    declared_skill_ids = sorted({int(item) for item in declared.skill_ids})
    if declared_skill_ids != actual_skill_ids:
        raise AgentContractInvalidError(
            "capability_usage.skill_ids did not match native Skill calls"
        )

    evidence_ids = set(evidence.by_id())
    declared_citations = set(agent_turn.citations)
    declared_rag_ids = set(declared.rag_evidence_ids)
    if not declared_citations.issubset(evidence_ids):
        raise AgentContractInvalidError(
            "Agent citation ID was outside this turn's RAG EvidenceSet"
        )
    if declared_rag_ids != declared_citations:
        raise AgentContractInvalidError(
            "capability_usage.rag_evidence_ids did not match citations"
        )

    actual_mcp_names = sorted(
        {str(item.tool_name) for item in mcp_invocations if item.tool_name}
    )
    declared_mcp_names = sorted({str(item) for item in declared.mcp_calls})
    if declared_mcp_names != actual_mcp_names:
        raise AgentContractInvalidError(
            "capability_usage.mcp_calls did not match runtime MCP calls"
        )

    mcp_names = set(actual_mcp_names)
    actual_tool_names = sorted(
        {
            str(call.get("tool_name") or "")
            for call in tool_calls
            if call.get("tool_name")
            and str(call.get("tool_name")) != "get_skill_instructions"
            and str(call.get("tool_name")) not in mcp_names
        }
    )
    declared_tool_names = sorted({str(item) for item in declared.tool_calls})
    if declared_tool_names != actual_tool_names:
        raise AgentContractInvalidError(
            "capability_usage.tool_calls did not match runtime Tool calls"
        )


def _lane_explanation(decision: LaneDecision) -> str:
    labels = {
        RuntimeLane.HANDOFF: "人工协同",
        RuntimeLane.PROPERTY_GOVERNED: "物业受控回答",
        RuntimeLane.ISOLATED_GENERAL: "隔离通用回答",
    }
    return f"{labels[decision.lane]}：{decision.reason}"


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


def _handoff_contract_for(
    decision: LaneDecision,
) -> HandoffExecutionContract:
    if decision.lane != RuntimeLane.HANDOFF:
        raise ValueError("Handoff execution contract requires effective A lane")
    return HandoffExecutionContract(
        kind=HandoffKind.USER_REQUESTED,
        reason_code="user_requested",
        queue="property_service",
        safety_override=False,
        response_mode=ResponseMode.HUMAN_HANDOFF,
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
    if decision.lane == RuntimeLane.HANDOFF:
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
            decision_reason="A类同轮创建普通人工协同并短路所有业务能力。",
        )
    if decision.lane == RuntimeLane.ISOLATED_GENERAL:
        return AnswerContract(
            response_mode=ResponseMode.SAFE_GENERAL,
            evidence_required=False,
            skill_policy="selected",
            rag_policy="selected",
            tool_policy="selected",
            write_policy="forbidden",
            handoff_policy="skipped",
            forbidden_claims=common_forbidden
            + ["property_official_fact", "harmful_instructions"],
            decision_reason="C仅装配所选隔离Agent自身绑定；没有增强能力仍正常回答。",
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
    query: Any,
    live_results: List[Dict[str, Any]],
    knowledge_versions: Dict[int, Dict[str, Any]],
    allowed_document_ids: set[int],
    top_k: int,
    context_threshold: float = 0.2,
    context_token_budget: int = 1800,
) -> Tuple[List[Dict[str, Any]], bool]:
    import rag_retrieval

    queries = [
        str(item).strip()
        for item in (query if isinstance(query, list) else [query])
        if str(item or "").strip()
    ]
    if not queries:
        raise ValueError("snapshot evidence validation requires a query")

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
    def with_adjacent_context(seeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand generic same-document neighbours inside one shared budget."""

        selected: Dict[Tuple[int, int], Dict[str, Any]] = {}
        document_order: List[int] = []
        used_tokens = 0

        def add(item: Dict[str, Any]) -> bool:
            nonlocal used_tokens
            key = (int(item["doc_id"]), int(item.get("chunk_index") or 0))
            if key in selected:
                return True
            estimated = int(_estimate_tokens(str(item.get("content") or "")) or 0)
            if selected and used_tokens + estimated > max(1, int(context_token_budget)):
                return False
            selected[key] = item
            used_tokens += estimated
            if key[0] not in document_order:
                document_order.append(key[0])
            return True

        seed_keys: List[Tuple[int, int]] = []
        for seed in seeds[:top_k]:
            key = (int(seed["doc_id"]), int(seed.get("chunk_index") or 0))
            seed_keys.append(key)
            add(seed)
        distance = 1
        while distance <= 2:
            added_any = False
            for doc_id, chunk_index in seed_keys:
                for adjacent_index in (chunk_index - distance, chunk_index + distance):
                    snapshot_chunk = published_chunks.get((doc_id, adjacent_index))
                    if not snapshot_chunk:
                        continue
                    document = snapshot_chunk["document"]
                    adjacent = {
                        "doc_id": doc_id,
                        "doc_title": document.get("title") or "",
                        "chunk_index": adjacent_index,
                        "content": str(snapshot_chunk.get("content") or ""),
                        "chunk_hash": snapshot_chunk.get("chunk_hash"),
                        "document_hash": document.get("document_hash"),
                        "document_version": document.get("document_version"),
                        "score": None,
                        "context_score": None,
                        "evidence_status": "accepted",
                        "evidence_reason": "same_document_adjacent_context",
                        "retrieval_sources": ["runtime_release_snapshot_adjacent"],
                        "retrieval_mode": "snapshot_adjacent",
                    }
                    if add(adjacent):
                        added_any = True
            if not added_any:
                break
            distance += 1
        order = {doc_id: index for index, doc_id in enumerate(document_order)}
        return sorted(
            selected.values(),
            key=lambda item: (
                order.get(int(item["doc_id"]), len(order)),
                int(item.get("chunk_index") or 0),
            ),
        )

    if verified:
        return with_adjacent_context(verified), False

    fallback: List[Dict[str, Any]] = []
    for (doc_id, chunk_index), snapshot_chunk in published_chunks.items():
        document = snapshot_chunk["document"]
        content = str(snapshot_chunk.get("content") or "")
        if rag_retrieval._is_structural_chunk(
            content,
            document.get("title") or "",
        ):
            continue
        accepted_scores: List[Tuple[float, int]] = []
        content_values = rag_retrieval._critical_values(content)
        for query_index, query_segment in enumerate(queries):
            query_values = rag_retrieval._required_evidence_values(query_segment)
            if query_values and not query_values.issubset(content_values):
                continue
            segment_score = rag_retrieval._context_relevance_score(
                query_segment,
                content,
            )
            if segment_score >= context_threshold:
                accepted_scores.append((segment_score, query_index))
        if not accepted_scores:
            continue
        context_score, matched_query_index = max(
            accepted_scores,
            key=lambda item: (item[0], -item[1]),
        )
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
                "matched_query_index": matched_query_index,
            }
        )
    fallback.sort(key=lambda item: float(item.get("context_score") or 0), reverse=True)
    return with_adjacent_context(fallback), True


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
        # Every visible bubble enters the same one-Router chain. Persisted
        # Draft/Proposal state never bypasses or pre-classifies a chat bubble.
        path = RuntimePath.CONSULTATION
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
            risk_level="L0",
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

            if state.lane_decision.lane == RuntimeLane.HANDOFF:
                async for event in self._stream_a_handoff(
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
                    "router_contract_invalid"
                    if isinstance(exc, RouterContractInvalidError)
                    else (
                        "agent_contract_invalid"
                        if isinstance(
                            exc,
                            (AgentContractInvalidError, InternalControlPayloadLeakError),
                        )
                        else "runtime_failure"
                    )
                )
            )
            public_error = (
                PROVIDER_FAILURE_PUBLIC_MESSAGE
                if failure_code == "provider_failure"
                else (
                    "本次回答未通过运行合同校验，已透明停止，请稍后重试。"
                    if failure_code
                    in {"router_contract_invalid", "agent_contract_invalid"}
                    else RUNTIME_FAILURE_PUBLIC_MESSAGE
                )
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
        if is_confirmation(message):
            latest_work_order = get_latest_action_proposal(
                session_id,
                "work_order.create",
            )
            if latest_work_order and latest_work_order.get("status") == "committed":
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

        from agents.router import route_session_once

        cards = router_agent_cards(snapshot.config)
        visible_messages = _visible_chat_history(session_id)
        if not visible_messages:
            raise RuntimeError("Router received no visible session messages")
        if visible_messages[-1].get("content") != message:
            raise RuntimeError("current bubble is not the final visible session message")

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
            raise RuntimeError("unified Router must use deepseek-v4-flash")

        router_started = time.time()
        router_thinking = _thinking_for_snapshot(snapshot.config, router_model_id)
        with provider_accounting_scope(
            trace_id=trace_id,
            session_id=session_id,
            stage="router",
            model_selection_reason="one-call A/B/C classification and B/C Agent selection",
            price_snapshot=_price_for_snapshot(snapshot.config, router_model_id),
            model_policy_version=str(
                (snapshot.config.get("model_policy") or {}).get("version") or "v1.8"
            ),
        ) as router_scope:
            result = await route_session_once(
                messages=visible_messages,
                vertical_agents=cards,
                user_id=user_id,
                session_id=f"{session_id}::router::{trace_id}",
                model=_build_model_from_snapshot(snapshot.config, router_model_id),
            )

        provider_status = str(result.get("provider_status") or "failed")
        if result.get("decision") is None:
            if provider_status != "success":
                raise ProviderFailureError("unified Router Provider call failed")
            raise RouterContractInvalidError(
                "unified Router contract invalid: "
                + str(result.get("validation_error") or "schema validation failed")
            )

        attempts = list(getattr(router_scope, "attempts", []) or [])
        if not attempts:
            attempts = [
                {
                    "provider_request_sequence": 1,
                    "provider_request_id": (result.get("provider_evidence") or {}).get(
                        "provider_request_id"
                    ),
                    "provider_response_model": (result.get("provider_evidence") or {}).get(
                        "provider_response_model"
                    ),
                    "usage": (result.get("provider_evidence") or {}).get("usage")
                    or result.get("metrics")
                    or {},
                    "status": provider_status,
                }
            ]
        for index, attempt in enumerate(attempts, start=1):
            usage = dict(attempt.get("usage") or {})
            response_model = attempt.get("provider_response_model")
            cost = build_cost_entry(
                stage="router",
                provider=_model_provider(snapshot.config, router_model_id),
                requested_model=router_model_id,
                response_model=None,
                provider_response_model=response_model,
                thinking_enabled=router_thinking,
                model_policy_version=str(
                    (snapshot.config.get("model_policy") or {}).get("version") or "v1.8"
                ),
                provider_usage=usage if usage else None,
                price_row=_price_for_snapshot(
                    snapshot.config, response_model or router_model_id
                ),
                local_estimate_tokens=None,
                provider_succeeded=provider_status == "success",
            )
            state.cost_entries.append(cost)
            state.model_calls.append(
                {
                    "stage": "router",
                    "model_id": router_model_id,
                    "requested_model": router_model_id,
                    "provider_response_model": response_model,
                    "provider_request_id": attempt.get("provider_request_id"),
                    "provider_request_sequence": int(
                        attempt.get("provider_request_sequence") or index
                    ),
                    "thinking_enabled": router_thinking,
                    "latency_ms": int((time.time() - router_started) * 1000),
                    "usage": usage,
                    "usage_source": cost.usage_source.value,
                    "status": attempt.get("status") or provider_status,
                }
            )

        state.lane_decision = result["decision"]
        state.answer_contract = _answer_contract_for(
            state.lane_decision, RuntimePath.CONSULTATION
        )
        lane_payload = state.lane_decision.model_dump(mode="json")
        lane_payload["explanation"] = _lane_explanation(state.lane_decision)
        ledger.set("lane_decision", lane_payload)
        ledger.set("answer_contract", state.answer_contract.model_dump(mode="json"))
        record_trace_event(
            trace_id,
            "router",
            "success",
            latency_ms=int((time.time() - router_started) * 1000),
            output_summary=lane_payload["explanation"],
            metadata={
                **lane_payload,
                "candidate_fields": ["agent_id", "name", "description", "scope"],
                "candidate_count": len(cards),
                "visible_message_count": len(visible_messages),
                "physical_request_count": len(attempts),
                "automatic_retry_count": 0,
                "selector_request_count": 0,
                "resolver_request_count": 0,
                "answer_contract": state.answer_contract.model_dump(mode="json"),
            },
        )
        return

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

        # The one Router result is final for this bubble. Persisted Handoff
        # fields remain historical state and must never rewrite a new A/B/C
        # decision or recreate legacy ordinary/safety/emergency subtypes.
        state.lane_decision = reported_decision

        if self._is_new_work_order_start_authorized(state.lane_decision):
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
        """Execute the one product-level A Handoff and short-circuit all capabilities."""

        if state.lane_decision is None or state.answer_contract is None:
            raise RuntimeError("A handoff started without contracts")
        handoff_contract = _handoff_contract_for(state.lane_decision)
        current_handoff = get_chat_session(session_id) or {}
        current_handoff_status = str(
            current_handoff.get("handoff_status") or "none"
        )
        handoff = request_handoff(
            session_id,
            state.lane_decision.reason,
            risk_level="L3",
            reason_code="user_requested",
            queue="property_service",
            handoff_package={
                "trace_id": trace_id,
                "release_id": snapshot.release_id,
                "trigger_message": message,
                "router_reason": state.lane_decision.reason,
                "semantic_lane": state.lane_decision.model_dump(mode="json"),
                "handoff_kind": "ordinary",
            },
        )
        if current_handoff_status == "waiting_user":
            handoff = resume_handoff_after_owner_message(session_id)
        handoff_state = str(
            handoff.get("handoff_status") or current_handoff_status or "requested"
        )
        handoff_policy = {
            "level": "L3",
            "reason_code": "user_requested",
            "queue": "property_service",
            "safety_override": False,
            "matched_signals": ["unified_router_a_lane"],
        }
        reply = (
            f"{state.lane_decision.reason} 已直接发起人工协同，无需再次确认；"
            "本轮未调用业务Agent、Skill、RAG、MCP、Tool或写入流程。"
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
                "reason_code": "handoff",
                "handoff_kind": "ordinary",
                "queue": "property_service",
                "safety_override": False,
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
            intent="handoff",
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
                "handoff_reason": "handoff",
                "handoff_queue": "property_service",
                "safety_override": False,
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
    def _is_new_work_order_start_authorized(decision: LaneDecision) -> bool:
        """Accept only the Router's exact structured authorization for a new Draft."""

        return bool(
            decision.lane == RuntimeLane.PROPERTY_GOVERNED
            and str(decision.business_intent or "").strip()
            == WORK_ORDER_CREATE_INTENT
        )

    @staticmethod
    def _is_work_order_action_context(session_id: str, message: str) -> bool:
        """Continue only persisted work-order state; never reclassify new text."""

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
        start_authorized = self._is_new_work_order_start_authorized(
            state.lane_decision
        )
        use_work_order = bool(
            start_authorized
            or self._is_work_order_action_context(
                session_id,
                message,
            )
        )
        if use_work_order:
            result = advance_work_order_workflow(
                session_id,
                message,
                trace_id=trace_id,
                release_id=snapshot.release_id,
                start_authorized=start_authorized,
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
        )
        retrieval_queries = _build_retrieval_queries(visible_history)
        retrieval_query = retrieval_queries[0]
        all_cards = vertical_agent_cards(snapshot.config)
        cards = _lane_candidates(all_cards, state.lane_decision.lane)
        if not cards:
            async for event in self._stream_unconfigured_lane_boundary(
                session_id, trace_id, snapshot, state, ledger, started
            ):
                yield event
            return

        candidates = [str(item["agent_id"]) for item in cards]
        selected = str(state.lane_decision.selected_agent_id or "")
        if selected not in candidates:
            raise RuntimeError("Router selected an Agent outside its fixed lane")
        selection = {
            "selected_agent_id": selected,
            "reason": state.lane_decision.reason,
            "selection_source": "unified_router",
        }
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
        record_trace_event(
            trace_id,
            "agent_frozen",
            "success",
            output_summary=f"selected={selected}",
            metadata={
                "lane": state.lane_decision.lane.value,
                "selected_agent_id": selected,
                "selection_source": "unified_router",
                "snapshot_id": snapshot.snapshot_id,
                "immutable_for_turn": True,
                "second_agent_request_count": 0,
                "automatic_retry_count": 0,
                "non_selected_agent_capabilities_loaded": False,
            },
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
        # The selected Agent's native tool loop receives its own published
        # read definitions. No keyword/regex ToolPlanner runs before it.
        read_tool_plans: List[Any] = []
        structured_realtime_query = False
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
            if state.answer_contract.rag_policy == "selected"
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
        if not structured_realtime_query:
            yield _sse(
                "progress",
                {"trace_id": trace_id, "stage": "rag.retrieve", "status": "running"},
            )
        if allowed_doc_ids and not structured_realtime_query:
            try:
                import rag_retrieval

                search_parts = await asyncio.gather(
                    *(
                        asyncio.to_thread(
                            rag_retrieval.advanced_search,
                            query_segment,
                            snapshot.config.get("retrieval_policy") or {},
                            allowed_document_ids=sorted(allowed_doc_ids),
                        )
                        for query_segment in retrieval_queries
                    ),
                    return_exceptions=True,
                )
                successful_parts = [
                    item for item in search_parts if isinstance(item, dict)
                ]
                failed_parts = [
                    item for item in search_parts if isinstance(item, BaseException)
                ]
                if not successful_parts:
                    raise RuntimeError(
                        str(failed_parts[0])
                        if failed_parts
                        else "contextual retrieval returned no result envelope"
                    )
                merged_results: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
                for query_index, part in enumerate(search_parts):
                    if not isinstance(part, dict):
                        continue
                    for raw_result in list(part.get("results") or []):
                        item = dict(raw_result)
                        key = (
                            item.get("doc_id", item.get("document_id")),
                            item.get("chunk_index"),
                        )
                        item["context_query_indexes"] = [query_index]
                        current = merged_results.get(key)
                        if current is None:
                            merged_results[key] = item
                            continue
                        indexes = sorted(
                            set(current.get("context_query_indexes") or [])
                            | {query_index}
                        )
                        if float(item.get("score") or 0.0) > float(
                            current.get("score") or 0.0
                        ):
                            item["context_query_indexes"] = indexes
                            merged_results[key] = item
                        else:
                            current["context_query_indexes"] = indexes
                results = sorted(
                    merged_results.values(),
                    key=lambda item: (
                        -float(item.get("score") or 0.0),
                        min(item.get("context_query_indexes") or [0]),
                    ),
                )
                first_part = successful_parts[0]
                retrieval = {
                    "results": results,
                    "query_count": len(retrieval_queries),
                    "failed_query_count": len(failed_parts),
                    "filter_summary": {"candidate_count": len(results)},
                    "evidence_policy": first_part.get("evidence_policy"),
                }
                if failed_parts:
                    ledger.violation(
                        "live_retrieval_failed",
                        "one or more contextual retrieval segments failed",
                        failed_query_count=len(failed_parts),
                        query_count=len(retrieval_queries),
                    )
                results, used_snapshot_fallback = _results_from_snapshot(
                    retrieval_queries,
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
                    int(
                        (snapshot.config.get("retrieval_policy") or {}).get(
                            "context_token_budget"
                        )
                        or 1800
                    ),
                )
                retrieval_status = (
                    "completed_snapshot_fallback"
                    if used_snapshot_fallback
                    else (
                        "completed_partial_failure"
                        if failed_parts
                        else "completed"
                    )
                )
            except Exception as exc:
                results, _ = _results_from_snapshot(
                    retrieval_queries,
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
                    int(
                        (snapshot.config.get("retrieval_policy") or {}).get(
                            "context_token_budget"
                        )
                        or 1800
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
            retrieval_query,
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
                "candidate_count": int(
                    ((retrieval or {}).get("filter_summary") or {}).get(
                        "candidate_count"
                    )
                    or len((retrieval or {}).get("results") or [])
                ),
                "loaded_chunk_count": len(evidence.items),
                "loaded_character_count": sum(
                    len(item.content_snapshot) for item in evidence.items
                ),
                "evidence_policy": (retrieval or {}).get("evidence_policy"),
                "filter_summary": (retrieval or {}).get("filter_summary"),
                "direct_knowledge_required": direct_knowledge_required,
                "query_message_count": len(visible_history),
                "query_segment_count": len(retrieval_queries),
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
        mcp_context, invocations = "", []
        preinvoked_tools = {
            (invocation.server_name, invocation.tool_name)
            for invocation in invocations
            if invocation.tool_name != "discovery"
            and invocation.invocation_status == "success"
        }
        model_native_toolkits = build_model_native_read_tools(
            snapshot.config,
            selected,
            message,
            excluded_tools=preinvoked_tools,
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

        evidence_prompt = prompt_evidence_allowlist(evidence)
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
                not structured_realtime_query
                and state.answer_contract.skill_policy == "selected"
            ),
        )
        state.activated_skills = []
        bound_skill_ids = [item.skill_id for item in build.bound_skills]
        tool_schema_count = sum(
            len(getattr(toolkit, "allowed_function_names", set()) or set())
            for toolkit in model_native_toolkits
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
                "skipped",
                "awaiting_same_agent_native_use",
                bound_skill_ids=bound_skill_ids,
                used_skill_ids=[],
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
                "skipped",
                "awaiting_same_agent_native_use",
                exposed_schema_count=tool_schema_count,
                tools=[],
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
                "status": "skipped",
                "reason_code": "awaiting_same_agent_native_use",
                "details": {
                    "bound_skill_ids": bound_skill_ids,
                    "used_skill_ids": [],
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
                "status": "skipped",
                "reason_code": "awaiting_same_agent_native_use",
                "details": {
                    "exposed_schema_count": tool_schema_count,
                    "tools": [],
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
        state.next_step = "answer"
        # Evidence availability is an Agent answer concern, never authority to
        # switch Agent, skip C, or replace a selected Agent's answer.
        model_invoked = True
        if knowledge_evidence_blocked or out_of_scope_without_agent:
            record_trace_event(
                trace_id,
                "evidence_gate",
                "success",
                output_summary=(
                    "no matching isolated Agent; frozen Agent will report its boundary"
                    if out_of_scope_without_agent
                    else "knowledge evidence insufficient; frozen Agent will report answer_status"
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
                (
                    "本轮由已冻结的物业Agent处理。只回答本轮用户问题；不得输出 Router、"
                    "LaneDecision、CapabilityDecision 或其他控制 JSON；不得直接提交任何业务写入。"
                    "若且仅若需要创建工单，必须按AgentTurnResult返回结构化proposal_request；"
                    "后端最多保存Draft或pending_confirmation Proposal，正式写入必须等待业主按钮确认。"
                )
                if state.lane_decision.lane == RuntimeLane.PROPERTY_GOVERNED
                else (
                    "本轮由已冻结的隔离通用Agent处理。只回答本轮用户问题；不得输出 Router、"
                    "LaneDecision、CapabilityDecision 或其他控制 JSON；proposal_request必须为空，"
                    "不得访问ActionGateway或发起任何物业业务写入。"
                )
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
        vertical_session_id = f"{session_id}::vertical::{selected}::{trace_id}"
        with accounting_context as provider_scope:
            response_stream = (
                build.agent.arun(
                    contextual_message,
                    user_id=user_id,
                    session_id=vertical_session_id,
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

        completed_run_output = await _completed_agent_run_output(
            build.agent,
            vertical_session_id,
        )
        if completed_run_output is not None:
            completed_content = _completed_agent_run_content(completed_run_output)
            if completed_content:
                full_content = str(completed_content)
                provisional_buffer = full_content
            for call in _extract_tool_calls(completed_run_output):
                if call not in tool_calls:
                    tool_calls.append(call)
            completed_metrics = _metrics_dict(completed_run_output)
            if completed_metrics:
                final_metrics.update(completed_metrics)
            merge_non_null(
                final_provider_evidence,
                provider_evidence_from_run(completed_run_output),
            )

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
            raise AgentContractInvalidError(
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

        if model_native_invocations and state.capability_decision is not None:
            previous_capability = state.capability_decision
            invoked_names = [
                f"{item.server_name}/{item.tool_name}"
                for item in model_native_invocations
            ]
            state.capability_decision = CapabilityDecision(
                selected_agent_id=selected,
                skill=previous_capability.skill,
                rag=previous_capability.rag,
                tool={
                    "status": "selected",
                    "reason_code": "selected_agent_native_tool_loop",
                    "details": {"tools": invoked_names},
                },
                write=previous_capability.write,
                handoff=previous_capability.handoff,
            )
            decision_summary["tool"] = _decision(
                "selected",
                "selected_agent_native_tool_loop",
                tools=invoked_names,
            )

        if provider_failure_reason:
            raise ProviderFailureError(
                f"model Provider returned failure text: {provider_failure_reason}"
            )

        (
            used_skills,
            used_skill_tool_calls,
            used_skill_evidence_sources,
        ) = resolve_model_used_skills(
            build,
            tool_calls,
        )
        build.activated_skills = used_skills
        build.skill_tool_calls = used_skill_tool_calls
        build.skill_evidence_sources = used_skill_evidence_sources
        state.activated_skills = used_skills
        for call in used_skill_tool_calls:
            call_status = str(call.get("status") or "success")
            record_trace_event(
                trace_id,
                f"skill.{call['skill_id']}.get_skill_instructions",
                call_status,
                output_summary=(
                    f"Skill {call['skill_id']} version={call['skill_version']} "
                    f"status={call_status}"
                ),
                metadata=call,
            )
            if call_status != "success":
                state.tool_invocations.append(
                    ToolInvocation(
                        server_name="skill_runtime",
                        tool_name="get_skill_instructions",
                        effect=ToolEffect.READ,
                        arguments=dict(call.get("arguments") or {}),
                        planner_source="model_native",
                        match_reason="selected_agent_bound_skill",
                        transport_status="failed",
                        invocation_status="failed",
                        business_status="failed",
                        error_summary=str(
                            call.get("error_summary") or "Skill execution failed"
                        ),
                    )
                )
        record_trace_event(
            trace_id,
            "skill_usage",
            "success",
            output_summary=(
                f"bound={len(build.bound_skills)}; used={len(used_skills)}"
            ),
            metadata={
                "bound_skill_ids": [item.skill_id for item in build.bound_skills],
                "used_skill_ids": [item.skill_id for item in used_skills],
                "failed_skill_ids": [
                    int(call["skill_id"])
                    for call in used_skill_tool_calls
                    if str(call.get("status") or "success") != "success"
                ],
            },
        )

        agent_turn = _parse_agent_turn_result(full_content)
        _validate_agent_capability_usage(
            agent_turn,
            activated_skills=used_skills,
            evidence=evidence,
            mcp_invocations=model_native_invocations,
            tool_calls=tool_calls,
        )
        full_content = agent_turn.answer
        declared_usage = agent_turn.capability_usage
        actual_mcp_names = sorted({item.tool_name for item in model_native_invocations})
        actual_tool_names = sorted(set(declared_usage.tool_calls))
        previous_capability = state.capability_decision
        if previous_capability is not None:
            state.capability_decision = CapabilityDecision(
                selected_agent_id=selected,
                skill={
                    "status": "selected" if used_skills else "skipped",
                    "reason_code": (
                        "same_agent_native_use" if used_skills else "not_used"
                    ),
                    "details": {
                        "bound_skill_ids": [
                            item.skill_id for item in build.bound_skills
                        ],
                        "used_skill_ids": [item.skill_id for item in used_skills],
                    },
                },
                rag={
                    "status": "selected" if agent_turn.citations else "skipped",
                    "reason_code": (
                        "cited_rag_evidence"
                        if agent_turn.citations
                        else "not_used_in_answer"
                    ),
                    "details": {
                        "candidate_evidence_count": len(evidence.items),
                        "used_evidence_ids": list(agent_turn.citations),
                    },
                },
                tool={
                    "status": (
                        "selected"
                        if actual_mcp_names or actual_tool_names
                        else "skipped"
                    ),
                    "reason_code": (
                        "same_agent_native_use"
                        if actual_mcp_names or actual_tool_names
                        else "not_used"
                    ),
                    "details": {
                        "exposed_schema_count": tool_schema_count,
                        "mcp_calls": actual_mcp_names,
                        "tool_calls": actual_tool_names,
                    },
                },
                write=previous_capability.write,
                handoff=previous_capability.handoff,
            )
        decision_summary["skill"] = _decision(
            "selected" if used_skills else "skipped",
            "same_agent_native_use" if used_skills else "not_used",
            bound_skill_ids=[item.skill_id for item in build.bound_skills],
            used_skill_ids=[item.skill_id for item in used_skills],
        )
        decision_summary["tool"] = _decision(
            "selected" if actual_mcp_names or actual_tool_names else "skipped",
            (
                "same_agent_native_use"
                if actual_mcp_names or actual_tool_names
                else "not_used"
            ),
            exposed_schema_count=tool_schema_count,
            mcp_calls=actual_mcp_names,
            tool_calls=actual_tool_names,
        )
        ledger.append(
            "evaluation_results",
            {
                "case": "capability_usage",
                "passed": True,
                "decision_summary": decision_summary,
            },
        )
        record_trace_event(
            trace_id,
            "capability_decision",
            "success",
            output_summary=(
                f"frozen_agent={selected}; "
                f"skill_bound={len(build.bound_skills)}; "
                f"skill_used={len(used_skills)}; "
                f"rag_used={len(agent_turn.citations)}; "
                f"mcp_used={len(actual_mcp_names)}; "
                f"tool_used={len(actual_tool_names)}"
            ),
            metadata={
                "passive_execution_record": True,
                "decision_summary": decision_summary,
            },
        )
        proposal_result: Optional[Dict[str, Any]] = None
        if agent_turn.proposal_request is not None:
            if state.lane_decision.lane != RuntimeLane.PROPERTY_GOVERNED:
                raise RuntimeError("only the selected B Agent may emit proposal_request")
            proposal_result = apply_structured_proposal_request(
                session_id=session_id,
                proposal_request=agent_turn.proposal_request.model_dump(mode="json"),
                trace_id=trace_id,
                release_id=snapshot.release_id,
                selected_agent_id=selected,
            )
            proposal_id = proposal_result.get("proposal_id")
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
                            risk_level=RiskLevel(proposal_row["risk_level"]),
                            payload=proposal_row.get("payload") or {},
                            parameter_hash=content_hash(proposal_row.get("payload") or {}),
                            idempotency_key=proposal_row["idempotency_key"],
                            status=proposal_row["status"],
                        )
                    )
            tool_calls.append(
                {
                    "tool_name": "action_gateway",
                    "action_type": "work_order.create",
                    "status": proposal_result.get("action"),
                    "phase": proposal_result.get("action"),
                    "proposal_id": proposal_id,
                    "session_id": session_id,
                    "proposal_status": proposal_result.get("proposal_status"),
                    "missing_fields": proposal_result.get("missing_field_keys") or [],
                    "invocation_mode": "selected_b_agent_structured_request",
                }
            )
            record_trace_event(
                trace_id,
                "work_order.proposal_request",
                "success",
                output_summary=str(proposal_result.get("action") or "draft_updated"),
                metadata={
                    "selected_agent_id": selected,
                    "proposal_id": proposal_id,
                    "proposal_status": proposal_result.get("proposal_status"),
                    "missing_field_keys": proposal_result.get("missing_field_keys") or [],
                    "formal_write": False,
                },
            )
            previous_capability = state.capability_decision
            if previous_capability is not None:
                state.capability_decision = CapabilityDecision(
                    selected_agent_id=selected,
                    skill=previous_capability.skill,
                    rag=previous_capability.rag,
                    tool=previous_capability.tool,
                    write={
                        "status": "required",
                        "reason_code": "selected_b_agent_proposal_request",
                        "details": {
                            "phase": proposal_result.get("action"),
                            "proposal_id": proposal_id,
                            "formal_write": False,
                        },
                    },
                    handoff=previous_capability.handoff,
                )

        rendered, citations, citation_violations = render_rag_citations(
            full_content,
            evidence,
            declared_evidence_ids=agent_turn.citations,
        )
        linked_skill_evidence = build_skill_evidence(
            full_content,
            build.skill_evidence_sources,
        )
        citation_required = False
        knowledge_grounding_failed = False
        state.citations = citations
        _record_citation_violations(ledger, citation_violations)

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
                    or agent_turn.answer_status
                    in {"insufficient_evidence", "insufficient_capability"}
                ),
                "required": direct_knowledge_required,
                "evidence_count": len(evidence.items),
                "skill_evidence_count": len(build.skill_evidence_sources),
                "tool_evidence_count": len(successful_tool_evidence),
                "domain_scope": domain_scope,
                "model_invoked": model_invoked,
                "decision": (
                    "rejected_insufficient"
                    if agent_turn.answer_status == "insufficient_evidence"
                    else "rejected_insufficient_capability"
                    if agent_turn.answer_status == "insufficient_capability"
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
                "bound_skill_ids": [item.skill_id for item in build.bound_skills],
                "used_skill_ids": [
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
                "tool_schema_exposed_count": tool_schema_count,
                "tool_actual_call_count": len(model_native_invocations)
                + len(actual_tool_names),
                "tool_result_character_count": sum(
                    len(str(item.result_summary or ""))
                    for item in model_native_invocations
                ),
                "decision_summary": decision_summary,
                "evidence_decision": (
                    "rejected_insufficient"
                    if agent_turn.answer_status == "insufficient_evidence"
                    else "rejected_insufficient_capability"
                    if agent_turn.answer_status == "insufficient_capability"
                    else "answered_with_evidence"
                ),
                "citation_violations": citation_violations,
                "answer_status": agent_turn.answer_status,
                "model_invoked": model_invoked,
                "second_agent_request_count": 0,
                "automatic_retry_count": 0,
                "selector_request_count": 0,
                "resolver_request_count": 0,
                "non_selected_agent_capabilities_loaded": False,
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
        auto_badcase = None
        try:
            auto_badcase = capture_runtime_badcase(
                ledger=ledger.contract,
                original_query=message,
                ai_response=rendered,
                source_message_id=saved.get("id"),
                agent_answer_status=agent_turn.answer_status,
                delivery_context={
                    "normal_completed": True,
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
        except Exception as exc:
            # The governed answer and its Trace are already complete.  A
            # secondary Badcase persistence failure is a structured runtime
            # observation, never authority to suppress that successful answer.
            ledger.append(
                "system_observations",
                {
                    "code": "badcase_capture_failed",
                    "component": "badcase_capture",
                    "error_type": type(exc).__name__,
                    "delivery_status": "preserved",
                },
            )
            try:
                ledger.persist("complete")
            except Exception:
                pass
            try:
                record_trace_event(
                    trace_id,
                    "badcase_capture",
                    "failed",
                    latency_ms=int((time.time() - started) * 1000),
                    output_summary="Badcase capture failed after successful answer",
                    metadata={
                        "error_type": type(exc).__name__,
                        "delivery_status": "preserved",
                    },
                )
            except Exception:
                pass

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
