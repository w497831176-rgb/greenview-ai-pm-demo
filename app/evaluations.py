"""Evaluation / Golden Set API for YIAI物业 V1.6.

This module deliberately evaluates the *product path*, not an isolated model
completion.  A case can assert route, Skill, Tool/MCP, RAG evidence, handoff
and hard business prohibitions.  A real model call happens only when an
operator explicitly runs one active case; creating, editing and reviewing a
case is free of model calls.
"""

import json
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.observability import (
    _background_budget_gate,
    _split_model_records,
    _token_source,
    _token_value,
)
from db.property_db import (
    add_badcase_action,
    create_badcase,
    create_evaluation_case,
    create_evaluation_run,
    evaluation_summary,
    get_badcase_by_evaluation_run,
    get_badcase,
    get_chat_trace,
    get_evidence_ledger,
    get_evaluation_case,
    get_evaluation_run,
    get_model_calls_for_trace,
    get_session_runtime_side_effects,
    list_evaluation_cases,
    list_evaluation_runs,
    list_trace_events,
    now_cn,
    record_trace_event,
    update_chat_trace,
    update_evaluation_case,
    update_evaluation_run,
    update_badcase,
)


router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])

CASE_STATUSES = {"draft", "active", "archived"}
RISK_LEVELS = {"L1", "L2", "L3", "L4"}
SOURCES = {"badcase", "sop", "expert", "adversarial", "synthetic"}


class EvaluationCaseCreate(BaseModel):
    case_key: str
    title: str
    user_message: str
    description: str = ""
    scenario: str = ""
    session_context: Dict[str, Any] = Field(default_factory=dict)
    risk_level: str = "L2"
    expected_agent_id: Optional[str] = None
    expected_skills: List[str] = Field(default_factory=list)
    expected_tools: List[str] = Field(default_factory=list)
    expected_citation_docs: List[str] = Field(default_factory=list)
    required_terms: List[str] = Field(default_factory=list)
    forbidden_terms: List[str] = Field(default_factory=list)
    expected_handoff: Optional[bool] = None
    rubric: Dict[str, Any] = Field(default_factory=dict)
    source: str = "expert"
    source_badcase_id: Optional[int] = None
    status: str = "draft"
    version_label: Optional[str] = None
    owner: Optional[str] = None


class EvaluationCaseUpdate(BaseModel):
    case_key: Optional[str] = None
    title: Optional[str] = None
    user_message: Optional[str] = None
    description: Optional[str] = None
    scenario: Optional[str] = None
    session_context: Optional[Dict[str, Any]] = None
    risk_level: Optional[str] = None
    expected_agent_id: Optional[str] = None
    expected_skills: Optional[List[str]] = None
    expected_tools: Optional[List[str]] = None
    expected_citation_docs: Optional[List[str]] = None
    required_terms: Optional[List[str]] = None
    forbidden_terms: Optional[List[str]] = None
    expected_handoff: Optional[bool] = None
    rubric: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    source_badcase_id: Optional[int] = None
    status: Optional[str] = None
    version_label: Optional[str] = None
    owner: Optional[str] = None


class EvaluationReviewRequest(BaseModel):
    passed: bool
    note: str


class EvaluationRunRequest(BaseModel):
    linked_badcase_id: Optional[int] = None


def _validate_case_payload(payload: Dict[str, Any]) -> None:
    risk = payload.get("risk_level")
    if risk is not None and risk not in RISK_LEVELS:
        raise HTTPException(status_code=400, detail=f"invalid risk_level: {risk}")
    status = payload.get("status")
    if status is not None and status not in CASE_STATUSES:
        raise HTTPException(status_code=400, detail=f"invalid status: {status}")
    source = payload.get("source")
    if source is not None and source not in SOURCES:
        raise HTTPException(status_code=400, detail=f"invalid source: {source}")
    key = payload.get("case_key")
    if key is not None and (not key.strip() or len(key) > 80):
        raise HTTPException(status_code=400, detail="case_key must be 1-80 characters")


def _text_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_tool_names(done: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for call in (done.get("mcp_calls") or []):
        server = str(call.get("server_name") or "").strip()
        tool = str(call.get("tool_name") or "").strip()
        if tool:
            names.append(tool)
        if server and tool:
            names.append(f"{server}.{tool}")
    for call in (done.get("tool_calls") or []):
        if isinstance(call, str):
            names.append(call)
        elif isinstance(call, dict):
            name = call.get("tool_name") or call.get("name")
            if name:
                names.append(str(name))
    return list(dict.fromkeys(names))


def _normalize_skill_names(done: Dict[str, Any]) -> List[str]:
    names = []
    for item in (done.get("activated_skills") or []):
        if isinstance(item, dict):
            name = item.get("name") or item.get("skill_name")
        else:
            name = item
        if name:
            names.append(str(name))
    return list(dict.fromkeys(names))


def _normalize_citation_titles(done: Dict[str, Any]) -> List[str]:
    return list(dict.fromkeys([
        str(item.get("doc_title") or "")
        for item in (done.get("citations") or [])
        if isinstance(item, dict) and item.get("doc_title")
    ]))


def _match_expected(expected: str, actuals: Iterable[str]) -> bool:
    """Match exact names, qualified server.tool names or a case-insensitive fallback."""
    normalized = expected.strip().lower()
    for actual in actuals:
        candidate = str(actual).strip().lower()
        if candidate == normalized or candidate.endswith("." + normalized):
            return True
    return False


def _rule(
    key: str,
    label: str,
    expected: Any,
    actual: Any,
    status: str,
    hard: bool = True,
    note: str = "",
) -> Dict[str, Any]:
    if status == "fail" and not note:
        note = "实际结果与预期不一致"
    return {
        "key": key,
        "label": label,
        "expected": expected,
        "actual": actual,
        "status": status,
        "hard": hard,
        "note": note,
    }


def _nested_value(value: Any, path: str) -> Any:
    current = value
    for part in str(path).split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _matches_contract(expected: Any, actual: Any) -> bool:
    if isinstance(expected, list):
        actual_items = actual if isinstance(actual, list) else []
        return all(item in actual_items for item in expected)
    return actual == expected


def _manual_rubric_required(case: Dict[str, Any]) -> bool:
    rubric = case.get("rubric") or {}
    return bool(isinstance(rubric, dict) and rubric.get("operator_rubric"))


def evaluate_runtime_evidence(case: Dict[str, Any], answer: str, done: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """Run deterministic checks and leave qualitative judgement to humans.

    This purposefully does not use an LLM-as-a-Judge.  In an interview demo it
    is more honest to show which rules are objectively verified and which still
    need business/operator review.
    """
    checks: List[Dict[str, Any]] = []
    actual_agent = str(done.get("current_agent_id") or done.get("route_intent") or "")
    skills = _normalize_skill_names(done)
    tools = _normalize_tool_names(done)
    citations = _normalize_citation_titles(done)
    answer_lower = (answer or "").lower()

    expected_agent = str(case.get("expected_agent_id") or "").strip()
    if expected_agent:
        checks.append(_rule(
            "agent", "Agent 路由", expected_agent, actual_agent,
            "pass" if actual_agent == expected_agent else "fail",
        ))
    else:
        checks.append(_rule("agent", "Agent 路由", "未配置", actual_agent, "not_configured", note="可由人工 Rubric 评审"))

    expected_skills = _text_list(case.get("expected_skills"))
    if expected_skills:
        missing = [item for item in expected_skills if not _match_expected(item, skills)]
        checks.append(_rule("skills", "Skill 命中", expected_skills, skills, "pass" if not missing else "fail", note=("缺少：" + "、".join(missing)) if missing else ""))
    else:
        checks.append(_rule("skills", "Skill 命中", "未配置", skills, "not_configured"))

    expected_tools = _text_list(case.get("expected_tools"))
    if expected_tools:
        missing = [item for item in expected_tools if not _match_expected(item, tools)]
        checks.append(_rule("tools", "Tool/MCP 调用", expected_tools, tools, "pass" if not missing else "fail", note=("缺少：" + "、".join(missing)) if missing else ""))
    else:
        checks.append(_rule("tools", "Tool/MCP 调用", "未配置", tools, "not_configured"))

    expected_docs = _text_list(case.get("expected_citation_docs"))
    if expected_docs:
        missing = [item for item in expected_docs if not _match_expected(item, citations)]
        checks.append(_rule("citations", "RAG 证据引用", expected_docs, citations, "pass" if not missing else "fail", note=("缺少：" + "、".join(missing)) if missing else ""))
    else:
        checks.append(_rule("citations", "RAG 证据引用", "未配置", citations, "not_configured"))

    required_terms = _text_list(case.get("required_terms"))
    if required_terms:
        missing = [item for item in required_terms if item.lower() not in answer_lower]
        checks.append(_rule("required_terms", "必须表达", required_terms, answer[:800], "pass" if not missing else "fail", note=("未出现：" + "、".join(missing)) if missing else ""))
    else:
        checks.append(_rule("required_terms", "必须表达", "未配置", "-", "not_configured"))

    forbidden_terms = _text_list(case.get("forbidden_terms"))
    if forbidden_terms:
        found = [item for item in forbidden_terms if item.lower() in answer_lower]
        checks.append(_rule("forbidden_terms", "禁止表达", forbidden_terms, answer[:800], "fail" if found else "pass", note=("出现：" + "、".join(found)) if found else ""))
    else:
        checks.append(_rule("forbidden_terms", "禁止表达", "未配置", "-", "not_configured"))

    expected_handoff = case.get("expected_handoff")
    if expected_handoff is not None:
        actual_handoff = bool(done.get("handoff"))
        checks.append(_rule("handoff", "人机协同边界", bool(expected_handoff), actual_handoff, "pass" if actual_handoff == bool(expected_handoff) else "fail"))
    else:
        checks.append(_rule("handoff", "人机协同边界", "未配置", bool(done.get("handoff")), "not_configured"))

    rubric = case.get("rubric") or {}
    assertions = rubric.get("deterministic_assertions") or {}
    decision_summary = done.get("decision_summary") or {}
    expected_decisions = assertions.get("expected_decision_summary") or {}
    for component, expected_fields in expected_decisions.items():
        actual_decision = decision_summary.get(component) or {}
        for field, expected_value in (expected_fields or {}).items():
            actual_value = _nested_value(actual_decision, field)
            passed = _matches_contract(expected_value, actual_value)
            checks.append(_rule(
                f"decision_{component}_{field}",
                f"{component} 决策 · {field}",
                expected_value,
                actual_value,
                "pass" if passed else "fail",
            ))

    citation_terms = _text_list(assertions.get("citation_required_terms"))
    if citation_terms:
        citation_text = "\n".join(
            " ".join(str(item.get(key) or "") for key in ("doc_title", "content_snapshot", "content"))
            for item in (done.get("citations") or [])
            if isinstance(item, dict)
        ).lower()
        missing = [term for term in citation_terms if term.lower() not in citation_text]
        checks.append(_rule(
            "citation_support", "引用内容支持关键事实", citation_terms,
            citation_text[:1200], "pass" if not missing else "fail",
            note=("引用中缺少：" + "、".join(missing)) if missing else "",
        ))

    if assertions.get("require_knowledge_insufficient"):
        passed = "当前知识依据不足" in (answer or "")
        checks.append(_rule(
            "knowledge_insufficient", "无依据时安全拒答", "当前知识依据不足",
            answer[:800], "pass" if passed else "fail",
        ))

    side_effects = done.get("side_effects") or {}
    if assertions.get("forbid_business_side_effects"):
        actual_writes = int(side_effects.get("business_writes") or 0)
        checks.append(_rule(
            "business_side_effects", "无 ActionGateway/工单业务写入", 0,
            {key: side_effects.get(key, 0) for key in (
                "business_writes", "work_orders", "work_order_drafts",
                "action_proposals", "action_receipts",
            )},
            "pass" if actual_writes == 0 else "fail",
        ))

    if assertions.get("require_mcp_business_success"):
        calls = done.get("mcp_calls") or []
        valid_calls = [call for call in calls if isinstance(call, dict)]
        successful = bool(valid_calls) and len(valid_calls) == len(calls) and all(
            str(call.get("invocation_status") or call.get("status") or "").lower() == "success"
            and str(call.get("business_status") or "success").lower() == "success"
            for call in valid_calls
        )
        checks.append(_rule(
            "mcp_business_success", "MCP 调用与业务结果成功", True, calls,
            "pass" if successful else "fail",
        ))

    model_calls, _ = _split_model_records(done.get("model_calls") or [])
    if assertions.get("forbid_normal_business_answer"):
        actual = {
            "handoff": bool(done.get("handoff")),
            "model_call_count": len(model_calls),
            "agent_decision": (decision_summary.get("agent") or {}).get("status"),
        }
        passed = (
            actual["handoff"]
            and actual["model_call_count"] == 0
            and actual["agent_decision"] == "skipped"
        )
        checks.append(_rule(
            "handoff_preempts_answer", "人工接管优先且未生成普通业务回答",
            {"handoff": True, "model_call_count": 0, "agent_decision": "skipped"},
            actual, "pass" if passed else "fail",
        ))

    if assertions.get("expected_model_call_count") is not None:
        expected_count = int(assertions["expected_model_call_count"])
        checks.append(_rule(
            "model_call_count", "模型调用次数", expected_count, len(model_calls),
            "pass" if len(model_calls) == expected_count else "fail",
        ))

    for config_key, label, actual_items in (
        ("expected_mcp_call_count", "MCP 调用次数", done.get("mcp_calls") or []),
        ("expected_citation_count", "Citation 数量", done.get("citations") or []),
        ("expected_skill_count", "Skill 命中数量", _normalize_skill_names(done)),
    ):
        if assertions.get(config_key) is None:
            continue
        expected_count = int(assertions[config_key])
        checks.append(_rule(
            config_key, label, expected_count, len(actual_items),
            "pass" if len(actual_items) == expected_count else "fail",
        ))

    forced_failure = assertions.get("controlled_failure")
    if forced_failure:
        checks.append(_rule(
            "controlled_failure", "受控故障注入（不计入黄金集）",
            forced_failure.get("expected", "故意设置为不满足"),
            forced_failure.get("actual", "安全只读链路"),
            "fail",
            note="此失败由验收用例显式注入，不代表真实能力故障。",
        ))

    hard_fail = any(item["hard"] and item["status"] == "fail" for item in checks)
    needs_manual = _manual_rubric_required(case)
    status = "failed" if hard_fail else "needs_manual_review" if needs_manual else "passed"
    return checks, status


class RuntimeExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        answer: str,
        done: Dict[str, Any],
        *,
        trace_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.answer = answer
        self.done = done
        self.trace_id = (
            str(done.get("trace_id") or trace_id or "").strip() or None
        )


async def _run_real_chat(message: str, session_id: str) -> Tuple[str, Dict[str, Any]]:
    """Run the canonical owner runtime and require one evidenced SSE success."""
    from app.chat import stream_chat_response

    answer = ""
    done: Dict[str, Any] = {}
    done_event_count = 0
    valid_done_payload_count = 0
    done_payload_invalid = False
    error_event_count = 0
    valid_error_payload_count = 0
    malformed_error_event_received = False
    payload_error_detail = ""
    evidence_trace_id: Optional[str] = None
    event_name = ""
    post_done_event_count = 0
    stream_exception_type = ""

    def consume_chunk(chunk: str) -> None:
        nonlocal answer, done, done_event_count, valid_done_payload_count
        nonlocal done_payload_invalid, error_event_count
        nonlocal valid_error_payload_count, malformed_error_event_received
        nonlocal payload_error_detail, evidence_trace_id, event_name
        nonlocal post_done_event_count

        for line in chunk.splitlines():
            if not line.strip():
                event_name = ""
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
                if done_event_count:
                    post_done_event_count += 1
                if event_name == "error":
                    error_event_count += 1
                elif event_name == "done":
                    done_event_count += 1
                continue
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw:
                if event_name == "error":
                    malformed_error_event_received = True
                elif event_name == "done":
                    done_payload_invalid = True
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                if event_name == "error":
                    malformed_error_event_received = True
                elif event_name == "done":
                    done_payload_invalid = True
                continue
            if not isinstance(payload, dict):
                if event_name == "error":
                    malformed_error_event_received = True
                elif event_name == "done":
                    done_payload_invalid = True
                continue

            if event_name == "error":
                valid_error_payload_count += 1
            elif event_name == "done":
                valid_done_payload_count += 1
            payload_trace_id = str(payload.get("trace_id") or "").strip()
            if payload_trace_id:
                evidence_trace_id = payload_trace_id
            if event_name == "delta" and payload.get("content"):
                answer += str(payload["content"])
            elif event_name in {"final", "done"}:
                terminal_answer = payload.get("content") or payload.get("answer")
                if terminal_answer:
                    answer = str(terminal_answer)
            if event_name == "done":
                done = payload
            if payload.get("error"):
                payload_error_detail = str(payload["error"])

    try:
        async for chunk in stream_chat_response(message, session_id, "evaluation"):
            consume_chunk(chunk)
    except Exception as exc:
        stream_exception_type = type(exc).__name__

    def reject(reason_code: str) -> None:
        failure_evidence = dict(done)
        failure_evidence["runtime_error_code"] = reason_code
        failure_evidence["done_event_count"] = done_event_count
        if stream_exception_type:
            failure_evidence["stream_exception_type"] = stream_exception_type
        detail = payload_error_detail.strip()
        message_text = f"{reason_code}: {detail}" if detail else reason_code
        raise RuntimeExecutionError(
            message_text,
            answer,
            failure_evidence,
            trace_id=evidence_trace_id,
        )

    if (
        malformed_error_event_received
        or error_event_count != valid_error_payload_count
    ):
        reject("evaluation_error_event_malformed")
    if error_event_count:
        reject("evaluation_error_event")
    if payload_error_detail:
        reject("evaluation_payload_error")
    if stream_exception_type:
        reject("evaluation_stream_interrupted")
    if done_event_count == 0:
        reject("evaluation_done_missing")
    if done_event_count != 1:
        reject("evaluation_multiple_done_events")
    if (
        done_payload_invalid
        or valid_done_payload_count != done_event_count
    ):
        reject("evaluation_done_malformed")
    if post_done_event_count:
        reject("evaluation_event_after_done")
    if "status" not in done:
        reject("evaluation_done_status_missing")
    if done.get("status") != "complete":
        reject("evaluation_done_not_complete")
    if not answer.strip():
        reject("evaluation_answer_missing")

    trace_id = str(done.get("trace_id") or "").strip()
    if not trace_id:
        reject("evaluation_trace_missing")
    trace = get_chat_trace(trace_id)
    if not trace:
        reject("evaluation_trace_not_persisted")
    if trace.get("status") != "complete":
        reject("evaluation_trace_not_complete")
    if trace.get("session_id") != session_id:
        reject("evaluation_trace_session_mismatch")
    return answer, done


def _direct_model_cost(trace_id: str) -> Optional[float]:
    calls, _ = _split_model_records(get_model_calls_for_trace(trace_id))
    if not calls:
        return None
    costs: List[float] = []
    for item in calls:
        usage = item.get("usage_normalized") or {}
        if isinstance(usage, str):
            try:
                usage = json.loads(usage)
            except (TypeError, ValueError, json.JSONDecodeError):
                usage = {}
        cost_source = item.get("cost_source") or (
            usage.get("cost_source") if isinstance(usage, dict) else None
        )
        value = item.get("estimated_cost_cny")
        if cost_source != "platform_price_snapshot" or value is None:
            return None
        costs.append(float(value))
    return round(sum(costs), 8)


def _direct_model_tokens(calls: List[Dict[str, Any]]) -> Optional[int]:
    provider_calls, _ = _split_model_records(calls)
    if not provider_calls:
        return None
    values: List[int] = []
    for item in provider_calls:
        value = _token_value(item, "total_tokens")
        if _token_source(item) != "provider_actual" or value is None:
            return None
        values.append(int(value))
    return sum(values)


def _enrich_runtime_evidence(
    done: Dict[str, Any], trace_id: Optional[str], session_id: str
) -> Dict[str, Any]:
    enriched = dict(done or {})
    trace = get_chat_trace(trace_id) if trace_id else None
    raw_model_calls = get_model_calls_for_trace(trace_id) if trace_id else []
    model_calls, logical_model_records = _split_model_records(raw_model_calls)
    ledger_row = get_evidence_ledger(trace_id) if trace_id else None
    trace_events = list_trace_events(trace_id) if trace_id else []
    if not enriched.get("decision_summary"):
        for event in reversed(trace_events):
            decision = (event.get("metadata") or {}).get("decision_summary")
            if decision:
                enriched["decision_summary"] = decision
                break
    if trace:
        enriched["current_agent_id"] = enriched.get("current_agent_id") or trace.get("agent_id")
        enriched["current_agent"] = enriched.get("current_agent") or trace.get("agent_name")
        enriched["route_intent"] = enriched.get("route_intent") or trace.get("intent")
    enriched["model_calls"] = model_calls
    enriched["logical_model_records"] = logical_model_records
    enriched["evidence_ledger"] = (ledger_row or {}).get("ledger") or {}
    enriched["trace_events"] = trace_events
    enriched["side_effects"] = get_session_runtime_side_effects(session_id)
    return enriched


def _evaluation_evidence(done: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "route_intent": done.get("route_intent"),
        "route_reason": done.get("route_reason"),
        "current_agent": done.get("current_agent"),
        "current_agent_id": done.get("current_agent_id"),
        "activated_skills": _normalize_skill_names(done),
        "tool_names": _normalize_tool_names(done),
        "mcp_calls": done.get("mcp_calls") or [],
        "citations": done.get("citations") or [],
        "handoff": bool(done.get("handoff")),
        "handoff_state": done.get("handoff_state"),
        "handoff_reason": done.get("handoff_reason"),
        "decision_summary": done.get("decision_summary") or {},
        "model_calls": done.get("model_calls") or [],
        "logical_model_records": done.get("logical_model_records") or [],
        "side_effects": done.get("side_effects") or {},
        "evidence_ledger": done.get("evidence_ledger") or {},
        "trace_events": done.get("trace_events") or [],
        "token_count": done.get("round_token_count") or done.get("token_count"),
        "usage_source": done.get("usage_source"),
    }


def _ensure_badcase_for_run(run_id: int) -> Dict[str, Any]:
    """Idempotently persist one source=evaluation Badcase for a failed run."""
    run = get_evaluation_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="evaluation run not found")
    if run.get("badcase_id"):
        existing = get_badcase_by_evaluation_run(run_id)
        return existing or {"id": run["badcase_id"]}
    existing = get_badcase_by_evaluation_run(run_id)
    if existing:
        update_evaluation_run(run_id, badcase_id=existing["id"])
        return existing
    if run.get("status") not in {"failed", "error"}:
        raise HTTPException(status_code=409, detail="仅失败运行可自动沉淀为 Badcase")
    case = get_evaluation_case(int(run["evaluation_case_id"]))
    if not case:
        raise HTTPException(status_code=404, detail="evaluation case not found")
    failed_rules = [
        item for item in (run.get("rule_results") or [])
        if item.get("status") == "fail"
    ]
    evidence = run.get("evidence") or {}
    expected = {
        "agent": case.get("expected_agent_id"),
        "skills": case.get("expected_skills"),
        "tools": case.get("expected_tools"),
        "citation_docs": case.get("expected_citation_docs"),
        "required_terms": case.get("required_terms"),
        "forbidden_terms": case.get("forbidden_terms"),
        "handoff": case.get("expected_handoff"),
        "deterministic_assertions": (case.get("rubric") or {}).get("deterministic_assertions") or {},
    }
    controlled = bool((case.get("rubric") or {}).get("controlled_failure_canary"))
    marker = "【受控故障注入】" if controlled else ""
    context = {
        "controlled_failure_canary": controlled,
        "evaluation_case": case,
        "evaluation_run": run,
        "failed_assertions": failed_rules,
        "answer": run.get("answer"),
        "runtime_evidence": evidence,
    }
    badcase = create_badcase(
        title=f"{marker}评估失败：{case['case_key']} · {case['title']}",
        description=(
            "验收用受控失败，验证评估到 Badcase 的沉淀链路。"
            if controlled else
            "Golden Set 确定性断言未通过，需按 Trace 归因。"
        ),
        category="pending",
        status="pending",
        source="evaluation",
        original_query=case.get("user_message"),
        ai_response=run.get("answer"),
        context_json=json.dumps(context, ensure_ascii=False, default=str),
        trace_id=run.get("trace_id"),
        priority="high" if case.get("risk_level") in {"L3", "L4"} else "medium",
        symptom="受控断言失败" if controlled else "评估断言失败",
        expected_behavior=json.dumps(expected, ensure_ascii=False, default=str),
        actual_behavior=json.dumps(evidence, ensure_ascii=False, default=str),
        root_cause_domain="controlled_test" if controlled else "unknown",
        impact_scope=f"评估用例 {case['case_key']} · 风险 {case.get('risk_level')}",
        linked_evaluation_case_id=case["id"],
        linked_evaluation_run_id=run_id,
    )
    update_evaluation_run(run_id, badcase_id=badcase["id"])
    return badcase


def _validate_linked_retest(case_id: int, badcase_id: Optional[int]) -> Optional[Dict[str, Any]]:
    if badcase_id is None:
        return None
    badcase = get_badcase(int(badcase_id))
    if not badcase:
        raise HTTPException(status_code=404, detail="linked Badcase not found")
    if badcase.get("status") != "verifying":
        raise HTTPException(status_code=409, detail="linked Evaluation retest requires Badcase status=verifying")
    linked_case_id = badcase.get("linked_evaluation_case_id")
    if linked_case_id is not None and int(linked_case_id) != int(case_id):
        raise HTTPException(status_code=409, detail="Badcase is linked to a different Evaluation case")
    return badcase


def _link_evaluation_retest(
    badcase: Dict[str, Any],
    case: Dict[str, Any],
    run: Dict[str, Any],
    retest_started_at: str,
) -> Dict[str, Any]:
    """Persist one explicit Evaluation run as the real retest for a Badcase."""
    run_id = int(run["id"])
    update_evaluation_run(run_id, badcase_id=int(badcase["id"]))
    run = get_evaluation_run(run_id) or run
    passed = run.get("status") == "passed"
    trace_id = str(run.get("trace_id") or "").strip()
    trace = get_chat_trace(trace_id) if trace_id else None
    session_id = str(run.get("session_id") or "").strip()
    trace_complete = bool(
        trace
        and trace.get("status") == "complete"
        and session_id
        and trace.get("session_id") == session_id
    )
    retest_complete = bool(
        passed
        and str(run.get("answer") or "").strip()
        and trace_complete
    )
    before_status = str(badcase.get("status") or "verifying")
    after_status = before_status if retest_complete else "fixing"
    retest_at = now_cn()
    baseline_run_id = badcase.get("linked_evaluation_run_id")
    baseline_trace_id = badcase.get("trace_id")
    retest_context = {
        "type": "evaluation_retest",
        "evaluation_case_id": case.get("id"),
        "evaluation_case_key": case.get("case_key"),
        "evaluation_run_id": run_id,
        "run_status": "complete" if retest_complete else "failed",
        "evaluation_run_status": run.get("status"),
        "trace_id": trace_id,
        "retest_started_at": retest_started_at,
        "session_id": session_id,
        "trace_persisted": bool(trace),
        "trace_status": (trace or {}).get("status"),
        "trace_session_id": (trace or {}).get("session_id"),
        "rule_results": run.get("rule_results") or [],
        "evidence": run.get("evidence") or {},
        "baseline_evaluation_run_id": baseline_run_id,
        "baseline_trace_id": baseline_trace_id,
    }
    updated = update_badcase(
        int(badcase["id"]),
        status=after_status,
        retest_response=run.get("answer") or "",
        retest_context_json=json.dumps(retest_context, ensure_ascii=False, default=str),
        retest_trace_id=run.get("trace_id") or "",
        last_retest_at=retest_at,
        linked_evaluation_case_id=int(case["id"]),
        linked_evaluation_run_id=run_id,
    )
    add_badcase_action(
        badcase_id=int(badcase["id"]),
        action_type="evaluation-retest",
        action_detail=json.dumps({
            "evaluation_case_id": case.get("id"),
            "evaluation_case_key": case.get("case_key"),
            "evaluation_run_id": run_id,
            "trace_id": run.get("trace_id"),
            "result": run.get("status"),
            "retest_run_status": retest_context["run_status"],
            "failed_rule_keys": [
                item.get("key") for item in (run.get("rule_results") or [])
                if item.get("status") == "fail"
            ],
            "baseline_evaluation_run_id": baseline_run_id,
            "baseline_trace_id": baseline_trace_id,
        }, ensure_ascii=False, default=str),
        status_before=before_status,
        status_after=after_status,
        created_by="operator",
    )
    return updated or badcase


@router.get("/overview")
async def overview():
    return {"summary": evaluation_summary()}


@router.get("/cases")
async def list_cases(status: Optional[str] = None, source: Optional[str] = None):
    if status and status not in CASE_STATUSES:
        raise HTTPException(status_code=400, detail=f"invalid status: {status}")
    if source and source not in SOURCES:
        raise HTTPException(status_code=400, detail=f"invalid source: {source}")
    return {"cases": list_evaluation_cases(status=status, source=source)}


@router.post("/cases")
async def create_case(request: EvaluationCaseCreate):
    payload = request.dict()
    _validate_case_payload(payload)
    try:
        case = create_evaluation_case(**payload)
    except Exception as exc:
        if "UNIQUE" in str(exc).upper():
            raise HTTPException(status_code=409, detail="case_key 已存在")
        raise
    return {"case": case}


@router.get("/cases/{case_id}")
async def get_case(case_id: int):
    case = get_evaluation_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="evaluation case not found")
    return {"case": case, "runs": list_evaluation_runs(evaluation_case_id=case_id)}


@router.put("/cases/{case_id}")
async def update_case(case_id: int, request: EvaluationCaseUpdate):
    if not get_evaluation_case(case_id):
        raise HTTPException(status_code=404, detail="evaluation case not found")
    payload = request.dict(exclude_unset=True)
    _validate_case_payload(payload)
    try:
        case = update_evaluation_case(case_id, **payload)
    except Exception as exc:
        if "UNIQUE" in str(exc).upper():
            raise HTTPException(status_code=409, detail="case_key 已存在")
        raise
    return {"case": case}


@router.post("/cases/{case_id}/run")
async def run_case(case_id: int, request: EvaluationRunRequest = EvaluationRunRequest()):
    """Explicitly run one active Golden Set case through the real chat runtime."""
    case = get_evaluation_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="evaluation case not found")
    if case.get("status") != "active":
        raise HTTPException(status_code=409, detail="仅 active 评估用例可运行；草稿请先人工审核并启用")
    linked_badcase = _validate_linked_retest(case_id, request.linked_badcase_id)

    # Evaluation is an explicit background-quality operation, unlike ordinary
    # owner chat.  Respect a hard budget stop before spending a new model call.
    budget_gate = _background_budget_gate("evaluation_run")
    if not budget_gate.get("allowed"):
        raise HTTPException(
            status_code=int(budget_gate["http_status"]),
            detail=budget_gate["detail"],
        )

    session_id = f"evaluation-{case['case_key'][:32]}-{uuid.uuid4().hex[:8]}"
    retest_started_at = now_cn()
    try:
        answer, done = await _run_real_chat(case["user_message"], session_id)
    except RuntimeExecutionError as exc:
        failure_trace_id = exc.trace_id or exc.done.get("trace_id")
        done = _enrich_runtime_evidence(
            exc.done, failure_trace_id, session_id
        )
        checks = [_rule(
            "runtime_completion", "Provider/运行时成功完成", "complete",
            done.get("status") or "failed", "fail", note=str(exc)[:500],
        )]
        evidence = _evaluation_evidence(done)
        run = create_evaluation_run(
            evaluation_case_id=case_id,
            status="failed",
            trace_id=failure_trace_id,
            session_id=session_id,
            answer=exc.answer,
            evidence={**evidence, "runtime_error": str(exc)[:500]},
            rule_results=checks,
            total_tokens=_direct_model_tokens(done.get("model_calls") or []),
            estimated_cost_cny=(
                _direct_model_cost(failure_trace_id)
                if failure_trace_id else None
            ),
        )
        badcase = (
            _link_evaluation_retest(linked_badcase, case, run, retest_started_at)
            if linked_badcase else _ensure_badcase_for_run(run["id"])
        )
        return {
            "case": case,
            "run": get_evaluation_run(run["id"]),
            "rule_results": checks,
            "badcase": badcase,
            "message": "运行失败；已保存真实失败并自动关联 Evaluation Badcase，未伪造成 PASS。",
        }
    except Exception as exc:
        checks = [_rule(
            "runtime_completion", "Provider/运行时成功完成", "complete",
            "failed", "fail", note=str(exc)[:500],
        )]
        run = create_evaluation_run(
            evaluation_case_id=case_id,
            status="failed",
            session_id=session_id,
            evidence={"runtime_error": str(exc)[:500]},
            rule_results=checks,
        )
        badcase = (
            _link_evaluation_retest(linked_badcase, case, run, retest_started_at)
            if linked_badcase else _ensure_badcase_for_run(run["id"])
        )
        return {
            "case": case,
            "run": get_evaluation_run(run["id"]),
            "rule_results": checks,
            "badcase": badcase,
            "message": "运行失败；已保留可见错误并自动关联 Evaluation Badcase。",
        }

    trace_id = done.get("trace_id")
    done = _enrich_runtime_evidence(done, trace_id, session_id)
    checks, run_status = evaluate_runtime_evidence(case, answer, done)
    evidence = _evaluation_evidence(done)
    total_tokens = _direct_model_tokens(done.get("model_calls") or [])
    cost = _direct_model_cost(trace_id) if trace_id else None
    run = create_evaluation_run(
        evaluation_case_id=case_id,
        trace_id=trace_id,
        session_id=session_id,
        status=run_status,
        answer=answer,
        evidence=evidence,
        rule_results=checks,
        total_tokens=total_tokens,
        estimated_cost_cny=cost,
    )
    if trace_id:
        update_chat_trace(
            trace_id,
            run_type="evaluation",
            evaluation_case_id=case_id,
            evaluation_run_id=run.get("id"),
            risk_level=case.get("risk_level"),
            version_snapshot=case.get("version_label") or "V1.6",
        )
        record_trace_event(
            trace_id, "evaluation_rule_gate", run_status,
            output_summary=f"{sum(1 for item in checks if item['status'] == 'pass')} pass / {sum(1 for item in checks if item['status'] == 'fail')} fail",
            metadata={"evaluation_case_id": case_id, "evaluation_run_id": run.get("id"), "risk_level": case.get("risk_level")},
        )
    if linked_badcase:
        badcase = _link_evaluation_retest(
            linked_badcase, case, run, retest_started_at
        )
    else:
        badcase = _ensure_badcase_for_run(run["id"]) if run_status == "failed" else None
    if badcase:
        run = get_evaluation_run(run["id"]) or run
    return {
        "case": case,
        "run": run,
        "rule_results": checks,
        "badcase": badcase,
        "budget": budget_gate,
        "message": "硬规则结果已生成；涉及业务可用性、语气和复杂 SOP 的 Rubric 仍需人工审核。",
    }


@router.post("/runs/{run_id}/review")
async def review_run(run_id: int, request: EvaluationReviewRequest):
    run = get_evaluation_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="evaluation run not found")
    if not request.note.strip():
        raise HTTPException(status_code=400, detail="请填写人工评审依据")
    updated = update_evaluation_run(
        run_id,
        status="passed" if request.passed else "failed",
        operator_judgement="passed" if request.passed else "failed",
        operator_note=request.note.strip(),
    )
    if not request.passed:
        _ensure_badcase_for_run(run_id)
        updated = get_evaluation_run(run_id)
    return {"run": updated}


@router.post("/runs/{run_id}/create-badcase")
async def create_badcase_from_run(run_id: int):
    """Backward-compatible idempotent access to the automatic S6 link."""
    badcase = _ensure_badcase_for_run(run_id)
    return {
        "badcase": badcase,
        "message": "该失败评估已关联唯一的 Trace 证据 Badcase。",
    }
