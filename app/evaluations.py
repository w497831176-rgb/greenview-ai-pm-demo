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

from app.observability import _check_budget
from db.property_db import (
    create_badcase,
    create_evaluation_case,
    create_evaluation_run,
    evaluation_summary,
    get_evaluation_case,
    get_evaluation_run,
    get_model_calls_for_trace,
    list_evaluation_cases,
    list_evaluation_runs,
    record_trace_event,
    update_badcase,
    update_chat_trace,
    update_evaluation_case,
    update_evaluation_run,
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
    return {
        "key": key,
        "label": label,
        "expected": expected,
        "actual": actual,
        "status": status,
        "hard": hard,
        "note": note,
    }


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

    hard_fail = any(item["hard"] and item["status"] == "fail" for item in checks)
    needs_manual = bool(case.get("rubric"))
    status = "failed" if hard_fail else "needs_manual_review" if needs_manual else "passed"
    return checks, status


async def _run_real_chat(message: str, session_id: str) -> Tuple[str, Dict[str, Any]]:
    """Run through the canonical owner runtime and reconstruct its SSE done event."""
    from app.chat import _stream_agent_response

    answer = ""
    done: Dict[str, Any] = {}
    async for chunk in _stream_agent_response(message, session_id, "evaluation"):
        for line in chunk.splitlines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("content"):
                answer += str(payload["content"])
            if payload.get("status") == "complete" or payload.get("message_id"):
                done = payload
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))
    if not done:
        done = {"status": "complete", "answer": answer}
    return answer, done


def _direct_model_cost(trace_id: str) -> Optional[float]:
    calls = get_model_calls_for_trace(trace_id)
    if not calls:
        return 0.0
    costs = [item.get("estimated_cost_cny") for item in calls]
    if any(value is None for value in costs):
        return None
    return round(sum(float(value or 0.0) for value in costs), 8)


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
async def run_case(case_id: int):
    """Explicitly run one active Golden Set case through the real chat runtime."""
    case = get_evaluation_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="evaluation case not found")
    if case.get("status") != "active":
        raise HTTPException(status_code=409, detail="仅 active 评估用例可运行；草稿请先人工审核并启用")

    # Evaluation is an explicit background-quality operation, unlike ordinary
    # owner chat.  Respect a hard budget stop before spending a new model call.
    budget = _check_budget("evaluation_run")
    if budget.get("alert_level") == "blocked":
        raise HTTPException(status_code=403, detail=budget.get("reason") or "预算已达上限，评估运行已阻止")

    session_id = f"evaluation-{case['case_key'][:32]}-{uuid.uuid4().hex[:8]}"
    try:
        answer, done = await _run_real_chat(case["user_message"], session_id)
    except Exception as exc:
        run = create_evaluation_run(
            evaluation_case_id=case_id,
            status="error",
            session_id=session_id,
            evidence={"error": str(exc)[:500]},
        )
        return {"case": case, "run": run, "message": "运行失败，已保留错误证据；未伪造评估结论。"}

    trace_id = done.get("trace_id")
    checks, run_status = evaluate_runtime_evidence(case, answer, done)
    evidence = {
        "route_intent": done.get("route_intent"),
        "route_reason": done.get("route_reason"),
        "current_agent": done.get("current_agent"),
        "current_agent_id": done.get("current_agent_id"),
        "activated_skills": _normalize_skill_names(done),
        "tool_names": _normalize_tool_names(done),
        "mcp_calls": done.get("mcp_calls") or [],
        "citations": done.get("citations") or [],
        "handoff": bool(done.get("handoff")),
        "token_count": done.get("round_token_count") or done.get("token_count"),
        "usage_source": done.get("usage_source"),
    }
    cost = _direct_model_cost(trace_id) if trace_id else None
    run = create_evaluation_run(
        evaluation_case_id=case_id,
        trace_id=trace_id,
        session_id=session_id,
        status=run_status,
        answer=answer,
        evidence=evidence,
        rule_results=checks,
        total_tokens=evidence.get("token_count"),
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
    return {
        "case": case,
        "run": run,
        "rule_results": checks,
        "budget": budget,
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
    return {"run": updated}


@router.post("/runs/{run_id}/create-badcase")
async def create_badcase_from_run(run_id: int):
    """Turn a failed evaluation into a trace-linked Badcase on operator click."""
    run = get_evaluation_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="evaluation run not found")
    if run.get("badcase_id"):
        return {"badcase_id": run["badcase_id"], "message": "该评估运行已关联 Badcase"}
    if run.get("status") not in {"failed", "needs_manual_review"}:
        raise HTTPException(status_code=409, detail="仅失败或待人工评审的运行可沉淀为 Badcase")
    case = get_evaluation_case(int(run["evaluation_case_id"]))
    if not case:
        raise HTTPException(status_code=404, detail="evaluation case not found")
    failed_rules = [item for item in (run.get("rule_results") or []) if item.get("status") == "fail"]
    evidence = run.get("evidence") or {}
    expected = {
        "agent": case.get("expected_agent_id"),
        "skills": case.get("expected_skills"),
        "tools": case.get("expected_tools"),
        "citation_docs": case.get("expected_citation_docs"),
        "required_terms": case.get("required_terms"),
        "forbidden_terms": case.get("forbidden_terms"),
        "handoff": case.get("expected_handoff"),
    }
    badcase = create_badcase(
        title=f"评估失败：{case['case_key']} · {case['title']}",
        description="Golden Set 规则未通过，需按 Trace 归因并修复。",
        category="pending",
        status="pending",
        source="evaluation",
        original_query=case.get("user_message"),
        ai_response=run.get("answer"),
        context_json=json.dumps({"evaluation_case": case, "evaluation_run": run, "failed_rules": failed_rules}, ensure_ascii=False, default=str),
        trace_id=run.get("trace_id"),
        priority="high" if case.get("risk_level") in {"L3", "L4"} else "medium",
        symptom="评估规则失败",
        expected_behavior=json.dumps(expected, ensure_ascii=False),
        actual_behavior=json.dumps(evidence, ensure_ascii=False, default=str),
        root_cause_domain="unknown",
        impact_scope=f"评估用例 {case['case_key']} · 风险 {case.get('risk_level')}",
        linked_evaluation_case_id=case["id"],
        linked_evaluation_run_id=run_id,
    )
    update_evaluation_run(run_id, badcase_id=badcase["id"])
    return {"badcase": badcase, "message": "已将失败评估沉淀为 Trace 关联 Badcase；尚未自动判定根因。"}

