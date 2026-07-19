"""Create trace-linked Badcases from governed V1.8 runtime failures."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.runtime.contracts import RunEvidenceLedger
from db.property_db import add_badcase_action, create_badcase, list_badcases


AUTO_SOURCES = {
    "runtime_contract",
    "evaluation",
    "tool_failure",
    "runtime_failure",
}


def _failed_evaluations(ledger: RunEvidenceLedger) -> list[Dict[str, Any]]:
    return [
        item
        for item in ledger.evaluation_results
        if item.get("passed") is False or item.get("status") in {"failed", "error"}
    ]


def _failed_tools(ledger: RunEvidenceLedger) -> list[Dict[str, Any]]:
    return [
        item
        for item in ledger.tool_invocations
        if item.get("transport_status") in {"failed", "timeout"}
        or item.get("invocation_status") == "failed"
        or item.get("business_status") in {"failed", "rejected"}
    ]


def runtime_badcase_trigger(
    ledger: RunEvidenceLedger,
) -> Optional[Dict[str, Any]]:
    """Return one deterministic capture decision, or None for a healthy run."""

    violations = list(ledger.contract_violations)
    failed_evaluations = _failed_evaluations(ledger)
    failed_tools = _failed_tools(ledger)
    if not violations and not failed_evaluations and not failed_tools:
        return None

    if failed_tools:
        source = "tool_failure"
        category = "mcp_capability"
        root_cause_domain = "tool_mcp"
        reason = "Tool/MCP 调用或业务结果失败"
    elif violations:
        source = "runtime_contract"
        codes = {str(item.get("code") or "") for item in violations}
        if any("citation" in code or "evidence" in code for code in codes):
            category = "knowledge_gap"
            root_cause_domain = "knowledge_rag"
        elif any("skill" in code for code in codes):
            category = "skill_prompt"
            root_cause_domain = "model_instruction"
        else:
            category = "response_quality"
            root_cause_domain = "authority_safety"
        reason = "V1.8 运行时契约被违反"
    else:
        source = "evaluation"
        category = "response_quality"
        root_cause_domain = "unknown"
        reason = "运行时 Evaluation 断言失败"

    return {
        "source": source,
        "category": category,
        "root_cause_domain": root_cause_domain,
        "reason": reason,
        "contract_violations": violations,
        "failed_evaluations": failed_evaluations,
        "failed_tools": failed_tools,
    }


def capture_runtime_badcase(
    *,
    ledger: RunEvidenceLedger,
    original_query: str,
    ai_response: str,
    source_message_id: Optional[int] = None,
    runtime_error: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Persist one idempotent Badcase for a failed V1.8 Trace."""

    trigger = runtime_badcase_trigger(ledger)
    if runtime_error and trigger is None:
        trigger = {
            "source": "runtime_failure",
            "category": "other",
            "root_cause_domain": "external_dependency",
            "reason": "V1.8 运行时异常",
            "contract_violations": [
                {"code": "runtime_failure", "detail": runtime_error[:500]}
            ],
            "failed_evaluations": [],
            "failed_tools": [],
        }
    if trigger is None:
        return None

    for existing in list_badcases():
        if (
            str(existing.get("trace_id") or "") == ledger.trace_id
            and str(existing.get("source") or "") in AUTO_SOURCES
        ):
            return existing

    evidence = {
        "trace_id": ledger.trace_id,
        "config_snapshot": ledger.config_snapshot,
        **trigger,
    }
    case = create_badcase(
        title=f"运行时自动捕获：{trigger['reason']} · {ledger.trace_id}",
        description=trigger["reason"],
        category=trigger["category"],
        status="pending",
        evidence=json.dumps(evidence, ensure_ascii=False, default=str),
        source_message_id=source_message_id,
        message_id=source_message_id,
        session_id=ledger.session_id,
        source=trigger["source"],
        original_query=original_query,
        ai_response=ai_response,
        feedback_reason=trigger["reason"],
        context_json=json.dumps(
            {
                "trace_id": ledger.trace_id,
                "route_decision": ledger.route_decision,
                "activated_skills": ledger.activated_skills,
                "retrieval_evidence": ledger.retrieval_evidence,
                "tool_invocations": ledger.tool_invocations,
                "evaluation_results": ledger.evaluation_results,
                "contract_violations": ledger.contract_violations,
            },
            ensure_ascii=False,
            default=str,
        ),
        trace_id=ledger.trace_id,
        priority="high" if runtime_error or trigger["failed_tools"] else "medium",
        symptom=trigger["reason"],
        expected_behavior="运行时满足已发布 V1.8 契约，失败时安全降级并保留证据。",
        actual_behavior=runtime_error or trigger["reason"],
        root_cause_domain=trigger["root_cause_domain"],
    )
    add_badcase_action(
        badcase_id=int(case["id"]),
        action_type="auto-capture",
        action_detail=json.dumps(evidence, ensure_ascii=False, default=str),
        status_before="pending",
        status_after="pending",
        created_by="runtime",
    )
    return case
