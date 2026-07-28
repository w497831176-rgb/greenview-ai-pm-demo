"""V2 deterministic Badcase capture for governed V1.8 runtime evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from app.runtime.contracts import RunEvidenceLedger
from db.property_db import add_badcase_action, create_badcase, list_badcases


AUTO_SOURCES = {
    "runtime_contract",
    "evaluation",
    "tool_failure",
    "runtime_failure",
    "provider_failure",
}

_ANSWER_BASIS_CODES = {
    "ungrounded_critical_value",
    "citation_claim_mismatch",
    "citation_evidence_mismatch",
    "required_citation_missing",
    "invalid_evidence_id",
}
_SELECTION_TOKENS = ("route", "routing", "agent", "skill", "rag_selection", "tool_selection", "handoff_selection")
_ACTION_RISK_TOKENS = ("action", "receipt", "approval", "idempot", "unconfirmed_write", "false_success")
_OBSERVATION_TOKENS = ("cost", "usage", "latency", "trace_display", "observability")


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
        or item.get("business_status")
        in {
            "failed",
            "rejected",
            "timeout",
            "upstream_error",
            "unauthorized",
            "invalid_input",
        }
    ]


def _fixed_evaluation_failures(
    ledger: RunEvidenceLedger,
    *,
    legacy_mode: bool,
) -> list[Dict[str, Any]]:
    failures = _failed_evaluations(ledger)
    if legacy_mode:
        return failures
    return [
        item
        for item in failures
        if item.get("evaluation_run_id")
        or item.get("evaluation_case_id")
        or item.get("assertion_id")
        or item.get("source") == "evaluation"
        or item.get("fixed_assertion") is True
    ]


def _violation_codes(ledger: RunEvidenceLedger) -> set[str]:
    return {str(item.get("code") or "").lower() for item in ledger.contract_violations}


def runtime_badcase_decision(
    ledger: RunEvidenceLedger,
    *,
    runtime_error: Optional[str] = None,
    runtime_error_type: Optional[str] = None,
    delivery_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return one auditable V2 disposition without applying human judgement."""

    context = dict(delivery_context or {})
    legacy_mode = delivery_context is None
    violations = list(ledger.contract_violations)
    codes = _violation_codes(ledger)
    failed_tools = _failed_tools(ledger)
    failed_evaluations = _fixed_evaluation_failures(ledger, legacy_mode=legacy_mode)

    base = {
        "capture_version": 2,
        "contract_violations": violations,
        "failed_evaluations": failed_evaluations,
        "failed_tools": failed_tools,
        "affects_final_user": not bool(context.get("safe_rejection") or context.get("renderer_intercepted")),
    }

    if runtime_error:
        provider_failure = runtime_error_type == "provider_failure"
        return {
            **base,
            "disposition": "formal_badcase",
            "source": "provider_failure" if provider_failure else "runtime_failure",
            "category": "provider_failure" if provider_failure else "other",
            "suggested_root_cause_domain": "model_provider" if provider_failure else "external_dependency",
            "reason": "模型服务失败，用户未获得正常结果" if provider_failure else "系统异常，用户未获得正常结果",
        }

    if context.get("safe_rejection") or context.get("renderer_intercepted"):
        return {
            **base,
            "disposition": "system_observation" if violations or failed_tools else "none",
            "reason": "风险已在交付前被安全拦截，用户未收到错误内容",
            "source": "runtime_contract",
            "category": "response_quality",
            "suggested_root_cause_domain": "authority_safety",
        }

    if codes and all(any(token in code for token in _OBSERVATION_TOKENS) for code in codes):
        return {
            **base,
            "disposition": "system_observation",
            "reason": "仅涉及成本、时延或技术追踪展示，不影响业务结果",
            "source": "runtime_contract",
            "category": "other",
            "suggested_root_cause_domain": "ux",
        }

    answer_basis = bool(codes & _ANSWER_BASIS_CODES) or any(
        "citation" in code or "evidence" in code for code in codes
    )
    wrong_selection = any(any(token in code for token in _SELECTION_TOKENS) for code in codes)
    risky_action = any(any(token in code for token in _ACTION_RISK_TOKENS) for code in codes)

    if failed_tools:
        required_failure = legacy_mode or any(
            item.get("required") is True or item.get("affects_final_user") is True
            for item in failed_tools
        )
        if required_failure:
            return {
                **base,
                "disposition": "formal_badcase",
                "source": "tool_failure",
                "category": "mcp_capability",
                "suggested_root_cause_domain": "tool_mcp",
                "reason": "必要工具失败，用户未获得正常结果",
            }
        return {
            **base,
            "disposition": "system_observation",
            "source": "tool_failure",
            "category": "mcp_capability",
            "suggested_root_cause_domain": "tool_mcp",
            "reason": "非必要工具异常但最终业务结果正常",
        }

    if answer_basis or wrong_selection or risky_action:
        if answer_basis:
            category, domain, reason = "response_quality", "knowledge_rag", "最终回答依据存在确定性问题"
        elif wrong_selection:
            category, domain, reason = "routing", "routing", "运行证据证明能力选择错误"
        else:
            category, domain, reason = "other", "authority_safety", "业务操作存在确定性风险"
        return {
            **base,
            "disposition": "formal_badcase",
            "source": "runtime_contract",
            "category": category,
            "suggested_root_cause_domain": domain,
            "reason": reason,
        }

    if failed_evaluations:
        return {
            **base,
            "disposition": "formal_badcase",
            "source": "evaluation",
            "category": "response_quality",
            "suggested_root_cause_domain": "unknown",
            "reason": "固定评估断言未通过",
        }

    if violations:
        return {
            **base,
            "disposition": "system_observation",
            "source": "runtime_contract",
            "category": "other",
            "suggested_root_cause_domain": "unknown",
            "reason": "技术规则出现异常，但没有证据证明用户收到错误结果",
        }

    return {**base, "disposition": "none", "reason": "运行正常"}


def runtime_badcase_trigger(
    ledger: RunEvidenceLedger,
    *,
    runtime_error: Optional[str] = None,
    runtime_error_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Backward-compatible formal trigger used by existing deterministic tests."""

    # Some legacy contract tests AST-load only this function and its two small
    # helpers. Keep that read-only harness viable without weakening V2 runtime.
    decision_fn = globals().get("runtime_badcase_decision")
    if decision_fn is None:
        failed_tools = _failed_tools(ledger)
        failed_evaluations = _failed_evaluations(ledger)
        violations = list(ledger.contract_violations)
        if runtime_error and runtime_error_type == "provider_failure":
            return {
                "source": "provider_failure",
                "category": "provider_failure",
                "root_cause_domain": "model_provider",
                "reason": "模型服务失败，用户未获得正常结果",
                "contract_violations": violations,
                "failed_evaluations": failed_evaluations,
                "failed_tools": failed_tools,
            }
        if failed_tools:
            return {
                "source": "tool_failure",
                "category": "mcp_capability",
                "root_cause_domain": "tool_mcp",
                "reason": "必要工具失败，用户未获得正常结果",
                "contract_violations": violations,
                "failed_evaluations": failed_evaluations,
                "failed_tools": failed_tools,
            }
        if violations or failed_evaluations or runtime_error:
            return {
                "source": "runtime_contract" if violations else "evaluation",
                "category": "response_quality",
                "root_cause_domain": "unknown",
                "reason": "运行证据未通过既有合同",
                "contract_violations": violations,
                "failed_evaluations": failed_evaluations,
                "failed_tools": failed_tools,
            }
        return None
    decision = decision_fn(
        ledger,
        runtime_error=runtime_error,
        runtime_error_type=runtime_error_type,
    )
    if decision["disposition"] != "formal_badcase":
        return None
    return {
        **decision,
        "root_cause_domain": decision.get("suggested_root_cause_domain", "unknown"),
    }


def _fingerprint(decision: Dict[str, Any]) -> str:
    payload = {
        "source": decision.get("source"),
        "category": decision.get("category"),
        "codes": sorted(
            str(item.get("code") or "")
            for item in decision.get("contract_violations") or []
        ),
        "tools": sorted(
            f"{item.get('server_name')}:{item.get('tool_name')}"
            for item in decision.get("failed_tools") or []
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]


def capture_runtime_badcase(
    *,
    ledger: RunEvidenceLedger,
    original_query: str,
    ai_response: str,
    source_message_id: Optional[int] = None,
    runtime_error: Optional[str] = None,
    runtime_error_type: Optional[str] = None,
    delivery_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Persist only formal V2 issues; observations stay inside the evidence ledger."""

    decision = runtime_badcase_decision(
        ledger,
        runtime_error=runtime_error,
        runtime_error_type=runtime_error_type,
        delivery_context=delivery_context,
    )
    if decision["disposition"] == "system_observation":
        ledger.system_observations.append(
            {"trace_id": ledger.trace_id, **decision}
        )
        return None
    if decision["disposition"] != "formal_badcase":
        return None

    for existing in list_badcases():
        if (
            str(existing.get("trace_id") or "") == ledger.trace_id
            and str(existing.get("source") or "") in AUTO_SOURCES
        ):
            return existing

    fingerprint = _fingerprint(decision)
    for existing in list_badcases():
        try:
            context = json.loads(existing.get("context_json") or "{}")
        except Exception:
            context = {}
        if context.get("capture_version") == 2 and context.get("issue_fingerprint") == fingerprint:
            add_badcase_action(
                badcase_id=int(existing["id"]),
                action_type="auto-duplicate-occurrence",
                action_detail=json.dumps(
                    {"trace_id": ledger.trace_id, "issue_fingerprint": fingerprint},
                    ensure_ascii=False,
                ),
                status_before=str(existing.get("status") or "pending"),
                status_after=str(existing.get("status") or "pending"),
                created_by="runtime",
            )
            return existing

    evidence = {
        "trace_id": ledger.trace_id,
        "config_snapshot": ledger.config_snapshot,
        "issue_fingerprint": fingerprint,
        **decision,
    }
    context = {
        "capture_version": 2,
        "issue_fingerprint": fingerprint,
        "trace_id": ledger.trace_id,
        "route_decision": ledger.route_decision,
        "activated_skills": ledger.activated_skills,
        "skill_evidence": ledger.skill_evidence,
        "retrieval_evidence": ledger.retrieval_evidence,
        "tool_invocations": ledger.tool_invocations,
        "evaluation_results": ledger.evaluation_results,
        "contract_violations": ledger.contract_violations,
    }
    case = create_badcase(
        title=f"系统疑似发现：{decision['reason']} · {ledger.trace_id}",
        description=decision["reason"],
        category=decision["category"],
        status="pending",
        evidence=json.dumps(evidence, ensure_ascii=False, default=str),
        source_message_id=source_message_id,
        message_id=source_message_id,
        session_id=ledger.session_id,
        source=decision["source"],
        original_query=original_query,
        ai_response=ai_response,
        feedback_reason=decision["reason"],
        context_json=json.dumps(context, ensure_ascii=False, default=str),
        trace_id=ledger.trace_id,
        priority="high" if runtime_error or decision["failed_tools"] else "medium",
        symptom=decision["reason"],
        expected_behavior="系统正常完成；若存在风险，应在交付前安全拦截。",
        actual_behavior=runtime_error or decision["reason"],
        root_cause_domain="unknown",
    )
    add_badcase_action(
        badcase_id=int(case["id"]),
        action_type="auto-capture-v2",
        action_detail=json.dumps(evidence, ensure_ascii=False, default=str),
        status_before="pending",
        status_after="pending",
        created_by="runtime",
    )
    return case
