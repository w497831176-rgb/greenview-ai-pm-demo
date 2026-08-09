"""Closed, deterministic Badcase capture for governed runtime evidence."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from enum import Enum
from typing import Any, Dict, Optional

from app.runtime.contracts import RunEvidenceLedger
from db.property_db import (
    add_badcase_action,
    create_badcase,
    list_badcases,
    update_badcase,
)


AUTO_SOURCES = {
    "runtime_contract",
    "evaluation",
    "tool_failure",
    "runtime_failure",
    "provider_failure",
    "agent_insufficient_evidence",
    "agent_insufficient_capability",
    "agent_capability_unavailable",
}


class BadcaseTriggerCode(str, Enum):
    """Closed runtime facts that may create an active suspected Badcase."""

    ROUTER_CONTRACT_INVALID = "router_contract_invalid"
    AGENT_CONTRACT_INVALID = "agent_contract_invalid"
    RUNTIME_FAILED = "runtime_failed"
    CAPABILITY_FAILED = "capability_failed"
    CITATION_INVALID = "citation_invalid"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INSUFFICIENT_CAPABILITY = "insufficient_capability"


_RUNTIME_FAILURE_CODES = {
    "provider_failure",
    "runtime_failure",
}
_ROUTER_CONTRACT_INVALID_CODES = {
    "router_contract_invalid",
}
_AGENT_CONTRACT_INVALID_CODES = {
    "agent_contract_invalid",
    "contract_invalid",
    "internal_control_payload_leak",
}
_CAPABILITY_FAILURE_CODES = {
    "live_retrieval_failed",
    "skill_selected_not_loaded",
}
_CITATION_INVALID_CODES = {
    "invalid_positional_citation",
    "invalid_evidence_id",
    "unsupported_evidence_citation",
    "unsupported_critical_value",
    "unstructured_reference_marker",
    "ungrounded_critical_value",
    "citation_claim_mismatch",
    "citation_evidence_mismatch",
    "required_citation_missing",
}
_ACTION_FAILURE_CODES = {
    "action_failed",
    "action_gateway_failed",
    "approval_missing",
    "false_success_without_receipt",
    "receipt_missing",
    "unconfirmed_write",
}
_FAILED_ACTION_STATUSES = {"error", "failed"}
_AGENT_ANSWER_STATUS_TRIGGERS = {
    "insufficient_evidence": BadcaseTriggerCode.INSUFFICIENT_EVIDENCE,
    "insufficient_capability": BadcaseTriggerCode.INSUFFICIENT_CAPABILITY,
    # Backward-compatible runtime alias; the persisted closed trigger is the
    # product term above rather than a second lifecycle category.
    "capability_unavailable": BadcaseTriggerCode.INSUFFICIENT_CAPABILITY,
}
_KNOWN_AGENT_ANSWER_STATUSES = {"answered", *_AGENT_ANSWER_STATUS_TRIGGERS}


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


def _violations_for_codes(
    ledger: RunEvidenceLedger,
    allowed: set[str],
) -> list[Dict[str, Any]]:
    return [
        item
        for item in ledger.contract_violations
        if str(item.get("code") or "").strip().lower() in allowed
    ]


def _failed_actions(ledger: RunEvidenceLedger) -> list[Dict[str, Any]]:
    return [
        item
        for item in ledger.action_receipts
        if str(item.get("status") or "").strip().lower()
        in _FAILED_ACTION_STATUSES
    ]


def _selected_agent_id(ledger: RunEvidenceLedger) -> str:
    for payload in (
        ledger.lane_decision,
        ledger.route_decision,
    ):
        if isinstance(payload, dict):
            value = payload.get("selected_agent_id") or payload.get("target_agent_id")
            if value:
                return str(value)
    return "unknown_agent"


def _release_id(ledger: RunEvidenceLedger) -> str:
    snapshot = ledger.config_snapshot or {}
    return str(
        snapshot.get("release_id")
        or snapshot.get("snapshot_hash")
        or snapshot.get("snapshot_id")
        or "unknown_release"
    )


def _component_list(*values: Any) -> str:
    components = sorted(
        {
            str(value).strip()
            for value in values
            if value is not None and str(value).strip()
        }
    )
    return "+".join(components) or "runtime"


def _violation_component(prefix: str, violations: list[Dict[str, Any]]) -> str:
    return _component_list(
        *(
            f"{prefix}:{str(item.get('code') or 'unknown').strip().lower()}"
            for item in violations
        )
    )


def _formal_decision(
    base: Dict[str, Any],
    *,
    trigger_code: BadcaseTriggerCode,
    source: str,
    category: str,
    root_cause_domain: str,
    reason: str,
    component: str,
) -> Dict[str, Any]:
    return {
        **base,
        "disposition": "formal_badcase",
        "trigger_code": trigger_code.value,
        "source": source,
        "category": category,
        "suggested_root_cause_domain": root_cause_domain,
        "reason": reason,
        "component": component,
    }


def runtime_badcase_decision(
    ledger: RunEvidenceLedger,
    *,
    runtime_error: Optional[str] = None,
    runtime_error_type: Optional[str] = None,
    agent_answer_status: Optional[str] = None,
    delivery_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return one auditable disposition from closed structured runtime facts.

    User-facing prose is deliberately absent from this decision. Unknown
    contract codes remain observations and cannot become active Badcases by
    keyword or substring coincidence.
    """

    context = dict(delivery_context or {})
    legacy_mode = delivery_context is None
    violations = list(ledger.contract_violations)
    codes = _violation_codes(ledger)
    failed_tools = _failed_tools(ledger)
    failed_evaluations = _fixed_evaluation_failures(ledger, legacy_mode=legacy_mode)
    failed_actions = _failed_actions(ledger)
    runtime_violations = _violations_for_codes(ledger, _RUNTIME_FAILURE_CODES)
    router_contract_violations = _violations_for_codes(
        ledger, _ROUTER_CONTRACT_INVALID_CODES
    )
    agent_contract_violations = _violations_for_codes(
        ledger, _AGENT_CONTRACT_INVALID_CODES
    )
    capability_violations = _violations_for_codes(
        ledger, _CAPABILITY_FAILURE_CODES
    )
    citation_violations = _violations_for_codes(ledger, _CITATION_INVALID_CODES)
    action_violations = _violations_for_codes(ledger, _ACTION_FAILURE_CODES)
    normalized_answer_status = (
        str(agent_answer_status).strip().lower()
        if agent_answer_status is not None
        else None
    )

    base = {
        "capture_version": 4,
        "contract_violations": violations,
        "failed_evaluations": failed_evaluations,
        "failed_tools": failed_tools,
        "failed_actions": failed_actions,
        "agent_answer_status": normalized_answer_status,
        "release_id": _release_id(ledger),
        "affects_final_user": not bool(context.get("renderer_intercepted")),
    }

    normalized_runtime_error_type = str(runtime_error_type or "").strip().lower()
    if (
        normalized_runtime_error_type in _ROUTER_CONTRACT_INVALID_CODES
        or router_contract_violations
    ):
        return _formal_decision(
            base,
            trigger_code=BadcaseTriggerCode.ROUTER_CONTRACT_INVALID,
            source="runtime_contract",
            category="routing",
            root_cause_domain="routing",
            reason="Router structured result failed contract validation",
            component=(
                _violation_component("router", router_contract_violations)
                if router_contract_violations
                else "router:contract"
            ),
        )

    if (
        normalized_runtime_error_type in _AGENT_CONTRACT_INVALID_CODES
        or agent_contract_violations
    ):
        return _formal_decision(
            base,
            trigger_code=BadcaseTriggerCode.AGENT_CONTRACT_INVALID,
            source="runtime_contract",
            category="response_quality",
            root_cause_domain="model_instruction",
            reason="Selected Agent structured result failed contract validation",
            component=(
                _violation_component("agent", agent_contract_violations)
                if agent_contract_violations
                else f"agent:{_selected_agent_id(ledger)}"
            ),
        )

    if runtime_error or runtime_violations:
        provider_failure = runtime_error_type == "provider_failure" or any(
            str(item.get("code") or "").strip().lower() == "provider_failure"
            for item in runtime_violations
        )
        return _formal_decision(
            base,
            trigger_code=BadcaseTriggerCode.RUNTIME_FAILED,
            source="provider_failure" if provider_failure else "runtime_failure",
            category="provider_failure" if provider_failure else "other",
            root_cause_domain=(
                "model_provider" if provider_failure else "external_dependency"
            ),
            reason=(
                "模型服务失败，用户未获得正常结果"
                if provider_failure
                else "系统异常，用户未获得正常结果"
            ),
            component=(
                "provider"
                if provider_failure
                else _violation_component("runtime", runtime_violations)
                if runtime_violations
                else "runtime"
            ),
        )

    if action_violations or failed_actions:
        action_components = [
            f"action:{item.get('proposal_id') or item.get('action_type') or 'gateway'}"
            for item in failed_actions
        ]
        return _formal_decision(
            base,
            trigger_code=BadcaseTriggerCode.RUNTIME_FAILED,
            source="runtime_contract",
            category="other",
            root_cause_domain="authority_safety",
            reason="结构化业务操作证据表明执行失败",
            component=_component_list(
                _violation_component("action", action_violations)
                if action_violations
                else None,
                *action_components,
            ),
        )

    if citation_violations or context.get("renderer_intercepted") is True:
        return _formal_decision(
            base,
            trigger_code=BadcaseTriggerCode.CITATION_INVALID,
            source="runtime_contract",
            category="response_quality",
            root_cause_domain="knowledge_rag",
            reason="结构化引用校验失败或依据在交付前被拦截",
            component=(
                _violation_component("citation", citation_violations)
                if citation_violations
                else "citation:renderer_intercepted"
            ),
        )

    if failed_tools:
        return _formal_decision(
            base,
            trigger_code=BadcaseTriggerCode.CAPABILITY_FAILED,
            source="tool_failure",
            category="mcp_capability",
            root_cause_domain="tool_mcp",
            reason="本轮实际能力调用出现结构化失败",
            component=_component_list(
                *(
                    f"tool:{item.get('server_name') or 'local'}/{item.get('tool_name') or 'unknown'}"
                    for item in failed_tools
                )
            ),
        )

    if capability_violations:
        return _formal_decision(
            base,
            trigger_code=BadcaseTriggerCode.CAPABILITY_FAILED,
            source="runtime_contract",
            category="mcp_capability",
            root_cause_domain="unknown",
            reason="已选能力出现结构化执行失败",
            component=_violation_component("capability", capability_violations),
        )

    agent_trigger = _AGENT_ANSWER_STATUS_TRIGGERS.get(
        normalized_answer_status or ""
    )
    if agent_trigger is not None:
        insufficient = agent_trigger == BadcaseTriggerCode.INSUFFICIENT_EVIDENCE
        return _formal_decision(
            base,
            trigger_code=agent_trigger,
            source=(
                "agent_insufficient_evidence"
                if insufficient
                else "agent_insufficient_capability"
            ),
            category="knowledge_gap" if insufficient else "mcp_capability",
            root_cause_domain="knowledge_rag" if insufficient else "unknown",
            reason=(
                "已选Agent结构化自报依据不足"
                if insufficient
                else "已选Agent结构化自报能力不可用"
            ),
            component=f"agent:{_selected_agent_id(ledger)}",
        )

    if (
        normalized_answer_status is not None
        and normalized_answer_status not in _KNOWN_AGENT_ANSWER_STATUSES
    ):
        return _formal_decision(
            base,
            trigger_code=BadcaseTriggerCode.AGENT_CONTRACT_INVALID,
            source="runtime_contract",
            category="response_quality",
            root_cause_domain="model_instruction",
            reason="Selected Agent returned an invalid answer_status",
            component=f"agent:{_selected_agent_id(ledger)}",
        )

    if failed_evaluations:
        return {
            **base,
            "disposition": "system_observation",
            "trigger_code": None,
            "source": "evaluation",
            "category": "response_quality",
            "suggested_root_cause_domain": "unknown",
            "reason": "Evaluation differences require an operator decision",
            "component": "evaluation",
        }

    if violations:
        return {
            **base,
            "disposition": "system_observation",
            "trigger_code": None,
            "source": "runtime_contract",
            "category": "other",
            "suggested_root_cause_domain": "unknown",
            "reason": "未注册的结构化合同码仅记录观察，不进入活跃Badcase",
            "component": _component_list(*(f"contract:{code}" for code in codes)),
        }

    return {
        **base,
        "disposition": "none",
        "trigger_code": None,
        "reason": "运行正常",
        "component": "runtime",
    }


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


def _problem_fingerprint(original_query: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(original_query or ""))
    normalized = " ".join(normalized.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _fingerprint(
    decision: Dict[str, Any],
    *,
    release_id: str,
    problem_fingerprint: str,
) -> str:
    payload = {
        "trigger_code": decision.get("trigger_code"),
        "component": decision.get("component"),
        "release_id": release_id,
        "problem_fingerprint": problem_fingerprint,
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
    agent_answer_status: Optional[str] = None,
    delivery_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Persist only closed structured issues; observations stay in the ledger."""

    decision = runtime_badcase_decision(
        ledger,
        runtime_error=runtime_error,
        runtime_error_type=runtime_error_type,
        agent_answer_status=agent_answer_status,
        delivery_context=delivery_context,
    )
    if decision["disposition"] == "system_observation":
        ledger.system_observations.append(
            {"trace_id": ledger.trace_id, **decision}
        )
        return None
    if decision["disposition"] != "formal_badcase":
        return None

    existing_cases = list_badcases()
    for existing in existing_cases:
        try:
            existing_context = json.loads(existing.get("context_json") or "{}")
        except Exception:
            existing_context = {}
        if (
            str(existing.get("trace_id") or "") == ledger.trace_id
            and str(existing.get("source") or "") in AUTO_SOURCES
            and str(existing_context.get("trigger_code") or "")
            == str(decision.get("trigger_code") or "")
        ):
            occurrence_count = max(
                1, int(existing_context.get("occurrence_count") or 1)
            ) + 1
            existing_context.update(
                {
                    "capture_version": 4,
                    "occurrence_count": occurrence_count,
                    "last_occurrence_trace_id": ledger.trace_id,
                }
            )
            updated = update_badcase(
                int(existing["id"]),
                context_json=json.dumps(
                    existing_context, ensure_ascii=False, default=str
                ),
            )
            add_badcase_action(
                badcase_id=int(existing["id"]),
                action_type="auto-duplicate-occurrence",
                action_detail=json.dumps(
                    {
                        "trace_id": ledger.trace_id,
                        "trigger_code": decision.get("trigger_code"),
                        "occurrence_count": occurrence_count,
                    },
                    ensure_ascii=False,
                ),
                status_before=str(existing.get("status") or "pending"),
                status_after=str(existing.get("status") or "pending"),
                created_by="runtime",
            )
            return updated or existing

    release_id = _release_id(ledger)
    problem_fingerprint = _problem_fingerprint(original_query)
    fingerprint = _fingerprint(
        decision,
        release_id=release_id,
        problem_fingerprint=problem_fingerprint,
    )
    evidence = {
        "trace_id": ledger.trace_id,
        "config_snapshot": ledger.config_snapshot,
        "issue_fingerprint": fingerprint,
        "problem_fingerprint": problem_fingerprint,
        **decision,
    }
    context = {
        "capture_version": 4,
        "issue_fingerprint": fingerprint,
        "problem_fingerprint": problem_fingerprint,
        "occurrence_count": 1,
        "occurrence_trace_ids": [ledger.trace_id],
        "last_occurrence_trace_id": ledger.trace_id,
        "trigger_code": decision.get("trigger_code"),
        "component": decision.get("component"),
        "release_id": release_id,
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
        action_type="auto-capture-v4",
        action_detail=json.dumps(evidence, ensure_ascii=False, default=str),
        status_before="pending",
        status_after="pending",
        created_by="runtime",
    )
    return case
