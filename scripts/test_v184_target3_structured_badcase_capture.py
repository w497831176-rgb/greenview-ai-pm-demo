"""Focused target-3 contracts; no Provider call is permitted."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


TEMP_DIR = tempfile.TemporaryDirectory(prefix="yiai-target3-")
os.environ["PROPERTY_DATA_DIR"] = TEMP_DIR.name

from app.badcase_schema import _enrich_badcase, user_status_label
from app.evaluations import evaluate_runtime_evidence
from app.runtime.badcase_capture import (
    BadcaseTriggerCode,
    capture_runtime_badcase,
    runtime_badcase_decision,
)
from app.runtime.contracts import RunEvidenceLedger
from db import property_db


checks: list[str] = []


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def ledger(
    trace_id: str,
    *,
    contract_violations: list[dict] | None = None,
    tool_invocations: list[dict] | None = None,
    action_receipts: list[dict] | None = None,
    evaluation_results: list[dict] | None = None,
) -> RunEvidenceLedger:
    return RunEvidenceLedger(
        trace_id=trace_id,
        session_id=f"session-{trace_id}",
        config_snapshot={
            "snapshot_id": "snapshot-fixture",
            "release_id": "release-fixture",
            "snapshot_hash": "hash-fixture",
        },
        lane_decision={
            "lane": "B_PROPERTY_GOVERNED",
            "selected_agent_id": "agent-fixture",
            "reason": "fixture",
        },
        contract_violations=contract_violations or [],
        tool_invocations=tool_invocations or [],
        action_receipts=action_receipts or [],
        evaluation_results=evaluation_results or [],
    )


def decide(value: RunEvidenceLedger, **kwargs) -> dict:
    return runtime_badcase_decision(
        value,
        delivery_context=kwargs.pop("delivery_context", {"normal_completed": True}),
        **kwargs,
    )


def violation(code: str) -> list[dict]:
    return [{"code": code, "detail": "fixture"}]


property_db.init_db()

check(
    "01 closed automatic trigger vocabulary",
    {item.value for item in BadcaseTriggerCode}
    == {
        "router_contract_invalid",
        "agent_contract_invalid",
        "runtime_failed",
        "capability_failed",
        "citation_invalid",
        "insufficient_evidence",
        "insufficient_capability",
    },
)
check(
    "02 router schema failure is distinct",
    decide(
        ledger("router-contract"),
        runtime_error="fixture",
        runtime_error_type="router_contract_invalid",
    )["trigger_code"]
    == "router_contract_invalid",
)
check(
    "03 agent schema failure is distinct",
    decide(
        ledger("agent-contract", contract_violations=violation("agent_contract_invalid"))
    )["trigger_code"]
    == "agent_contract_invalid",
)
check(
    "04 Provider failure is runtime_failed",
    decide(
        ledger("runtime"),
        runtime_error="fixture",
        runtime_error_type="provider_failure",
    )["trigger_code"]
    == "runtime_failed",
)
check(
    "05 required capability failure is structural",
    decide(
        ledger(
            "capability",
            tool_invocations=[
                {
                    "tool_name": "tool-fixture",
                    "transport_status": "failed",
                    "required": True,
                }
            ],
        )
    )["trigger_code"]
    == "capability_failed",
)
failed_optional_tool = [
    {
        "tool_name": "tool-fixture",
        "transport_status": "failed",
        "required": False,
        "affects_final_user": False,
    }
]
check(
    "05a Agent capability self-report retains capability failure",
    decide(
        ledger("capability-self-report", tool_invocations=failed_optional_tool),
        agent_answer_status="insufficient_capability",
    )["trigger_code"]
    == "capability_failed",
)
check(
    "05b every actually executed failed Tool is a suspected capability failure",
    decide(
        ledger("capability-optional", tool_invocations=failed_optional_tool),
        agent_answer_status="answered",
    )["trigger_code"]
    == "capability_failed",
)
check(
    "06 citation contract failure is structural",
    decide(
        ledger("citation", contract_violations=violation("invalid_evidence_id"))
    )["trigger_code"]
    == "citation_invalid",
)
check(
    "07 action runtime failure stays inside closed runtime trigger",
    decide(
        ledger(
            "action",
            action_receipts=[{"receipt_id": "receipt-fixture", "status": "failed"}],
        )
    )["trigger_code"]
    == "runtime_failed",
)

insufficient_evidence = decide(
    ledger("self-report-evidence"), agent_answer_status="insufficient_evidence"
)
check(
    "08 Agent self-report evidence has its own source",
    insufficient_evidence["trigger_code"] == "insufficient_evidence"
    and insufficient_evidence["source"] == "agent_insufficient_evidence",
)
insufficient_capability = decide(
    ledger("self-report-capability"), agent_answer_status="capability_unavailable"
)
check(
    "09 legacy Agent alias persists canonical capability trigger",
    insufficient_capability["trigger_code"] == "insufficient_capability"
    and insufficient_capability["source"] == "agent_insufficient_capability",
)
check(
    "10 invalid Agent answer status is an Agent contract failure",
    decide(ledger("invalid-answer-status"), agent_answer_status="invalid-fixture")[
        "trigger_code"
    ]
    == "agent_contract_invalid",
)
check(
    "11 evaluation difference alone is only an observation",
    decide(
        ledger(
            "evaluation",
            evaluation_results=[
                {
                    "evaluation_case_id": 1,
                    "assertion_id": "fixture",
                    "passed": False,
                    "status": "failed",
                }
            ],
        )
    )["disposition"]
    == "system_observation",
)
check(
    "12 unregistered runtime code cannot open a Badcase",
    decide(
        ledger("unknown", contract_violations=violation("unregistered-fixture"))
    )["disposition"]
    == "system_observation",
)

golden_before = len(property_db.list_evaluation_cases())
badcases_before = len(property_db.list_badcases())
check(
    "13 answer prose alone creates no Badcase",
    capture_runtime_badcase(
        ledger=ledger("prose-only"),
        original_query="symbolic-input",
        ai_response="symbolic-output",
        agent_answer_status="answered",
        delivery_context={"normal_completed": True},
    )
    is None
    and len(property_db.list_badcases()) == badcases_before,
)

first = capture_runtime_badcase(
    ledger=ledger("same-trace"),
    original_query="symbolic-input",
    ai_response="symbolic-output",
    agent_answer_status="insufficient_evidence",
    delivery_context={"normal_completed": True},
)
second = capture_runtime_badcase(
    ledger=ledger("same-trace"),
    original_query="symbolic-input",
    ai_response="symbolic-output",
    agent_answer_status="insufficient_evidence",
    delivery_context={"normal_completed": True},
)
same_context = json.loads(
    (property_db.get_badcase(int(first["id"])) or {}).get("context_json") or "{}"
)
check(
    "14 same trace and trigger dedupe to one row",
    first["id"] == second["id"] and same_context.get("occurrence_count") == 2,
)
different_trigger = capture_runtime_badcase(
    ledger=ledger("same-trace", contract_violations=violation("invalid_evidence_id")),
    original_query="symbolic-input",
    ai_response="symbolic-output",
    agent_answer_status="answered",
    delivery_context={"normal_completed": True},
)
check(
    "15 same trace with a different closed trigger is a distinct issue",
    different_trigger["id"] != first["id"],
)
different_trace = capture_runtime_badcase(
    ledger=ledger("different-trace"),
    original_query="symbolic-input",
    ai_response="symbolic-output",
    agent_answer_status="insufficient_evidence",
    delivery_context={"normal_completed": True},
)
check(
    "16 different trace remains a distinct suspected Badcase",
    different_trace["id"] != first["id"],
)
check(
    "17 automatic capture never creates Golden data",
    len(property_db.list_evaluation_cases()) == golden_before,
)

check(
    "18 four human status labels",
    [user_status_label(value) for value in ("pending", "fixing", "verifying", "closed")]
    == ["待处理", "处理中", "待验证", "已关闭"],
)
presented = _enrich_badcase(
    {
        "id": 999,
        "status": "pending",
        "source": "agent_insufficient_capability",
        "category": "mcp_capability",
        "actions": [],
    }
)
check(
    "19 Agent self-report is visible but remains pending human handling",
    presented["source_label"] == "Agent自报：能力不可用"
    and presented["user_status_label"] == "待处理",
)

golden_case = {
    "expected_agent_id": "agent-fixture",
    "expected_handoff": False,
    "required_terms": ["must-not-be-used"],
    "forbidden_terms": ["must-not-be-used"],
    "rubric": {
        "expected_lane": "B",
        "capability_expectations": {
            "skill": "must",
            "rag": "forbid",
            "mcp": "ignore",
            "tool": "must",
        },
        "operator_rubric": "human-only-fixture",
    },
}
golden_done = {
    "status": "complete",
    "lane_decision": {"lane": "B", "selected_agent_id": "agent-fixture"},
    "current_agent_id": "agent-fixture",
    "activated_skills": [{"name": "skill-fixture"}],
    "citations": [],
    "mcp_calls": [],
    "tool_calls": [{"tool_name": "tool-fixture", "status": "success"}],
    "handoff": False,
}
golden_checks, golden_status = evaluate_runtime_evidence(
    golden_case, "symbolic-output", golden_done
)
by_key = {item["key"]: item for item in golden_checks}
check(
    "20 Golden structural checks cover lane and capability modes",
    golden_status == "needs_manual_review"
    and by_key["lane"]["status"] == "pass"
    and by_key["capability_skill"]["status"] == "pass"
    and by_key["capability_rag"]["status"] == "pass"
    and by_key["capability_tool"]["status"] == "pass",
)
check(
    "21 answer term lists are not part of Golden core checks",
    "required_terms" not in by_key
    and "forbidden_terms" not in by_key
    and "knowledge_insufficient" not in by_key,
)
mismatch_done = {**golden_done, "lane_decision": {"lane": "C"}, "tool_calls": []}
mismatch_checks, mismatch_status = evaluate_runtime_evidence(
    golden_case, "symbolic-output", mismatch_done
)
check(
    "22 structural mismatch still awaits the operator verdict",
    mismatch_status == "needs_manual_review"
    and any(item["status"] == "fail" for item in mismatch_checks),
)

evaluation_source = (
    Path(__file__).resolve().parents[1] / "app" / "evaluations.py"
).read_text(encoding="utf-8")
check(
    "23 evaluation runner does not auto-create Badcases",
    "if linked_badcase else None" in evaluation_source
    and 'badcase = _ensure_badcase_for_run(run["id"]) if run_status == "failed" else None'
    not in evaluation_source,
)
check(
    "24 review failure does not auto-create Badcases",
    "if not request.passed:\n        _ensure_badcase_for_run" not in evaluation_source,
)
frontend_source = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")
active_golden = frontend_source.rsplit(
    "async function renderEvaluationsPage(container)", 1
)[1].split("async function renderCostGovernancePage", 1)[0]
check(
    "25 active Golden UI has no answer-keyword fields",
    "required_terms" not in active_golden
    and "forbidden_terms" not in active_golden
    and "evaluation-case-required" not in active_golden,
)
active_badcase = frontend_source.rsplit(
    "async function renderBadcaseDetailPage(container)", 1
)[1].split("async function renderLegacyEvaluationsPage", 1)[0]
check(
    "26 active Badcase UI exposes only the four-stage human loop",
    "addEventListener" in active_badcase
    and "/extract-knowledge" not in active_badcase
    and "/switch-model-retry" not in active_badcase
    and "待验证" in active_badcase
    and "人工复测真实链路" in active_badcase
    and f"/api/badcases/${{id}}/retest" in active_badcase
    and "人工确认并关闭" in active_badcase
    and "const hasRetest = Boolean(bc.retest_trace_id)" in active_badcase,
)
badcase_api_source = (
    Path(__file__).resolve().parents[1] / "app" / "badcases.py"
).read_text(encoding="utf-8")
check(
    "27 AI suggestions cannot advance Badcase status",
    '"status_changed": False' in badcase_api_source
    and 'new_status = "fixing"' in badcase_api_source
    and 'action_type="ai-suggestion"' not in badcase_api_source,
)
check(
    "28 AI draft creation endpoint is retired",
    "AI draft creation is retired" in badcase_api_source
    and badcase_api_source.index("AI draft creation is retired")
    < badcase_api_source.index("created_drafts: List"),
)

print(f"PASS: {len(checks)} deterministic target-3 checks")
