"""Focused target-3 Badcase contract checks; no Provider call is permitted."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


TEMP_DIR = tempfile.TemporaryDirectory(prefix="yiai-target3-")
os.environ["PROPERTY_DATA_DIR"] = TEMP_DIR.name

from app.badcase_schema import _enrich_badcase, user_status_label
from app.runtime.badcase_capture import (
    BadcaseTriggerCode,
    _problem_fingerprint,
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
    release_id: str = "rr-test-1",
    agent_id: str = "agent-test-1",
    contract_violations: list[dict] | None = None,
    tool_invocations: list[dict] | None = None,
    action_receipts: list[dict] | None = None,
    evaluation_results: list[dict] | None = None,
) -> RunEvidenceLedger:
    return RunEvidenceLedger(
        trace_id=trace_id,
        session_id=f"session-{trace_id}",
        config_snapshot={
            "snapshot_id": f"snapshot-{release_id}",
            "release_id": release_id,
            "snapshot_hash": f"hash-{release_id}",
        },
        lane_decision={
            "lane": "B_PROPERTY_GOVERNED",
            "selected_agent_id": agent_id,
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
    return [{"code": code, "detail": code}]


property_db.init_db()

# Only the seven closed runtime facts can open an automatic suspected Badcase.
check(
    "01 closed trigger vocabulary",
    {item.value for item in BadcaseTriggerCode}
    == {
        "runtime_failed",
        "contract_invalid",
        "capability_failed",
        "citation_invalid",
        "action_failed",
        "insufficient_evidence",
        "capability_unavailable",
    },
)
check(
    "02 runtime failure maps to runtime_failed",
    decide(
        ledger("runtime"),
        runtime_error="provider unavailable",
        runtime_error_type="provider_failure",
    )["trigger_code"]
    == "runtime_failed",
)
check(
    "03 fixed evaluation maps to contract_invalid",
    decide(
        ledger(
            "evaluation",
            evaluation_results=[
                {
                    "evaluation_case_id": 7,
                    "assertion_id": "fixed-agent",
                    "passed": False,
                    "status": "failed",
                }
            ],
        )
    )["trigger_code"]
    == "contract_invalid",
)
check(
    "04 capability code maps without text inference",
    decide(
        ledger(
            "capability",
            contract_violations=violation("skill_selected_not_loaded"),
        )
    )["trigger_code"]
    == "capability_failed",
)
check(
    "05 citation code maps without text inference",
    decide(
        ledger(
            "citation",
            contract_violations=violation("invalid_evidence_id"),
        )
    )["trigger_code"]
    == "citation_invalid",
)
check(
    "06 failed receipt maps to action_failed",
    decide(
        ledger(
            "action",
            action_receipts=[
                {
                    "receipt_id": "receipt-test",
                    "proposal_id": "proposal-test",
                    "status": "failed",
                }
            ],
        )
    )["trigger_code"]
    == "action_failed",
)

# Agent self-reports are strict answer_status values and retain their own source.
insufficient = decide(
    ledger("agent-insufficient"), agent_answer_status="insufficient_evidence"
)
check(
    "07 insufficient evidence keeps agent source",
    insufficient["trigger_code"] == "insufficient_evidence"
    and insufficient["source"] == "agent_insufficient_evidence"
    and insufficient["component"] == "agent:agent-test-1",
)
unavailable = decide(
    ledger("agent-unavailable"), agent_answer_status="capability_unavailable"
)
check(
    "08 capability unavailable keeps agent source",
    unavailable["trigger_code"] == "capability_unavailable"
    and unavailable["source"] == "agent_capability_unavailable",
)
check(
    "09 unknown answer status cannot open Badcase",
    decide(ledger("agent-unknown"), agent_answer_status="maybe_unavailable")[
        "disposition"
    ]
    == "system_observation",
)

# Former substring/answer-prefix triggers are inert without a registered fact.
check(
    "10 former selection substring is only observation",
    decide(
        ledger(
            "unknown-contract",
            contract_violations=violation("tool_selection_wrong"),
        )
    )["disposition"]
    == "system_observation",
)
golden_before = len(property_db.list_evaluation_cases())
badcases_before = len(property_db.list_badcases())
text_only = capture_runtime_badcase(
    ledger=ledger("text-only"),
    original_query="fixture question",
    ai_response="当前知识依据不足，这只是回答正文。",
    agent_answer_status="answered",
    delivery_context={"normal_completed": True},
)
check(
    "11 answer wording alone creates no Badcase",
    text_only is None and len(property_db.list_badcases()) == badcases_before,
)
check(
    "12 automatic capture never creates Golden case",
    len(property_db.list_evaluation_cases()) == golden_before,
)

# Problem fingerprints are deterministic normalization, not keyword semantics.
check(
    "13 problem fingerprint normalizes width case and whitespace",
    _problem_fingerprint("  ＦＩＸＴＵＲＥ   Question ")
    == _problem_fingerprint("fixture question"),
)

# Dedupe key = trigger + component + Release + problem fingerprint.
first = capture_runtime_badcase(
    ledger=ledger("dedupe-trace-1"),
    original_query="Repeated fixture question",
    ai_response="structured answer",
    agent_answer_status="insufficient_evidence",
    delivery_context={"normal_completed": True},
)
second = capture_runtime_badcase(
    ledger=ledger("dedupe-trace-2"),
    original_query=" repeated   fixture question ",
    ai_response="structured answer",
    agent_answer_status="insufficient_evidence",
    delivery_context={"normal_completed": True},
)
check("14 same structured issue dedupes", first["id"] == second["id"])
deduped = property_db.get_badcase(int(first["id"])) or {}
dedupe_context = json.loads(deduped.get("context_json") or "{}")
check(
    "15 duplicate occurrence accumulates without replacing primary Trace",
    dedupe_context.get("occurrence_count") == 2
    and dedupe_context.get("occurrence_trace_ids")
    == ["dedupe-trace-1", "dedupe-trace-2"]
    and deduped.get("trace_id") == "dedupe-trace-1",
)
same_occurrence_again = capture_runtime_badcase(
    ledger=ledger("dedupe-trace-2"),
    original_query="Repeated fixture question",
    ai_response="structured answer",
    agent_answer_status="insufficient_evidence",
    delivery_context={"normal_completed": True},
)
same_occurrence_context = json.loads(
    (property_db.get_badcase(int(first["id"])) or {}).get("context_json") or "{}"
)
check(
    "16 one Trace occurrence is idempotent",
    same_occurrence_again["id"] == first["id"]
    and same_occurrence_context.get("occurrence_count") == 2,
)

different_release = capture_runtime_badcase(
    ledger=ledger("dedupe-release", release_id="rr-test-2"),
    original_query="Repeated fixture question",
    ai_response="structured answer",
    agent_answer_status="insufficient_evidence",
    delivery_context={"normal_completed": True},
)
different_agent = capture_runtime_badcase(
    ledger=ledger("dedupe-agent", agent_id="agent-test-2"),
    original_query="Repeated fixture question",
    ai_response="structured answer",
    agent_answer_status="insufficient_evidence",
    delivery_context={"normal_completed": True},
)
different_problem = capture_runtime_badcase(
    ledger=ledger("dedupe-problem"),
    original_query="A different fixture question",
    ai_response="structured answer",
    agent_answer_status="insufficient_evidence",
    delivery_context={"normal_completed": True},
)
check(
    "17 release component and problem all partition dedupe",
    len(
        {
            first["id"],
            different_release["id"],
            different_agent["id"],
            different_problem["id"],
        }
    )
    == 4,
)

# A recurrence after a human terminal decision creates a new pending record.
terminal_first = capture_runtime_badcase(
    ledger=ledger("terminal-trace-1"),
    original_query="Terminal recurrence fixture",
    ai_response="structured answer",
    agent_answer_status="capability_unavailable",
    delivery_context={"normal_completed": True},
)
property_db.update_badcase(int(terminal_first["id"]), status="closed")
terminal_second = capture_runtime_badcase(
    ledger=ledger("terminal-trace-2"),
    original_query="Terminal recurrence fixture",
    ai_response="structured answer",
    agent_answer_status="capability_unavailable",
    delivery_context={"normal_completed": True},
)
terminal_history = property_db.get_badcase(int(terminal_first["id"])) or {}
check(
    "18 terminal recurrence preserves history and opens a new pending record",
    terminal_second["id"] != terminal_first["id"]
    and terminal_second["status"] == "pending"
    and terminal_history.get("status") == "closed"
    and terminal_history.get("trace_id") == "terminal-trace-1",
)

# Existing human lifecycle and presentation remain authoritative.
check(
    "19 four human status groups remain unchanged",
    [user_status_label(value) for value in ("pending", "fixing", "verifying", "closed")]
    == ["待审核", "处理中", "待验证", "已结束"],
)
agent_source = _enrich_badcase(
    {
        "id": 999,
        "status": "pending",
        "source": "agent_insufficient_evidence",
        "category": "knowledge_gap",
        "actions": [],
    }
)
check(
    "20 agent self-report is visibly distinct but still pending human review",
    agent_source["source_label"] == "Agent自报：依据不足"
    and agent_source["user_status_label"] == "待审核",
)
check(
    "21 target3 capture still leaves Golden data unchanged",
    len(property_db.list_evaluation_cases()) == golden_before,
)

# Public runtime wiring must pass the parsed Agent status and must not infer it
# from user-facing answer prose.
coordinator_source = (
    Path(__file__).resolve().parents[1] / "app" / "runtime" / "coordinator.py"
).read_text(encoding="utf-8")
check(
    "22 coordinator passes the frozen Agent structured answer status",
    "agent_answer_status=agent_turn.answer_status" in coordinator_source,
)
check(
    "23 coordinator does not classify Badcases from answer text prefixes",
    'rendered.startswith("当前知识依据不足")' not in coordinator_source,
)

print(f"PASS: {len(checks)} deterministic target-3 checks")
