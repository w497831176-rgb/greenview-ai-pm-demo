"""Run N201-N300 once through the final production A/B/C Router only.

Every case uses one fresh session and at most one Flash Router request. The
script never invokes a vertical Agent, RAG, Skill, Tool or business write. It
does not retry and completes all 100 cases even when individual cases fail.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from app.runtime.agent_factory import vertical_agent_cards
from app.runtime.contracts import CapabilityDecision, ResponseMode, RunState, RunStatus, RuntimeLane, RuntimePath
from app.runtime.coordinator import RuntimeCoordinator, _lane_candidates
from app.runtime.evidence_ledger import EvidenceLedger
from app.runtime.snapshot_resolver import resolve_snapshot
from db.property_db import create_chat_trace, ensure_chat_session, record_trace_event, update_chat_trace


EXPECTED_OVERRIDES = {
    "N229": RuntimeLane.ISOLATED_GENERAL.value,
    "N267": RuntimeLane.ISOLATED_GENERAL.value,
}
EXPECTED_DISTRIBUTION = {
    RuntimeLane.SAFETY_HANDOFF.value: 30,
    RuntimeLane.PROPERTY_GOVERNED.value: 36,
    RuntimeLane.ISOLATED_GENERAL.value: 34,
}


def load_cases(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 100 or len({str(row.get("id")) for row in rows}) != 100:
        raise ValueError("final Router evaluation requires exactly 100 unique cases")
    for row in rows:
        case_id = str(row.get("id") or "")
        row["effective_expected_lane"] = EXPECTED_OVERRIDES.get(case_id, str(row.get("expected_lane") or ""))
    distribution = Counter(str(row["effective_expected_lane"]) for row in rows)
    if dict(distribution) != EXPECTED_DISTRIBUTION:
        raise ValueError(f"unexpected final A/B/C distribution: {dict(distribution)}")
    return rows


def agent_selection_readiness(snapshot_config: Dict[str, Any], lane: RuntimeLane) -> Dict[str, Any]:
    cards = vertical_agent_cards(snapshot_config)
    candidates = _lane_candidates(cards, lane)
    if lane == RuntimeLane.SAFETY_HANDOFF:
        status = "not_required"
    elif not candidates:
        status = "no_same_domain_agent"
    elif len(candidates) == 1:
        status = "single_candidate_ready"
    else:
        status = "downstream_selection_required"
    expected_scope = "property" if lane == RuntimeLane.PROPERTY_GOVERNED else "isolated_general"
    valid = lane == RuntimeLane.SAFETY_HANDOFF or all(
        str(item.get("domain_scope") or "property") == expected_scope for item in candidates
    )
    return {
        "status": status,
        "candidate_count": len(candidates),
        "candidate_ids": [str(item.get("agent_id") or "") for item in candidates],
        "same_domain_only": valid,
        "agent_invoked": False,
    }


def downstream_contract_check(state: RunState) -> Dict[str, Any]:
    decision = state.lane_decision
    contract = state.answer_contract
    if not decision or not contract:
        return {"passed": False, "reason": "lane_or_answer_contract_missing"}
    if decision.lane == RuntimeLane.SAFETY_HANDOFF:
        passed = bool(
            contract.response_mode == ResponseMode.EMERGENCY_HANDOFF
            and contract.handoff_policy == "required"
            and contract.skill_policy == contract.rag_policy == contract.tool_policy == "skipped"
            and contract.write_policy == "forbidden"
        )
    elif decision.lane == RuntimeLane.ISOLATED_GENERAL:
        passed = bool(
            contract.response_mode == ResponseMode.SAFE_GENERAL
            and not contract.evidence_required
            and contract.skill_policy == contract.rag_policy == contract.tool_policy == "skipped"
            and contract.write_policy == "forbidden"
            and contract.handoff_policy == "skipped"
        )
    else:
        passed = bool(
            (
                contract.response_mode == ResponseMode.GROUNDED_ANSWER
                and contract.evidence_required
                and contract.write_policy == "forbidden"
            )
            or (
                contract.response_mode == ResponseMode.CONTROLLED_WRITE
                and contract.write_policy == "allowed_after_confirmation"
                and "receipt" in contract.evidence_requirements
            )
        )
    return {
        "passed": passed,
        "response_mode": contract.response_mode.value,
        "evidence_required": contract.evidence_required,
        "skill_policy": contract.skill_policy,
        "rag_policy": contract.rag_policy,
        "tool_policy": contract.tool_policy,
        "write_policy": contract.write_policy,
        "handoff_policy": contract.handoff_policy,
    }


async def run(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.dataset))
    coordinator = RuntimeCoordinator()
    results: List[Dict[str, Any]] = []
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    expected_counts: Counter[str] = Counter()
    actual_counts: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    total_cost = 0.0
    semantic_correct = 0
    schema_valid_count = 0
    agent_contract_pass = 0
    downstream_contract_pass = 0
    provider_contract_pass = 0

    for case in cases:
        case_id = str(case["id"])
        expected = str(case["effective_expected_lane"])
        expected_counts[expected] += 1
        session_id = f"s10b2-final-router-eval-{case_id.lower()}-{uuid.uuid4().hex[:8]}"
        trace_id = uuid.uuid4().hex[:16]
        ensure_chat_session(session_id)
        snapshot = resolve_snapshot(session_id)
        state = RunState(
            run_id=f"eval_{uuid.uuid4().hex}",
            trace_id=trace_id,
            session_id=session_id,
            snapshot_id=snapshot.snapshot_id,
            path=RuntimePath.EXTENSION_ACCEPTANCE,
            status=RunStatus.RUNNING,
            next_step="final_three_lane_router_evaluation",
        )
        create_chat_trace(
            trace_id=trace_id,
            session_id=session_id,
            user_message=str(case["input"]),
            risk_level="L0",
            version_snapshot=snapshot.snapshot_hash,
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
            runtime_path="evaluation_router_only",
        )
        started = time.time()
        failure = None
        actual = "SCHEMA_OR_PROVIDER_FAILURE"
        schema_valid = False
        try:
            await coordinator._resolve_semantic_lane(
                str(case["input"]),
                session_id,
                "router-evaluation",
                trace_id,
                snapshot,
                state,
                ledger,
            )
            actual = state.lane_decision.lane.value
            schema_valid = True
        except Exception as exc:
            failure = f"{type(exc).__name__}: {str(exc)[:240]}"

        if schema_valid:
            schema_valid_count += 1
        lane_correct = bool(schema_valid and actual == expected)
        semantic_correct += int(lane_correct)
        actual_counts[actual] += 1
        confusion[expected][actual] += 1

        readiness = (
            agent_selection_readiness(snapshot.config, state.lane_decision.lane)
            if state.lane_decision
            else {"status": "not_evaluated", "same_domain_only": False, "agent_invoked": False}
        )
        readiness_pass = bool(readiness.get("same_domain_only"))
        agent_contract_pass += int(readiness_pass)
        downstream = downstream_contract_check(state)
        downstream_pass = bool(downstream.get("passed"))
        downstream_contract_pass += int(downstream_pass)

        router_calls = [item for item in state.model_calls if item.get("stage") == "router"]
        provider_ok = bool(
            len(state.model_calls) == 1
            and len(router_calls) == 1
            and str(router_calls[0].get("requested_model") or "").lower() == "deepseek-v4-flash"
            and str(router_calls[0].get("usage_source") or "") == "provider_actual"
            and len(state.cost_entries) == 1
            and state.cost_entries[0].usage_source.value == "provider_actual"
            and state.cost_entries[0].input_cache_hit_tokens is not None
            and state.cost_entries[0].input_cache_miss_tokens is not None
            and state.cost_entries[0].output_tokens is not None
        )
        provider_contract_pass += int(provider_ok)

        state.capability_decision = CapabilityDecision(
            selected_agent_id=None,
            skill={"status": "skipped", "reason_code": "router_only_evaluation"},
            rag={"status": "skipped", "reason_code": "router_only_evaluation"},
            tool={"status": "skipped", "reason_code": "router_only_evaluation"},
            write={"status": "not_required", "reason_code": "router_only_evaluation"},
            handoff={"status": "not_required", "reason_code": "router_only_evaluation"},
        ) if state.lane_decision else None

        passed = bool(lane_correct and readiness_pass and downstream_pass and provider_ok)
        state.status = RunStatus.COMPLETED if passed else RunStatus.FAILED
        state.next_step = None
        ledger.capture_state(state)
        ledger.append(
            "evaluation_results",
            {
                "case": "final_three_lane_router_counterexample",
                "case_id": case_id,
                "expected_lane": expected,
                "actual_lane": actual,
                "semantic_lane_correct": lane_correct,
                "schema_valid": schema_valid,
                "agent_selection": readiness,
                "downstream_contract": downstream,
                "provider_contract_pass": provider_ok,
                "vertical_agent_invoked": False,
                "business_capabilities_invoked": False,
            },
        )
        ledger.persist("complete" if passed else "failed")
        update_chat_trace(
            trace_id,
            intent="final_three_lane_router_evaluation",
            agent_name="三分类语义 Router",
            agent_id="semantic_lane_router",
            status="complete" if passed else "failed",
        )
        record_trace_event(
            trace_id,
            "final_three_lane_router_evaluation",
            "success" if passed else "failed",
            latency_ms=int((time.time() - started) * 1000),
            output_summary=f"{case_id}: expected={expected}; actual={actual}",
            metadata={
                "case_id": case_id,
                "semantic_lane_correct": lane_correct,
                "schema_valid": schema_valid,
                "agent_selection": readiness,
                "downstream_contract_pass": downstream_pass,
                "provider_request_count": len(state.model_calls),
                "vertical_agent_invoked": False,
            },
        )

        for cost in state.cost_entries:
            usage["provider_requests"] += 1
            usage["input_cache_hit_tokens"] += int(cost.input_cache_hit_tokens or 0)
            usage["input_cache_miss_tokens"] += int(cost.input_cache_miss_tokens or 0)
            usage["output_tokens"] += int(cost.output_tokens or 0)
            usage["total_tokens"] += int(cost.total_tokens or 0)
            total_cost += float(cost.amount or 0)

        row = {
            "id": case_id,
            "expected_lane": expected,
            "actual_lane": actual,
            "semantic_lane_correct": lane_correct,
            "schema_valid": schema_valid,
            "agent_selection": readiness,
            "downstream_contract": downstream,
            "provider_contract_pass": provider_ok,
            "passed": passed,
            "trace_id": trace_id,
            "session_id": session_id,
            "failure": failure,
            "provider_request_count": len(state.model_calls),
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        "status": "PASS" if all(item["passed"] for item in results) else "PARTIAL",
        "executed": len(results),
        "expected_distribution": dict(expected_counts),
        "actual_distribution": dict(actual_counts),
        "confusion_matrix": {key: dict(value) for key, value in confusion.items()},
        "semantic_lane_accuracy": f"{semantic_correct}/100",
        "schema_valid": f"{schema_valid_count}/100",
        "agent_selection_contract": f"{agent_contract_pass}/100",
        "downstream_execution_contract": f"{downstream_contract_pass}/100",
        "provider_contract": f"{provider_contract_pass}/100",
        "provider_usage": dict(usage),
        "cost_cny": round(total_cost, 8),
        "vertical_agent_calls": 0,
        "rag_calls": 0,
        "tool_calls": 0,
        "business_writes": 0,
        "pro_calls": 0,
        "darwin_calls": 0,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))
    print(f"RESULT_FILE={output_path}")
    return 0 if summary["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="/tmp/s10b2-final-router-eval.json")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
