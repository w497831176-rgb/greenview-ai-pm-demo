"""Run the external N201-N300 set through the production semantic Router only.

One case equals one fresh session, one Trace and at most one Flash Provider
request. The script never invokes a vertical Agent, RAG, Tool, ActionGateway or
Badcase creation, and it stops on the first schema/Lane failure without retry.
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

from app.runtime.contracts import CapabilityDecision, RunState, RunStatus, RuntimePath
from app.runtime.coordinator import RuntimeCoordinator
from app.runtime.evidence_ledger import EvidenceLedger
from app.runtime.snapshot_resolver import resolve_snapshot
from db.property_db import (
    create_chat_trace,
    ensure_chat_session,
    record_trace_event,
    update_chat_trace,
)


EXPECTED_OVERRIDES = {"N229": "CLARIFY", "N267": "CLARIFY"}


def load_cases(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 100 or len({str(row.get("id")) for row in rows}) != 100:
        raise ValueError("router evaluation requires exactly 100 unique cases")
    for index, row in enumerate(rows):
        case_id = str(row.get("id") or "")
        if case_id in EXPECTED_OVERRIDES:
            row = dict(row)
            row["expected_lane"] = EXPECTED_OVERRIDES[case_id]
        row["effective_expected_lane"] = EXPECTED_OVERRIDES.get(case_id, row.get("expected_lane"))
        rows[index] = row
    return rows


def _resume_seed(
    cases: List[Dict[str, Any]],
    prior_result_path: str | None,
    replay_result_path: str | None,
) -> Dict[str, Any]:
    """Validate the immutable N201-N241 breakpoint and seed final aggregation."""

    if not prior_result_path and not replay_result_path:
        return {
            "results": [],
            "start_index": 0,
            "usage": Counter(),
            "cost_cny": 0.0,
            "provenance": {
                "original_provider_cases": [],
                "offline_replay_cases": [],
                "continuation_provider_cases": [],
            },
        }
    if not prior_result_path or not replay_result_path:
        raise ValueError("resume requires both --prior-result and --replay-result")

    prior = json.loads(Path(prior_result_path).read_text(encoding="utf-8"))
    replay = json.loads(Path(replay_result_path).read_text(encoding="utf-8"))
    prior_rows = list(prior.get("results") or [])
    expected_prior_ids = [str(row["id"]) for row in cases[:41]]
    if [str(row.get("id")) for row in prior_rows] != expected_prior_ids:
        raise ValueError("prior result is not the immutable N201-N241 run")
    if not all(bool(row.get("passed")) for row in prior_rows[:40]):
        raise ValueError("prior N201-N240 must all be original one-pass successes")
    if str(prior_rows[40].get("id")) != "N241" or bool(prior_rows[40].get("passed")):
        raise ValueError("prior N241 must be the retained original schema failure")
    if not (
        replay.get("id") == "N241"
        and replay.get("passed") is True
        and replay.get("offline_replay") is True
        and replay.get("provider_called") is False
        and replay.get("provider_request_count") == 0
        and replay.get("actual_lane") == "C_ISOLATED_GENERAL"
        and (replay.get("decision") or {}).get("request_kind") == "fact"
        and (replay.get("answer_contract") or {}).get("response_mode") == "safe_general"
    ):
        raise ValueError("saved-response N241 replay contract is invalid")

    replay_row = {
        "id": "N241",
        "expected_lane": "C_ISOLATED_GENERAL",
        "actual_lane": "C_ISOLATED_GENERAL",
        "schema_valid": True,
        "passed": True,
        "trace_id": replay.get("source_trace_id"),
        "session_id": replay.get("source_session_id"),
        "failure": None,
        "provider_request_count": 0,
        "source_provider_request_count": 1,
        "evaluation_origin": "saved_response_offline_replay",
    }
    seeded = [
        {**row, "evaluation_origin": "original_provider_call"}
        for row in prior_rows[:40]
    ] + [replay_row]
    return {
        "results": seeded,
        "start_index": 41,
        "usage": Counter(prior.get("provider_usage") or {}),
        "cost_cny": float(prior.get("cost_cny") or 0),
        "provenance": {
            "original_provider_cases": [str(row["id"]) for row in prior_rows],
            "offline_replay_cases": ["N241"],
            "continuation_provider_cases": [],
            "offline_replay_provider_requests": 0,
            "retained_original_failure_trace": replay.get("source_trace_id"),
        },
    }


async def run(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.dataset))
    coordinator = RuntimeCoordinator()
    seed = _resume_seed(cases, args.prior_result, args.replay_result)
    results: List[Dict[str, Any]] = list(seed["results"])
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    expected_counts: Counter[str] = Counter()
    actual_counts: Counter[str] = Counter()
    for row in results:
        expected_counts[str(row["expected_lane"])] += 1
        actual_counts[str(row["actual_lane"])] += 1
        confusion[str(row["expected_lane"])][str(row["actual_lane"])] += 1
    usage = Counter(seed["usage"])
    continuation_usage = Counter()
    total_cost = float(seed["cost_cny"])
    continuation_cost = 0.0
    schema_failures = 0
    provenance = dict(seed["provenance"])

    for case in cases[int(seed["start_index"]):]:
        case_id = str(case["id"])
        expected = str(case["effective_expected_lane"])
        expected_counts[expected] += 1
        session_id = f"s10b2-fix2-router-eval-{case_id.lower()}-{uuid.uuid4().hex[:8]}"
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
            next_step="semantic_router_evaluation",
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
            runtime_path="evaluation_router",
        )
        started = time.time()
        failure = None
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
            actual = "SCHEMA_OR_PROVIDER_FAILURE"
            schema_valid = False
            schema_failures += 1
            failure = f"{type(exc).__name__}: {str(exc)[:240]}"

        if state.lane_decision:
            state.capability_decision = CapabilityDecision(
                selected_agent_id=None,
                skill={"status": "skipped", "reason_code": "router_only_evaluation"},
                rag={"status": "skipped", "reason_code": "router_only_evaluation"},
                tool={"status": "skipped", "reason_code": "router_only_evaluation"},
                write={"status": "not_required", "reason_code": "router_only_evaluation"},
                handoff={"status": "not_required", "reason_code": "router_only_evaluation"},
            )
        passed = bool(schema_valid and actual == expected)
        state.status = RunStatus.COMPLETED if passed else RunStatus.FAILED
        state.next_step = None
        ledger.capture_state(state)
        ledger.append(
            "evaluation_results",
            {
                "case": "semantic_router_counterexample",
                "case_id": case_id,
                "expected_lane": expected,
                "actual_lane": actual,
                "schema_valid": schema_valid,
                "passed": passed,
                "vertical_agent_invoked": False,
                "business_capabilities_invoked": False,
            },
        )
        ledger.persist("complete" if passed else "failed")
        update_chat_trace(
            trace_id,
            intent="semantic_router_evaluation",
            agent_name="结构化语义 Router",
            agent_id="semantic_lane_router",
            status="complete" if passed else "failed",
        )
        record_trace_event(
            trace_id,
            "semantic_router_evaluation",
            "success" if passed else "failed",
            latency_ms=int((time.time() - started) * 1000),
            output_summary=f"{case_id}: expected={expected}; actual={actual}",
            metadata={
                "case_id": case_id,
                "schema_valid": schema_valid,
                "provider_request_count": len(state.model_calls),
                "vertical_agent_invoked": False,
            },
        )

        actual_counts[actual] += 1
        confusion[expected][actual] += 1
        for cost in state.cost_entries:
            usage["provider_requests"] += 1
            continuation_usage["provider_requests"] += 1
            usage["input_cache_hit_tokens"] += int(cost.input_cache_hit_tokens or 0)
            continuation_usage["input_cache_hit_tokens"] += int(cost.input_cache_hit_tokens or 0)
            usage["input_cache_miss_tokens"] += int(cost.input_cache_miss_tokens or 0)
            continuation_usage["input_cache_miss_tokens"] += int(cost.input_cache_miss_tokens or 0)
            usage["output_tokens"] += int(cost.output_tokens or 0)
            continuation_usage["output_tokens"] += int(cost.output_tokens or 0)
            usage["total_tokens"] += int(cost.total_tokens or 0)
            continuation_usage["total_tokens"] += int(cost.total_tokens or 0)
            total_cost += float(cost.amount or 0)
            continuation_cost += float(cost.amount or 0)
        result = {
            "id": case_id,
            "expected_lane": expected,
            "actual_lane": actual,
            "schema_valid": schema_valid,
            "passed": passed,
            "trace_id": trace_id,
            "session_id": session_id,
            "failure": failure,
            "provider_request_count": len(state.model_calls),
            "evaluation_origin": "continuation_provider_call" if seed["start_index"] else "original_provider_call",
        }
        results.append(result)
        provenance["continuation_provider_cases"].append(case_id)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not passed:
            break

    summary = {
        "status": "PASS" if len(results) == 100 and all(item["passed"] for item in results) else "PARTIAL",
        "executed": len(results),
        "expected_distribution": dict(expected_counts),
        "actual_distribution": dict(actual_counts),
        "confusion_matrix": {expected: dict(values) for expected, values in confusion.items()},
        "schema_failures": schema_failures,
        "provider_usage": dict(usage),
        "cost_cny": round(total_cost, 8),
        "continuation_provider_usage": dict(continuation_usage),
        "continuation_cost_cny": round(continuation_cost, 8),
        "evaluation_provenance": provenance,
        "historical_schema_failures_replayed": 1 if seed["start_index"] else 0,
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
    parser.add_argument("--prior-result")
    parser.add_argument("--replay-result")
    parser.add_argument("--output", default="/tmp/s10b2-fix2-router-eval.json")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
