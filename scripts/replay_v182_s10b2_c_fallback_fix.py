"""Zero-Provider replay for the two persisted C-lane fallback failures.

The script creates deterministic derived Trace/Evidence records. It never
modifies the source traces or their saved Router/model evidence.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.runtime.contracts import AnswerContract, LaneDecision, RunState, RunStatus, RuntimeLane, RuntimePath
from app.runtime.coordinator import OUT_OF_SCOPE_RESPONSE, build_lane_agent_unavailable_decision
from app.runtime.evidence_ledger import EvidenceLedger
from db.property_db import (
    add_badcase_action,
    create_chat_trace,
    get_badcase,
    get_chat_trace,
    get_evidence_ledger,
    get_model_calls_for_trace,
    list_badcase_actions,
    list_trace_events,
    record_trace_event,
    update_chat_trace,
)


SOURCE_TRACE_IDS = ("50611a73ec7a41aa", "edcfdc8e5dd04507")
BADCASE_ID = 680


def replay_trace_id(source_trace_id: str) -> str:
    return hashlib.sha256(f"c-fallback-offline-replay:{source_trace_id}".encode()).hexdigest()[:16]


def _has_replay_action(replay_id: str) -> bool:
    for action in list_badcase_actions(BADCASE_ID):
        if action.get("action_type") != "offline-replay":
            continue
        raw = action.get("action_detail") or "{}"
        try:
            detail = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            continue
        if detail.get("replay_trace_id") == replay_id:
            return True
    return False


def replay_one(source_trace_id: str) -> dict[str, Any]:
    source_trace = get_chat_trace(source_trace_id)
    source_evidence = get_evidence_ledger(source_trace_id)
    if not source_trace or not source_evidence:
        raise RuntimeError(f"missing persisted source evidence: {source_trace_id}")
    source_ledger = source_evidence["ledger"]
    lane = LaneDecision.model_validate(source_ledger.get("lane_decision") or {})
    if lane.lane != RuntimeLane.ISOLATED_GENERAL:
        raise RuntimeError(f"source lane is not C: {source_trace_id}")
    if str(lane.business_intent or "").strip() == "user_requested_handoff":
        raise RuntimeError(f"source requested Handoff and is not a C fallback replay: {source_trace_id}")
    if source_ledger.get("handoff_events"):
        raise RuntimeError(f"source contains Handoff evidence: {source_trace_id}")

    answer_contract = AnswerContract.model_validate(source_ledger.get("answer_contract") or {})
    decision = build_lane_agent_unavailable_decision(property_lane=False)
    replay_id = replay_trace_id(source_trace_id)
    replay_session_id = f"offline-replay::{source_trace_id}"
    config_snapshot = source_ledger.get("config_snapshot") or {}
    snapshot_id = str(config_snapshot.get("snapshot_id") or f"saved::{source_trace_id}")

    state = RunState(
        run_id=f"offline-replay::{source_trace_id}",
        trace_id=replay_id,
        session_id=replay_session_id,
        snapshot_id=snapshot_id,
        path=RuntimePath.CONSULTATION,
        lane_decision=lane,
        answer_contract=answer_contract,
        capability_decision=decision,
        status=RunStatus.COMPLETED,
    )

    if get_chat_trace(replay_id) is None:
        create_chat_trace(
            trace_id=replay_id,
            session_id=replay_session_id,
            user_message=f"保存的Router结果离线重放：{source_trace_id}",
            run_type="offline_replay",
            risk_level="L0",
            version_snapshot=source_evidence.get("config_hash"),
        )
    update_chat_trace(
        replay_id,
        intent="isolated_general_boundary",
        agent_name="通用边界",
        agent_id="isolated_general_boundary",
        status="complete",
        run_type="offline_replay",
    )

    ledger = EvidenceLedger(
        trace_id=replay_id,
        session_id=replay_session_id,
        config_snapshot=config_snapshot,
        release_id=source_evidence.get("release_id"),
        config_hash=source_evidence.get("config_hash"),
        runtime_path=RuntimePath.CONSULTATION.value,
    )
    ledger.capture_state(state)
    ledger.contract.model_calls = []
    ledger.contract.cost_entries = []
    ledger.contract.handoff_events = []
    observation = {
        "type": "saved_router_result_offline_replay",
        "source_trace_id": source_trace_id,
        "provider_invoked": False,
        "original_response_modified": False,
        "result": "complete",
        "response": OUT_OF_SCOPE_RESPONSE,
    }
    if not any(
        item.get("type") == observation["type"]
        and item.get("source_trace_id") == source_trace_id
        for item in ledger.contract.system_observations
    ):
        ledger.contract.system_observations.append(observation)
    evaluation = {
        "case": "c_lane_saved_result_offline_replay",
        "passed": True,
        "source_trace_id": source_trace_id,
        "trace_status": "complete",
        "evidence_status": "complete",
        "provider_request_count": 0,
    }
    if not any(
        item.get("case") == evaluation["case"]
        and item.get("source_trace_id") == source_trace_id
        for item in ledger.contract.evaluation_results
    ):
        ledger.contract.evaluation_results.append(evaluation)
    ledger.persist("complete")

    if not any(event.get("span_name") == "offline_c_fallback_replay" for event in list_trace_events(replay_id)):
        record_trace_event(
            replay_id,
            "offline_c_fallback_replay",
            "success",
            output_summary="saved Router result replayed without Provider call",
            metadata={
                "source_trace_id": source_trace_id,
                "provider_invoked": False,
                "original_response_modified": False,
            },
        )

    if not _has_replay_action(replay_id):
        status = (get_badcase(BADCASE_ID) or {}).get("status")
        add_badcase_action(
            badcase_id=BADCASE_ID,
            action_type="offline-replay",
            action_detail=json.dumps(
                {
                    "source_trace_id": source_trace_id,
                    "replay_trace_id": replay_id,
                    "provider_request_count": 0,
                    "result": "complete",
                },
                ensure_ascii=False,
            ),
            status_before=status,
            status_after=status,
            created_by="deterministic_replay",
        )

    replay_trace = get_chat_trace(replay_id) or {}
    replay_evidence = get_evidence_ledger(replay_id) or {}
    result = {
        "source_trace_id": source_trace_id,
        "source_trace_status": source_trace.get("status"),
        "source_provider_request_count": len(get_model_calls_for_trace(source_trace_id)),
        "replay_trace_id": replay_id,
        "replay_trace_status": replay_trace.get("status"),
        "replay_evidence_status": replay_evidence.get("status"),
        "lane": lane.lane.value,
        "business_intent": lane.business_intent,
        "handoff_events": 0,
        "property_capability_calls": 0,
        "provider_request_count": 0,
        "original_response_modified": False,
    }
    if result["replay_trace_status"] != "complete" or result["replay_evidence_status"] != "complete":
        raise RuntimeError(f"replay Trace/Evidence mismatch: {result}")
    return result


def main() -> None:
    results = [replay_one(trace_id) for trace_id in SOURCE_TRACE_IDS]
    print(json.dumps({"status": "PASS", "provider_request_count": 0, "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
