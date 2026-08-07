"""Offline full-flow contract for Target 1 A-lane human collaboration.

The test consumes the real ``RuntimeCoordinator.stream`` and reads the real
temporary SQLite Evidence Ledger through the FastAPI runtime Trace endpoint.
Only the semantic Router result and the Handoff transport are controlled.  No
Provider key, network request, production database, browser, or business write
is used.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Tuple


TEMP_DIR = tempfile.TemporaryDirectory(
    prefix="yiai-target1-a-handoff-",
    ignore_cleanup_errors=True,
)
os.environ["PROPERTY_DATA_DIR"] = TEMP_DIR.name
for key in (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
):
    os.environ[key] = ""


from db.property_db import (  # noqa: E402
    _get_conn,
    get_chat_session,
    get_chat_trace,
    get_evidence_ledger,
    init_db,
    list_trace_events,
    claim_handoff,
    request_handoff as persist_handoff,
    resume_handoff_after_owner_message as persist_resume_handoff,
    wait_for_handoff_user,
)


# app.settings builds its default model object at import time and reads model
# configuration.  Initialize only this brand-new temporary database first.
init_db()


from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import agents.router as router_module  # noqa: E402
import app.runtime.coordinator as coordinator_module  # noqa: E402
from app.runtime.api import router as runtime_router  # noqa: E402
from app.runtime.contracts import (  # noqa: E402
    LaneDecision,
    LaneDecisionSource,
    RunConfigSnapshot,
    RuntimeLane,
)
from app.runtime.coordinator import RuntimeCoordinator  # noqa: E402


USER_MESSAGE = "我明确要求工作人员接手处理。"
SAFETY_MESSAGE = "现场存在现实安全风险，请立即协同处理。"
ORDINARY_C_MESSAGE = "请解释一个普通的非物业概念。"
DECLINE_MESSAGE = "暂时不用转人工。"
RULES_MESSAGE = "转人工需要什么条件？"
EXISTING_ORDINARY_MESSAGE = "这是给接管工作人员的补充信息。"
EXISTING_SAFETY_MESSAGE = "继续补充现场情况。"

DECISIONS: Dict[str, LaneDecision] = {
    # Deliberately reproduce the stale Router output.  The runtime invariant,
    # not a sentence-specific test branch, must convert this to effective A.
    USER_MESSAGE: LaneDecision(
        lane=RuntimeLane.ISOLATED_GENERAL,
        business_intent="user_requested_handoff",
        reason="业主明确要求工作人员接手。",
        decision_source=LaneDecisionSource.ROUTER_MODEL,
    ),
    SAFETY_MESSAGE: LaneDecision(
        lane=RuntimeLane.SAFETY_HANDOFF,
        business_intent="safety_risk",
        reason="存在需要立即人工协同的现实安全风险。",
        decision_source=LaneDecisionSource.ROUTER_MODEL,
    ),
    ORDINARY_C_MESSAGE: LaneDecision(
        lane=RuntimeLane.ISOLATED_GENERAL,
        business_intent="general_question",
        reason="普通非物业问题。",
        decision_source=LaneDecisionSource.ROUTER_MODEL,
    ),
    DECLINE_MESSAGE: LaneDecision(
        lane=RuntimeLane.ISOLATED_GENERAL,
        business_intent="decline_handoff",
        reason="业主明确否定本轮人工协同。",
        decision_source=LaneDecisionSource.ROUTER_MODEL,
    ),
    RULES_MESSAGE: LaneDecision(
        lane=RuntimeLane.ISOLATED_GENERAL,
        business_intent="ask_handoff_rules",
        reason="仅咨询人工协同规则。",
        decision_source=LaneDecisionSource.ROUTER_MODEL,
    ),
    EXISTING_ORDINARY_MESSAGE: LaneDecision(
        lane=RuntimeLane.ISOLATED_GENERAL,
        business_intent="general_follow_up",
        reason="Router只看见一条普通补充消息。",
        decision_source=LaneDecisionSource.ROUTER_MODEL,
    ),
    # A stale Router may relabel a follow-up as an ordinary user request.  A
    # persisted emergency Handoff must still keep the safety subtype.
    EXISTING_SAFETY_MESSAGE: LaneDecision(
        lane=RuntimeLane.SAFETY_HANDOFF,
        business_intent="user_requested_handoff",
        reason="Router识别到人工协同，但未保留既有安全子类型。",
        decision_source=LaneDecisionSource.ROUTER_MODEL,
    ),
}

HANDOFF_CALLS: List[Dict[str, Any]] = []
RESUME_CALLS: List[str] = []
DOWNSTREAM_HITS: List[str] = []
PATCHES: List[Tuple[Any, str, Any]] = []


def _snapshot(session_id: str) -> RunConfigSnapshot:
    return RunConfigSnapshot(
        snapshot_id=f"snapshot-{session_id}",
        release_id="rr-target1-a-offline",
        snapshot_hash="target1-a-offline-snapshot",
        session_id=session_id,
        created_at="2026-08-07T00:00:00+08:00",
        config={
            # There is intentionally no vertical Agent.  A must finish before
            # Agent selection, and C must reach the existing no-Agent boundary.
            "agents": [
                {
                    "agent_id": "router",
                    "name": "Semantic Router",
                    "category": "router",
                    "enabled": True,
                    "model_id": "deepseek-v4-flash",
                }
            ],
            "skills": [],
            "knowledge": [],
            "mcp_servers": [],
            "model_policy": {
                "version": "target1-a-offline",
                "default": {
                    "model_id": "deepseek-v4-flash",
                    "provider": "deepseek",
                    "model_params": {"use_thinking": True},
                },
                "available": [],
            },
        },
    )


async def _fake_router(message: str, **_: Any) -> Dict[str, Any]:
    return {
        "decision": DECISIONS[message],
        "raw": DECISIONS[message].model_dump_json(),
        "metrics": {},
        "provider_evidence": {},
        "provider_status": "success",
        "validation_error": None,
    }


def _fake_handoff(
    session_id: str,
    reason: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Capture the execution contract and persist it only in temporary SQLite."""

    HANDOFF_CALLS.append(
        {"session_id": session_id, "reason": reason, **kwargs}
    )
    return persist_handoff(session_id, reason, **kwargs)


def _fake_resume_handoff(session_id: str) -> Dict[str, Any]:
    RESUME_CALLS.append(session_id)
    return persist_resume_handoff(session_id)


def _forbidden(name: str) -> Callable[..., Any]:
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        DOWNSTREAM_HITS.append(name)
        raise AssertionError(f"A lane entered forbidden downstream capability: {name}")

    return fail


def _forbidden_async(name: str) -> Callable[..., Any]:
    async def fail(*_args: Any, **_kwargs: Any) -> Any:
        DOWNSTREAM_HITS.append(name)
        raise AssertionError(f"A lane entered forbidden downstream capability: {name}")

    return fail


def _patch(target: Any, name: str, replacement: Any) -> None:
    PATCHES.append((target, name, getattr(target, name)))
    setattr(target, name, replacement)


def _install_fakes_and_spies() -> None:
    _patch(coordinator_module, "resolve_snapshot", _snapshot)
    _patch(router_module, "classify_lane_decision", _fake_router)
    _patch(
        coordinator_module,
        "provider_accounting_scope",
        lambda **_kwargs: nullcontext(),
    )
    _patch(
        coordinator_module,
        "_build_model_from_snapshot",
        lambda *_args, **_kwargs: object(),
    )
    _patch(coordinator_module, "request_handoff", _fake_handoff)
    _patch(
        coordinator_module,
        "resume_handoff_after_owner_message",
        _fake_resume_handoff,
    )

    # These spies cover every capability that A must preempt.  The C fixtures
    # have no same-domain Agent, so they truthfully stop at the existing C
    # boundary without touching any of these patched functions either.
    _patch(
        RuntimeCoordinator,
        "_select_agent_after_lane",
        _forbidden_async("agent_selector"),
    )
    _patch(
        coordinator_module,
        "build_agent_from_snapshot",
        _forbidden("vertical_agent_or_skill_loader"),
    )
    _patch(coordinator_module, "_results_from_snapshot", _forbidden("rag"))
    _patch(coordinator_module, "plan_tools", _forbidden("mcp_planner"))
    _patch(
        coordinator_module,
        "preinvoke_read_tools",
        _forbidden_async("mcp_preinvoke"),
    )
    _patch(
        coordinator_module,
        "build_model_native_read_tools",
        _forbidden("mcp_model_native"),
    )
    _patch(
        coordinator_module,
        "advance_work_order_workflow",
        _forbidden("draft_or_proposal"),
    )
    _patch(
        coordinator_module.action_gateway,
        "propose",
        _forbidden("action_gateway_propose"),
    )
    _patch(
        coordinator_module.action_gateway,
        "approve",
        _forbidden("action_gateway_approve"),
    )
    _patch(
        coordinator_module.action_gateway,
        "reject",
        _forbidden("action_gateway_reject"),
    )
    _patch(
        coordinator_module.action_gateway,
        "execute",
        _forbidden("action_gateway_execute"),
    )
    _patch(
        coordinator_module.action_gateway,
        "execute_async",
        _forbidden_async("action_gateway_execute_async"),
    )


def _restore_patches() -> None:
    for target, name, original in reversed(PATCHES):
        setattr(target, name, original)
    PATCHES.clear()


def _scalar(sql: str, params: Tuple[Any, ...] = ()) -> int:
    conn = _get_conn()
    try:
        return int(conn.execute(sql, params).fetchone()[0])
    finally:
        conn.close()


def _provider_attempt_count() -> int:
    return _scalar(
        "SELECT COUNT(*) FROM model_calls WHERE record_kind = 'provider_attempt'"
    )


def _parse_sse(block: str) -> Dict[str, Any]:
    event_name = "message"
    data_lines: List[str] = []
    for line in block.splitlines():
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
    payload: Any = None
    if data_lines:
        payload = json.loads("\n".join(data_lines))
    return {"event": event_name, "data": payload}


async def _consume(message: str, session_id: str) -> List[Dict[str, Any]]:
    # A network call here is always a test failure.  TestClient uses ASGI
    # in-process and therefore does not need socket.create_connection.
    original_connect = socket.create_connection

    def no_network(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("network access is forbidden in Target 1 A test")

    socket.create_connection = no_network
    try:
        blocks = [
            block
            async for block in RuntimeCoordinator().stream(
                message,
                session_id,
                "offline-owner",
            )
        ]
    finally:
        socket.create_connection = original_connect
    return [_parse_sse(block) for block in blocks]


APP = FastAPI()
APP.include_router(runtime_router)
CLIENT = TestClient(APP)


def _event(events: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    matches = [item["data"] for item in events if item["event"] == name]
    assert len(matches) == 1, {"event": name, "matches": matches, "events": events}
    assert isinstance(matches[0], dict)
    return matches[0]


def _assert_trace_consistency(
    events: List[Dict[str, Any]],
    expected_lane: RuntimeLane,
) -> Tuple[str, Dict[str, Any]]:
    assert not [item for item in events if item["event"] == "error"], events
    start = _event(events, "start")
    lane = _event(events, "lane")
    done = _event(events, "done")
    assert done["status"] == "complete"
    trace_id = str(start["trace_id"])
    assert done["trace_id"] == trace_id

    response = CLIENT.get(f"/api/runtime/traces/{trace_id}/evidence")
    assert response.status_code == 200, response.text
    api_payload = response.json()
    api_ledger = api_payload["ledger"]
    persisted = get_evidence_ledger(trace_id)
    assert persisted and persisted["ledger"] == api_ledger

    lane_trace_events = [
        item for item in list_trace_events(trace_id)
        if item.get("span_name") == "lane_decision"
    ]
    assert len(lane_trace_events) == 1, lane_trace_events
    trace_lane = (lane_trace_events[0].get("metadata") or {}).get("lane")
    expected = expected_lane.value
    assert lane["lane"] == expected
    assert done["lane_decision"]["lane"] == expected
    assert api_ledger["lane_decision"]["lane"] == expected
    assert trace_lane == expected
    if expected_lane == RuntimeLane.SAFETY_HANDOFF:
        assert done["handler"] == "human_copilot"
        handoff_events = api_ledger.get("handoff_events") or []
        assert len(handoff_events) == 1, handoff_events
        assert handoff_events[0]["handler"] == "human_copilot"
        handoff_details = (
            ((api_ledger.get("capability_decision") or {}).get("handoff") or {}).get(
                "details"
            )
            or {}
        )
        assert handoff_details["handler"] == "human_copilot"
    return trace_id, api_ledger


def _assert_no_business_capability(ledger: Dict[str, Any]) -> None:
    assert ledger.get("route_decision") is None
    assert ledger.get("activated_skills") == []
    assert ledger.get("retrieval_evidence") == []
    assert ledger.get("tool_invocations") == []
    assert ledger.get("action_proposals") == []
    assert ledger.get("action_receipts") == []
    assert ledger.get("contract_violations") == []
    capability = ledger.get("capability_decision") or {}
    assert capability.get("selected_agent_id") is None
    for key in ("skill", "rag", "tool"):
        assert (capability.get(key) or {}).get("status") == "skipped"
    assert (capability.get("write") or {}).get("status") == "not_required"
    assert ((capability.get("handoff") or {}).get("details") or {}).get(
        "handler"
    ) == "human_copilot"


def test_user_requested_c_becomes_ordinary_a() -> None:
    before_handoffs = len(HANDOFF_CALLS)
    events = asyncio.run(_consume(USER_MESSAGE, "target1-a-user"))
    trace_id, ledger = _assert_trace_consistency(
        events,
        RuntimeLane.SAFETY_HANDOFF,
    )
    lane = _event(events, "lane")
    done = _event(events, "done")
    final = _event(events, "final")
    assert lane["business_intent"] == "user_requested_handoff"
    assert done["handoff"] is True
    assert done["handoff_state"] == "requested"
    assert done["handoff_reason"] == "user_requested"
    assert done["handoff_queue"] == "property_service"
    assert done["safety_override"] is False
    assert final["current_agent_id"] == "human_copilot"
    assert not any(marker in final["content"] for marker in ("119", "120", "燃气"))
    assert len(HANDOFF_CALLS) == before_handoffs + 1
    call = HANDOFF_CALLS[-1]
    assert call["reason_code"] == "user_requested"
    assert call["queue"] == "property_service"
    assert (call.get("handoff_package") or {}).get("safety_override") is False
    assert get_chat_session("target1-a-user")["handoff_status"] == "requested"
    assert get_chat_session("target1-a-user")["handoff_reason_code"] == "user_requested"

    assert len(ledger.get("handoff_events") or []) == 1
    handoff = ledger["handoff_events"][0]
    assert handoff["reason_code"] == "user_requested"
    assert handoff["queue"] == "property_service"
    assert handoff["safety_override"] is False
    observations = ledger.get("system_observations") or []
    assert any(
        item.get("type") == "effective_lane_invariant"
        and item.get("router_reported_lane") == RuntimeLane.ISOLATED_GENERAL.value
        and item.get("effective_lane") == RuntimeLane.SAFETY_HANDOFF.value
        for item in observations
    ), observations
    _assert_no_business_capability(ledger)
    trace = get_chat_trace(trace_id)
    assert trace and trace["agent_id"] == "human_copilot"


def test_safety_risk_remains_emergency_a() -> None:
    before_handoffs = len(HANDOFF_CALLS)
    events = asyncio.run(_consume(SAFETY_MESSAGE, "target1-a-safety"))
    trace_id, ledger = _assert_trace_consistency(
        events,
        RuntimeLane.SAFETY_HANDOFF,
    )
    done = _event(events, "done")
    final = _event(events, "final")
    assert done["handoff"] is True
    assert done["handoff_state"] == "requested"
    assert done["handoff_reason"] == "safety_risk"
    assert done["handoff_queue"] == "emergency"
    assert done["safety_override"] is True
    assert any(marker in final["content"] for marker in ("119", "120", "110"))
    assert len(HANDOFF_CALLS) == before_handoffs + 1
    call = HANDOFF_CALLS[-1]
    assert call["reason_code"] == "safety_risk"
    assert call["queue"] == "emergency"
    assert (call.get("handoff_package") or {}).get("safety_override") is True
    assert get_chat_session("target1-a-safety")["handoff_status"] == "requested"
    assert get_chat_session("target1-a-safety")["handoff_reason_code"] == "safety_risk"

    assert len(ledger.get("handoff_events") or []) == 1
    handoff = ledger["handoff_events"][0]
    assert handoff["reason_code"] == "safety_risk"
    assert handoff["queue"] == "emergency"
    assert handoff["safety_override"] is True
    assert not ledger.get("system_observations")
    _assert_no_business_capability(ledger)
    trace = get_chat_trace(trace_id)
    assert trace and trace["agent_id"] == "human_copilot"


def _seed_waiting_handoff(
    session_id: str,
    *,
    reason_code: str,
    queue: str,
) -> None:
    persist_handoff(
        session_id,
        f"seed {reason_code} handoff in temporary database",
        risk_level="L3",
        reason_code=reason_code,
        queue=queue,
        handoff_package={"fixture": "target1-a-existing-handoff"},
    )
    claim_handoff(session_id, "offline-staff")
    wait_for_handoff_user(session_id, "offline-staff", "请补充信息")
    seeded = get_chat_session(session_id)
    assert seeded["handoff_status"] == "waiting_user"
    assert seeded["handoff_reason_code"] == reason_code
    assert seeded["handoff_queue"] == queue


def test_existing_ordinary_waiting_handoff_stays_a_and_resumes() -> None:
    session_id = "target1-a-existing-ordinary"
    _seed_waiting_handoff(
        session_id,
        reason_code="user_requested",
        queue="property_service",
    )
    before_requests = len(HANDOFF_CALLS)
    before_resumes = len(RESUME_CALLS)
    events = asyncio.run(_consume(EXISTING_ORDINARY_MESSAGE, session_id))
    _trace_id, ledger = _assert_trace_consistency(
        events,
        RuntimeLane.SAFETY_HANDOFF,
    )
    lane = _event(events, "lane")
    done = _event(events, "done")
    assert lane["business_intent"] == "user_requested_handoff"
    assert done["handoff_state"] == "active"
    assert done["handoff_reason"] == "user_requested"
    assert done["handoff_queue"] == "property_service"
    assert done["safety_override"] is False
    assert len(HANDOFF_CALLS) == before_requests
    assert RESUME_CALLS[before_resumes:] == [session_id]
    session = get_chat_session(session_id)
    assert session["handoff_status"] == "active"
    assert session["handoff_reason_code"] == "user_requested"
    assert session["handoff_queue"] == "property_service"
    handoff = ledger["handoff_events"][0]
    assert handoff["reason_code"] == "user_requested"
    assert handoff["queue"] == "property_service"
    assert handoff["safety_override"] is False
    _assert_no_business_capability(ledger)


def test_existing_safety_waiting_handoff_cannot_downgrade() -> None:
    session_id = "target1-a-existing-safety"
    _seed_waiting_handoff(
        session_id,
        reason_code="safety_risk",
        queue="emergency",
    )
    before_requests = len(HANDOFF_CALLS)
    before_resumes = len(RESUME_CALLS)
    events = asyncio.run(_consume(EXISTING_SAFETY_MESSAGE, session_id))
    _trace_id, ledger = _assert_trace_consistency(
        events,
        RuntimeLane.SAFETY_HANDOFF,
    )
    lane = _event(events, "lane")
    done = _event(events, "done")
    assert lane["business_intent"] == "safety_risk"
    assert done["handoff_state"] == "active"
    assert done["handoff_reason"] == "safety_risk"
    assert done["handoff_queue"] == "emergency"
    assert done["safety_override"] is True
    assert len(HANDOFF_CALLS) == before_requests
    assert RESUME_CALLS[before_resumes:] == [session_id]
    session = get_chat_session(session_id)
    assert session["handoff_status"] == "active"
    assert session["handoff_reason_code"] == "safety_risk"
    assert session["handoff_queue"] == "emergency"
    handoff = ledger["handoff_events"][0]
    assert handoff["reason_code"] == "safety_risk"
    assert handoff["queue"] == "emergency"
    assert handoff["safety_override"] is True
    _assert_no_business_capability(ledger)


def test_c_and_non_requests_never_handoff() -> None:
    cases = (
        (ORDINARY_C_MESSAGE, "target1-c-ordinary", "general_question"),
        (DECLINE_MESSAGE, "target1-c-decline", "decline_handoff"),
        (RULES_MESSAGE, "target1-c-rules", "ask_handoff_rules"),
    )
    for message, session_id, intent in cases:
        before_handoffs = len(HANDOFF_CALLS)
        events = asyncio.run(_consume(message, session_id))
        _trace_id, ledger = _assert_trace_consistency(
            events,
            RuntimeLane.ISOLATED_GENERAL,
        )
        done = _event(events, "done")
        assert done["lane_decision"]["business_intent"] == intent
        assert len(HANDOFF_CALLS) == before_handoffs
        assert get_chat_session(session_id)["handoff_status"] == "none"
        assert ledger.get("handoff_events") == []
        assert ledger.get("activated_skills") == []
        assert ledger.get("retrieval_evidence") == []
        assert ledger.get("tool_invocations") == []
        assert ledger.get("action_proposals") == []
        assert ledger.get("action_receipts") == []


def main() -> None:
    baseline_provider_attempts = _provider_attempt_count()
    baseline_business_rows = {
        table: _scalar(f"SELECT COUNT(*) FROM {table}")
        for table in (
            "work_order_drafts",
            "action_proposals",
            "action_receipts",
        )
    }
    _install_fakes_and_spies()
    try:
        tests = (
            test_user_requested_c_becomes_ordinary_a,
            test_safety_risk_remains_emergency_a,
            test_existing_ordinary_waiting_handoff_stays_a_and_resumes,
            test_existing_safety_waiting_handoff_cannot_downgrade,
            test_c_and_non_requests_never_handoff,
        )
        for test in tests:
            test()
            print(f"PASS {test.__name__}")
        assert not DOWNSTREAM_HITS, DOWNSTREAM_HITS
        assert _provider_attempt_count() == baseline_provider_attempts == 0
        for table, baseline in baseline_business_rows.items():
            assert _scalar(f"SELECT COUNT(*) FROM {table}") == baseline, table
        print("Target 1 A handoff full-flow: PASS (Provider attempts: 0)")
    finally:
        _restore_patches()


if __name__ == "__main__":
    try:
        main()
    finally:
        TEMP_DIR.cleanup()
