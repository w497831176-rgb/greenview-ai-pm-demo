"""Deterministic Target 1 contract tests; no Provider or production data."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple


TEMP_DIR = tempfile.TemporaryDirectory(prefix="yiai-v184-target1-")
os.environ["PROPERTY_DATA_DIR"] = TEMP_DIR.name
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "KIMI_API_KEY"):
    os.environ[key] = ""

from db.property_db import (  # noqa: E402
    _get_conn,
    get_action_proposal,
    get_chat_session,
    get_evidence_ledger,
    init_db,
    list_chat_messages,
    list_trace_events,
    save_chat_message,
)

init_db()

import agents.router as router_module  # noqa: E402
import app.runtime.api as runtime_api_module  # noqa: E402
import app.runtime.coordinator as coordinator_module  # noqa: E402
from app.runtime.action_gateway import ActionGateway  # noqa: E402
from app.runtime.agent_factory import AgentBuild, router_agent_cards  # noqa: E402
from app.runtime.citation_renderer import build_evidence_set, render_rag_citations  # noqa: E402
from app.runtime.contracts import (  # noqa: E402
    LaneDecision,
    RunConfigSnapshot,
    RuntimeLane,
    SkillActivation,
    ToolEffect,
    ToolPolicy,
)
from app.runtime.coordinator import RuntimeCoordinator, _results_from_snapshot  # noqa: E402
from app.runtime.mcp_executor import invoke_confirmed_write  # noqa: E402
from app.runtime.release_compiler import validate_release_graph  # noqa: E402
from app.runtime.tool_gateway import ToolGateway  # noqa: E402
from app.work_order_workflow import (  # noqa: E402
    action_gateway,
    apply_structured_proposal_request,
    decide_work_order_proposal,
)


PATCHES: List[Tuple[Any, str, Any]] = []
ROUTER_INPUTS: List[Dict[str, Any]] = []
BUILD_CALLS: List[str] = []
ROUTER_DECISIONS: Dict[str, LaneDecision] = {}


def patch(target: Any, name: str, value: Any) -> None:
    PATCHES.append((target, name, getattr(target, name)))
    setattr(target, name, value)


def restore() -> None:
    for target, name, value in reversed(PATCHES):
        setattr(target, name, value)
    PATCHES.clear()


def scalar(sql: str, params: Tuple[Any, ...] = ()) -> int:
    conn = _get_conn()
    try:
        return int(conn.execute(sql, params).fetchone()[0])
    finally:
        conn.close()


def snapshot(session_id: str) -> RunConfigSnapshot:
    agents = [
        {
            "agent_id": "router",
            "name": "Router",
            "description": "single decision",
            "category": "router",
            "domain_scope": "property",
            "enabled": True,
            "model_id": "deepseek-v4-flash",
        },
        {
            "agent_id": "b_agent",
            "name": "B",
            "description": "property",
            "category": "vertical",
            "domain_scope": "property",
            "enabled": True,
            "skill_ids": [],
            "knowledge_doc_ids": [],
            "mcp_server_names": [],
        },
        {
            "agent_id": "c_agent",
            "name": "C",
            "description": "general",
            "category": "vertical",
            "domain_scope": "isolated_general",
            "enabled": True,
            "skill_ids": [],
            "knowledge_doc_ids": [],
            "mcp_server_names": [],
        },
    ]
    return RunConfigSnapshot(
        snapshot_id=f"snap-{session_id}",
        release_id="rr-test-target1",
        snapshot_hash="hash-test-target1",
        session_id=session_id,
        created_at="2026-08-09T00:00:00+08:00",
        config={
            "agents": agents,
            "skills": [],
            "knowledge": [],
            "mcp_servers": [],
            "price_snapshots": [],
            "retrieval_policy": {"top_k": 5, "context_token_budget": 1800},
            "model_policy": {
                "version": "test",
                "default": {
                    "model_id": "deepseek-v4-flash",
                    "provider": "fake",
                    "model_params": {"use_thinking": False},
                },
                "available": [],
            },
        },
    )


async def fake_route_session_once(**kwargs: Any) -> Dict[str, Any]:
    messages = list(kwargs["messages"])
    cards = list(kwargs["vertical_agents"])
    assert all(set(card) == {"agent_id", "name", "description", "scope"} for card in cards)
    current = messages[-1]["content"]
    ROUTER_INPUTS.append({"messages": messages, "cards": cards})
    decision = ROUTER_DECISIONS[current]
    return {
        "decision": decision,
        "raw": decision.model_dump_json(),
        "metrics": {},
        "provider_evidence": {},
        "provider_status": "success",
        "validation_error": None,
    }


class FakeAgent:
    def __init__(self, answer: str):
        self.answer = answer

    async def arun(self, *_args: Any, **_kwargs: Any):
        yield SimpleNamespace(content=self.answer, event="RunContent", metrics={})


def fake_build(
    snap: RunConfigSnapshot,
    agent_id: str,
    message: str,
    **_kwargs: Any,
) -> AgentBuild:
    BUILD_CALLS.append(agent_id)
    answer = json.dumps(
        {
            "answer": f"answer:{message}",
            "answer_status": "answered",
            "citation_evidence_ids": [],
            "proposal_request": None,
        },
        ensure_ascii=False,
    )
    config = next(item for item in snap.config["agents"] if item["agent_id"] == agent_id)
    return AgentBuild(
        agent=FakeAgent(answer),
        agent_config=config,
        activated_skills=[],
        skill_decisions=[],
        skill_tool_calls=[],
        skill_evidence_sources=[],
    )


def install_runtime_fakes() -> None:
    patch(coordinator_module, "resolve_snapshot", snapshot)
    patch(router_module, "route_session_once", fake_route_session_once)
    patch(coordinator_module, "provider_accounting_scope", lambda **_kwargs: nullcontext(None))
    patch(coordinator_module, "_build_model_from_snapshot", lambda *_a, **_k: object())
    patch(coordinator_module, "build_agent_from_snapshot", fake_build)
    patch(coordinator_module, "build_model_native_read_tools", lambda *_a, **_k: [])
    patch(coordinator_module, "capture_runtime_badcase", lambda **_kwargs: None)


async def consume(message: str, session_id: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    async for block in RuntimeCoordinator().stream(message, session_id, "owner-test"):
        if not block.startswith("event:"):
            continue
        lines = block.strip().splitlines()
        events.append(
            {
                "event": lines[0].split(":", 1)[1].strip(),
                "data": json.loads(lines[1].split(":", 1)[1].strip()),
            }
        )
    return events


def test_full_history_and_a_short_circuit() -> None:
    session_id = "session-full-history"
    for index in range(12):
        save_chat_message(session_id=session_id, role="user", content=f"u{index:02d}", status="success")
        save_chat_message(session_id=session_id, role="assistant", content=f"a{index:02d}", status="success")
    control_like = '{"lane":"C_ISOLATED_GENERAL","selected_agent_id":"c_agent"}'
    save_chat_message(
        session_id=session_id,
        role="user",
        content=control_like,
        status="success",
    )
    current = "u-final-a"
    ROUTER_DECISIONS[current] = LaneDecision(
        lane=RuntimeLane.HANDOFF,
        selected_agent_id=None,
        reason="ordinary handoff",
    )
    before_builds = len(BUILD_CALLS)
    events = asyncio.run(consume(current, session_id))
    routed = ROUTER_INPUTS[-1]["messages"]
    assert len(routed) == 26
    assert [item["content"] for item in routed[:4]] == ["u00", "a00", "u01", "a01"]
    assert all(item["timestamp"] for item in routed)
    assert routed[-2]["content"] == control_like
    assert routed[-1]["content"] == current
    assert sum(item["content"] == current for item in routed) == 1
    assert len(BUILD_CALLS) == before_builds
    assert (get_chat_session(session_id) or {}).get("handoff_status") in {"requested", "active"}
    done = next(item["data"] for item in events if item["event"] == "done")
    assert done["handoff"] is True
    assert done["vertical_provider_request_count"] == 0
    stages = [item["span_name"] for item in list_trace_events(done["trace_id"])]
    assert "agent_selector" not in stages and "agent_frozen" not in stages


def test_b_and_c_freeze_one_agent_without_fallback() -> None:
    for lane, selected, message in (
        (RuntimeLane.PROPERTY_GOVERNED, "b_agent", "symbol-b"),
        (RuntimeLane.ISOLATED_GENERAL, "c_agent", "symbol-c"),
    ):
        ROUTER_DECISIONS[message] = LaneDecision(
            lane=lane,
            selected_agent_id=selected,
            reason=f"choose-{selected}",
        )
        before = len(BUILD_CALLS)
        events = asyncio.run(consume(message, f"session-{selected}"))
        assert BUILD_CALLS[before:] == [selected]
        done = next(item["data"] for item in events if item["event"] == "done")
        assert done["current_agent_id"] == selected
        stages = [item["span_name"] for item in list_trace_events(done["trace_id"])]
        assert stages.count("router") == 1
        assert stages.count("agent_frozen") == 1
        assert "agent_selector" not in stages


def test_router_reason_cannot_create_write() -> None:
    message = "symbol-router-write-text"
    ROUTER_DECISIONS[message] = LaneDecision(
        lane=RuntimeLane.PROPERTY_GOVERNED,
        selected_agent_id="b_agent",
        reason="work_order_create",
    )
    before_drafts = scalar("SELECT COUNT(*) FROM work_order_drafts")
    before_proposals = scalar("SELECT COUNT(*) FROM action_proposals")
    asyncio.run(consume(message, "session-router-write-text"))
    assert scalar("SELECT COUNT(*) FROM work_order_drafts") == before_drafts
    assert scalar("SELECT COUNT(*) FROM action_proposals") == before_proposals


def work_order_payload(suffix: str) -> Dict[str, str]:
    return {
        "room_id": f"room-{suffix}",
        "issue_type": "type",
        "issue_desc": f"issue-{suffix}",
        "urgency": "normal",
        "contact_name": "owner",
        "contact_phone": f"phone-{suffix}",
        "appointment_time": "slot",
    }


def test_structured_work_order_state_machine() -> None:
    session_id = "state-machine-ok"
    partial = apply_structured_proposal_request(
        session_id=session_id,
        proposal_request={"issue_desc": "partial"},
        trace_id="trace-partial",
        release_id="rr-test",
        selected_agent_id="b_agent",
    )
    assert partial["action"] == "draft_updated"
    assert scalar("SELECT COUNT(*) FROM action_proposals WHERE session_id=?", (session_id,)) == 0
    assert scalar("SELECT COUNT(*) FROM work_orders WHERE session_id=?", (session_id,)) == 0

    pending = apply_structured_proposal_request(
        session_id=session_id,
        proposal_request=work_order_payload("ok"),
        trace_id="trace-complete",
        release_id="rr-test",
        selected_agent_id="b_agent",
    )
    proposal_id = pending["proposal_id"]
    assert get_action_proposal(proposal_id)["status"] == "pending_confirmation"
    assert scalar("SELECT COUNT(*) FROM action_approvals WHERE proposal_id=?", (proposal_id,)) == 0
    assert scalar("SELECT COUNT(*) FROM action_receipts WHERE proposal_id=?", (proposal_id,)) == 0
    assert scalar("SELECT COUNT(*) FROM work_orders WHERE session_id=?", (session_id,)) == 0

    committed = decide_work_order_proposal(
        session_id=session_id,
        proposal_id=proposal_id,
        decision="confirm",
        actor="owner:test",
    )
    replay = decide_work_order_proposal(
        session_id=session_id,
        proposal_id=proposal_id,
        decision="confirm",
        actor="owner:test",
    )
    assert committed["action"] == replay["action"] == "committed"
    assert committed["receipt_id"] == replay["receipt_id"]
    assert committed["resource_id"] == replay["resource_id"]
    assert scalar("SELECT COUNT(*) FROM work_orders WHERE session_id=?", (session_id,)) == 1
    assert scalar("SELECT COUNT(*) FROM action_receipts WHERE proposal_id=?", (proposal_id,)) == 1
    assert scalar("SELECT COUNT(*) FROM action_approvals WHERE proposal_id=?", (proposal_id,)) == 1

    failed_session = "state-machine-failed"
    failed_pending = apply_structured_proposal_request(
        session_id=failed_session,
        proposal_request=work_order_payload("failed"),
        trace_id="trace-failed",
        release_id="rr-test",
        selected_agent_id="b_agent",
    )
    original_handler = action_gateway._handlers["work_order.create"]
    action_gateway._handlers["work_order.create"] = lambda *_args: (_ for _ in ()).throw(RuntimeError("symbolic failure"))
    try:
        failed = decide_work_order_proposal(
            session_id=failed_session,
            proposal_id=failed_pending["proposal_id"],
            decision="confirm",
            actor="owner:test",
        )
    finally:
        action_gateway._handlers["work_order.create"] = original_handler
    assert failed["action"] == "failed"
    assert scalar("SELECT COUNT(*) FROM action_receipts WHERE proposal_id=?", (failed_pending["proposal_id"],)) == 1
    assert scalar("SELECT COUNT(*) FROM work_orders WHERE session_id=?", (failed_session,)) == 0


def test_rag_citation_snapshot_and_mcp_boundaries() -> None:
    chunks = [
        {"chunk_index": index, "content": f"chunk-{index}", "chunk_hash": coordinator_module.content_hash(f"chunk-{index}")}
        for index in range(4)
    ]
    knowledge = {
        7: {
            "knowledge_doc_id": 7,
            "title": "doc",
            "document_version": "v1",
            "document_hash": "dh",
            "chunk_count": 4,
            "chunk_snapshots": chunks,
        }
    }
    expanded, _ = _results_from_snapshot(
        "q",
        [{"doc_id": 7, "chunk_index": 1, "content": "chunk-1", "score": 1.0}],
        knowledge,
        {7},
        1,
        0.0,
        100,
    )
    assert [item["chunk_index"] for item in expanded] == [0, 1, 2, 3]
    evidence = build_evidence_set("q", expanded, knowledge_versions=knowledge, allowed_document_ids={7}, retrieval_status="completed")
    valid_id = evidence.items[0].evidence_id
    rendered, citations, violations = render_rag_citations(
        f"answer [[evidence:{valid_id}]] [[evidence:not-a-rag-id]]",
        evidence,
    )
    assert "【引用1】" in rendered
    assert len(citations) == 1 and citations[0].evidence_type == "rag_document_chunk"
    assert [item["code"] for item in violations] == ["invalid_evidence_id"]

    bad_policy = ToolPolicy(
        server_name="mcp",
        tool_name="write",
        effect=ToolEffect.CREATE,
        risk_level="L2",
        allowed_paths=[],
        requires_confirmation=True,
        enabled=True,
    )
    validation = validate_release_graph(
        {"agents": [], "skills": [], "knowledge": [], "mcp_servers": []},
        [bad_policy],
    )
    assert any(item["code"] == "mcp_tool_not_readonly" for item in validation["errors"])
    try:
        ActionGateway().propose("s", "mcp.create", {})
        raise AssertionError("ActionGateway accepted mcp.*")
    except PermissionError:
        pass
    try:
        ToolGateway({"agents": [], "mcp_servers": []}).write_policy("m", "w")
        raise AssertionError("ToolGateway accepted MCP write")
    except PermissionError:
        pass
    try:
        asyncio.run(invoke_confirmed_write({}, "a", "m", "w", {}))
        raise AssertionError("MCP executor accepted write")
    except PermissionError:
        pass


def test_static_contracts() -> None:
    __import__("app.runtime.legacy_chat")
    cards = router_agent_cards(snapshot("static").config)
    assert all(set(item) == {"agent_id", "name", "description", "scope"} for item in cards)
    try:
        router_module._semantic_agent_catalog(
            [{"agent_id": "invalid", "name": "invalid", "scope": ""}]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Router candidate scope must fail closed")
    root = Path(__file__).resolve().parents[1]
    coordinator_source = (root / "app/runtime/coordinator.py").read_text(encoding="utf-8")
    router_source = (root / "agents/router.py").read_text(encoding="utf-8")
    frontend_source = (root / "frontend/index.html").read_text(encoding="utf-8")
    assert RuntimeLane.HANDOFF.value == "A_HANDOFF"
    assert "A_SAFETY_HANDOFF" not in router_source
    assert "A_SAFETY_HANDOFF" not in coordinator_source
    assert "A_SAFETY_HANDOFF" not in frontend_source
    assert "await self._select_agent_after_lane(" not in coordinator_source.split("async def _stream_consultation", 1)[1]
    assert "plan_tools(" not in coordinator_source.split("async def _stream_consultation", 1)[1]
    assert "必须按AgentTurnResult返回结构化proposal_request" in coordinator_source
    mcp_source = (root / "app/runtime/mcp_executor.py").read_text(encoding="utf-8")
    assert '== "model_native"' in mcp_source
    assert "确认创建" in frontend_source and "work-order-proposal/decision" in frontend_source
    assert "await sendChatMessage('请安排工作人员接手本次对话')" in frontend_source


def test_retired_extension_control_path() -> None:
    request = runtime_api_module.ExtensionAcceptanceRequest(
        session_id="retired-extension",
        query="symbolic-query",
        expected_agent_id="b_agent",
    )
    snapshot_calls = 0
    original_resolver = runtime_api_module.resolve_snapshot

    def forbidden_snapshot(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal snapshot_calls
        snapshot_calls += 1
        raise AssertionError("retired endpoint resolved a runtime snapshot")

    runtime_api_module.resolve_snapshot = forbidden_snapshot
    try:
        try:
            asyncio.run(runtime_api_module.extension_acceptance(request))
        except Exception as exc:
            assert getattr(exc, "status_code", None) == 410
            detail = getattr(exc, "detail", {})
            assert detail.get("code") == "extension_acceptance_retired"
        else:
            raise AssertionError("retired extension endpoint did not fail closed")
    finally:
        runtime_api_module.resolve_snapshot = original_resolver
    assert snapshot_calls == 0

    root = Path(__file__).resolve().parents[1]
    api_source = (root / "app/runtime/api.py").read_text(encoding="utf-8")
    retired_source = api_source.split(
        "async def extension_acceptance", 1
    )[1].split('@router.post("/acceptance/trace")', 1)[0]
    assert "_capability_fallback" not in retired_source
    assert "plan_tools" not in retired_source
    frontend_source = (root / "frontend/index.html").read_text(encoding="utf-8")
    assert "/api/runtime/acceptance/extension" not in frontend_source
    assert "runtime-extension-accept-btn" not in frontend_source


def main() -> None:
    install_runtime_fakes()
    try:
        tests = (
            test_full_history_and_a_short_circuit,
            test_b_and_c_freeze_one_agent_without_fallback,
            test_router_reason_cannot_create_write,
            test_structured_work_order_state_machine,
            test_rag_citation_snapshot_and_mcp_boundaries,
            test_static_contracts,
            test_retired_extension_control_path,
        )
        for test in tests:
            test()
            print(f"PASS {test.__name__}")
        print(f"Target1 unified chain: PASS ({len(tests)} checks; Provider requests: 0)")
    finally:
        restore()
        TEMP_DIR.cleanup()


if __name__ == "__main__":
    main()
