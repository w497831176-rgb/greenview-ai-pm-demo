"""Offline contracts for target 1: the necessary A/B/C demo paths.

This script uses a fresh temporary SQLite database, never opens the network,
and never invokes a model Provider. Existing focused tests cover A and C; this
file closes the B-RAG, B-Tool, and controlled-write behavior gaps.
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
    prefix="yiai-target1-demo-",
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
    ensure_chat_session,
    get_action_proposal,
    get_chat_session,
    get_evidence_ledger,
    get_work_order,
    init_db,
)


# app.settings constructs the default model object during import and reads the
# model configuration table. Initialize only this fresh temporary database
# before importing any app runtime module.
init_db()


from app.runtime import mcp_executor  # noqa: E402
import agents.router as router_module  # noqa: E402
import app.runtime.coordinator as coordinator_module  # noqa: E402
from app.runtime.agent_factory import build_agent_from_snapshot  # noqa: E402
from app.runtime.contracts import (  # noqa: E402
    LaneDecision,
    RunConfigSnapshot,
    RunStatus,
    RuntimeLane,
    RuntimePath,
    ToolEffect,
    content_hash,
)
from app.runtime.coordinator import (  # noqa: E402
    RuntimeCoordinator,
    _answer_contract_for,
    _requires_rag_citation,
    _results_from_snapshot,
)
from app.runtime.tool_planner import plan_tools  # noqa: E402
from app.work_order_workflow import advance_work_order_workflow  # noqa: E402


RAG_QUERY = "紧急维修的登记和到场时限是什么？"
TOOL_QUERY = "查询我最近的维修工单。"
INCOMPLETE_WRITE = (
    "我家厨房水龙头持续漏水，房号3-2-1201，请帮我创建一张普通维修工单。"
    "请先让我确认，不要直接提交。"
)
COMPLETE_WRITE = (
    "请创建维修工单：房号3-2-1201，厨房水龙头持续漏水，紧急程度中等，"
    "联系人王先生，电话13800138000，预约明天下午上门。请先让我确认，不要直接提交。"
)
QUERY_ONLY = "查询最近工单，仅查询，不要创建。"
OTHER_B = "请说明公共区域维修的服务范围。"
POISONED_A = "这是一个安全风险场景。"
POISONED_C = "请解释一个非物业概念。"
CHINESE_INTENT = "请创建厨房水龙头维修工单。"

ENTRY_DECISIONS: Dict[str, LaneDecision] = {
    INCOMPLETE_WRITE: LaneDecision(
        lane=RuntimeLane.PROPERTY_GOVERNED,
        business_intent="work_order_create",
        reason="业主明确要求启动维修工单创建流程并先确认。",
    ),
    COMPLETE_WRITE: LaneDecision(
        lane=RuntimeLane.PROPERTY_GOVERNED,
        business_intent="work_order_create",
        reason="业主明确要求启动维修工单创建流程并先确认。",
    ),
    QUERY_ONLY: LaneDecision(
        lane=RuntimeLane.PROPERTY_GOVERNED,
        business_intent="read_recent_work_orders",
        reason="只查询已有工单。",
    ),
    OTHER_B: LaneDecision(
        lane=RuntimeLane.PROPERTY_GOVERNED,
        business_intent="maintenance_service_scope",
        reason="普通物业咨询。",
    ),
    CHINESE_INTENT: LaneDecision(
        lane=RuntimeLane.PROPERTY_GOVERNED,
        business_intent="创建厨房水龙头维修工单",
        reason="故意返回非保留值以验证精确匹配。",
    ),
    POISONED_A: LaneDecision(
        lane=RuntimeLane.SAFETY_HANDOFF,
        business_intent="work_order_create",
        reason="A即使携带保留值也不得进入写流程。",
    ),
    POISONED_C: LaneDecision(
        lane=RuntimeLane.ISOLATED_GENERAL,
        business_intent="work_order_create",
        reason="C即使携带保留值也不得进入写流程。",
    ),
}

PATCHES: List[Tuple[Any, str, Any]] = []
FORBIDDEN_HITS: List[str] = []
CONSULTATION_SESSIONS: List[str] = []
HANDOFF_SESSIONS: List[str] = []
ROUTER_MESSAGES: List[str] = []


def _scalar(sql: str, params: Tuple[Any, ...] = ()) -> int:
    conn = _get_conn()
    try:
        return int(conn.execute(sql, params).fetchone()[0])
    finally:
        conn.close()


def _rows(sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    conn = _get_conn()
    conn.row_factory = __import__("sqlite3").Row
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _provider_attempt_count() -> int:
    return _scalar(
        "SELECT COUNT(*) FROM model_calls WHERE record_kind = 'provider_attempt'"
    )


async def _fake_entry_router(message: str, **_: Any) -> Dict[str, Any]:
    ROUTER_MESSAGES.append(message)
    decision = ENTRY_DECISIONS.get(message)
    assert decision is not None, message
    return {
        "decision": decision,
        "raw": decision.model_dump_json(),
        "metrics": {},
        "provider_evidence": {},
        "provider_status": "success",
        "validation_error": None,
    }


def _patch(target: Any, name: str, replacement: Any) -> None:
    PATCHES.append((target, name, getattr(target, name)))
    setattr(target, name, replacement)


def _restore_patches() -> None:
    for target, name, original in reversed(PATCHES):
        setattr(target, name, original)
    PATCHES.clear()


def _forbidden(name: str) -> Callable[..., Any]:
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        FORBIDDEN_HITS.append(name)
        raise AssertionError(f"controlled write entered forbidden capability: {name}")

    return fail


def _forbidden_async(name: str) -> Callable[..., Any]:
    async def fail(*_args: Any, **_kwargs: Any) -> Any:
        FORBIDDEN_HITS.append(name)
        raise AssertionError(f"controlled write entered forbidden capability: {name}")

    return fail


async def _fake_consultation(
    _self: RuntimeCoordinator,
    _message: str,
    session_id: str,
    _user_id: str,
    trace_id: str,
    snapshot: RunConfigSnapshot,
    state: Any,
    ledger: Any,
    _started: float,
):
    CONSULTATION_SESSIONS.append(session_id)
    state.status = RunStatus.COMPLETED
    state.next_step = None
    ledger.capture_state(state)
    ledger.persist("complete")
    yield coordinator_module._sse(
        "done",
        {
            "status": "complete",
            "trace_id": trace_id,
            "runtime_path": RuntimePath.CONSULTATION.value,
            "release_id": snapshot.release_id,
            "snapshot_id": snapshot.snapshot_id,
        },
    )


async def _fake_handoff(
    _self: RuntimeCoordinator,
    _message: str,
    session_id: str,
    trace_id: str,
    snapshot: RunConfigSnapshot,
    state: Any,
    ledger: Any,
    _started: float,
):
    HANDOFF_SESSIONS.append(session_id)
    state.status = RunStatus.COMPLETED
    state.next_step = None
    ledger.capture_state(state)
    ledger.persist("complete")
    yield coordinator_module._sse(
        "done",
        {
            "status": "complete",
            "trace_id": trace_id,
            "runtime_path": RuntimePath.CONSULTATION.value,
            "release_id": snapshot.release_id,
            "snapshot_id": snapshot.snapshot_id,
            "handoff": True,
        },
    )


def _install_entry_fakes() -> None:
    FORBIDDEN_HITS.clear()
    CONSULTATION_SESSIONS.clear()
    HANDOFF_SESSIONS.clear()
    ROUTER_MESSAGES.clear()
    _patch(coordinator_module, "resolve_snapshot", _snapshot)
    _patch(router_module, "classify_lane_decision", _fake_entry_router)
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
    _patch(RuntimeCoordinator, "_stream_consultation", _fake_consultation)
    _patch(RuntimeCoordinator, "_stream_a_handoff", _fake_handoff)

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
        RuntimeCoordinator,
        "_advance_dynamic_mcp_action",
        _forbidden_async("dynamic_mcp_action"),
    )
    _patch(
        coordinator_module.action_gateway,
        "approve",
        _forbidden("action_gateway_approve"),
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
    original_connect = socket.create_connection

    def no_network(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("network access is forbidden in Target 1 B-write test")

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


def _event(events: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    matches = [item["data"] for item in events if item["event"] == name]
    assert len(matches) == 1, {"event": name, "matches": matches, "events": events}
    assert isinstance(matches[0], dict)
    return matches[0]


def _delta_text(events: List[Dict[str, Any]]) -> str:
    return "".join(
        str((item.get("data") or {}).get("content") or "")
        for item in events
        if item.get("event") == "delta"
    )


def _session_counts(session_id: str) -> Dict[str, int]:
    return {
        "drafts": _scalar(
            "SELECT COUNT(*) FROM work_order_drafts WHERE session_id = ?",
            (session_id,),
        ),
        "proposals": _scalar(
            "SELECT COUNT(*) FROM action_proposals WHERE session_id = ?",
            (session_id,),
        ),
        "approvals": _scalar(
            """SELECT COUNT(*) FROM action_approvals a
               JOIN action_proposals p ON p.proposal_id = a.proposal_id
               WHERE p.session_id = ?""",
            (session_id,),
        ),
        "receipts": _scalar(
            """SELECT COUNT(*) FROM action_receipts r
               JOIN action_proposals p ON p.proposal_id = r.proposal_id
               WHERE p.session_id = ?""",
            (session_id,),
        ),
        "work_orders": _scalar(
            "SELECT COUNT(*) FROM work_orders WHERE session_id = ?",
            (session_id,),
        ),
    }


def _assert_no_committed_write(session_id: str) -> None:
    counts = _session_counts(session_id)
    assert counts["approvals"] == 0, counts
    assert counts["receipts"] == 0, counts
    assert counts["work_orders"] == 0, counts


def test_structured_work_order_entry_full_flow() -> None:
    before_attempts = _provider_attempt_count()
    _install_entry_fakes()
    try:
        incomplete_session = "target1-b-write-incomplete"
        incomplete_events = asyncio.run(
            _consume(INCOMPLETE_WRITE, incomplete_session)
        )
        assert not [
            item for item in incomplete_events if item.get("event") == "error"
        ], incomplete_events
        incomplete_lane = _event(incomplete_events, "lane")
        incomplete_done = _event(incomplete_events, "done")
        incomplete_tools = _event(incomplete_events, "tool_calls")
        incomplete_reply = _delta_text(incomplete_events)
        assert incomplete_lane["lane"] == RuntimeLane.PROPERTY_GOVERNED.value
        assert incomplete_lane["business_intent"] == "work_order_create"
        assert incomplete_done["status"] == "paused"
        assert incomplete_done["runtime_path"] == RuntimePath.CONTROLLED_ACTION.value
        assert incomplete_done.get("proposal_id") is None
        assert incomplete_tools["tool_calls"][0]["arguments"]["phase"] == "draft_updated"
        assert "3-2-1201" in incomplete_reply and "厨房水龙头持续漏水" in incomplete_reply
        for label in ("紧急程度", "联系电话", "预约上门时间"):
            assert label in incomplete_reply, incomplete_reply
        assert _session_counts(incomplete_session) == {
            "drafts": 1,
            "proposals": 0,
            "approvals": 0,
            "receipts": 0,
            "work_orders": 0,
        }
        incomplete_draft = _rows(
            "SELECT * FROM work_order_drafts WHERE session_id = ?",
            (incomplete_session,),
        )[0]
        assert incomplete_draft["room_id"] == "3-2-1201"
        assert "厨房水龙头持续漏水" in incomplete_draft["issue_desc"]
        assert incomplete_draft["urgency"] == ""
        assert incomplete_draft["contact_phone"] == ""
        assert incomplete_draft["appointment_time"] == ""
        incomplete_trace = _event(incomplete_events, "start")["trace_id"]
        incomplete_ledger_row = get_evidence_ledger(incomplete_trace)
        assert incomplete_ledger_row["runtime_path"] == RuntimePath.CONTROLLED_ACTION.value
        incomplete_ledger = incomplete_ledger_row["ledger"]
        assert incomplete_ledger.get("action_proposals") == []
        assert incomplete_ledger.get("approval_events") == []
        assert incomplete_ledger.get("action_receipts") == []
        assert incomplete_ledger.get("activated_skills") == []
        assert incomplete_ledger.get("retrieval_evidence") == []
        assert incomplete_ledger.get("tool_invocations") == []
        assert incomplete_ledger.get("handoff_events") == []

        complete_session = "target1-b-write-complete"
        complete_events = asyncio.run(_consume(COMPLETE_WRITE, complete_session))
        assert not [
            item for item in complete_events if item.get("event") == "error"
        ], complete_events
        complete_lane = _event(complete_events, "lane")
        complete_done = _event(complete_events, "done")
        complete_tools = _event(complete_events, "tool_calls")
        complete_reply = _delta_text(complete_events)
        assert complete_lane["lane"] == RuntimeLane.PROPERTY_GOVERNED.value
        assert complete_lane["business_intent"] == "work_order_create"
        assert complete_done["status"] == "paused"
        assert complete_done["runtime_path"] == RuntimePath.CONTROLLED_ACTION.value
        proposal_id = str(complete_done.get("proposal_id") or "")
        assert proposal_id
        assert complete_tools["tool_calls"][0]["arguments"]["phase"] == "awaiting_confirmation"
        assert "3-2-1201" in complete_reply and "厨房水龙头持续漏水" in complete_reply
        assert "中" in complete_reply and "王先生" in complete_reply
        assert "13800138000" in complete_reply and "明天下午" in complete_reply
        assert _session_counts(complete_session) == {
            "drafts": 1,
            "proposals": 1,
            "approvals": 0,
            "receipts": 0,
            "work_orders": 0,
        }
        proposal = get_action_proposal(proposal_id)
        assert proposal and proposal["status"] == "pending_confirmation"
        payload = proposal.get("payload") or {}
        assert payload["room_id"] == "3-2-1201"
        assert "厨房水龙头持续漏水" in payload["issue_desc"]
        assert payload["urgency"] == "中"
        assert payload["contact_name"] == "王先生"
        assert payload["contact_phone"] == "13800138000"
        assert "明天下午" in payload["appointment_time"]
        _assert_no_committed_write(complete_session)
        complete_trace = _event(complete_events, "start")["trace_id"]
        complete_ledger_row = get_evidence_ledger(complete_trace)
        assert complete_ledger_row["runtime_path"] == RuntimePath.CONTROLLED_ACTION.value
        complete_ledger = complete_ledger_row["ledger"]
        assert len(complete_ledger.get("action_proposals") or []) == 1
        assert complete_ledger.get("approval_events") == []
        assert complete_ledger.get("action_receipts") == []
        assert complete_ledger.get("activated_skills") == []
        assert complete_ledger.get("retrieval_evidence") == []
        assert complete_ledger.get("tool_invocations") == []
        assert complete_ledger.get("handoff_events") == []

        negative_cases = (
            (QUERY_ONLY, "target1-b-write-query", "consultation"),
            (OTHER_B, "target1-b-write-other-b", "consultation"),
            (CHINESE_INTENT, "target1-b-write-chinese-intent", "consultation"),
            (POISONED_A, "target1-b-write-poisoned-a", "handoff"),
            (POISONED_C, "target1-b-write-poisoned-c", "consultation"),
        )
        for message, session_id, terminal in negative_cases:
            events = asyncio.run(_consume(message, session_id))
            lane = _event(events, "lane")
            expected = ENTRY_DECISIONS[message]
            assert lane["lane"] == expected.lane.value
            assert lane["business_intent"] == expected.business_intent
            assert _session_counts(session_id) == {
                "drafts": 0,
                "proposals": 0,
                "approvals": 0,
                "receipts": 0,
                "work_orders": 0,
            }
            if terminal == "handoff":
                assert session_id in HANDOFF_SESSIONS
            else:
                assert session_id in CONSULTATION_SESSIONS

        assert FORBIDDEN_HITS == []
        assert _provider_attempt_count() == before_attempts == 0
    finally:
        _restore_patches()


def _install_state_machine_fakes() -> None:
    FORBIDDEN_HITS.clear()
    ROUTER_MESSAGES.clear()
    _patch(coordinator_module, "resolve_snapshot", _snapshot)
    _patch(router_module, "classify_lane_decision", _fake_entry_router)
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
        RuntimeCoordinator,
        "_advance_dynamic_mcp_action",
        _forbidden_async("dynamic_mcp_action"),
    )


def test_persisted_work_order_state_machine_through_coordinator() -> None:
    before_attempts = _provider_attempt_count()
    _install_state_machine_fakes()
    try:
        commit_session = "target1-b-write-coordinator-commit"
        start_events = asyncio.run(_consume(COMPLETE_WRITE, commit_session))
        start_done = _event(start_events, "done")
        assert start_done["status"] == "paused"
        assert start_done["tool_calls"][0]["arguments"]["phase"] == "awaiting_confirmation"
        assert _session_counts(commit_session) == {
            "drafts": 1,
            "proposals": 1,
            "approvals": 0,
            "receipts": 0,
            "work_orders": 0,
        }

        commit_events = asyncio.run(_consume("确认创建", commit_session))
        commit_done = _event(commit_events, "done")
        assert commit_done["status"] == "completed"
        assert commit_done["tool_calls"][0]["arguments"]["phase"] == "committed"
        assert len(commit_done.get("action_receipts") or []) == 1
        assert _session_counts(commit_session) == {
            "drafts": 0,
            "proposals": 1,
            "approvals": 1,
            "receipts": 1,
            "work_orders": 1,
        }

        replay_events = asyncio.run(_consume("确认创建", commit_session))
        replay_done = _event(replay_events, "done")
        assert replay_done["status"] == "completed"
        assert replay_done["tool_calls"][0]["arguments"]["phase"] == "idempotent_replay"
        assert _session_counts(commit_session) == {
            "drafts": 0,
            "proposals": 1,
            "approvals": 1,
            "receipts": 1,
            "work_orders": 1,
        }

        cancel_session = "target1-b-write-coordinator-cancel"
        cancel_start_events = asyncio.run(_consume(COMPLETE_WRITE, cancel_session))
        assert _event(cancel_start_events, "done")["status"] == "paused"
        cancel_events = asyncio.run(_consume("取消这个工单", cancel_session))
        cancel_done = _event(cancel_events, "done")
        assert cancel_done["status"] == "completed"
        assert cancel_done["tool_calls"][0]["arguments"]["phase"] == "rejected"
        assert _session_counts(cancel_session) == {
            "drafts": 0,
            "proposals": 1,
            "approvals": 1,
            "receipts": 0,
            "work_orders": 0,
        }

        assert ROUTER_MESSAGES == [COMPLETE_WRITE, COMPLETE_WRITE]
        assert FORBIDDEN_HITS == []
        assert _provider_attempt_count() == before_attempts == 0
    finally:
        _restore_patches()


def _snapshot(session_id: str):
    knowledge_content = (
        "第二章 响应时效承诺\n"
        "一、紧急维修：接到报修后，物业客服中心在 5 分钟内完成工单登记并通知工程人员，"
        "工程人员 30 分钟内到场处置。"
    )
    tool_policy = {
        "server_id": 7,
        "server_name": "workorder-server",
        "tool_name": "get_my_recent_work_orders",
        "effect": "read",
        "risk_level": "L0",
        "allowed_paths": ["consultation"],
        "requires_confirmation": False,
        "enabled": True,
        "policy_reason": "target1 offline fixture",
    }
    return RunConfigSnapshot(
        snapshot_id=f"snapshot-{session_id}",
        release_id="rr-target1-offline",
        snapshot_hash="target1-offline-snapshot",
        session_id=session_id,
        created_at="2026-08-07T00:00:00+08:00",
        config={
            "agents": [
                {
                    "agent_id": "maintenance",
                    "name": "维修 Agent",
                    "enabled": True,
                    "category": "maintenance",
                    "domain_scope": "property",
                    "instructions": "处理维修咨询和只读工单查询。",
                    "skill_ids": [8],
                    "mcp_server_names": ["workorder-server"],
                    "knowledge_doc_ids": [1],
                }
            ],
            "skills": [
                {
                    "skill_id": 8,
                    "name": "维修工单处理",
                    "description": "维修咨询与工单处理规则",
                    "version": "1.0.0",
                    "enabled": True,
                    "trigger_condition": "维修,报修,工单",
                    "metadata": {
                        "positive_triggers": ["维修", "报修", "工单"]
                    },
                    "content_hash": "skill-eight-target1",
                    "reference_snapshots": [],
                    "instructions_fallback": "先核实维修事项；只读查询不得写入业务数据。",
                }
            ],
            "knowledge": [
                {
                    "knowledge_doc_id": 1,
                    "title": "物业维修服务承诺",
                    "document_hash": "doc-one-target1",
                    "document_version": "v1",
                    "chunk_snapshots": [
                        {
                            "chunk_index": 1,
                            "content": knowledge_content,
                            "chunk_hash": content_hash(knowledge_content),
                        }
                    ],
                }
            ],
            "mcp_servers": [
                {
                    "id": 7,
                    "name": "workorder-server",
                    "enabled": True,
                    "command": "fake-workorder",
                    "args": [],
                    "tools": [
                        {
                            "name": "get_my_recent_work_orders",
                            "description": "查询我的最近维修工单",
                            "input_schema": {
                                "type": "object",
                                "properties": {"limit": {"type": "integer"}},
                            },
                            "policy": tool_policy,
                        }
                    ],
                }
            ],
            "retrieval_policy": {"top_k": 5, "context_threshold": 0.2},
            "model_policy": {
                "default": {
                    "model_id": "deepseek-v4-flash",
                    "provider": "deepseek",
                    "model_params": {"use_thinking": True},
                },
                "available": [],
            },
        },
    )


def test_b_rag_uses_bound_snapshot_and_skill() -> None:
    snapshot = _snapshot("target1-b-rag")
    maintenance = next(
        item
        for item in snapshot.config.get("agents") or []
        if item.get("agent_id") == "maintenance" and item.get("enabled")
    )
    allowed_document_ids = {
        int(item) for item in maintenance.get("knowledge_doc_ids") or []
    }
    knowledge_versions = {
        int(item["knowledge_doc_id"]): item
        for item in snapshot.config.get("knowledge") or []
        if int(item.get("knowledge_doc_id") or 0) in allowed_document_ids
    }
    results, used_snapshot = _results_from_snapshot(
        RAG_QUERY,
        [],
        knowledge_versions,
        allowed_document_ids,
        int((snapshot.config.get("retrieval_policy") or {}).get("top_k") or 5),
        float(
            (snapshot.config.get("retrieval_policy") or {}).get(
                "context_threshold"
            )
            or 0.2
        ),
    )
    matching = [
        item
        for item in results
        if int(item.get("doc_id") or 0) == 1
        and int(item.get("chunk_index") or -1) == 1
    ]
    assert used_snapshot
    assert matching, results
    compact = str(matching[0].get("content") or "").replace(" ", "")
    assert "5分钟" in compact and "30分钟" in compact

    build = build_agent_from_snapshot(snapshot, "maintenance", RAG_QUERY)
    assert [item.skill_id for item in build.activated_skills] == [8], {
        "activated": [item.skill_id for item in build.activated_skills],
        "decisions": build.skill_decisions,
        "bound_skill_ids": maintenance.get("skill_ids") or [],
    }
    assert [item.get("skill_id") for item in build.skill_tool_calls] == [8]
    assert all(item.get("status") == "success" for item in build.skill_tool_calls)
    # The Skill is already loaded into immutable instructions, so its access
    # tool is intentionally hidden from the model after pre-invocation.
    assert build.agent.skills is None


class _FakeFunction:
    async def entrypoint(self, **arguments: Any) -> dict[str, Any]:
        assert arguments == {"limit": 5}
        return {
            "status": "success",
            "data": [
                {
                    "work_order_id": "WO-TARGET1-001",
                    "status": "处理中",
                }
            ],
        }


class _FakeMCPTools:
    def __init__(self, **_: Any):
        self.functions: dict[str, Any] = {}

    async def __aenter__(self):
        self.functions = {"get_my_recent_work_orders": _FakeFunction()}
        return self

    async def close(self) -> None:
        return None


def test_b_tool_is_successful_evidence_without_unrelated_rag() -> None:
    snapshot = _snapshot("target1-b-tool")
    plans = plan_tools(
        snapshot.config,
        "maintenance",
        TOOL_QUERY,
        RuntimePath.CONSULTATION,
        effects=[ToolEffect.READ],
        execution_modes=["auto_preinvoke"],
    )
    assert [(item.server_name, item.tool_name) for item in plans] == [
        ("workorder-server", "get_my_recent_work_orders")
    ]
    assert plans[0].arguments == {"limit": 5}

    original = mcp_executor.MCPTools
    mcp_executor.MCPTools = _FakeMCPTools
    try:
        context, invocations = asyncio.run(
            mcp_executor.preinvoke_read_tools(
                snapshot.config,
                "maintenance",
                TOOL_QUERY,
            )
        )
    finally:
        mcp_executor.MCPTools = original

    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.invocation_status == "success"
    assert invocation.business_status == "success"
    assert "WO-TARGET1-001" in context

    contract = _answer_contract_for(
        LaneDecision(
            lane=RuntimeLane.PROPERTY_GOVERNED,
            business_intent="read_recent_work_orders",
        )
    )
    assert not _requires_rag_citation(
        contract,
        evidence_count=4,
        linked_skill_evidence_count=0,
        successful_tool_evidence_count=1,
    )
    assert _requires_rag_citation(
        contract,
        evidence_count=1,
        linked_skill_evidence_count=0,
        successful_tool_evidence_count=0,
    )


def test_controlled_write_requires_confirmation_and_is_idempotent() -> None:
    session_id = "target1-controlled-write"
    ensure_chat_session(session_id)
    before_orders = _scalar("SELECT COUNT(*) FROM work_orders")
    first = advance_work_order_workflow(
        session_id,
        (
            "请创建维修工单：房号 3-2-1201，厨房水槽持续漏水，紧急，"
            "联系人测试业主，电话 13800138000，尽快上门。"
        ),
        trace_id="trace-target1-proposal",
        release_id="rr-target1-test",
        start_authorized=True,
    )
    assert first and first["action"] == "awaiting_confirmation"
    assert _scalar("SELECT COUNT(*) FROM work_orders") == before_orders
    proposal = get_action_proposal(first["proposal_id"])
    assert proposal and proposal["status"] == "pending_confirmation"
    assert get_chat_session(session_id)["handoff_status"] == "none"

    committed = advance_work_order_workflow(
        session_id,
        "确认创建",
        trace_id="trace-target1-confirm",
        release_id="rr-target1-test",
    )
    assert committed and committed["action"] == "committed"
    receipt = committed.get("receipt") or {}
    assert receipt.get("status") == "committed"
    assert receipt.get("receipt_id") and receipt.get("resource_id")
    assert get_work_order(receipt["resource_id"])
    assert _scalar("SELECT COUNT(*) FROM work_orders") == before_orders + 1

    replay = advance_work_order_workflow(session_id, "确认创建")
    assert replay and replay["action"] == "idempotent_replay"
    assert replay["work_order_id"] == committed["work_order_id"]
    assert (replay.get("receipt") or {}).get("receipt_id") == receipt["receipt_id"]
    assert _scalar("SELECT COUNT(*) FROM work_orders") == before_orders + 1
    assert get_chat_session(session_id)["handoff_status"] == "none"

    cancel_session = "target1-controlled-write-cancel"
    ensure_chat_session(cancel_session)
    cancel_first = advance_work_order_workflow(
        cancel_session,
        (
            "请创建维修工单：房号 3-2-1201，厨房水槽持续漏水，紧急程度中等，"
            "联系人王先生，电话 13800138000，预约明天下午上门。"
        ),
        trace_id="trace-target1-cancel-proposal",
        release_id="rr-target1-test",
        start_authorized=True,
    )
    assert cancel_first and cancel_first["action"] == "awaiting_confirmation"
    cancelled = advance_work_order_workflow(
        cancel_session,
        "取消这个工单",
        trace_id="trace-target1-cancel",
        release_id="rr-target1-test",
    )
    assert cancelled and cancelled["action"] == "rejected"
    assert _session_counts(cancel_session) == {
        "drafts": 0,
        "proposals": 1,
        "approvals": 1,
        "receipts": 0,
        "work_orders": 0,
    }
    cancelled_proposal = get_action_proposal(cancel_first["proposal_id"])
    assert cancelled_proposal and cancelled_proposal["status"] == "rejected"


def main() -> None:
    before_attempts = _provider_attempt_count()
    tests = [
        test_b_rag_uses_bound_snapshot_and_skill,
        test_b_tool_is_successful_evidence_without_unrelated_rag,
        test_structured_work_order_entry_full_flow,
        test_persisted_work_order_state_machine_through_coordinator,
        test_controlled_write_requires_confirmation_and_is_idempotent,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    assert _provider_attempt_count() == before_attempts == 0
    print("Target 1 necessary demo paths: PASS (Provider attempts: 0)")


if __name__ == "__main__":
    try:
        main()
    finally:
        TEMP_DIR.cleanup()
