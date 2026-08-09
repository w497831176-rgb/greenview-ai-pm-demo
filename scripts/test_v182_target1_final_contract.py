"""Offline deterministic contract for Target 1 final A/B/C convergence.

Run this file directly. It uses only symbolic messages and identifiers, a new
temporary SQLite directory, controlled model doubles, and read-only source
inspection. It never contacts a Provider, never reads production configuration,
and never approves, rejects, deletes, or executes an Action Proposal.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import inspect
import json
import os
import socket
import tempfile
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Dict, Iterable, Iterator, List


ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = tempfile.TemporaryDirectory(
    prefix="yiai-target1-final-contract-",
    ignore_cleanup_errors=True,
)
os.environ["PROPERTY_DATA_DIR"] = TEMP_DIR.name
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _key in (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
):
    os.environ[_key] = ""


from db.property_db import (  # noqa: E402
    _get_conn,
    ensure_chat_session,
    get_action_proposal,
    get_evidence_ledger,
    get_model_calls_for_trace,
    init_db,
    list_chat_messages,
    list_trace_events,
    save_chat_message,
)


# app.settings reads model configuration during import. Initialize only the new
# temporary database first so this script cannot touch production business data.
init_db()


import agents.router as router_module  # noqa: E402
import app.runtime.action_gateway as action_gateway_module  # noqa: E402
import app.runtime.agent_factory as agent_factory_module  # noqa: E402
import app.runtime.citation_renderer as citation_renderer_module  # noqa: E402
import app.runtime.contracts as contracts_module  # noqa: E402
import app.runtime.coordinator as coordinator_module  # noqa: E402
import app.runtime.mcp_executor as mcp_executor_module  # noqa: E402
import app.runtime.release_compiler as release_compiler_module  # noqa: E402
import app.runtime.tool_gateway as tool_gateway_module  # noqa: E402
import app.work_order_workflow as work_order_module  # noqa: E402
from app.runtime.action_gateway import ActionGateway  # noqa: E402
from app.runtime.citation_renderer import build_evidence_set, render_citations  # noqa: E402
from app.runtime.contracts import (  # noqa: E402
    AgentResponseEnvelope,
    RouterDecisionPayload,
    RunConfigSnapshot,
    RuntimeLane,
    RuntimePath,
    ToolEffect,
    WorkOrderCreateProposalRequest,
)
from app.runtime.coordinator import RuntimeCoordinator  # noqa: E402
from app.runtime.provider_accounting import (  # noqa: E402
    begin_provider_attempt,
    capture_active_provider_evidence,
    finalize_provider_attempt,
    mark_provider_attempt_dispatched,
    provider_accounting_scope,
    reset_active_provider_attempt,
)


SYMBOLIC_MESSAGES = [
    {
        "role": ("owner", "staff", "user", "assistant")[(index - 1) % 4],
        "content": f"msg_{index:02d}",
        "timestamp": f"2026-08-08T00:{index:02d}:00+08:00",
    }
    for index in range(1, 13)
]


def _config() -> Dict[str, Any]:
    """Return a fresh Published-like graph with deliberately secret sentinels."""

    return {
        "agents": [
            {
                "agent_id": "router",
                "name": "router_name",
                "description": "router_description",
                "category": "router",
                "enabled": True,
                "model_id": "deepseek-v4-flash",
                "instructions": "router_secret_instruction",
            },
            {
                "agent_id": "agent_b_01",
                "name": "agent_b_name",
                "description": "agent_b_description",
                "category": "vertical",
                "enabled": True,
                "domain_scope": "property",
                "model_id": "model_b",
                "instructions": "agent_b_secret_instruction",
                "skill_ids": [101],
                "knowledge_doc_ids": [201],
                "mcp_server_names": ["server_b_01"],
            },
            {
                "agent_id": "agent_b_02",
                "name": "agent_b2_name",
                "description": "agent_b2_description",
                "category": "vertical",
                "enabled": True,
                "domain_scope": "property",
                "model_id": "model_b2",
                "instructions": "agent_b2_secret_instruction",
                "skill_ids": [102],
                "knowledge_doc_ids": [202],
                "mcp_server_names": [],
            },
            {
                "agent_id": "agent_c_01",
                "name": "agent_c_name",
                "description": "agent_c_description",
                "category": "vertical",
                "enabled": True,
                "domain_scope": "isolated_general",
                "model_id": "model_c",
                "instructions": "agent_c_secret_instruction",
                "skill_ids": [103],
                "knowledge_doc_ids": [203],
                "mcp_server_names": ["server_c_01"],
            },
        ],
        "skills": [
            {"skill_id": 101, "name": "skill_b_01", "enabled": True, "instructions": "skill_b_secret"},
            {"skill_id": 102, "name": "skill_b_02", "enabled": True, "instructions": "skill_b2_secret"},
            {"skill_id": 103, "name": "skill_c_01", "enabled": True, "instructions": "skill_c_secret"},
        ],
        "knowledge": [
            {"knowledge_doc_id": 201, "title": "doc_b_01", "content": "doc_b_secret"},
            {"knowledge_doc_id": 202, "title": "doc_b_02", "content": "doc_b2_secret"},
            {"knowledge_doc_id": 203, "title": "doc_c_01", "content": "doc_c_secret"},
        ],
        "mcp_servers": [
            {
                "id": 301,
                "name": "server_b_01",
                "enabled": True,
                "description": "server_b_secret",
                "tools": [
                    {
                        "name": "read_b_01",
                        "policy": {
                            "server_id": 301,
                            "server_name": "server_b_01",
                            "tool_name": "read_b_01",
                            "effect": "read",
                            "risk_level": "L1",
                            "allowed_paths": ["consultation"],
                            "requires_confirmation": False,
                            "enabled": True,
                            "policy_reason": "read_only",
                        },
                    }
                ],
            },
            {
                "id": 302,
                "name": "server_c_01",
                "enabled": True,
                "description": "server_c_secret",
                "tools": [
                    {
                        "name": "read_c_01",
                        "policy": {
                            "server_id": 302,
                            "server_name": "server_c_01",
                            "tool_name": "read_c_01",
                            "effect": "read",
                            "risk_level": "L1",
                            "allowed_paths": ["consultation"],
                            "requires_confirmation": False,
                            "enabled": True,
                            "policy_reason": "read_only",
                        },
                    }
                ],
            },
        ],
        "model_policy": {
            "version": "target1_final_contract",
            "default": {
                "model_id": "deepseek-v4-flash",
                "provider": "deepseek",
                "model_params": {"use_thinking": True},
            },
            "available": [],
        },
    }


def _snapshot(session_id: str) -> RunConfigSnapshot:
    return RunConfigSnapshot(
        snapshot_id=f"snapshot_{session_id}",
        release_id="rr_symbolic_01",
        snapshot_hash="snapshot_hash_01",
        session_id=session_id,
        created_at="2026-08-08T00:00:00+08:00",
        config=_config(),
    )


def _snapshot_c_without_capabilities(session_id: str) -> RunConfigSnapshot:
    config = _config()
    selected = next(
        item for item in config["agents"] if item["agent_id"] == "agent_c_01"
    )
    selected["skill_ids"] = []
    selected["knowledge_doc_ids"] = []
    selected["mcp_server_names"] = []
    return RunConfigSnapshot(
        snapshot_id=f"snapshot_{session_id}",
        release_id="rr_symbolic_01",
        snapshot_hash="snapshot_hash_c_empty_01",
        session_id=session_id,
        created_at="2026-08-08T00:00:00+08:00",
        config=config,
    )


def _scalar(sql: str, params: Iterable[Any] = ()) -> int:
    conn = _get_conn()
    try:
        row = conn.execute(sql, tuple(params)).fetchone()
        return int(row[0])
    finally:
        conn.close()


def _session_counts(session_id: str) -> Dict[str, int]:
    return {
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


@contextmanager
def _patch(target: Any, name: str, replacement: Any) -> Iterator[None]:
    sentinel = object()
    if isinstance(target, type):
        original = inspect.getattr_static(target, name, sentinel)
        had_direct_attribute = original is not sentinel
    else:
        namespace = vars(target)
        had_direct_attribute = name in namespace
        original = namespace.get(name, sentinel)
    setattr(target, name, replacement)
    try:
        yield
    finally:
        if not had_direct_attribute:
            delattr(target, name)
        else:
            setattr(target, name, original)


def _maybe_patch(stack: ExitStack, target: Any, name: str, replacement: Any) -> None:
    if hasattr(target, name):
        stack.enter_context(_patch(target, name, replacement))


def _forbidden(name: str, hits: List[str]) -> Callable[..., Any]:
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        hits.append(name)
        raise AssertionError(f"forbidden production path reached: {name}")

    return fail


def _forbidden_async(name: str, hits: List[str]) -> Callable[..., Any]:
    async def fail(*_args: Any, **_kwargs: Any) -> Any:
        hits.append(name)
        raise AssertionError(f"forbidden production path reached: {name}")

    return fail


class _FakeRun:
    def __init__(self, content: Any):
        self.content = content
        self.metrics: Dict[str, Any] = {}
        self.model_provider_data: Dict[str, Any] = {}


class _FakeRouterAgent:
    def __init__(self, response: Any, calls: List[Dict[str, Any]]):
        self.response = response
        self.calls = calls

    async def arun(self, prompt: Any, **kwargs: Any) -> _FakeRun:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return _FakeRun(self.response)


def _router_result(lane: RuntimeLane, selected_agent_id: Any, reason: str) -> Dict[str, Any]:
    decision = RouterDecisionPayload(
        lane=lane,
        selected_agent_id=selected_agent_id,
        reason=reason,
    )
    return {
        "decision": decision,
        "raw": decision.model_dump_json(),
        "metrics": {},
        "provider_evidence": {},
        "provider_status": "success",
        "validation_error": None,
    }


def _parse_sse(block: str) -> Dict[str, Any]:
    event_name = "message"
    data_lines: List[str] = []
    for line in block.splitlines():
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
    data: Any = None
    if data_lines:
        data = json.loads("\n".join(data_lines))
    return {"event": event_name, "data": data}


async def _consume(message: str, session_id: str) -> List[Dict[str, Any]]:
    original_connect = socket.create_connection

    def no_network(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("network access is forbidden in final contract tests")

    socket.create_connection = no_network
    try:
        blocks = [
            block
            async for block in RuntimeCoordinator().stream(
                message,
                session_id,
                "owner_symbolic_01",
            )
        ]
    finally:
        socket.create_connection = original_connect
    return [_parse_sse(item) for item in blocks]


async def _terminal_stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[str]:
    yield coordinator_module._sse("done", {"status": "complete"})


def _coordinator_shell(
    stack: ExitStack,
    classifier: Callable[..., Any],
) -> None:
    stack.enter_context(_patch(coordinator_module, "resolve_snapshot", _snapshot))
    stack.enter_context(_patch(router_module, "classify_lane_decision", classifier))
    stack.enter_context(
        _patch(
            coordinator_module,
            "provider_accounting_scope",
            lambda **_kwargs: nullcontext(),
        )
    )
    stack.enter_context(
        _patch(
            coordinator_module,
            "_build_model_from_snapshot",
            lambda *_args, **_kwargs: object(),
        )
    )
    _maybe_patch(stack, RuntimeCoordinator, "_stream_selected_agent", _terminal_stream)


def _assert_no_error(events: List[Dict[str, Any]]) -> None:
    errors = [item for item in events if item["event"] == "error"]
    if errors:
        trace_id = str((errors[0].get("data") or {}).get("trace_id") or "")
        raise AssertionError(
            {"errors": errors, "trace_events": list_trace_events(trace_id)}
        )


def test_router_gets_complete_timestamped_sequence() -> None:
    cards = agent_factory_module.router_agent_cards(_config())
    calls: List[Dict[str, Any]] = []
    response = {
        "lane": RuntimeLane.PROPERTY_GOVERNED.value,
        "selected_agent_id": "agent_b_01",
        "reason": "reason_b_01",
    }
    fake = _FakeRouterAgent(json.dumps(response), calls)
    with _patch(
        router_module,
        "create_semantic_lane_router",
        lambda **_kwargs: fake,
    ):
        result = asyncio.run(
            router_module.classify_lane_decision(
                messages=copy.deepcopy(SYMBOLIC_MESSAGES),
                vertical_agents=cards,
                user_id="owner_symbolic_01",
                session_id="session_router_01",
                model=object(),
            )
        )
    assert result["decision"] == RouterDecisionPayload.model_validate(response)
    assert len(calls) == 1
    payload = json.loads(str(calls[0]["prompt"]))
    assert payload["messages"] == SYMBOLIC_MESSAGES
    assert "current_user_message" not in payload
    assert "visible_conversation" not in payload
    assert "history" not in payload
    assert len(payload["messages"]) > 5

    visible_rows = [
        {
            "role": item["role"],
            "content": item["content"],
            "created_at": item["timestamp"],
        }
        for item in SYMBOLIC_MESSAGES
    ]
    with _patch(
        coordinator_module,
        "list_chat_messages",
        lambda _session_id: copy.deepcopy(visible_rows),
    ):
        assert coordinator_module._visible_chat_history("session_router_01") == SYMBOLIC_MESSAGES


def test_every_bubble_routes_once_with_draft_or_proposal() -> None:
    calls: List[Dict[str, Any]] = []

    async def classify(*, messages: List[Dict[str, Any]], vertical_agents: List[Dict[str, Any]], **_kwargs: Any) -> Dict[str, Any]:
        calls.append({"messages": copy.deepcopy(messages), "cards": copy.deepcopy(vertical_agents)})
        return _router_result(
            RuntimeLane.PROPERTY_GOVERNED,
            "agent_b_01",
            "reason_b_02",
        )

    for session_id in (
        "session_plain_01",
        "session_draft_state_01",
        "session_proposal_state_01",
    ):
        ensure_chat_session(session_id, "owner_symbolic_01")
        state_payload = {
            "room_id": "room_state_01",
            "issue_type": "issue_state_01",
            "issue_desc": "desc_state_01",
            "urgency": "urgency_state_01",
            "contact_name": "contact_state_01",
            "contact_phone": "phone_state_01",
            "appointment_time": "time_state_01",
        }
        if session_id == "session_draft_state_01":
            work_order_module.save_work_order_draft(
                session_id=session_id,
                **state_payload,
            )
        elif session_id == "session_proposal_state_01":
            work_order_module.action_gateway.propose(
                session_id=session_id,
                action_type="work_order.create",
                payload=state_payload,
                trace_id="trace_existing_proposal_01",
            )
        before = len(calls)
        with ExitStack() as stack:
            _coordinator_shell(stack, classify)
            for forbidden_name in ("_select_path", "_stream_controlled_action"):
                _maybe_patch(
                    stack,
                    RuntimeCoordinator,
                    forbidden_name,
                    _forbidden(forbidden_name, []),
                )
            events = asyncio.run(_consume("msg_current_01", session_id))
        _assert_no_error(events)
        assert len(calls) == before + 1, {"session_id": session_id, "calls": calls}


def test_router_cards_and_schema_are_strict() -> None:
    cards = agent_factory_module.router_agent_cards(_config())
    assert cards == [
        {"agent_id": "agent_b_01", "name": "agent_b_name", "description": "agent_b_description", "scope": "property"},
        {"agent_id": "agent_b_02", "name": "agent_b2_name", "description": "agent_b2_description", "scope": "property"},
        {"agent_id": "agent_c_01", "name": "agent_c_name", "description": "agent_c_description", "scope": "isolated_general"},
    ]
    assert all(set(card) == {"agent_id", "name", "description", "scope"} for card in cards)
    serialized = json.dumps(cards, sort_keys=True)
    for secret in (
        "secret_instruction",
        "skill_b_secret",
        "doc_b_secret",
        "server_b_secret",
        "capability",
        "binding",
        "tool",
    ):
        assert secret not in serialized

    assert RouterDecisionPayload.model_config.get("extra") == "forbid"
    invalid = (
        {"lane": "A_SAFETY_HANDOFF", "reason": "reason_00"},
        {"lane": "A_SAFETY_HANDOFF", "selected_agent_id": "agent_b_01", "reason": "reason_01"},
        {"lane": "B_PROPERTY_GOVERNED", "selected_agent_id": None, "reason": "reason_02"},
        {"lane": "C_ISOLATED_GENERAL", "selected_agent_id": "", "reason": "reason_03"},
        {"lane": "B_PROPERTY_GOVERNED", "selected_agent_id": "agent_b_01", "reason": "reason_04", "business_intent": "work_order_create"},
    )
    for value in invalid:
        try:
            RouterDecisionPayload.model_validate(value)
        except Exception:
            continue
        raise AssertionError(f"invalid Router payload was accepted: {value}")


def test_a_unified_handoff_short_circuits() -> None:
    router_calls: List[List[Dict[str, Any]]] = []
    handoff_calls: List[Dict[str, Any]] = []
    forbidden_hits: List[str] = []

    async def classify(*, messages: List[Dict[str, Any]], **_kwargs: Any) -> Dict[str, Any]:
        router_calls.append(copy.deepcopy(messages))
        return _router_result(RuntimeLane.SAFETY_HANDOFF, None, "reason_a_01")

    def request_handoff(session_id: str, reason: str, **kwargs: Any) -> Dict[str, Any]:
        handoff_calls.append({"session_id": session_id, "reason": reason, **kwargs})
        return {"session_id": session_id, "handoff_status": "requested"}

    with ExitStack() as stack:
        _coordinator_shell(stack, classify)
        stack.enter_context(_patch(coordinator_module, "request_handoff", request_handoff))
        for target, name, async_call in (
            (RuntimeCoordinator, "_select_agent_after_lane", True),
            (coordinator_module, "build_agent_from_snapshot", False),
            (coordinator_module, "_results_from_snapshot", False),
            (coordinator_module, "preinvoke_read_tools", True),
            (coordinator_module, "build_model_native_read_tools", False),
            (coordinator_module, "advance_structured_work_order_workflow", False),
            (coordinator_module, "advance_work_order_workflow", False),
        ):
            replacement = _forbidden_async(name, forbidden_hits) if async_call else _forbidden(name, forbidden_hits)
            _maybe_patch(stack, target, name, replacement)
        for name in ("propose", "approve", "reject", "execute", "execute_async"):
            replacement = _forbidden_async(name, forbidden_hits) if name == "execute_async" else _forbidden(name, forbidden_hits)
            _maybe_patch(stack, coordinator_module.action_gateway, name, replacement)
        events = asyncio.run(_consume("msg_a_01", "session_a_01"))

    _assert_no_error(events)
    assert len(router_calls) == 1
    assert len(handoff_calls) == 1
    assert forbidden_hits == []
    rendered = json.dumps(events, ensure_ascii=False)
    assert "reason_a_01" in rendered
    branch_source = inspect.getsource(RuntimeCoordinator._stream_unified_handoff)
    for token in ("HandoffKind", "safety_override", "emergency"):
        assert token not in branch_source
    assert branch_source.count("request_handoff(") == 1


def test_selected_agent_is_frozen_on_b_and_c_failure() -> None:
    source = "\n".join(
        inspect.getsource(getattr(RuntimeCoordinator, name))
        for name in ("stream", "_resolve_semantic_lane", "_stream_selected_agent")
        if hasattr(RuntimeCoordinator, name)
    )
    for forbidden in (
        "_select_agent_after_lane(",
        "select_lane_agent(",
        "agent_selector",
        "_capability_fallback(",
        "_first_enabled(",
        "build_lane_agent_unavailable_decision(",
        "advance_work_order_workflow(",
        "is_confirmation(",
        "plan_tools(",
        "_select_path(",
        "_stream_controlled_action(",
    ):
        assert forbidden not in source
    assert "selected_agent_id" in source
    assert "build_agent_from_snapshot" in source


def test_c_without_capabilities_completes_one_frozen_agent_run() -> None:
    router_calls: List[Dict[str, Any]] = []
    build_calls: List[str] = []
    tool_build_calls: List[str] = []
    vertical_calls: List[Dict[str, Any]] = []
    forbidden_hits: List[str] = []
    vertical_payload = {
        "value": AgentResponseEnvelope(answer="answer_c_dynamic_01").model_dump_json()
    }

    async def classify(*, messages: List[Dict[str, Any]], vertical_agents: List[Dict[str, Any]], **_kwargs: Any) -> Dict[str, Any]:
        router_calls.append(
            {"messages": copy.deepcopy(messages), "cards": copy.deepcopy(vertical_agents)}
        )
        return _router_result(
            RuntimeLane.ISOLATED_GENERAL,
            "agent_c_01",
            "reason_c_dynamic_01",
        )

    class FakeVerticalAgent:
        def arun(self, prompt: Any, **kwargs: Any) -> AsyncIterator[Any]:
            vertical_calls.append({"prompt": prompt, "kwargs": kwargs})

            async def chunks() -> AsyncIterator[Any]:
                yield SimpleNamespace(
                    content={"status": "success", "value": "tool_event_ignored_01"},
                    event="ToolCallCompleted",
                    metrics={},
                )
                yield SimpleNamespace(
                    content=vertical_payload["value"],
                    event="RunContentCompleted",
                    metrics={},
                )

            return chunks()

    def build_agent(_snapshot_value: Any, agent_id: str, *_args: Any, **_kwargs: Any) -> Any:
        build_calls.append(agent_id)
        return SimpleNamespace(
            agent=FakeVerticalAgent(),
            activated_skills=[],
            skill_tool_calls=[],
        )

    def build_tools(_config_value: Dict[str, Any], agent_id: str, *_args: Any, **_kwargs: Any) -> List[Any]:
        tool_build_calls.append(agent_id)
        return []

    @contextmanager
    def accounting_scope(**kwargs: Any) -> Iterator[Any]:
        attempts: List[Dict[str, Any]] = []
        if kwargs.get("stage") == "vertical_agent":
            attempts.append(
                {
                    "provider_request_sequence": 1,
                    "provider_response_model": "model_c",
                    "provider_request_id": "request_c_dynamic_01",
                    "status": "success",
                    "usage": {
                        "input_cache_hit_tokens": 0,
                        "input_cache_miss_tokens": 3,
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "reasoning_tokens": 1,
                        "total_tokens": 5,
                    },
                }
            )
        yield SimpleNamespace(attempts=attempts)

    with ExitStack() as stack:
        stack.enter_context(
            _patch(coordinator_module, "resolve_snapshot", _snapshot_c_without_capabilities)
        )
        stack.enter_context(_patch(router_module, "classify_lane_decision", classify))
        stack.enter_context(
            _patch(coordinator_module, "provider_accounting_scope", accounting_scope)
        )
        stack.enter_context(
            _patch(
                coordinator_module,
                "_build_model_from_snapshot",
                lambda *_args, **_kwargs: object(),
            )
        )
        stack.enter_context(
            _patch(coordinator_module, "build_agent_from_snapshot", build_agent)
        )
        stack.enter_context(
            _patch(coordinator_module, "build_model_native_read_tools", build_tools)
        )
        for name in (
            "advance_structured_work_order_workflow",
            "get_work_order_draft",
            "get_pending_action_proposal",
        ):
            stack.enter_context(
                _patch(coordinator_module, name, _forbidden(name, forbidden_hits))
            )
        events = asyncio.run(_consume("msg_c_dynamic_01", "session_c_dynamic_01"))
        vertical_payload["value"] = '{"answer":"invalid_c_01","extra":"forbidden"}'
        failed_events = asyncio.run(
            _consume("msg_c_dynamic_02", "session_c_dynamic_failure_01")
        )

    _assert_no_error(events)
    assert len(router_calls) == 2
    assert build_calls == ["agent_c_01", "agent_c_01"]
    assert tool_build_calls == ["agent_c_01", "agent_c_01"]
    assert len(vertical_calls) == 2
    assert forbidden_hits == []
    done = [item["data"] for item in events if item["event"] == "done"][-1]
    assert done["status"] == "completed"
    assert done["current_agent_id"] == "agent_c_01"
    assert done["vertical_provider_request_count"] == 1
    assert done["content"] == "answer_c_dynamic_01"
    error = [item["data"] for item in failed_events if item["event"] == "error"][-1]
    assert error["error_code"] == "agent_envelope_error"
    failed_ledger = get_evidence_ledger(error["trace_id"])
    assert failed_ledger is not None
    assert any(
        item.get("stage") == "vertical_agent"
        and item.get("provider_request_id") == "request_c_dynamic_01"
        for item in failed_ledger["ledger"]["model_calls"]
    )


def test_b_bindings_and_reference_allowlist() -> None:
    config = _config()
    assert agent_factory_module.validate_agent_binding_isolation(
        config,
        "agent_b_01",
        expected_scope="property",
    ) == "property"
    shared = _config()
    shared["agents"][3]["skill_ids"].append(101)
    try:
        agent_factory_module.validate_agent_binding_isolation(
            shared,
            "agent_c_01",
            expected_scope="isolated_general",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("cross-domain shared binding was accepted")

    evidence = build_evidence_set(
        "query_01",
        [
            {
                "doc_id": 201,
                "doc_title": "doc_b_01",
                "chunk_id": "chunk_b_01",
                "chunk_index": 0,
                "content": "source_01",
                "source": "retrieval_01",
            }
        ],
        allowed_document_ids={201},
    )
    evidence_id = evidence.items[0].evidence_id
    hits: List[str] = []
    with ExitStack() as stack:
        _maybe_patch(
            stack,
            citation_renderer_module,
            "_citation_is_supported",
            _forbidden("semantic_citation_override", hits),
        )
        rendered, citations, violations = render_citations(
            f"claim_99 [[evidence:{evidence_id}]]",
            evidence,
        )
    assert hits == []
    assert citations and citations[0].evidence_id == evidence_id
    assert violations == []
    assert "【引用1】" in rendered

    _rendered, invalid_citations, invalid_violations = render_citations(
        "claim_98 [[evidence:ev_not_from_run]]",
        evidence,
    )
    assert invalid_citations == []
    assert any(item.get("code") == "unknown_evidence_id" for item in invalid_violations)

    if mcp_executor_module.GovernedMCPTools is not None:
        toolkit = object.__new__(mcp_executor_module.GovernedMCPTools)
        object.__setattr__(toolkit, "server_name", "server_b_01")
        object.__setattr__(toolkit, "result_contracts", {})
        object.__setattr__(toolkit, "recorded_invocations", [])

        def read_tool(**_kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "value": "tool_value_01"}

        exposed = asyncio.run(
            toolkit._wrap_entrypoint(read_tool, "read_b_01")(
                key="argument_01"
            )
        )
        recorded = toolkit.recorded_invocations
        assert len(recorded) == 1
        assert exposed["citation_id"] == recorded[0].invocation_id
        assert exposed["result"]["value"] == "tool_value_01"


def test_c_optional_bindings_and_property_isolation() -> None:
    config = _config()
    assert agent_factory_module.validate_agent_binding_isolation(
        config,
        "agent_c_01",
        expected_scope="isolated_general",
    ) == "isolated_general"
    policies = tool_gateway_module.ToolGateway(config).policies_for_agent(
        "agent_c_01",
        RuntimePath.CONSULTATION,
    )
    assert [(item.server_name, item.tool_name) for item in policies] == [
        ("server_c_01", "read_c_01")
    ]

    empty = _config()
    empty["agents"][3]["skill_ids"] = []
    empty["agents"][3]["knowledge_doc_ids"] = []
    empty["agents"][3]["mcp_server_names"] = []
    assert agent_factory_module.validate_agent_binding_isolation(
        empty,
        "agent_c_01",
        expected_scope="isolated_general",
    ) == "isolated_general"
    assert tool_gateway_module.ToolGateway(empty).policies_for_agent(
        "agent_c_01",
        RuntimePath.CONSULTATION,
    ) == []

    sentinel_key = "SYMBOLIC_UNDECLARED_PROPERTY_ENV"
    os.environ[sentinel_key] = "symbolic_secret_value"
    try:
        assert sentinel_key not in mcp_executor_module._server_subprocess_env(
            {"env_keys": []}
        )
        assert mcp_executor_module._server_subprocess_env(
            {"env_keys": [sentinel_key]}
        )[sentinel_key] == "symbolic_secret_value"
    finally:
        os.environ.pop(sentinel_key, None)

    envelope = AgentResponseEnvelope(answer="answer_c_01")
    assert envelope.validate_for_scope("isolated_general") == envelope
    try:
        AgentResponseEnvelope(
            answer="answer_c_02",
            proposal_request=WorkOrderCreateProposalRequest(
                action_type="work_order.create",
                room_id="room_01",
                issue_type="issue_01",
                issue_desc="desc_01",
                urgency="urgency_01",
                contact_name="contact_01",
                contact_phone="phone_01",
                appointment_time="time_01",
            ),
        ).validate_for_scope("isolated_general")
    except ValueError:
        pass
    else:
        raise AssertionError("C envelope gained a property write capability")


def test_router_write_text_cannot_open_workflow() -> None:
    session_id = "session_router_text_01"
    ensure_chat_session(session_id, "owner_symbolic_01")
    before = _session_counts(session_id)
    calls: List[Dict[str, Any]] = []
    fake = _FakeRouterAgent(
        json.dumps(
            {
                "lane": "B_PROPERTY_GOVERNED",
                "selected_agent_id": "agent_b_01",
                "reason": "work_order_create",
            }
        ),
        calls,
    )
    with _patch(router_module, "create_semantic_lane_router", lambda **_kwargs: fake):
        result = asyncio.run(
            router_module.classify_lane_decision(
                messages=copy.deepcopy(SYMBOLIC_MESSAGES),
                vertical_agents=agent_factory_module.router_agent_cards(_config()),
                user_id="owner_symbolic_01",
                session_id=session_id,
                model=object(),
            )
        )
    assert result["decision"] is not None
    assert len(calls) == 1
    assert _session_counts(session_id) == before


def test_only_selected_b_structured_proposal_creates_pending() -> None:
    signature = inspect.signature(work_order_module.advance_structured_work_order_workflow)
    assert "message" not in signature.parameters
    assert "selected_agent_id" in signature.parameters
    assert "selected_agent_scope" in signature.parameters
    assert "proposal_request" in signature.parameters

    request = WorkOrderCreateProposalRequest(
        action_type="work_order.create",
        room_id="room_02",
        issue_type="issue_02",
        issue_desc="desc_02",
        urgency="urgency_02",
        contact_name="contact_02",
        contact_phone="phone_02",
        appointment_time="time_02",
    )
    session_id = "session_proposal_01"
    ensure_chat_session(session_id, "owner_symbolic_01")
    result = work_order_module.advance_structured_work_order_workflow(
        session_id=session_id,
        selected_agent_id="agent_b_01",
        selected_agent_scope="property",
        trace_id="trace_proposal_01",
        proposal_request=request,
    )
    assert result and result["action"] == "awaiting_confirmation"
    proposal = get_action_proposal(result["proposal_id"])
    assert proposal and proposal["status"] == "pending_confirmation"
    assert result["selected_agent_id"] == "agent_b_01"

    for agent_id, scope in (("agent_c_01", "isolated_general"), ("", "property")):
        try:
            work_order_module.advance_structured_work_order_workflow(
                session_id=f"session_rejected_{agent_id or 'empty'}",
                selected_agent_id=agent_id,
                selected_agent_scope=scope,
                proposal_request=request,
            )
        except PermissionError:
            continue
        raise AssertionError("a non-selected B Agent opened the write workflow")


def test_pending_proposal_has_no_approval_receipt_or_work_order() -> None:
    session_id = "session_proposal_pending_01"
    ensure_chat_session(session_id, "owner_symbolic_01")
    request = WorkOrderCreateProposalRequest(
        action_type="work_order.create",
        room_id="room_03",
        issue_type="issue_03",
        issue_desc="desc_03",
        urgency="urgency_03",
        contact_name="contact_03",
        contact_phone="phone_03",
        appointment_time="time_03",
    )
    result = work_order_module.advance_structured_work_order_workflow(
        session_id=session_id,
        selected_agent_id="agent_b_01",
        selected_agent_scope="property",
        trace_id="trace_proposal_pending_01",
        proposal_request=request,
    )
    assert result and result["action"] == "awaiting_confirmation"
    assert _session_counts(session_id) == {
        "proposals": 1,
        "approvals": 0,
        "receipts": 0,
        "work_orders": 0,
    }

    stale_session = "session_stale_proposal_01"
    ensure_chat_session(stale_session, "owner_symbolic_01")
    stale_proposal = SimpleNamespace(
        proposal_id="proposal_stale_01",
        status="committed",
    )
    with _patch(
        work_order_module.action_gateway,
        "propose",
        lambda **_kwargs: stale_proposal,
    ):
        try:
            work_order_module.advance_structured_work_order_workflow(
                session_id=stale_session,
                selected_agent_id="agent_b_01",
                selected_agent_scope="property",
                proposal_request=request,
            )
        except RuntimeError as exc:
            assert "no longer pending" in str(exc)
        else:
            raise AssertionError("a stale Proposal was misreported as pending")


def test_mcp_write_is_rejected_at_all_layers() -> None:
    server = {"id": 401, "name": "server_write_01", "enabled": True}
    for effect in (ToolEffect.CREATE, ToolEffect.UPDATE, ToolEffect.DELETE):
        policy = release_compiler_module.compile_tool_policy(
            server,
            {
                "name": f"tool_{effect.value}",
                "tool_metadata": {
                    "effect": effect.value,
                    "effect_source": "operator_declared",
                    "risk_level": "L2",
                },
            },
        )
        assert policy.enabled is False
        assert policy.allowed_paths == []

    write_config = _config()
    write_config["agents"][1]["mcp_server_names"] = ["server_write_01"]
    write_config["mcp_servers"] = [
        {
            "id": 401,
            "name": "server_write_01",
            "enabled": True,
            "tools": [
                {
                    "name": "tool_create",
                    "policy": {
                        "server_id": 401,
                        "server_name": "server_write_01",
                        "tool_name": "tool_create",
                        "effect": "create",
                        "risk_level": "L2",
                        "allowed_paths": ["controlled_action"],
                        "requires_confirmation": True,
                        "enabled": True,
                        "policy_reason": "legacy_write_fixture",
                    },
                }
            ],
        }
    ]
    try:
        tool_gateway_module.ToolGateway(write_config).write_policy(
            "server_write_01",
            "tool_create",
            agent_id="agent_b_01",
        )
    except tool_gateway_module.ToolPolicyError:
        pass
    else:
        raise AssertionError("runtime ToolGateway exposed an MCP write policy")

    session_id = "session_mcp_write_01"
    ensure_chat_session(session_id, "owner_symbolic_01")
    before = _session_counts(session_id)
    try:
        ActionGateway().propose(
            session_id=session_id,
            action_type="mcp.server_write_01.tool_create",
            payload={"arg_01": "value_01"},
            trace_id="trace_mcp_write_01",
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("ActionGateway accepted an mcp.* Proposal")
    assert _session_counts(session_id) == before

    source = inspect.getsource(ActionGateway.execute_async)
    gateway_boundary = source + inspect.getsource(action_gateway_module._internal_action_type)
    assert "invoke_confirmed_write" not in source
    assert "mcp." in gateway_boundary and "raise" in gateway_boundary
    executor_source = inspect.getsource(mcp_executor_module.invoke_confirmed_write)
    assert "raise" in executor_source


def test_trace_has_no_selector_and_one_row_per_physical_call() -> None:
    live_source = "\n".join(
        inspect.getsource(getattr(RuntimeCoordinator, name))
        for name in ("stream", "_resolve_semantic_lane", "_stream_selected_agent")
        if hasattr(RuntimeCoordinator, name)
    )
    assert "agent_selector" not in live_source
    selected_source = inspect.getsource(RuntimeCoordinator._stream_selected_agent)
    assert selected_source.index("state.tool_invocations = []") < selected_source.index(
        "if agent_failure is not None"
    )

    trace_id = "trace_accounting_01"
    for sequence, stage in enumerate(("router", "vertical_agent", "vertical_agent"), 1):
        with provider_accounting_scope(
            trace_id=trace_id,
            session_id="session_accounting_01",
            stage=stage,
            model_selection_reason="reason_accounting_01",
            price_snapshot={
                "input_cache_hit_price_per_million": 0,
                "input_cache_miss_price_per_million": 0,
                "output_price_per_million": 0,
            },
            model_policy_version="policy_01",
        ):
            attempt, token = begin_provider_attempt(
                requested_model="model_symbolic_01",
                thinking_enabled=True,
                stream=False,
            )
            try:
                mark_provider_attempt_dispatched(attempt)
                capture_active_provider_evidence(
                    {
                        "provider_response_model": "model_symbolic_01",
                        "provider_request_id": f"request_{sequence:02d}",
                        "usage": {
                            "input_cache_hit_tokens": sequence,
                            "input_cache_miss_tokens": sequence,
                            "input_tokens": sequence * 2,
                            "output_tokens": sequence,
                            "reasoning_tokens": sequence,
                            "total_tokens": sequence * 3,
                        },
                    }
                )
                normalized = finalize_provider_attempt(attempt, normal_completion=True)
                assert normalized["status"] == "success"
            finally:
                reset_active_provider_attempt(token)
    calls = get_model_calls_for_trace(trace_id)
    assert len(calls) == 3
    assert [item["stage"] for item in calls] == ["router", "vertical_agent", "vertical_agent"]
    assert len({item["provider_request_id"] for item in calls}) == 3


def test_frozen_history_and_write_evidence_are_read_only() -> None:
    verifier_path = ROOT / "scripts" / "verify_v182_s10f_history_readonly.py"
    source = verifier_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(verifier_path))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str))
    }
    for expected in (4096, 12373, 6714, 5009, 23183, 0.02588292):
        assert expected in literals
    assert "mode=ro" in source
    assert "database_artifact_fingerprint" in source
    assert "UPDATE " not in source.upper()
    assert "DELETE FROM" not in source.upper()


def test_imports_and_target_syntax() -> None:
    targets = (
        "agents/router.py",
        "app/runtime/contracts.py",
        "app/runtime/coordinator.py",
        "app/runtime/agent_factory.py",
        "app/runtime/citation_renderer.py",
        "app/work_order_workflow.py",
        "app/runtime/release_compiler.py",
        "app/runtime/tool_gateway.py",
        "app/runtime/mcp_executor.py",
        "app/runtime/action_gateway.py",
        "app/main.py",
    )
    for relative in targets:
        path = ROOT / relative
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    envelope_fields = set(AgentResponseEnvelope.model_fields)
    assert envelope_fields == {
        "answer",
        "citation_ids",
        "proposal_request",
        "confirmation_request",
    }
    assert AgentResponseEnvelope.model_config.get("extra") == "forbid"
    for invalid_envelope in (
        '```json\n{"answer":"answer_01"}\n```',
        '{"answer":"answer_01","unexpected":"value_01"}',
    ):
        try:
            coordinator_module._parse_agent_envelope(invalid_envelope)
        except Exception:
            continue
        raise AssertionError("non-strict Agent envelope was accepted")
    coordinator_source = inspect.getsource(RuntimeCoordinator)
    assert "router_agent_cards" in coordinator_source
    assert "advance_structured_work_order_workflow" in coordinator_source


TESTS = (
    test_router_gets_complete_timestamped_sequence,
    test_every_bubble_routes_once_with_draft_or_proposal,
    test_router_cards_and_schema_are_strict,
    test_a_unified_handoff_short_circuits,
    test_selected_agent_is_frozen_on_b_and_c_failure,
    test_c_without_capabilities_completes_one_frozen_agent_run,
    test_b_bindings_and_reference_allowlist,
    test_c_optional_bindings_and_property_isolation,
    test_router_write_text_cannot_open_workflow,
    test_only_selected_b_structured_proposal_creates_pending,
    test_pending_proposal_has_no_approval_receipt_or_work_order,
    test_mcp_write_is_rejected_at_all_layers,
    test_trace_has_no_selector_and_one_row_per_physical_call,
    test_frozen_history_and_write_evidence_are_read_only,
    test_imports_and_target_syntax,
)


def main() -> None:
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS final_contract_tests={len(TESTS)}")


if __name__ == "__main__":
    main()
