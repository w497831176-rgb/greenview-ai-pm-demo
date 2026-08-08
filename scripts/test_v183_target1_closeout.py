"""Focused, offline construction checks for Target 1 close-out.

The test imports the existing Target 1 fixture, which initializes a brand-new
temporary SQLite database and clears Provider credentials before application
imports.  Provider behavior is faked only at the Router and selected-Agent
boundaries.  The real ``RuntimeCoordinator.stream`` and its real
``_stream_selected_agent`` implementation remain in the call path.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, Iterator, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.test_v182_target1_final_contract as base  # noqa: E402
import app.runtime.api as runtime_api_module  # noqa: E402
import app.runtime.legacy_chat as legacy_chat_module  # noqa: E402
from app.runtime.contracts import (  # noqa: E402
    AgentResponseEnvelope,
    RunConfigSnapshot,
    RuntimeLane,
    WorkOrderConfirmationRequest,
    WorkOrderCreateProposalRequest,
)
from fastapi import HTTPException  # noqa: E402


def _snapshot_b_without_capabilities(session_id: str) -> RunConfigSnapshot:
    config = {
        "agents": [
            {
                "agent_id": "router",
                "name": "router_name",
                "description": "router_description",
                "category": "router",
                "enabled": True,
                "model_id": "deepseek-v4-flash",
            },
            {
                "agent_id": "maintenance",
                "name": "maintenance_name",
                "description": "maintenance_description",
                "category": "vertical",
                "enabled": True,
                "domain_scope": "property",
                "model_id": "model_b",
                "skill_ids": [],
                "knowledge_doc_ids": [],
                "mcp_server_names": [],
            },
            {
                "agent_id": "billing",
                "name": "billing_name",
                "description": "billing_description",
                "category": "vertical",
                "enabled": True,
                "domain_scope": "property",
                "model_id": "model_b_alt",
                "skill_ids": [],
                "knowledge_doc_ids": [],
                "mcp_server_names": [],
            },
            {
                "agent_id": "mars-greenhouse-agent",
                "name": "general_name",
                "description": "general_description",
                "category": "vertical",
                "enabled": True,
                "domain_scope": "isolated_general",
                "model_id": "model_c",
                "skill_ids": [],
                "knowledge_doc_ids": [],
                "mcp_server_names": [],
            },
        ],
        "skills": [],
        "knowledge": [],
        "mcp_servers": [],
        "model_policy": {
            "version": "target1_closeout_contract",
            "default": {
                "model_id": "deepseek-v4-flash",
                "provider": "deepseek",
                "model_params": {"use_thinking": True},
            },
            "available": [],
        },
    }
    return RunConfigSnapshot(
        snapshot_id=f"snapshot_{session_id}",
        release_id="rr_symbolic_target1_closeout",
        snapshot_hash="snapshot_hash_target1_closeout",
        session_id=session_id,
        created_at="2026-08-08T00:00:00+08:00",
        config=config,
    )


def _proposal_request(suffix: str) -> WorkOrderCreateProposalRequest:
    return WorkOrderCreateProposalRequest(
        action_type="work_order.create",
        room_id=f"room_{suffix}",
        issue_type=f"issue_{suffix}",
        issue_desc=f"description_{suffix}",
        urgency=f"urgency_{suffix}",
        contact_name=f"contact_{suffix}",
        contact_phone=f"phone_{suffix}",
        appointment_time=f"time_{suffix}",
    )


def _done(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = [item["data"] for item in events if item["event"] == "done"]
    assert values, events
    return values[-1]


def _confirmation(proposal_id: str) -> WorkOrderConfirmationRequest:
    return WorkOrderConfirmationRequest(
        action_type="work_order.create",
        proposal_id=proposal_id,
        decision="approve",
    )


def test_active_a_lane_and_retired_side_channels() -> None:
    assert RuntimeLane.HANDOFF.value == "A_HANDOFF"
    assert RuntimeLane("A_SAFETY_HANDOFF") is RuntimeLane.HANDOFF
    assert RuntimeLane.SAFETY_HANDOFF is RuntimeLane.HANDOFF

    live_source = "\n".join(
        inspect.getsource(getattr(base.RuntimeCoordinator, name))
        for name in ("stream", "_resolve_semantic_lane", "_stream_unified_handoff")
    )
    router_source = inspect.getsource(base.router_module.classify_lane_decision)
    assert "A_SAFETY_HANDOFF" not in live_source
    assert "A_SAFETY_HANDOFF" not in router_source
    assert "A_HANDOFF" in router_source

    session_id = "session_retired_handoff_entry_01"
    base.ensure_chat_session(session_id, "owner_symbolic_01")
    before = base.coordinator_module.get_chat_session(session_id)

    async def invoke_retired_entries() -> None:
        requests = (
            legacy_chat_module.chat_handoff(
                legacy_chat_module.HandoffRequest(
                    session_id=session_id,
                    reason="symbolic_reason_01",
                )
            ),
            legacy_chat_module.chat_handoff_policy(
                legacy_chat_module.HandoffPolicyDiagnosticRequest(
                    message="symbolic_message_01"
                )
            ),
            runtime_api_module.extension_acceptance(),
        )
        for request in requests:
            try:
                await request
            except HTTPException as exc:
                assert exc.status_code == 410
            else:
                raise AssertionError("retired shortcut did not return HTTP 410")

    asyncio.run(invoke_retired_entries())
    after = base.coordinator_module.get_chat_session(session_id)
    assert after == before
    assert base._session_counts(session_id) == {
        "proposals": 0,
        "approvals": 0,
        "receipts": 0,
        "work_orders": 0,
    }

    router_calls: List[List[Dict[str, Any]]] = []
    forbidden_hits: List[str] = []

    async def classify_a(
        *, messages: List[Dict[str, Any]], **_kwargs: Any
    ) -> Dict[str, Any]:
        router_calls.append(json.loads(json.dumps(messages)))
        return base._router_result(
            RuntimeLane.HANDOFF,
            None,
            "reason_unified_handoff_01",
        )

    @contextmanager
    def router_accounting_scope(**_kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(attempts=[])

    active_session = "session_active_a_handoff_01"
    with ExitStack() as stack:
        stack.enter_context(
            base._patch(
                base.coordinator_module,
                "resolve_snapshot",
                _snapshot_b_without_capabilities,
            )
        )
        stack.enter_context(base._patch(base.router_module, "classify_lane_decision", classify_a))
        stack.enter_context(
            base._patch(
                base.coordinator_module,
                "provider_accounting_scope",
                router_accounting_scope,
            )
        )
        stack.enter_context(
            base._patch(
                base.coordinator_module,
                "_build_model_from_snapshot",
                lambda *_args, **_kwargs: object(),
            )
        )
        for name in (
            "_select_agent_after_lane",
            "_stream_a_handoff",
            "_stream_consultation",
            "_stream_controlled_action",
            "_select_path",
            "_maybe_handoff",
        ):
            stack.enter_context(
                base._patch(
                    base.RuntimeCoordinator,
                    name,
                    base._forbidden(name, forbidden_hits),
                )
            )
        active_events = asyncio.run(base._consume("bubble_a_handoff_01", active_session))

    base._assert_no_error(active_events)
    active_done = _done(active_events)
    assert len(router_calls) == 1
    assert forbidden_hits == []
    assert active_done["lane_decision"]["lane"] == "A_HANDOFF"
    assert "A_SAFETY_HANDOFF" not in json.dumps(active_events, ensure_ascii=False)
    active_trace_id = str(active_done["trace_id"])
    active_ledger = base.get_evidence_ledger(active_trace_id)
    assert active_ledger is not None
    assert active_ledger["ledger"]["lane_decision"]["lane"] == "A_HANDOFF"
    assert [
        item["span_name"] for item in base.list_trace_events(active_trace_id)
    ].count("agent_selector") == 0
    assert base._session_counts(active_session) == {
        "proposals": 0,
        "approvals": 0,
        "receipts": 0,
        "work_orders": 0,
    }

    extension_source = inspect.getsource(runtime_api_module.extension_acceptance)
    assert "_capability_fallback" not in extension_source
    assert "plan_tools" not in extension_source
    frontend_source = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "/api/runtime/acceptance/extension" not in frontend_source
    assert "post('/api/chat/handoff'," not in frontend_source


def test_public_stream_full_state_machine_and_frozen_b_refusal() -> None:
    envelopes: List[AgentResponseEnvelope] = []
    router_calls: List[Dict[str, Any]] = []
    build_calls: List[str] = []
    vertical_calls: List[Dict[str, Any]] = []
    provider_sequence = {"value": 0}
    forbidden_hits: List[str] = []

    async def classify(
        *,
        messages: List[Dict[str, Any]],
        vertical_agents: List[Dict[str, Any]],
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        router_calls.append(
            {
                "messages": json.loads(json.dumps(messages)),
                "cards": json.loads(json.dumps(vertical_agents)),
            }
        )
        return base._router_result(
            RuntimeLane.PROPERTY_GOVERNED,
            "maintenance",
            "reason_b_frozen_01",
        )

    class FakeVerticalAgent:
        def arun(self, prompt: Any, **kwargs: Any) -> AsyncIterator[Any]:
            vertical_calls.append({"prompt": prompt, "kwargs": kwargs})
            assert envelopes, "missing fake selected-Agent response"
            envelope = envelopes.pop(0)

            async def chunks() -> AsyncIterator[Any]:
                yield SimpleNamespace(
                    content=envelope.model_dump_json(),
                    event="RunContentCompleted",
                    metrics={},
                )

            return chunks()

    def build_agent(
        _snapshot: Any,
        agent_id: str,
        *_args: Any,
        **_kwargs: Any,
    ) -> Any:
        build_calls.append(agent_id)
        return SimpleNamespace(
            agent=FakeVerticalAgent(),
            activated_skills=[],
            skill_tool_calls=[],
        )

    @contextmanager
    def accounting_scope(**kwargs: Any) -> Iterator[Any]:
        attempts: List[Dict[str, Any]] = []
        if kwargs.get("stage") == "vertical_agent":
            provider_sequence["value"] += 1
            sequence = provider_sequence["value"]
            attempts.append(
                {
                    "provider_request_sequence": 1,
                    "provider_response_model": "model_b",
                    "provider_request_id": f"request_b_{sequence:02d}",
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

    gateway = base.work_order_module.action_gateway
    original_execute = gateway.execute
    execute_calls: List[str] = []

    def counted_execute(proposal_id: str) -> Any:
        execute_calls.append(proposal_id)
        return original_execute(proposal_id)

    with ExitStack() as stack:
        stack.enter_context(
            base._patch(base.coordinator_module, "resolve_snapshot", _snapshot_b_without_capabilities)
        )
        stack.enter_context(base._patch(base.router_module, "classify_lane_decision", classify))
        stack.enter_context(
            base._patch(base.coordinator_module, "provider_accounting_scope", accounting_scope)
        )
        stack.enter_context(
            base._patch(
                base.coordinator_module,
                "_build_model_from_snapshot",
                lambda *_args, **_kwargs: object(),
            )
        )
        stack.enter_context(
            base._patch(base.coordinator_module, "build_agent_from_snapshot", build_agent)
        )
        stack.enter_context(
            base._patch(
                base.coordinator_module,
                "build_model_native_read_tools",
                lambda *_args, **_kwargs: [],
            )
        )
        stack.enter_context(base._patch(gateway, "execute", counted_execute))
        for name in (
            "_select_agent_after_lane",
            "_stream_a_handoff",
            "_stream_consultation",
            "_stream_controlled_action",
            "_select_path",
            "_maybe_handoff",
        ):
            stack.enter_context(
                base._patch(
                    base.RuntimeCoordinator,
                    name,
                    base._forbidden(name, forbidden_hits),
                )
            )

        success_session = "session_public_state_machine_success_01"
        proposal_envelope = AgentResponseEnvelope(
            answer="answer_proposal_01",
            proposal_request=_proposal_request("success_01"),
        )
        envelopes.append(proposal_envelope)
        proposal_events = asyncio.run(base._consume("bubble_proposal_01", success_session))
        base._assert_no_error(proposal_events)
        proposal_done = _done(proposal_events)
        proposal_id = str(proposal_done["proposal_id"])
        assert proposal_done["status"] == "paused"
        assert base.work_order_module.get_work_order_draft(success_session)
        assert base._session_counts(success_session) == {
            "proposals": 1,
            "approvals": 0,
            "receipts": 0,
            "work_orders": 0,
        }
        assert execute_calls == []

        envelopes.append(
            AgentResponseEnvelope(
                answer="answer_confirmation_01",
                confirmation_request=_confirmation(proposal_id),
            )
        )
        confirmation_events = asyncio.run(
            base._consume("bubble_confirmation_01", success_session)
        )
        base._assert_no_error(confirmation_events)
        confirmation_done = _done(confirmation_events)
        assert confirmation_done["status"] == "completed"
        assert len(execute_calls) == 1
        assert base._session_counts(success_session) == {
            "proposals": 1,
            "approvals": 1,
            "receipts": 1,
            "work_orders": 1,
        }
        receipt = confirmation_done["action_receipts"][0]
        assert receipt["status"] == "committed"
        assert receipt["resource_id"]

        envelopes.append(
            AgentResponseEnvelope(
                answer="answer_idempotent_01",
                confirmation_request=_confirmation(proposal_id),
            )
        )
        replay_events = asyncio.run(base._consume("bubble_confirmation_02", success_session))
        base._assert_no_error(replay_events)
        assert _done(replay_events)["status"] == "completed"
        assert len(execute_calls) == 1
        assert base._session_counts(success_session) == {
            "proposals": 1,
            "approvals": 1,
            "receipts": 1,
            "work_orders": 1,
        }

        failure_session = "session_public_state_machine_failure_01"
        envelopes.append(
            AgentResponseEnvelope(
                answer="answer_failure_proposal_01",
                proposal_request=_proposal_request("failure_01"),
            )
        )
        failure_proposal_events = asyncio.run(
            base._consume("bubble_failure_proposal_01", failure_session)
        )
        base._assert_no_error(failure_proposal_events)
        failure_proposal_id = str(_done(failure_proposal_events)["proposal_id"])
        assert base._session_counts(failure_session) == {
            "proposals": 1,
            "approvals": 0,
            "receipts": 0,
            "work_orders": 0,
        }

        original_handler = gateway._handlers["work_order.create"]

        def fail_internal_service(_payload: Dict[str, Any], _session_id: str) -> Dict[str, Any]:
            raise RuntimeError("symbolic_internal_service_failure")

        gateway._handlers["work_order.create"] = fail_internal_service
        try:
            envelopes.append(
                AgentResponseEnvelope(
                    answer="answer_failure_confirmation_01",
                    confirmation_request=_confirmation(failure_proposal_id),
                )
            )
            failure_events = asyncio.run(
                base._consume("bubble_failure_confirmation_01", failure_session)
            )
        finally:
            gateway._handlers["work_order.create"] = original_handler
        base._assert_no_error(failure_events)
        failure_done = _done(failure_events)
        assert failure_done["status"] == "completed"
        assert failure_done["action_receipts"][0]["status"] == "failed"
        assert base._session_counts(failure_session) == {
            "proposals": 1,
            "approvals": 1,
            "receipts": 1,
            "work_orders": 0,
        }

        refusal_session = "session_public_b_refusal_01"
        before_builds = len(build_calls)
        envelopes.append(AgentResponseEnvelope(answer="b_agent_refusal_01"))
        refusal_events = asyncio.run(base._consume("bubble_b_refusal_01", refusal_session))
        base._assert_no_error(refusal_events)
        refusal_done = _done(refusal_events)
        assert refusal_done["content"] == "b_agent_refusal_01"
        assert refusal_done["current_agent_id"] == "maintenance"
        assert build_calls[before_builds:] == ["maintenance"]
        assert base._session_counts(refusal_session) == {
            "proposals": 0,
            "approvals": 0,
            "receipts": 0,
            "work_orders": 0,
        }

    assert envelopes == []
    assert len(router_calls) == 6
    assert len(vertical_calls) == 6
    assert build_calls == ["maintenance"] * 6
    assert len(execute_calls) == 2
    assert forbidden_hits == []

    trace_ids = [
        _done(events)["trace_id"]
        for events in (
            proposal_events,
            confirmation_events,
            replay_events,
            failure_proposal_events,
            failure_events,
            refusal_events,
        )
    ]
    for trace_id in trace_ids:
        stages = [item["span_name"] for item in base.list_trace_events(trace_id)]
        assert "agent_selector" not in stages
        ledger = base.get_evidence_ledger(trace_id)
        assert ledger is not None
        model_stages = [
            item.get("stage") for item in ledger["ledger"].get("model_calls") or []
        ]
        assert model_stages == ["router", "vertical_agent"]
        capability = ledger["ledger"].get("capability_decision") or {}
        assert capability.get("selected_agent_id") == "maintenance"

    selected_source = inspect.getsource(base.RuntimeCoordinator._stream_selected_agent)
    assert "if state.capability_decision" not in selected_source
    assert selected_source.index("state.selected_agent") < selected_source.index(
        "state.capability_decision ="
    )


TESTS = (
    test_active_a_lane_and_retired_side_channels,
    test_public_stream_full_state_machine_and_frozen_b_refusal,
)


def main() -> None:
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS target1_closeout_tests={len(TESTS)}")


if __name__ == "__main__":
    main()
