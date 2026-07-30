"""Deterministic contract for V1.8.2-S10-B.2-C-Fallback-Fix.

No Provider, network, browser or runtime database is used.
"""

from __future__ import annotations

import asyncio
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import app.runtime.coordinator as coordinator_module
from app.runtime.contracts import (
    CapabilityDecision,
    LaneDecision,
    RunState,
    RunStatus,
    RuntimeLane,
    RuntimePath,
)
from app.runtime.coordinator import (
    OUT_OF_SCOPE_RESPONSE,
    RuntimeCoordinator,
    _answer_contract_for,
    build_lane_agent_unavailable_decision,
)


ROOT = Path(__file__).resolve().parents[1]


def check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"PASS {label}")


class FakeLedger:
    def __init__(self) -> None:
        self.captured: dict[str, Any] | None = None
        self.persisted: list[str] = []
        self.items: dict[str, list[dict[str, Any]]] = {}

    def capture_state(self, state: RunState) -> None:
        self.captured = {
            "status": state.status.value,
            "lane_decision": state.lane_decision.model_dump(mode="json"),
            "capability_decision": state.capability_decision.model_dump(mode="json"),
        }

    def append(self, field: str, value: dict[str, Any]) -> None:
        self.items.setdefault(field, []).append(value)

    def persist(self, status: str) -> dict[str, Any]:
        self.persisted.append(status)
        return {"status": status}


def decision_contract_checks() -> None:
    no_agent = build_lane_agent_unavailable_decision(property_lane=False)
    check(no_agent.selected_agent_id is None, "C类无Agent不跨域选择物业Agent")
    check(no_agent.skill.status == "skipped", "C类无Agent跳过Skill")
    check(no_agent.rag.status == "skipped", "C类无Agent跳过RAG")
    check(no_agent.tool.status == "skipped", "C类无Agent跳过Tool")
    check(no_agent.write.status == "not_required", "C类无Agent写入状态使用合法唯一枚举")
    check(no_agent.handoff.status == "not_required", "C类无Agent人工协同状态使用合法唯一枚举")

    property_no_agent = build_lane_agent_unavailable_decision(property_lane=True)
    check(property_no_agent.write.status == "not_required", "B类无Agent不要求业务写入")
    check(property_no_agent.handoff.status == "available", "B类无Agent仍可由业主选择人工")

    found_general_agent = CapabilityDecision(
        selected_agent_id="isolated-general-agent",
        skill={"status": "skipped", "reason_code": "no_match"},
        rag={"status": "skipped", "reason_code": "isolated_general"},
        tool={"status": "skipped", "reason_code": "isolated_general"},
        write={"status": "not_required", "reason_code": "consultation_path"},
        handoff={"status": "available", "reason_code": "owner_can_request"},
    )
    check(found_general_agent.selected_agent_id == "isolated-general-agent", "C类找到通用Agent时合同有效")

    safe_refusal = CapabilityDecision(
        selected_agent_id=None,
        skill={"status": "skipped", "reason_code": "unsafe_request_refused"},
        rag={"status": "skipped", "reason_code": "unsafe_request_refused"},
        tool={"status": "skipped", "reason_code": "unsafe_request_refused"},
        write={"status": "not_required", "reason_code": "unsafe_request_refused"},
        handoff={"status": "not_required", "reason_code": "unsafe_request_refused"},
    )
    check(safe_refusal.write.status == "not_required", "C类安全拒绝不调用物业写入")

    try:
        CapabilityDecision(
            selected_agent_id=None,
            skill={"status": "skipped", "reason_code": "test"},
            rag={"status": "skipped", "reason_code": "test"},
            tool={"status": "skipped", "reason_code": "test"},
            write={"status": "skipped", "reason_code": "test"},
            handoff={"status": "skipped", "reason_code": "test"},
        )
    except Exception:
        check(True, "合同继续拒绝Write/Handoff的非法skipped状态")
    else:
        raise AssertionError("非法状态不得被合同接受")


async def no_agent_runtime_checks() -> None:
    original_save = coordinator_module.save_chat_message
    original_update = coordinator_module.update_chat_trace
    original_event = coordinator_module.record_trace_event
    updates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    def fake_save(**kwargs: Any) -> dict[str, Any]:
        return {"id": 901, **kwargs}

    def fake_update(trace_id: str, **kwargs: Any) -> dict[str, Any]:
        updates.append({"trace_id": trace_id, **kwargs})
        return updates[-1]

    def fake_event(trace_id: str, span_name: str, status: str, **kwargs: Any) -> dict[str, Any]:
        events.append({"trace_id": trace_id, "span_name": span_name, "status": status, **kwargs})
        return events[-1]

    try:
        coordinator_module.save_chat_message = fake_save
        coordinator_module.update_chat_trace = fake_update
        coordinator_module.record_trace_event = fake_event
        lane = LaneDecision(
            lane=RuntimeLane.ISOLATED_GENERAL,
            business_intent="ask_handoff_rules",
            reason="用户询问规则而非请求人工接手",
        )
        state = RunState(
            run_id="run-test",
            trace_id="trace-test",
            session_id="session-test",
            snapshot_id="snapshot-test",
            path=RuntimePath.CONSULTATION,
            lane_decision=lane,
            answer_contract=_answer_contract_for(lane),
        )
        ledger = FakeLedger()
        chunks = [
            item
            async for item in RuntimeCoordinator()._stream_unconfigured_lane_boundary(
                "session-test",
                "trace-test",
                SimpleNamespace(release_id="rr-test", snapshot_id="snapshot-test"),
                state,
                ledger,
                0.0,
            )
        ]
        events_by_name = {
            line.split("\n", 1)[0].removeprefix("event: "): json.loads(
                line.split("data: ", 1)[1]
            )
            for line in chunks
        }
        check(state.status == RunStatus.COMPLETED, "C类无Agent回退终态complete")
        check(ledger.persisted == ["complete"], "Evidence以complete持久化")
        check(ledger.captured is not None, "Evidence捕获C类能力决策")
        check(ledger.captured["capability_decision"]["write"]["status"] == "not_required", "Trace与Evidence使用同一Write状态")
        check(ledger.captured["capability_decision"]["handoff"]["status"] == "not_required", "Trace与Evidence使用同一Handoff状态")
        check(events_by_name["done"]["status"] == "complete", "SSE正常结束到done")
        check(events_by_name["final"]["content"] == OUT_OF_SCOPE_RESPONSE, "C类返回自然能力边界说明")
        check(updates[-1]["status"] == "complete", "Trace顶层状态complete")
        check(events[-1]["span_name"] == "lane_agent_boundary" and events[-1]["status"] == "success", "Trace记录C类边界成功")
        check(len(state.model_calls) == 0, "确定性回退不新增Provider请求")
        check(all(token not in events_by_name["final"]["content"] for token in ("CapabilityDecision", "validation error", "input_value")), "内部状态错误不展示给用户")
    finally:
        coordinator_module.save_chat_message = original_save
        coordinator_module.update_chat_trace = original_update
        coordinator_module.record_trace_event = original_event


def source_boundary_checks() -> None:
    source = (ROOT / "app/runtime/coordinator.py").read_text(encoding="utf-8")
    router = (ROOT / "agents/router.py").read_text(encoding="utf-8")
    check("build_lane_agent_unavailable_decision" in source, "C类回退由单一构造器生产状态")
    for forbidden in ("暂时不用转人工", "转人工需要什么条件", "那转一下人工"):
        check(forbidden not in source and forbidden not in router, "生产代码没有验收句特例")
    check("re.search" not in source[source.index("def build_lane_agent_unavailable_decision"):source.index("class ProviderFailureError")], "修复没有新增正则或关键词判断")
    for relative in ("app/runtime/coordinator.py", "app/runtime/contracts.py"):
        ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
    check(True, "受影响Python语法通过")


def main() -> None:
    source_boundary_checks()
    decision_contract_checks()
    asyncio.run(no_agent_runtime_checks())
    print("S10-B.2-C-Fallback-Fix deterministic contract: PASS")


if __name__ == "__main__":
    main()
