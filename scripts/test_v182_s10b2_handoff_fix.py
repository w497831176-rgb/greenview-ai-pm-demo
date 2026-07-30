"""Deterministic contract for V1.8.2-S10-B.2-Handoff-Fix.

No Provider, network, browser or runtime database is used.
"""

from __future__ import annotations

import asyncio
import ast
import inspect
from pathlib import Path
from typing import Any

import agents.router as router_module
import app.runtime.coordinator as coordinator_module
import app.runtime.legacy_chat as chat_module
from app.runtime.contracts import LaneDecision, RuntimeLane
from app.runtime.coordinator import RuntimeCoordinator
from app.runtime.legacy_chat import HandoffRequest


ROOT = Path(__file__).resolve().parents[1]


def check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"PASS {label}")


async def coordinator_checks() -> None:
    original_get = coordinator_module.get_chat_session
    original_request = coordinator_module.request_handoff
    original_resume = coordinator_module.resume_handoff_after_owner_message
    calls: list[dict[str, Any]] = []

    def fake_request(session_id: str, reason: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"session_id": session_id, "reason": reason, **kwargs})
        return {"session_id": session_id, "handoff_status": "requested"}

    try:
        coordinator_module.get_chat_session = lambda _session: {"handoff_status": "none"}
        coordinator_module.request_handoff = fake_request
        coordinator_module.resume_handoff_after_owner_message = lambda _session: {
            "handoff_status": "active"
        }
        runtime = RuntimeCoordinator()
        for lane in (RuntimeLane.PROPERTY_GOVERNED, RuntimeLane.ISOLATED_GENERAL):
            result = await runtime._maybe_handoff(
                "明确要求工作人员接手",
                f"session-{lane.value}",
                "trace-test",
                "rr-test",
                LaneDecision(
                    lane=lane,
                    business_intent="user_requested_handoff",
                    reason="用户明确要求结束AI对话并由工作人员接手",
                ),
            )
            check(result is not None, f"{lane.value}人工意图全局短路")
            check(result[0] == "已发起人工协同：等待工作人员领取。", "新请求返回真实等待状态")
            check(result[2]["reason_code"] == "user_requested", "Handoff原因统一为user_requested")

        call_count = len(calls)
        for intent in ("decline_handoff", "ask_handoff_rules", "discuss_future_handoff"):
            result = await runtime._maybe_handoff(
                "普通非请求语义",
                "session-no-handoff",
                "trace-test",
                "rr-test",
                LaneDecision(lane=RuntimeLane.ISOLATED_GENERAL, business_intent=intent),
            )
            check(result is None, f"{intent}不触发人工协同")
        check(len(calls) == call_count, "否定、规则询问和未来讨论不写Handoff")

        coordinator_module.get_chat_session = lambda _session: {"handoff_status": "requested"}
        repeated = await runtime._maybe_handoff(
            "再次明确要求人工",
            "session-repeat",
            "trace-test",
            "rr-test",
            LaneDecision(
                lane=RuntimeLane.ISOLATED_GENERAL,
                business_intent="user_requested_handoff",
            ),
        )
        check(repeated[0] == "人工协同已在等待领取。", "重复请求返回已在等待")
        check(repeated[1] == "requested", "重复请求保持requested")

        coordinator_module.get_chat_session = lambda _session: {"handoff_status": "active"}

        def active_request(session_id: str, reason: str, **kwargs: Any) -> dict[str, Any]:
            calls.append({"session_id": session_id, "reason": reason, **kwargs})
            return {"session_id": session_id, "handoff_status": "active"}

        coordinator_module.request_handoff = active_request
        active = await runtime._maybe_handoff(
            "再次要求人工",
            "session-active",
            "trace-test",
            "rr-test",
            LaneDecision(
                lane=RuntimeLane.PROPERTY_GOVERNED,
                business_intent="user_requested_handoff",
            ),
        )
        check(active[0] == "工作人员已领取，当前正在人工协同处理中。", "已领取状态如实显示")

        def failed_request(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("persist failed")

        coordinator_module.get_chat_session = lambda _session: {"handoff_status": "none"}
        coordinator_module.request_handoff = failed_request
        try:
            await runtime._maybe_handoff(
                "明确要求人工",
                "session-failed",
                "trace-test",
                "rr-test",
                LaneDecision(
                    lane=RuntimeLane.ISOLATED_GENERAL,
                    business_intent="user_requested_handoff",
                ),
            )
        except RuntimeError as exc:
            check(str(exc) == "persist failed", "Handoff失败保持可见")
        else:
            raise AssertionError("Handoff失败不得返回成功")
    finally:
        coordinator_module.get_chat_session = original_get
        coordinator_module.request_handoff = original_request
        coordinator_module.resume_handoff_after_owner_message = original_resume


async def button_checks() -> None:
    original = chat_module._request_handoff_with_context
    original_get = chat_module.get_chat_session
    captured: dict[str, Any] = {}

    def fake_request(session_id: str, policy: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured.update({"session_id": session_id, "policy": policy, **kwargs})
        return {"session_id": session_id, "handoff_status": "requested"}

    try:
        chat_module.get_chat_session = lambda _session: {"handoff_status": "none"}
        chat_module._request_handoff_with_context = fake_request
        result = await chat_module.chat_handoff(
            HandoffRequest(session_id="button-session", reason="点击转人工")
        )
        check(result["handoff_state"] == "requested", "按钮直接创建等待领取状态")
        check(result["policy"]["reason_code"] == "user_requested", "按钮原因统一为user_requested")
        check(result["message"] == "已发起人工协同：等待工作人员领取。", "按钮返回后端真实状态文案")
        check(captured["actor"] == "owner", "按钮由业主直接发起")
        source = inspect.getsource(chat_module.chat_handoff)
        check("classify_lane_decision" not in source and "MODEL" not in source, "按钮不调用Router或模型")
    finally:
        chat_module._request_handoff_with_context = original
        chat_module.get_chat_session = original_get


def source_checks() -> None:
    router_source = inspect.getsource(router_module.create_semantic_lane_router)
    handoff_source = inspect.getsource(RuntimeCoordinator._maybe_handoff)
    stream_source = inspect.getsource(RuntimeCoordinator.stream)
    check("user_requested_handoff" in router_source, "Router输出统一人工协同业务意图")
    check("owner_handoff_request" not in handoff_source, "运行时不再依赖旧人工意图")
    check("decision.lane ==" not in handoff_source, "人工协同意图不受A/B/C限制")
    check("handoff_preempted" in stream_source, "人工协同后Agent、Skill、RAG和Tool均跳过")
    check(stream_source.index("_maybe_handoff") < stream_source.index("_stream_consultation"), "Handoff在Agent/Evidence前短路")
    for forbidden in ("那转一下人工", "不能转人工吗", "暂时不用转人工"):
        check(forbidden not in router_source, "生产Router没有验收句特例")
    check("re.search" not in handoff_source and "keyword" not in handoff_source.lower(), "短路不使用正则或关键词分支")

    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    check("payload.handoff_state" in html, "按钮展示后端真实Handoff状态")
    check("if (!res.ok) throw" in html, "按钮失败不会显示成功")
    for relative in (
        "agents/router.py",
        "app/runtime/coordinator.py",
        "app/runtime/legacy_chat.py",
    ):
        ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
    check(True, "受影响Python语法通过")


def main() -> None:
    source_checks()
    asyncio.run(coordinator_checks())
    asyncio.run(button_checks())
    print("S10-B.2-Handoff-Fix deterministic contract: PASS")


if __name__ == "__main__":
    main()
