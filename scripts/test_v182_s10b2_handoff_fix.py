"""Deterministic contract for V1.8.2-S10-B.2-Handoff-Fix.

No Provider, network, browser or production database is used.
"""

from __future__ import annotations

import asyncio
import ast
import inspect
import os
import tempfile
from pathlib import Path
from typing import Any

TEMP_DIR = tempfile.TemporaryDirectory(
    prefix="yiai-s10b2-handoff-",
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

from db.property_db import init_db

init_db()

import agents.router as router_module
import app.runtime.coordinator as coordinator_module
import app.runtime.legacy_chat as chat_module
from app.runtime.contracts import (
    HandoffKind,
    LaneDecision,
    ResponseMode,
    RuntimeLane,
)
from app.runtime.coordinator import (
    RuntimeCoordinator,
    _effective_lane_decision,
    _handoff_contract_for,
)
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
        for reported_lane in (
            RuntimeLane.PROPERTY_GOVERNED,
            RuntimeLane.ISOLATED_GENERAL,
        ):
            effective, normalized_from = _effective_lane_decision(
                LaneDecision(
                    lane=reported_lane,
                    business_intent="user_requested_handoff",
                    reason="业主明确要求结束AI对话并由工作人员接手",
                )
            )
            check(effective.lane == RuntimeLane.SAFETY_HANDOFF, f"{reported_lane.value}人工意图在正式落账前归一为A")
            check(normalized_from == reported_lane.value, "保留Router原始Lane供审计")
            contract = _handoff_contract_for(effective)
            check(contract.kind == HandoffKind.USER_REQUESTED, "普通转人工使用user_requested子类型")
            check(contract.queue == "property_service", "普通转人工进入物业服务队列")
            check(contract.safety_override is False, "普通转人工不冒充安全风险")
            check(contract.response_mode == ResponseMode.HUMAN_HANDOFF, "普通转人工使用普通协同文案合同")

        safety_contract = _handoff_contract_for(
            LaneDecision(
                lane=RuntimeLane.SAFETY_HANDOFF,
                business_intent="safety_risk",
            )
        )
        check(safety_contract.kind == HandoffKind.SAFETY_RISK, "安全A保持safety_risk子类型")
        check(safety_contract.queue == "emergency", "安全A进入紧急队列")
        check(safety_contract.safety_override is True, "安全A保留安全优先")
        check(safety_contract.response_mode == ResponseMode.EMERGENCY_HANDOFF, "安全A保留紧急文案合同")

        existing_safety, _ = _effective_lane_decision(
            LaneDecision(
                lane=RuntimeLane.SAFETY_HANDOFF,
                business_intent="user_requested_handoff",
            ),
            handoff_status="active",
            handoff_reason_code="safety_risk",
            handoff_queue="emergency",
        )
        check(existing_safety.business_intent == "safety_risk", "既有安全Handoff不会被降级为普通队列")

        continued_safety, original_lane = _effective_lane_decision(
            LaneDecision(lane=RuntimeLane.ISOLATED_GENERAL, business_intent="general"),
            handoff_status="requested",
            handoff_reason_code="safety_risk",
            handoff_queue="emergency",
        )
        check(continued_safety.lane == RuntimeLane.SAFETY_HANDOFF, "既有安全Handoff后续轮仍保持A")
        check(continued_safety.business_intent == "safety_risk", "既有安全Handoff后续轮保持安全子类型")
        check(original_lane == RuntimeLane.ISOLATED_GENERAL.value, "后续轮保留Router原始Lane供审计")

        effective_user = LaneDecision(
            lane=RuntimeLane.SAFETY_HANDOFF,
            business_intent="user_requested_handoff",
            reason="业主明确要求结束AI对话并由工作人员接手",
        )
        result = await runtime._maybe_handoff(
            "明确要求工作人员接手",
            "session-user-requested",
            "trace-test",
            "rr-test",
            effective_user,
        )
        check(result is not None, "有效A创建真实Handoff")
        check(result[0] == "已发起人工协同：等待工作人员领取。", "新请求返回真实等待状态")
        check(result[2]["reason_code"] == "user_requested", "Handoff原因统一为user_requested")
        check(result[2]["queue"] == "property_service", "Handoff写入普通物业队列")
        check(result[2]["safety_override"] is False, "Handoff写入非安全覆盖证据")

        call_count = len(calls)
        for intent in ("decline_handoff", "ask_handoff_rules", "discuss_future_handoff"):
            ordinary_c = LaneDecision(
                lane=RuntimeLane.ISOLATED_GENERAL,
                business_intent=intent,
            )
            effective, normalized_from = _effective_lane_decision(ordinary_c)
            check(effective == ordinary_c and normalized_from is None, f"{intent}保持普通C")
        check(len(calls) == call_count, "否定、规则询问和未来讨论不写Handoff")

        coordinator_module.get_chat_session = lambda _session: {"handoff_status": "requested"}
        repeated = await runtime._maybe_handoff(
            "再次明确要求人工",
            "session-repeat",
            "trace-test",
            "rr-test",
            effective_user,
        )
        check(repeated[0] == "人工协同已在等待领取。", "重复请求返回已在等待")
        check(repeated[1] == "requested", "重复请求保持requested")
        check(len(calls) == call_count, "既有requested不会重复刷新Handoff")

        coordinator_module.get_chat_session = lambda _session: {
            "handoff_status": "waiting_user",
            "handoff_reason_code": "user_requested",
            "handoff_queue": "property_service",
        }
        waiting = await runtime._maybe_handoff(
            "补充说明",
            "session-waiting",
            "trace-test",
            "rr-test",
            effective_user,
        )
        check(waiting[0] == "已将补充信息同步给接管工作人员，人工处理已恢复。", "waiting_user后续轮恢复人工处理")
        check(waiting[1] == "active", "waiting_user真实迁移到active")
        check(len(calls) == call_count, "waiting_user恢复不重复创建Handoff")

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
            effective_user,
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
                effective_user,
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
    a_stream_source = inspect.getsource(RuntimeCoordinator._stream_a_handoff)
    check("user_requested_handoff" in router_source, "Router输出统一人工协同业务意图")
    check("必须同时输出A_SAFETY_HANDOFF" in router_source, "Router要求明确转人工正式输出A")
    check("该业务意图不改变A、B、C" not in router_source, "废弃横挂B/C的旧合同")
    check("owner_handoff_request" not in handoff_source, "运行时不再依赖旧人工意图")
    check("_stream_a_handoff" in stream_source, "A在通用咨询前进入统一人工协同执行器")
    check("_maybe_handoff(" not in stream_source, "B/C不再横向创建正式Handoff")
    check("handoff_preempted" in a_stream_source, "人工协同后Agent、Skill、RAG、Tool和写入均跳过")
    for forbidden in ("那转一下人工", "不能转人工吗", "暂时不用转人工"):
        check(forbidden not in router_source, "生产Router没有验收句特例")
    check("re.search" not in handoff_source and "keyword" not in handoff_source.lower(), "短路不使用正则或关键词分支")

    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    check("A路：人工协同" in html, "Trace主层统一显示A路人工协同")
    check("普通人工协同" in html and "紧急人工协同" in html, "Trace主层区分普通与安全人工协同")
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
    try:
        main()
    finally:
        TEMP_DIR.cleanup()
