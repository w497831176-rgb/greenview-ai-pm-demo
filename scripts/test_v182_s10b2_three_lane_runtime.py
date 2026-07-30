"""Deterministic S10-B.2-Fix1 structured three-lane contract.

All routing-model behavior is supplied by FakeModel/FakeAgent.  The script
opens no network connection, calls no Provider and writes no business data.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

import agents.router as router_module
import app.runtime.coordinator as coordinator_module
from app.handoff_policy import evaluate_handoff_policy
from app.runtime.contracts import (
    CapabilityDecision,
    LaneDecision,
    RuntimeLane,
)
from app.runtime.coordinator import (
    _decide_lane,
    _is_internal_control_payload,
    _lane_candidates,
    _lane_fallback_agent,
    _render_visible_history_context,
    _visible_chat_history,
)


ROOT = Path(__file__).resolve().parents[1]


PROPERTY_CARDS = [
    {
        "agent_id": "customer_service",
        "name": "客服 Agent",
        "description": "物业服务咨询、失物寻物和信息澄清",
        "domain_scope": "property",
        "enabled": True,
        "capability_card": {},
    },
    {
        "agent_id": "maintenance",
        "name": "维修 Agent",
        "description": "报修、维修和工单",
        "domain_scope": "property",
        "enabled": True,
        "capability_card": {},
    },
]
GENERAL_CARDS = [
    {
        "agent_id": "general-chat",
        "name": "通用闲聊 Agent",
        "description": "处理与物业场景无关的低风险通用、闲聊问题",
        "domain_scope": "isolated_general",
        "enabled": True,
        "capability_card": {"service_scope": "通用闲聊"},
    }
]
ALL_CARDS = PROPERTY_CARDS + GENERAL_CARDS


class FakeRunOutput:
    def __init__(self, content: str):
        self.content = content
        self.metrics: Dict[str, Any] = {}


class FakeModel:
    """Marker proving that tests never construct a real Provider model."""


class FakeAgent:
    def __init__(self, output: str):
        self.output = output
        self.model = FakeModel()
        self.last_prompt = ""

    async def arun(self, prompt: str, **_: Any) -> FakeRunOutput:
        self.last_prompt = prompt
        return FakeRunOutput(self.output)


def main() -> None:
    passed: List[str] = []

    def check(name: str, condition: bool) -> None:
        assert condition, name
        passed.append(name)

    lane_cases = {
        RuntimeLane.SAFETY_HANDOFF: [
            "不用转人工，但电梯有人被困",
            "楼道有燃气味",
            "水漫到插座",
            "找不到孩子了",
            "有人持刀",
        ],
        RuntimeLane.PROPERTY_GOVERNED: [
            "我的神奇火箭侠不见了",
            "找不到大大泡泡糖",
            "手机可能落在小区里",
            "电梯里捡到钥匙",
            "快递放门口后不见了",
            "门禁卡丢了",
            "客厅灯坏了",
            "物业费多少钱",
            "查询工单状态",
            "小区有没有免费搬家服务",
        ],
        RuntimeLane.ISOLATED_GENERAL: [
            "给我写一个火箭侠故事",
            "解释火箭为什么能飞",
            "今天奇门遁甲起运怎么弄",
            "解释量子纠缠",
            "写一首诗",
            "游戏账号丢了怎么办",
            "电脑文件找不到了",
        ],
    }
    for expected_lane, messages in lane_cases.items():
        for message in messages:
            decision = _decide_lane(message)
            check(
                f"lane {expected_lane.value}: {message}",
                decision.lane == expected_lane
                and bool(decision.reason_code)
                and bool(decision.business_intent),
            )

    for message in ("我的东西不见了", "找不到了怎么办", "那个东西掉在外面了"):
        decision = _decide_lane(message)
        check(
            f"ambiguous physical loss stays in B: {message}",
            decision.lane == RuntimeLane.PROPERTY_GOVERNED
            and decision.business_intent == "lost_and_found"
            and decision.needs_clarification,
        )

    t4 = _decide_lane("不用转人工，但电梯里有人被困。")
    check(
        "safety overrides handoff negation",
        t4.lane == RuntimeLane.SAFETY_HANDOFF
        and "safety_override" in t4.matched_signals,
    )
    for message in ("转人工", "找客服", "不想和AI沟通", "让工作人员处理"):
        policy = evaluate_handoff_policy(message)
        check(
            f"explicit owner handoff preempts lanes: {message}",
            policy["should_request_handoff"]
            and policy["reason_code"] == "owner_requested",
        )
    for message in ("暂时不用转人工", "不需要人工", "先别转人工"):
        policy = evaluate_handoff_policy(message)
        check(
            f"negated handoff remains direct AI: {message}",
            not policy["should_request_handoff"]
            and policy["reason_code"] == "negated_by_user",
        )

    b_candidates = _lane_candidates(ALL_CARDS, RuntimeLane.PROPERTY_GOVERNED)
    c_candidates = _lane_candidates(ALL_CARDS, RuntimeLane.ISOLATED_GENERAL)
    a_candidates = _lane_candidates(ALL_CARDS, RuntimeLane.SAFETY_HANDOFF)
    check("B candidates contain property only", all(card["domain_scope"] == "property" for card in b_candidates))
    check("C candidates contain isolated_general only", all(card["domain_scope"] == "isolated_general" for card in c_candidates))
    check("A has no ordinary Agent candidates", a_candidates == [])
    check("B fallback is customer_service", (_lane_fallback_agent(b_candidates, RuntimeLane.PROPERTY_GOVERNED) or {}).get("agent_id") == "customer_service")
    check("C fallback is configured general Agent", (_lane_fallback_agent(c_candidates, RuntimeLane.ISOLATED_GENERAL) or {}).get("agent_id") == "general-chat")
    check("C without a general Agent has no property fallback", _lane_fallback_agent(PROPERTY_CARDS, RuntimeLane.ISOLATED_GENERAL) is None)

    try:
        LaneDecision.model_validate(
            {
                "lane": "D_FREE_ROUTING",
                "reason_code": "invalid",
                "business_intent": "invalid",
                "confidence": 2,
                "decision_source": "router_model",
                "matched_signals": [],
                "allowed_domain_scopes": ["property"],
            }
        )
    except ValidationError:
        passed.append("invalid LaneDecision schema is rejected")
    else:
        raise AssertionError("invalid LaneDecision schema is rejected")

    capability = CapabilityDecision.model_validate(
        {
            "selected_agent_id": "customer_service",
            "skill": {"status": "skipped", "reason_code": "no_match"},
            "rag": {"status": "skipped", "reason_code": "no_bound_knowledge"},
            "tool": {"status": "skipped", "reason_code": "not_required"},
            "write": {"status": "not_required", "reason_code": "consultation_path"},
            "handoff": {"status": "available", "reason_code": "owner_can_request"},
        }
    )
    check("CapabilityDecision is structured", capability.selected_agent_id == "customer_service")

    fake_agent = FakeAgent(
        '{"target_agent_id":"general-chat","reason":"attempted cross-lane route"}'
    )
    original_create_router_agent = router_module.create_router_agent
    router_module.create_router_agent = lambda *args, **kwargs: fake_agent
    try:
        result = asyncio.run(
            router_module.classify_intent(
                "我的东西不见了",
                vertical_agents=PROPERTY_CARDS,
                session_id="business::router::trace",
                model=FakeModel(),
                visible_history=[
                    {"role": "user", "content": "上一轮用户可见问题"},
                    {"role": "assistant", "content": "上一轮用户可见回答"},
                ],
            )
        )
    finally:
        router_module.create_router_agent = original_create_router_agent
    check(
        "cross-lane target is blocked by candidate contract",
        result["model_target_agent_id"] == "general-chat"
        and result["target_agent_id"] in {"customer_service", "maintenance"},
    )
    check("Router receives visible history", "上一轮用户可见问题" in fake_agent.last_prompt)
    check("Router current message appears once", fake_agent.last_prompt.count("我的东西不见了") == 1)

    fake_rows = [
        {"role": "user", "content": "成功问题", "status": "success", "trace_id": "old-1"},
        {"role": "assistant", "content": "成功回答", "status": "success", "trace_id": "old-1"},
        {"role": "assistant", "content": '{"target_agent_id":"maintenance"}', "status": "success", "trace_id": "old-control"},
        {"role": "assistant", "content": "失败回答", "status": "failed", "trace_id": "old-2"},
        {"role": "user", "content": "当前问题", "status": "success", "trace_id": "current"},
    ]
    original_list_chat_messages = coordinator_module.list_chat_messages
    coordinator_module.list_chat_messages = lambda _session_id: fake_rows
    try:
        history = _visible_chat_history("business-session", current_trace_id="current", rounds=5)
    finally:
        coordinator_module.list_chat_messages = original_list_chat_messages
    check("chat_messages is the visible-history source", history == [{"role": "user", "content": "成功问题"}, {"role": "assistant", "content": "成功回答"}])
    context = _render_visible_history_context(history, "当前问题", boundary="只读咨询")
    check("multi-turn visible history remains effective", "成功问题" in context and "成功回答" in context)
    check("current user message is injected once", context.count("当前问题") == 1)
    check("Router JSON is absent from vertical input", "target_agent_id" not in context)

    check("whole Router JSON is detected", _is_internal_control_payload('{"target_agent_id":"maintenance","reason":"x"}'))
    check("whole Lane JSON is detected", _is_internal_control_payload('{"lane":"B_PROPERTY_GOVERNED","business_intent":"x"}'))
    check("fenced control JSON is detected", _is_internal_control_payload('```json\n{"lane":"C_ISOLATED_GENERAL"}\n```'))
    check("natural prose is not a control payload", not _is_internal_control_payload("建议先回忆物品最后出现的位置。"))

    coordinator_source = (ROOT / "app/runtime/coordinator.py").read_text(encoding="utf-8")
    router_source = (ROOT / "agents/router.py").read_text(encoding="utf-8")
    factory_source = (ROOT / "app/runtime/agent_factory.py").read_text(encoding="utf-8")
    exact_sentences = [
        "我的神奇火箭侠不见了怎么办",
        "今天奇门遁甲起运怎么弄",
        "小区有没有免费搬家服务",
    ]
    check("no exact acceptance-sentence patch", all(sentence not in coordinator_source for sentence in exact_sentences))
    check("Router automatic Agno history is disabled", "add_history_to_context=False" in router_source and "read_chat_history=False" in router_source)
    check("vertical automatic Agno history is disabled", "add_history_to_context=False" in factory_source and "num_history_runs=0" in factory_source)
    check("Router has an isolated stage session", '::router::' in coordinator_source)
    check("vertical Agent has an isolated stage session", '::vertical::' in coordinator_source)
    check("control payload leak is a named violation", "internal_control_payload_leak" in coordinator_source)
    check("Lane decision precedes Agent candidate filtering", coordinator_source.index("state.lane_decision = _decide_lane") < coordinator_source.index("cards = _lane_candidates"))

    counterexample_count = sum(len(items) for items in lane_cases.values()) + 3
    print(
        {
            "status": "PASS",
            "checks": len(passed),
            "counterexamples": counterexample_count,
            "provider_calls": 0,
        }
    )


if __name__ == "__main__":
    main()
