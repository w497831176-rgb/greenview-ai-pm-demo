"""No-model contracts for the final A/B/C Router.

The fakes return materialised responses. This script opens no network, calls no
Provider and writes no runtime or business data.
"""

from __future__ import annotations

import asyncio
import ast
import inspect
import json
from pathlib import Path
from typing import Any

import agents.router as router_module
import app.runtime.coordinator as coordinator_module
import scripts.run_v182_s10b2_final_router_eval as eval_module
from app.runtime.contracts import LaneDecision, ResponseMode, RuntimeLane, RuntimePath
from app.runtime.coordinator import (
    RuntimeCoordinator,
    _answer_contract_for,
    _is_internal_control_payload,
    _lane_candidates,
    _visible_chat_history,
)


ROOT = Path(__file__).resolve().parents[1]
CARDS = [
    {
        "agent_id": "customer_service",
        "name": "客服 Agent",
        "description": "物业服务咨询",
        "domain_scope": "property",
        "enabled": True,
        "capability_card": {},
    },
    {
        "agent_id": "maintenance",
        "name": "维修 Agent",
        "description": "物业维修服务",
        "domain_scope": "property",
        "enabled": True,
        "capability_card": {},
    },
    {
        "agent_id": "isolated_general",
        "name": "隔离通用 Agent",
        "description": "非物业低风险一般回答",
        "domain_scope": "isolated_general",
        "enabled": True,
        "capability_card": {},
    },
]


class FakeRun:
    def __init__(self, content: str):
        self.content = content
        self.model_provider_data = {
            "id": "provider-request-test",
            "response_model": "deepseek-v4-flash",
            "usage": {
                "input_cache_hit_tokens": 10,
                "input_cache_miss_tokens": 20,
                "output_tokens": 5,
                "total_tokens": 35,
            },
        }


class FakeAgent:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0
        self.prompt = ""

    async def arun(self, prompt: str, **_: Any) -> FakeRun:
        self.calls += 1
        self.prompt = prompt
        return FakeRun(self.content)


def check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"PASS {label}")


async def async_checks() -> None:
    original_router_factory = router_module.create_semantic_lane_router
    original_selector_factory = router_module.create_lane_agent_selector
    try:
        fenced = FakeAgent(
            "```json\n"
            '{"lane":"C_ISOLATED_GENERAL","request_kind":"fact",'
            '"target_agent_id":null,"confidence":0.91,"allowed_domain":"isolated_general",'
            '"business_intent":"普通知识问答","reason":"不需要物业能力"}'
            "\n```"
        )
        router_module.create_semantic_lane_router = lambda **_: fenced
        result = await router_module.classify_lane_decision(
            "当前问题",
            vertical_agents=CARDS,
            visible_history=[
                {"role": "user", "content": "上一轮用户可见问题"},
                {"role": "assistant", "content": "上一轮用户可见回答"},
            ],
            model=object(),
        )
        check(result["decision"].lane == RuntimeLane.ISOLATED_GENERAL, "复合旧字段不推翻正确C分类")
        check(fenced.calls == 1, "Router每轮只调用一次")
        payload = json.loads(fenced.prompt)
        check(payload["current_user_message"] == "当前问题", "Router收到当前用户消息")
        check(len(payload["visible_conversation"]) == 2, "Router收到用户可见对话前文")
        check("agent_candidates" not in fenced.prompt, "Router不接收Agent候选控制字段")
        check("request_kind" not in fenced.prompt, "Router请求Schema不含request_kind")
        check("target_agent_id" not in fenced.prompt, "Router请求Schema不含target_agent_id")

        selection = await router_module.select_lane_agent(
            "普通问题",
            lane=RuntimeLane.ISOLATED_GENERAL,
            vertical_agents=CARDS,
            model=object(),
        )
        check(selection["selected_agent_id"] == "isolated_general", "C仅选择隔离通用Agent")
        check(selection["selection_source"] == "single_candidate", "同域唯一Agent无需Provider")

        no_agent = await router_module.select_lane_agent(
            "普通问题",
            lane=RuntimeLane.ISOLATED_GENERAL,
            vertical_agents=CARDS[:2],
            model=object(),
        )
        check(no_agent["selected_agent_id"] is None, "C无Agent时允许无选择")
        check(no_agent["provider_status"] == "not_applicable", "C无Agent时不调用Provider")

        safety = await router_module.select_lane_agent(
            "现实危险",
            lane=RuntimeLane.SAFETY_HANDOFF,
            vertical_agents=CARDS,
            model=object(),
        )
        check(safety["selected_agent_id"] is None, "A不选择普通Agent")

        selector = FakeAgent('{"target_agent_id":"maintenance","reason":"维修职责匹配"}')
        router_module.create_lane_agent_selector = lambda **_: selector
        property_selection = await router_module.select_lane_agent(
            "需要物业协助",
            lane=RuntimeLane.PROPERTY_GOVERNED,
            vertical_agents=CARDS,
            visible_history=[{"role": "user", "content": "前文"}],
            model=object(),
        )
        check(property_selection["selected_agent_id"] == "maintenance", "B只在物业Agent内选择")
        check(selector.calls == 1, "多Agent选择器至多调用一次")

        invalid_selector = FakeAgent('{"target_agent_id":"isolated_general","reason":"越域"}')
        router_module.create_lane_agent_selector = lambda **_: invalid_selector
        invalid = await router_module.select_lane_agent(
            "需要物业协助",
            lane=RuntimeLane.PROPERTY_GOVERNED,
            vertical_agents=CARDS,
            model=object(),
        )
        check(invalid["selected_agent_id"] is None, "越域Agent选择不被采用")
        check(invalid["selection_source"] == "selector_failed", "Agent失败单独记录为下游状态")
    finally:
        router_module.create_semantic_lane_router = original_router_factory
        router_module.create_lane_agent_selector = original_selector_factory


def synchronous_checks() -> None:
    check(router_module._strict_json_object({"lane": "C_ISOLATED_GENERAL"})["lane"] == "C_ISOLATED_GENERAL", "原生结构化对象仍可解析")
    check(router_module._strict_json_object("\ufeff {\"lane\":\"C_ISOLATED_GENERAL\"}")["lane"] == "C_ISOLATED_GENERAL", "BOM加裸JSON仍可解析")
    check(
        {item.value for item in RuntimeLane}
        == {"A_SAFETY_HANDOFF", "B_PROPERTY_GOVERNED", "C_ISOLATED_GENERAL"},
        "Lane枚举只有A/B/C",
    )
    for lane in RuntimeLane:
        decision = LaneDecision(lane=lane)
        check(decision.lane == lane, f"{lane.value}只凭Lane即可有效")

    legacy = LaneDecision.model_validate(
        {
            "lane": "C_ISOLATED_GENERAL",
            "request_kind": "fact",
            "target_agent_id": None,
            "confidence": 0.5,
            "allowed_domain": "isolated_general",
        }
    )
    check(legacy.lane == RuntimeLane.ISOLATED_GENERAL, "N241/N255式额外字段被忽略")

    a = _answer_contract_for(LaneDecision(lane=RuntimeLane.SAFETY_HANDOFF))
    check(a.response_mode == ResponseMode.EMERGENCY_HANDOFF, "A固定安全协同")
    check(a.handoff_policy == "required", "A强制Handoff")
    check(all(value == "skipped" for value in (a.skill_policy, a.rag_policy, a.tool_policy)), "A跳过普通能力")

    b = _answer_contract_for(LaneDecision(lane=RuntimeLane.PROPERTY_GOVERNED))
    check(b.response_mode == ResponseMode.GROUNDED_ANSWER, "B固定物业依据回答")
    check(b.evidence_required, "B要求合法Evidence")
    controlled = _answer_contract_for(
        LaneDecision(lane=RuntimeLane.PROPERTY_GOVERNED), RuntimePath.CONTROLLED_ACTION
    )
    check(controlled.response_mode == ResponseMode.CONTROLLED_WRITE, "明确业务流程继续受控写入")
    check(controlled.write_policy == "allowed_after_confirmation", "受控写入仍要求确认")

    c = _answer_contract_for(LaneDecision(lane=RuntimeLane.ISOLATED_GENERAL))
    check(c.response_mode == ResponseMode.SAFE_GENERAL, "C固定隔离通用回答")
    check(not c.evidence_required, "C不要求物业Evidence")
    check(all(value == "skipped" for value in (c.skill_policy, c.rag_policy, c.tool_policy)), "C跳过物业能力")
    check(c.write_policy == "forbidden" and c.handoff_policy == "skipped", "C禁止写入和物业协同")

    check(_lane_candidates(CARDS, RuntimeLane.SAFETY_HANDOFF) == [], "A候选Agent为0")
    check({x["agent_id"] for x in _lane_candidates(CARDS, RuntimeLane.PROPERTY_GOVERNED)} == {"customer_service", "maintenance"}, "B候选仅property")
    check([x["agent_id"] for x in _lane_candidates(CARDS, RuntimeLane.ISOLATED_GENERAL)] == ["isolated_general"], "C候选仅isolated_general")

    control_json = '{"lane":"B_PROPERTY_GOVERNED","business_intent":"内部"}'
    check(_is_internal_control_payload(control_json), "Router内部JSON不会作为可见回答历史")

    original_messages = coordinator_module.list_chat_messages
    try:
        coordinator_module.list_chat_messages = lambda _session: [
            {"role": "user", "content": "可见问题", "status": "success", "trace_id": "old-1"},
            {"role": "assistant", "content": "可见回答", "status": "complete", "trace_id": "old-2"},
            {"role": "assistant", "content": control_json, "status": "success", "trace_id": "old-3"},
            {"role": "assistant", "content": "失败回答", "status": "failed", "trace_id": "old-4"},
            {"role": "user", "content": "当前消息", "status": "success", "trace_id": "current"},
        ]
        history = _visible_chat_history("session", current_trace_id="current")
        check(history == [{"role": "user", "content": "可见问题"}, {"role": "assistant", "content": "可见回答"}], "Router历史只含用户可见成功对话")
    finally:
        coordinator_module.list_chat_messages = original_messages

    saved = {
        "get_pending_action_proposal": coordinator_module.get_pending_action_proposal,
        "get_work_order_draft": coordinator_module.get_work_order_draft,
        "_is_draft_follow_up": coordinator_module._is_draft_follow_up,
        "_latest_committed_dynamic_action": RuntimeCoordinator._latest_committed_dynamic_action,
    }
    try:
        coordinator_module.get_pending_action_proposal = lambda _session: None
        coordinator_module.get_work_order_draft = lambda _session: {"missing_fields": ["location"]}
        coordinator_module._is_draft_follow_up = lambda _message, _draft: True
        RuntimeCoordinator._latest_committed_dynamic_action = staticmethod(lambda *_: None)
        check(RuntimeCoordinator._select_path("s", "补充资料", {}) == RuntimePath.CONTROLLED_ACTION, "业务资料补充不中断原流程")
        coordinator_module._is_draft_follow_up = lambda _message, _draft: False
        check(RuntimeCoordinator._select_path("s", "换个话题", {}) == RuntimePath.CONSULTATION, "明确换题重新进入Router")
    finally:
        coordinator_module.get_pending_action_proposal = saved["get_pending_action_proposal"]
        coordinator_module.get_work_order_draft = saved["get_work_order_draft"]
        coordinator_module._is_draft_follow_up = saved["_is_draft_follow_up"]
        RuntimeCoordinator._latest_committed_dynamic_action = saved["_latest_committed_dynamic_action"]

    router_source = inspect.getsource(router_module.create_semantic_lane_router)
    classify_source = inspect.getsource(router_module.classify_lane_decision)
    for forbidden in ("CLARIFY", "request_kind", "target_agent_id", "confidence", "allowed_domain"):
        check(forbidden not in router_source + classify_source, f"生产Router合同不含{forbidden}")
    check("keyword" not in classify_source.lower() and "regex" not in classify_source.lower(), "生产分类器没有关键词或正则分支")
    for principle in (
        "当前产品环境是物业服务助手",
        "控制性包装",
        "现实安全描述优先于包装方式",
        "纯虚构故事且没有现实事件指向时选C",
    ):
        check(principle in router_source, f"Router包含通用原则：{principle}")
    for forbidden_case_text in (
        "收房当天发现卧室墙面贯穿裂缝",
        "关闭证据校验",
        "配电柜正不断冒火星",
    ):
        check(forbidden_case_text not in router_source, "Router未写入失败题原句")

    check(eval_module.EXPECTED_OVERRIDES["N268"] == RuntimeLane.ISOLATED_GENERAL.value, "N268按信息不足标为C")
    check(
        eval_module.EXPECTED_DISTRIBUTION
        == {
            RuntimeLane.SAFETY_HANDOFF.value: 30,
            RuntimeLane.PROPERTY_GOVERNED.value: 35,
            RuntimeLane.ISOLATED_GENERAL.value: 35,
        },
        "最终评估分布为A30/B35/C35",
    )

    coordinator_source = inspect.getsource(RuntimeCoordinator._stream_consultation)
    for forbidden in ("lane_decision.target_agent_id", "lane_decision.request_kind", "lane_decision.confidence", "lane_decision.allowed_domain"):
        check(forbidden not in coordinator_source, f"下游不再依赖{forbidden}")

    for relative in ("agents/router.py", "app/runtime/contracts.py", "app/runtime/coordinator.py"):
        ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
    check(True, "三处受影响Python语法通过")


def main() -> None:
    synchronous_checks()
    asyncio.run(async_checks())
    print("S10-B.2-Final deterministic contract: PASS")


if __name__ == "__main__":
    main()
