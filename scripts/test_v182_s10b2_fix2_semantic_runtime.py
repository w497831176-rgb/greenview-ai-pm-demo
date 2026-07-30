"""No-model contract checks for S10-B.2-Fix2.

FakeAgent supplies already-materialised responses. This file opens no network,
calls no Provider and writes no runtime or business data.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

import agents.router as router_module
import app.runtime.coordinator as coordinator_module
from app.runtime.contracts import LaneDecision, ResponseMode, RuntimeLane
from app.runtime.coordinator import (
    _answer_contract_for,
    _is_internal_control_payload,
    _knowledge_evidence_decision,
    _lane_candidates,
    _render_visible_history_context,
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
        "agent_id": "general-dynamic",
        "name": "动态通用 Agent",
        "description": "非物业低风险一般回答",
        "domain_scope": "isolated_general",
        "enabled": True,
        "capability_card": {},
    },
]


def lane(**values: Any) -> LaneDecision:
    return LaneDecision.model_validate(values)


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
    def __init__(self, content: str, *, fail: bool = False):
        self.content = content
        self.fail = fail
        self.calls = 0
        self.prompt = ""

    async def arun(self, prompt: str, **_: Any) -> FakeRun:
        self.calls += 1
        self.prompt = prompt
        if self.fail:
            raise RuntimeError("provider-secret-internal-error")
        return FakeRun(self.content)


def main() -> None:
    checks: List[str] = []

    def check(name: str, condition: bool) -> None:
        assert condition, name
        checks.append(name)

    decisions = {
        "A": lane(
            lane="A_SAFETY_HANDOFF",
            business_intent="electrical_public_hazard",
            reason_code="imminent_safety_risk",
            reason="存在正在发生的公共电气风险。",
            confidence=0.99,
            requires_clarification=False,
            clarification_question=None,
            request_kind="emergency",
            allowed_domain="safety",
            target_agent_id=None,
        ),
        "B": lane(
            lane="B_PROPERTY_GOVERNED",
            business_intent="property_service_fact",
            reason_code="property_service_required",
            reason="需要物业提供有依据的服务事实。",
            confidence=0.96,
            requires_clarification=False,
            clarification_question=None,
            request_kind="fact",
            allowed_domain="property",
            target_agent_id="customer_service",
        ),
        "B_REALTIME": lane(
            lane="B_PROPERTY_GOVERNED",
            business_intent="work_order_status",
            reason_code="property_service_required",
            reason="请求读取当前工单状态。",
            confidence=0.98,
            requires_clarification=False,
            clarification_question=None,
            request_kind="realtime_read",
            allowed_domain="property",
            target_agent_id="maintenance",
        ),
        "B_WRITE": lane(
            lane="B_PROPERTY_GOVERNED",
            business_intent="create_repair_request",
            reason_code="property_service_required",
            reason="请求创建物业维修记录。",
            confidence=0.98,
            requires_clarification=False,
            clarification_question=None,
            request_kind="state_change",
            allowed_domain="property",
            target_agent_id="maintenance",
        ),
        "B_REFUSE": lane(
            lane="B_PROPERTY_GOVERNED",
            business_intent="unsafe_property_access_request",
            reason_code="property_service_required",
            reason="请求在物业领域实施危险或越权操作。",
            confidence=0.99,
            requires_clarification=False,
            clarification_question=None,
            request_kind="unsafe_request",
            allowed_domain="property",
            target_agent_id="customer_service",
        ),
        "C_FACT": lane(
            lane="C_ISOLATED_GENERAL",
            business_intent="general_cultural_fact",
            reason_code="non_property_general",
            reason="真实诉求是非物业的一般知识事实。",
            confidence=0.97,
            requires_clarification=False,
            clarification_question=None,
            request_kind="fact",
            allowed_domain="isolated_general",
            target_agent_id="general-dynamic",
        ),
        "C": lane(
            lane="C_ISOLATED_GENERAL",
            business_intent="general_cultural_question",
            reason_code="non_property_general",
            reason="真实诉求是非物业的一般知识。",
            confidence=0.97,
            requires_clarification=False,
            clarification_question=None,
            request_kind="general",
            allowed_domain="isolated_general",
            target_agent_id="general-dynamic",
        ),
        "C_REFUSE": lane(
            lane="C_ISOLATED_GENERAL",
            business_intent="harmful_chemical_instructions",
            reason_code="unsafe_non_property_request",
            reason="请求提供可造成伤害的具体实施信息。",
            confidence=0.99,
            requires_clarification=False,
            clarification_question=None,
            request_kind="unsafe_request",
            allowed_domain="isolated_general",
            target_agent_id="general-dynamic",
        ),
        "CLARIFY": lane(
            lane="CLARIFY",
            business_intent="ambiguous_missing_object",
            reason_code="insufficient_context",
            reason="对象和地点不足以确定处理范围。",
            confidence=0.61,
            requires_clarification=True,
            clarification_question="它是人、宠物、实体物品还是虚构角色，最后在哪里见到？",
            request_kind="ambiguous",
            allowed_domain="none",
            target_agent_id=None,
        ),
    }

    contracts = {name: _answer_contract_for(value) for name, value in decisions.items()}
    check("A compiles emergency_handoff", contracts["A"].response_mode == ResponseMode.EMERGENCY_HANDOFF)
    check("A skips all ordinary capabilities", contracts["A"].skill_policy == contracts["A"].rag_policy == contracts["A"].tool_policy == "skipped" and contracts["A"].handoff_policy == "required")
    check("B compiles grounded_answer", contracts["B"].response_mode == ResponseMode.GROUNDED_ANSWER and contracts["B"].evidence_required)
    check("B realtime requires successful Tool", contracts["B_REALTIME"].response_mode == ResponseMode.REALTIME_READ and contracts["B_REALTIME"].evidence_requirements == ["successful_current_tool_result"])
    check("B write preserves confirmation contract", contracts["B_WRITE"].response_mode == ResponseMode.CONTROLLED_WRITE and contracts["B_WRITE"].write_policy == "allowed_after_confirmation" and "receipt" in contracts["B_WRITE"].evidence_requirements)
    check("unsafe B compiles safe_refusal", contracts["B_REFUSE"].response_mode == ResponseMode.SAFE_REFUSAL and contracts["B_REFUSE"].write_policy == "forbidden")
    check("C fact compiles safe_general", contracts["C_FACT"].response_mode == ResponseMode.SAFE_GENERAL and not contracts["C_FACT"].evidence_required)
    check("C safe general skips property capabilities", contracts["C"].response_mode == ResponseMode.SAFE_GENERAL and contracts["C"].skill_policy == contracts["C"].rag_policy == contracts["C"].tool_policy == "skipped")
    check("unsafe C compiles safe_refusal", contracts["C_REFUSE"].response_mode == ResponseMode.SAFE_REFUSAL)
    check("CLARIFY selects no Agent", decisions["CLARIFY"].target_agent_id is None and contracts["CLARIFY"].response_mode == ResponseMode.CLARIFY_ONLY)

    b_cards = _lane_candidates(CARDS, RuntimeLane.PROPERTY_GOVERNED)
    c_cards = _lane_candidates(CARDS, RuntimeLane.ISOLATED_GENERAL)
    check("B candidates are property only", bool(b_cards) and all(item["domain_scope"] == "property" for item in b_cards))
    check("C candidates are isolated only", bool(c_cards) and all(item["domain_scope"] == "isolated_general" for item in c_cards))
    check("A and CLARIFY have no business candidates", _lane_candidates(CARDS, RuntimeLane.SAFETY_HANDOFF) == [] and _lane_candidates(CARDS, RuntimeLane.CLARIFY) == [])

    no_evidence = _knowledge_evidence_decision(contracts["B"], 0, False, set(), domain_scope="property")
    with_rag = _knowledge_evidence_decision(contracts["B"], 1, False, {1}, domain_scope="property")
    realtime_without_tool = _knowledge_evidence_decision(contracts["B_REALTIME"], 2, False, {1}, domain_scope="property", tool_evidence_count=0)
    realtime_with_tool = _knowledge_evidence_decision(contracts["B_REALTIME"], 0, True, set(), domain_scope="property", tool_evidence_count=1)
    check("B without Evidence is blocked before vertical model", no_evidence["blocked"] and not no_evidence["model_invoked"])
    check("B adopted RAG satisfies evidence", not with_rag["blocked"])
    check("realtime ignores non-Tool evidence", realtime_without_tool["blocked"])
    check("successful current Tool satisfies realtime", not realtime_with_tool["blocked"])

    missing_request_kind = decisions["B"].model_dump(mode="json")
    missing_request_kind.pop("request_kind")
    invalid_payloads = {
        "A general": {**decisions["A"].model_dump(mode="json"), "request_kind": "general"},
        "A state_change": {**decisions["A"].model_dump(mode="json"), "request_kind": "state_change"},
        "C property state_change": {
            **decisions["C_FACT"].model_dump(mode="json"),
            "request_kind": "state_change",
            "allowed_domain": "property",
            "target_agent_id": "maintenance",
        },
        "CLARIFY realtime_read": {**decisions["CLARIFY"].model_dump(mode="json"), "request_kind": "realtime_read"},
        "CLARIFY state_change": {**decisions["CLARIFY"].model_dump(mode="json"), "request_kind": "state_change"},
        "lane domain conflict": {**decisions["B"].model_dump(mode="json"), "allowed_domain": "isolated_general"},
        "missing target": {**decisions["B"].model_dump(mode="json"), "target_agent_id": None},
        "missing request_kind": missing_request_kind,
        "illegal request_kind": {**decisions["B"].model_dump(mode="json"), "request_kind": "not_a_kind"},
        "clarify missing question": {**decisions["CLARIFY"].model_dump(mode="json"), "clarification_question": None},
    }
    for name, payload in invalid_payloads.items():
        try:
            LaneDecision.model_validate(payload)
        except ValidationError:
            checks.append(f"invalid schema rejected: {name}")
        else:
            raise AssertionError(f"invalid schema rejected: {name}")

    fake = FakeAgent(decisions["C"].model_dump_json())
    original_factory = router_module.create_semantic_lane_router
    router_module.create_semantic_lane_router = lambda **_: fake
    try:
        semantic = asyncio.run(
            router_module.classify_lane_decision(
                "任意自然语言",
                vertical_agents=CARDS,
                model=object(),
                visible_history=[{"role": "user", "content": "上一轮可见问题"}],
            )
        )
    finally:
        router_module.create_semantic_lane_router = original_factory
    check("semantic Router is called exactly once", fake.calls == 1)
    check("semantic Router returns strict decision", semantic["decision"] == decisions["C"] and semantic["validation_error"] is None)
    check("dynamic Published Agent remains eligible", semantic["decision"].target_agent_id == "general-dynamic")
    check("visible conversation reaches Router only", "上一轮可见问题" in fake.prompt)
    check("Provider request evidence survives validation", semantic["provider_evidence"]["provider_request_id"] == "provider-request-test")

    fenced_fact = FakeAgent(f"```json\n{decisions['C_FACT'].model_dump_json()}\n```")
    router_module.create_semantic_lane_router = lambda **_: fenced_fact
    try:
        replay_shape = asyncio.run(
            router_module.classify_lane_decision(
                "任意非物业知识事实问题",
                vertical_agents=CARDS,
                model=object(),
            )
        )
    finally:
        router_module.create_semantic_lane_router = original_factory
    check("C fact in an outer JSON fence passes strict schema", replay_shape["decision"] == decisions["C_FACT"])
    check("C fact compiles safe_general after parsing", _answer_contract_for(replay_shape["decision"]).response_mode == ResponseMode.SAFE_GENERAL)

    cross_lane = FakeAgent(decisions["C"].model_copy(update={"target_agent_id": "maintenance"}).model_dump_json())
    router_module.create_semantic_lane_router = lambda **_: cross_lane
    try:
        blocked = asyncio.run(router_module.classify_lane_decision("任意自然语言", vertical_agents=CARDS, model=object()))
    finally:
        router_module.create_semantic_lane_router = original_factory
    check("cross-Lane target does not fallback", blocked["decision"] is None and "lane_agent_scope_mismatch" in blocked["validation_error"])

    failed_agent = FakeAgent("", fail=True)
    router_module.create_semantic_lane_router = lambda **_: failed_agent
    try:
        provider_failed = asyncio.run(router_module.classify_lane_decision("任意自然语言", vertical_agents=CARDS, model=object()))
    finally:
        router_module.create_semantic_lane_router = original_factory
    check("Provider failure does not produce a fallback decision", provider_failed["decision"] is None and provider_failed["provider_status"] == "failed")
    check("Provider internal error is never a user answer", "provider-secret" not in coordinator_module.RUNTIME_FAILURE_PUBLIC_MESSAGE)

    visible_context = _render_visible_history_context(
        [{"role": "user", "content": "成功问题"}, {"role": "assistant", "content": "成功回答"}],
        "本轮问题",
        boundary="只读咨询",
    )
    check("visible history remains effective", "成功问题" in visible_context and "成功回答" in visible_context)
    check("control JSON is absent from business context", "target_agent_id" not in visible_context and "reason_code" not in visible_context)
    check("control JSON is blocked at delivery", _is_internal_control_payload('{"response_mode":"grounded_answer","reason_code":"x"}'))

    coordinator_source = (ROOT / "app/runtime/coordinator.py").read_text(encoding="utf-8")
    router_source = (ROOT / "agents/router.py").read_text(encoding="utf-8")
    frontend_source = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    forbidden_production_symbols = [
        "_decide_lane",
        "PROPERTY_BUSINESS_TERMS",
        "ISOLATED_GENERAL_MARKERS",
        "unknown_scope_clarify_in_property_lane",
        "from agents.router import classify_intent",
    ]
    check("production coordinator has no keyword Lane or default B", all(value not in coordinator_source for value in forbidden_production_symbols))
    check("production coordinator never reads an Evaluation dataset", ".jsonl" not in coordinator_source and "N201" not in coordinator_source)
    check("one semantic Router owns production Lane", "classify_lane_decision" in coordinator_source and "::semantic_router::" in coordinator_source)
    check("Router disables automatic Agno history", "add_history_to_context=False" in router_source and "read_chat_history=False" in router_source)
    check("frontend shows Lane and AnswerContract in default layer", "本轮判断与回答边界" in frontend_source and "answer_contract" in frontend_source)
    check("technical decision JSON remains collapsed", "高级信息：查看结构化决策Schema" in frontend_source)

    print({"status": "PASS", "checks": len(checks), "provider_calls": 0})


if __name__ == "__main__":
    main()
