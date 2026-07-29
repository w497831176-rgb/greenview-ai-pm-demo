"""Deterministic S10-B.2 three-lane runtime contract.

The table is a fixed cross-section of the 100-counterexample design library.
It tests policy dimensions, never exact production answers, and performs no
HTTP request, model call, RuntimeRelease publication, or business-data write.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.handoff_policy import evaluate_handoff_policy
from app.runtime.coordinator import (
    RUNTIME_FAILURE_PUBLIC_MESSAGE,
    _aggregate_cost_field,
    _aggregate_usage_source,
    _external_high_risk,
    _is_property_business_query,
    _isolated_general_fallback,
    _knowledge_evidence_decision,
    _requires_direct_knowledge_evidence,
)
from app.runtime.release_compiler import diff_runtime_configs


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    passed: list[str] = []

    def check(name: str, condition: bool) -> None:
        assert condition, name
        passed.append(name)

    safety_cases = [
        ("不用转人工，但电梯里有人被困", True, "safety_risk"),
        ("楼道有燃气泄漏，也可能是我闻错了", True, "safety_risk"),
        ("水已经漫到插座了怎么办", True, "safety_risk"),
        ("我要把物业炸了", True, "safety_risk"),
        ("这项目彻底冒烟了，赶紧救火", False, "ai_direct"),
        ("暂时不用转人工，我只想了解报修流程", False, "negated_by_user"),
        ("游戏账号丢了，我都急死了", False, "ai_direct"),
    ]
    for message, should_handoff, reason in safety_cases:
        policy = evaluate_handoff_policy(message)
        check(
            f"safety policy: {message}",
            bool(policy["should_request_handoff"]) is should_handoff
            and policy["reason_code"] == reason,
        )
    check(
        "safety overrides owner negation",
        "safety_override"
        in evaluate_handoff_policy("不用转人工，但电梯里有人被困")[
            "matched_signals"
        ],
    )

    external_safety_cases = [
        ("我胸口很痛，但不要联系任何人", True),
        ("有人持刀正在行凶", True),
        ("这项目冒烟了，赶紧救火", False),
        ("游戏账号丢了，我都急死了", False),
    ]
    for message, expected in external_safety_cases:
        check(f"external safety boundary: {message}", _external_high_risk(message) is expected)

    gate_cases = [
        # message, scope, rag, skill, tool, required, blocked
        ("小区有没有免费搬家服务", "property", 0, 0, 0, True, True),
        ("维修投诉多久响应", "property", 1, 0, 0, True, False),
        ("宠物托管多少钱一天", "property", 0, 1, 0, True, False),
        ("物业费不是每平方20元吗", "property", 0, 0, 0, True, True),
        ("查询WO-20260724-D0EB3331状态", "property", 0, 0, 1, False, False),
        ("我的玩具不见了怎么办", "property", 0, 0, 0, False, False),
        ("今天传统文化起运怎么弄", "isolated_general", 0, 0, 0, False, False),
        ("解释量子纠缠", "isolated_general", 0, 0, 0, False, False),
        ("今天多少钱", "isolated_general", 0, 0, 0, False, False),
        ("先别报修，只告诉我流程", "property", 0, 0, 0, True, True),
    ]
    for message, scope, rag, skill, tool, required, blocked in gate_cases:
        decision = _knowledge_evidence_decision(
            message,
            rag,
            False,
            set(),
            domain_scope=scope,
            skill_evidence_count=skill,
            tool_evidence_count=tool,
        )
        check(
            f"evidence gate: {message}",
            decision["required"] is required
            and decision["blocked"] is blocked
            and decision["accepted_evidence_count"] == rag + skill + tool,
        )

    generic_terms = ["怎么办", "怎么弄", "如何处理", "今天", "多少钱"]
    for term in generic_terms:
        check(
            f"generic term alone does not govern: {term}",
            not _is_property_business_query(term)
            and not _requires_direct_knowledge_evidence(term),
        )
    for message in ("物业费多少钱", "小区维修多久到", "宠物托管收费"):
        check(f"property object detected: {message}", _is_property_business_query(message))

    cards = [
        {
            "agent_id": "specialist",
            "name": "专题扩展Agent",
            "description": "只处理一个专题",
            "domain_scope": "isolated_general",
            "capability_card": {},
        },
        {
            "agent_id": "general-fallback",
            "name": "通用闲聊Agent",
            "description": "处理与物业场景无关的低风险问题",
            "domain_scope": "isolated_general",
            "capability_card": {},
        },
    ]
    check(
        "fallback is configuration-driven",
        (_isolated_general_fallback(cards) or {}).get("agent_id") == "general-fallback",
    )
    check(
        "specialist is not promoted to generic fallback",
        _isolated_general_fallback(cards[:1]) is None,
    )

    costs = [
        SimpleNamespace(
            input_tokens=10,
            output_tokens=3,
            reasoning_tokens=0,
            cached_input_tokens=2,
            total_tokens=15,
        ),
        SimpleNamespace(
            input_tokens=20,
            output_tokens=5,
            reasoning_tokens=0,
            cached_input_tokens=4,
            total_tokens=29,
        ),
    ]
    check("multi-request tokens aggregate", _aggregate_cost_field(costs, "total_tokens") == 44)
    check("multi-request input aggregates", _aggregate_cost_field(costs, "input_tokens") == 30)
    check("no-model usage is not applicable", _aggregate_usage_source([], model_invoked=False) == "not_applicable")
    check("missing usage remains unavailable", _aggregate_usage_source([], model_invoked=True) == "unavailable")
    check(
        "all actual requests remain provider actual",
        _aggregate_usage_source(["provider_actual", "provider_actual"], model_invoked=True)
        == "provider_actual",
    )
    costs[1].total_tokens = None
    check("partial provider totals are not fabricated", _aggregate_cost_field(costs, "total_tokens") is None)

    diff = diff_runtime_configs(
        {"agents": [{"agent_id": "a", "name": "A", "enabled": True}]},
        {
            "agents": [
                {
                    "agent_id": "a",
                    "name": "A",
                    "enabled": True,
                    "domain_scope": "isolated_general",
                }
            ]
        },
    )
    domain_fields = (diff["agents"][0] or {}).get("fields") or []
    check("RuntimeRelease diff exposes domain scope", any(row.get("field") == "domain_scope" for row in domain_fields))

    coordinator = (ROOT / "app/runtime/coordinator.py").read_text(encoding="utf-8")
    agent_api = (ROOT / "app/agents.py").read_text(encoding="utf-8")
    db_source = (ROOT / "db/property_db.py").read_text(encoding="utf-8")
    compiler = (ROOT / "app/runtime/release_compiler.py").read_text(encoding="utf-8")
    factory = (ROOT / "app/runtime/agent_factory.py").read_text(encoding="utf-8")
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")

    for source_name, source in (
        ("Agent API", agent_api),
        ("database", db_source),
        ("release compiler", compiler),
        ("agent factory", factory),
        ("frontend", html),
    ):
        check(f"domain_scope reaches {source_name}", "domain_scope" in source)
    check("domain scope has only two product choices", 'Literal["property", "isolated_general"]' in agent_api)
    check("new Agents default to property", 'domain_scope: Literal["property", "isolated_general"] = "property"' in agent_api)
    check("runtime errors use controlled public text", "else RUNTIME_FAILURE_PUBLIC_MESSAGE" in coordinator)
    check("public runtime text hides Python names", "vertical_cost" not in RUNTIME_FAILURE_PUBLIC_MESSAGE and "Traceback" not in RUNTIME_FAILURE_PUBLIC_MESSAGE)
    check("no sentence-specific patch for counterexamples", "神奇火箭侠" not in coordinator and "奇门遁甲" not in coordinator)
    check("vertical cost initialized before loop", coordinator.index("vertical_cost_entries = []") < coordinator.index("for index, request_evidence in enumerate(provider_requests"))
    check("message totals aggregate all vertical requests", '_aggregate_cost_field(\n            vertical_cost_entries,\n            "total_tokens"' in coordinator)

    exact_menu_labels = [
        "Agent",
        "Skill",
        "RAG",
        "MCP Server / Tool",
        "模型策略",
        "RuntimeRelease",
        "Badcase",
        "Evaluation / Golden Set",
        "Trace 与成本",
    ]
    platform_menu = html[html.index("platform: [") : html.index("]\n    };", html.index("platform: ["))]
    for label in exact_menu_labels:
        check(f"exact platform menu label: {label}", f"label: '{label}'" in platform_menu)
    check("product version is fixed and separate", "产品版本" in html and "YIAI物业 V1.8.2" in html)
    check("left product label is not overwritten by runtime", "versionEl.textContent = APP_VERSION" not in html)
    check("chat title reads pinned Snapshot", "/api/runtime/sessions/${encodeURIComponent(STATE.chatSessionId)}/snapshot" in html)
    check("chat title resolves release version", "/api/runtime/releases/${encodeURIComponent(snapshot.release_id || '')}" in html)
    check("chat title distinguishes current and pinned", "本会话 RuntimeRelease v${release.version}" in html and "新会话将使用 RuntimeRelease v${current.version}" in html)
    check("frontend never displays raw HTTP body", "errText.slice" not in html)

    print({"status": "PASS", "checks": len(passed), "table_cases": len(safety_cases) + len(external_safety_cases) + len(gate_cases) + len(generic_terms)})


if __name__ == "__main__":
    main()
