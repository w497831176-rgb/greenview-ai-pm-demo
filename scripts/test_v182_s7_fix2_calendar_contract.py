"""Deterministic S7-Fix2 calendar Tool selection contract.

This test uses an in-memory published configuration fixture.  It never calls a
model Provider, MCP server, or the NAS database.
"""

from __future__ import annotations

from pathlib import Path

from agents.router import _capability_fallback
from app.runtime.contracts import AnswerContract, ResponseMode, RuntimePath, ToolEffect
from app.runtime.coordinator import _knowledge_evidence_decision
from app.runtime.tool_planner import plan_tools


CASE_2_QUESTION = (
    "请问小区是否提供免费搬家服务？请告诉我专用预约号码、服务时段、免费额度和办理步骤。"
    "请只使用知识库直接证据；没有证据就明确说依据不足，不要猜测，也不要转人工。"
)

CALENDAR_KEYWORDS = [
    "今天几号",
    "今天是几号",
    "今天星期几",
    "今天是星期几",
    "现在几点",
    "现在是几点",
    "当前日期",
    "当前时间",
    "日期计算",
]


def _policy(server_id: int, server_name: str, tool_name: str) -> dict:
    return {
        "server_id": server_id,
        "server_name": server_name,
        "tool_name": tool_name,
        "effect": "read",
        "risk_level": "L1",
        "allowed_paths": ["consultation"],
        "requires_confirmation": False,
        "enabled": True,
        "policy_reason": "S7-Fix2 deterministic fixture",
    }


def _config() -> dict:
    calendar_metadata = {
        "effect": "read",
        "risk_level": "L1",
        "result_contract": {
            "success_statuses": ["success"],
            "non_success_statuses": [
                "empty",
                "not_found",
                "invalid_input",
                "unauthorized",
                "timeout",
                "upstream_error",
                "unknown",
            ],
            "claim_rule": "Only an explicit success business status may be described as success.",
        },
        "natural_language_intents": ["查询当前日期或时间"],
        "trigger_keywords": CALENDAR_KEYWORDS,
        "trigger_mode": "any",
        "execution_mode": "auto_preinvoke",
        "argument_bindings": {},
        "effect_source": "operator_declared",
    }
    return {
        "agents": [
            {
                "agent_id": "customer_service",
                "enabled": True,
                "mcp_server_names": ["calendar-server", "workorder-server"],
            }
        ],
        "mcp_servers": [
            {
                "id": 55,
                "name": "calendar-server",
                "enabled": True,
                "tools": [
                    {
                        "name": "get_current_datetime",
                        "description": "获取当前北京时间日期和时间。",
                        "input_schema": {"type": "object", "properties": {}},
                        "tool_metadata": calendar_metadata,
                        "policy": _policy(55, "calendar-server", "get_current_datetime"),
                    }
                ],
            },
            {
                "id": 56,
                "name": "workorder-server",
                "enabled": True,
                "tools": [
                    {
                        "name": "get_my_work_order_by_id",
                        "description": "按工单号查询我的工单。",
                        "input_schema": {"type": "object", "properties": {}},
                        "policy": _policy(56, "workorder-server", "get_my_work_order_by_id"),
                    }
                ],
            },
        ],
    }


def _plans(message: str):
    return plan_tools(
        _config(),
        "customer_service",
        message,
        RuntimePath.CONSULTATION,
        effects=[ToolEffect.READ],
        execution_modes=["auto_preinvoke", "model_native"],
    )


def _calendar_plans(message: str):
    return [plan for plan in _plans(message) if plan.server_name == "calendar-server"]


def main() -> None:
    checks: list[str] = []

    def check(name: str, condition: bool) -> None:
        assert condition, name
        checks.append(name)

    # Calendar-positive phrases ask for current date/time information.
    for message in ("今天几号", "现在几点", "今天星期几"):
        plans = _calendar_plans(message)
        check(
            f"calendar selected for {message}",
            len(plans) == 1 and plans[0].tool_name == "get_current_datetime",
        )

    # Appointment/service phrases are business intents, not clock queries.
    for message in (
        "如何预约物业服务",
        "有没有免费搬家服务",
        CASE_2_QUESTION,
        "预约维修",
    ):
        check(f"calendar skipped for {message[:16]}", not _calendar_plans(message))

    mixed = _calendar_plans("今天几号，我想预约维修")
    check("mixed date and appointment selects calendar", len(mixed) == 1)
    check(
        "mixed match reason is the date phrase",
        "今天几号" in mixed[0].match_reason and "预约" not in mixed[0].match_reason,
    )

    workorder = _plans("请查询工单号 WO-20260727-001")
    check(
        "exact work-order lookup remains available",
        any(
            plan.server_name == "workorder-server"
            and plan.tool_name == "get_my_work_order_by_id"
            for plan in workorder
        ),
    )

    evidence_gate = _knowledge_evidence_decision(
        AnswerContract(
            response_mode=ResponseMode.GROUNDED_ANSWER,
            evidence_required=True,
            evidence_requirements=["activated_skill", "accepted_rag", "successful_tool"],
            skill_policy="selected",
            rag_policy="selected",
            tool_policy="selected",
            write_policy="forbidden",
            handoff_policy="optional",
            forbidden_claims=["unsupported_property_fact"],
            decision_reason="物业事实回答必须由当次运行的合法Evidence支撑。",
        ),
        evidence_count=0,
        structured_realtime_query=False,
        allowed_document_ids={1},
    )
    check(
        "no-evidence refusal remains enforced",
        evidence_gate["blocked"]
        and evidence_gate["evidence_decision"] == "rejected_insufficient"
        and evidence_gate["model_invoked"] is False,
    )

    customer = {
        "agent_id": "customer_service",
        "name": "客服 Agent",
        "enabled": True,
        "description": "物业服务、小区服务与搬家服务咨询",
        "capability_card": {"routing_hints": "物业服务咨询；知识不足时安全拒答"},
    }
    mars = {
        "agent_id": "mars-greenhouse-agent",
        "name": "火星温室 Agent",
        "enabled": True,
        "description": "火星温室种植舱、标准营养液、温室补给、份数与配比计算。",
        "capability_card": {
            "routing_hints": "仅回答火星温室、种植舱、标准营养液与温室补给计算。"
        },
    }
    mars_winner, _, _ = _capability_fallback(
        "4个火星温室种植舱连续7天需要多少份标准营养液？",
        [customer, mars],
    )
    case_winner, _, _ = _capability_fallback(CASE_2_QUESTION, [customer, mars])
    check("Agent 74 domain route remains intact", mars_winner == "mars-greenhouse-agent")
    check("Case 2 route remains customer service", case_winner == "customer_service")

    coordinator_source = (
        Path(__file__).resolve().parents[1] / "app" / "runtime" / "coordinator.py"
    ).read_text(encoding="utf-8")
    check(
        "skipped tool reason is persisted to Trace evidence",
        '"selected" if read_tool_plans else "skipped"' in coordinator_source
        and 'else ("matched_intent" if read_tool_plans else "not_required")'
        in coordinator_source
        and '"decision_summary": decision_summary' in coordinator_source,
    )

    print({"status": "PASS", "checks": len(checks), "names": checks})


if __name__ == "__main__":
    main()
