"""
Router Agent
============

Classifies owner intent and dispatches to the appropriate vertical agent.
"""

import json
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from agno.agent import Agent

from app.settings import MODEL, agent_db


def _base_router_instructions(vertical_agents: List[Dict[str, Any]]) -> List[str]:
    """Build router instructions from the current enabled vertical agents."""
    valid_targets = [
        {"agent_id": a.get("agent_id"), "name": a.get("name"), "description": a.get("description", "")}
        for a in vertical_agents
        if a.get("agent_id") and a.get("enabled")
    ]
    if not valid_targets:
        valid_targets = [
            {"agent_id": "maintenance", "name": "维修 Agent", "description": "维修报修"},
            {"agent_id": "billing", "name": "费用 Agent", "description": "费用缴费"},
            {"agent_id": "complaint", "name": "投诉 Agent", "description": "投诉纠纷"},
            {"agent_id": "customer_service", "name": "客服 Agent", "description": "一般咨询"},
        ]
    target_lines = "\n".join(
        f'- {t["agent_id"]}（{t["name"]}）：{t["description"] or "无描述"}'
        for t in valid_targets
    )
    return [
        "你是YIAI物业的路由 Agent，负责识别业主意图并分发给合适的垂直 Agent。",
        f"你只能从以下启用的垂直 Agent 中选择目标，输出其 agent_id：\n{target_lines}",
        '输出格式必须严格为 JSON：{"target_agent_id": "<agent_id>", "reason": "<一句话理由>"}',
        "如果用户问题同时涉及多个 Agent，选择最核心、最紧急的意图对应的 Agent。",
        "如果无法判断，选择 customer_service 或其他最接近的垂直 Agent；不要编造不存在的 agent_id。",
    ]


def create_router_agent(vertical_agents: Optional[List[Dict[str, Any]]] = None) -> Agent:
    return Agent(
        id="router_agent",
        name="路由 Agent",
        description="识别业主意图并分发给垂直 Agent。",
        model=MODEL,
        db=agent_db,
        instructions=_base_router_instructions(vertical_agents or []),
        add_datetime_to_context=True,
        add_history_to_context=True,
        read_chat_history=True,
        num_history_runs=3,
        markdown=False,
    )


async def _collect_response(generator) -> str:
    """Collect text from an Agno async generator or return a single response."""
    response = ""
    try:
        if isinstance(generator, str):
            return generator
        if hasattr(generator, "__aiter__"):
            async for chunk in generator:
                if hasattr(chunk, "content") and chunk.content:
                    response += str(chunk.content)
                elif hasattr(chunk, "delta") and chunk.delta:
                    response += str(chunk.delta)
                elif isinstance(chunk, str):
                    response += chunk
            return response.strip()
        result = await generator
        if hasattr(result, "content"):
            return str(result.content).strip()
        if isinstance(result, str):
            return result.strip()
        return ""
    except Exception:
        import traceback
        traceback.print_exc()
        return ""


async def classify_intent(
    message: str,
    vertical_agents: Optional[List[Dict[str, Any]]] = None,
    user_id: str = "web-user",
    session_id: str = "",
) -> Dict[str, Any]:
    """Use the router agent to classify the user message intent."""
    vertical_agents = vertical_agents or []
    enabled_ids = {a.get("agent_id") for a in vertical_agents if a.get("enabled") and a.get("agent_id")}
    if not enabled_ids:
        enabled_ids = {"maintenance", "billing", "complaint", "customer_service"}
    valid_lines = "\n".join(
        f'- {a.get("agent_id")}（{a.get("name")}）：{a.get("description") or "无描述"}'
        for a in vertical_agents if a.get("enabled") and a.get("agent_id")
    ) or "- maintenance（维修 Agent）\n- billing（费用 Agent）\n- complaint（投诉 Agent）\n- customer_service（客服 Agent）"
    prompt = (
        "请判断以下业主问题的意图，并从当前启用的垂直 Agent 中选择一个目标。只输出 JSON，不要添加其他解释。\n\n"
        f"用户问题：{message}\n\n"
        "可选目标：\n" + valid_lines
    )
    try:
        router_agent = create_router_agent(vertical_agents)
        response = await _collect_response(
            router_agent.arun(
                prompt,
                user_id=user_id,
                session_id=session_id or f"router-{id(message)}",
                stream=False,
            )
        )

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            target = parsed.get("target_agent_id", parsed.get("intent", "customer_service"))
            reason = parsed.get("reason", "")
        else:
            target = _keyword_intent(message, vertical_agents)
            reason = "基于关键词回退分类"

        if target not in enabled_ids:
            target = _keyword_intent(message, vertical_agents)
            reason = f"模型返回了未启用的 agent_id，已回退：{reason}"
        return {"target_agent_id": target, "reason": reason, "raw": response}
    except Exception:
        import traceback
        traceback.print_exc()
        return {
            "target_agent_id": _keyword_intent(message, vertical_agents),
            "reason": "路由异常，使用关键词回退",
            "raw": "",
        }


def _keyword_intent(message: str, vertical_agents: Optional[List[Dict[str, Any]]] = None) -> str:
    """Fallback keyword-based intent classification."""
    lowered = message.lower()
    # Canonical category keywords preserved for backward compatibility.
    maintenance_keywords = ["报修", "漏水", "跳闸", "灯不亮", "门锁", "窗户", "电梯", "下水道", "维修", "工单", "师傅", "上门", "天气", "气温", "下雨"]
    billing_keywords = ["收费", "缴费", "多少钱", "费用", "物业费", "停车费", "账单", "价格", "收费标准"]
    complaint_keywords = ["投诉", "不满意", "纠纷", "邻居", "噪音", "责任", "赔偿", "物业不作为", "举报"]
    if any(k in lowered for k in maintenance_keywords):
        return "maintenance" if _agent_enabled("maintenance", vertical_agents) else _first_enabled(vertical_agents)
    if any(k in lowered for k in billing_keywords):
        return "billing" if _agent_enabled("billing", vertical_agents) else _first_enabled(vertical_agents)
    if any(k in lowered for k in complaint_keywords):
        return "complaint" if _agent_enabled("complaint", vertical_agents) else _first_enabled(vertical_agents)
    return "customer_service" if _agent_enabled("customer_service", vertical_agents) else _first_enabled(vertical_agents)


def _agent_enabled(agent_id: str, vertical_agents: Optional[List[Dict[str, Any]]]) -> bool:
    if not vertical_agents:
        return True
    return any(a.get("agent_id") == agent_id and a.get("enabled") for a in vertical_agents)


def _first_enabled(vertical_agents: Optional[List[Dict[str, Any]]]) -> str:
    if vertical_agents:
        for a in vertical_agents:
            if a.get("enabled") and a.get("agent_id"):
                return a["agent_id"]
    return "customer_service"
