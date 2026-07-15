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

ROUTER_INSTRUCTIONS = [
    "你是YIAI物业的路由 Agent，负责识别业主意图并分发给合适的垂直 Agent。",
    "你只能从以下类别中选择一个输出：maintenance（维修/工单）、billing（费用/缴费）、complaint（投诉/纠纷）、customer_service（一般客服/咨询）、other（其他/无法判断）。",
    "输出格式必须严格为 JSON：{\"intent\": \"<category>\", \"reason\": \"<一句话理由>\"}",
    "维修类关键词：报修、漏水、跳闸、灯不亮、门锁、窗户、电梯、下水道、维修、工单。",
    "费用类关键词：收费、缴费、多少钱、费用、物业费、停车费、收费标准、账单。",
    "投诉类关键词：投诉、不满意、纠纷、邻居、噪音、责任、赔偿、物业不作为。",
    "客服类关键词：服务承诺、联系方式、咨询电话、小区规定、上班时间、托管、宠物。",
    "天气查询（如'天气'、'气温'、'下雨'）属于维修/工单场景，因为涉及上门维修安排，应归类为 maintenance。",
    "如果用户问题同时涉及多个类别，选择最核心、最紧急的意图。",
]


def create_router_agent() -> Agent:
    return Agent(
        id="router_agent",
        name="路由 Agent",
        description="识别业主意图并分发给垂直 Agent。",
        model=MODEL,
        db=agent_db,
        instructions=ROUTER_INSTRUCTIONS,
        add_datetime_to_context=True,
        add_history_to_context=True,
        read_chat_history=True,
        num_history_runs=3,
        markdown=False,
    )


router_agent = create_router_agent()


async def _collect_response(generator) -> str:
    """Collect text from an Agno async generator or return a single response."""
    response = ""
    try:
        # If it's already a string/coroutine, await it.
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
        # Some Agno versions return a RunResponse object directly.
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


async def classify_intent(message: str, user_id: str = "web-user", session_id: str = "") -> Dict[str, Any]:
    """Use the router agent to classify the user message intent."""
    prompt = (
        "请判断以下业主问题的意图类别。只输出 JSON，不要添加其他解释。\n\n"
        f"用户问题：{message}\n\n"
        "可选类别：maintenance、billing、complaint、customer_service、other"
    )
    try:
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
            intent = parsed.get("intent", "other")
            reason = parsed.get("reason", "")
        else:
            # Fallback: keyword classification.
            intent = _keyword_intent(message)
            reason = "基于关键词回退分类"

        valid_intents = {"maintenance", "billing", "complaint", "customer_service", "other"}
        if intent not in valid_intents:
            intent = "other"
        return {"intent": intent, "reason": reason, "raw": response}
    except Exception:
        import traceback
        traceback.print_exc()
        return {"intent": _keyword_intent(message), "reason": "路由异常，使用关键词回退", "raw": ""}


def _keyword_intent(message: str) -> str:
    """Fallback keyword-based intent classification."""
    lowered = message.lower()
    maintenance_keywords = ["报修", "漏水", "跳闸", "灯不亮", "门锁", "窗户", "电梯", "下水道", "维修", "工单", "师傅", "上门", "天气", "气温", "下雨"]
    billing_keywords = ["收费", "缴费", "多少钱", "费用", "物业费", "停车费", "账单", "价格", "收费标准"]
    complaint_keywords = ["投诉", "不满意", "纠纷", "邻居", "噪音", "责任", "赔偿", "物业不作为", "举报"]
    if any(k in lowered for k in maintenance_keywords):
        return "maintenance"
    if any(k in lowered for k in billing_keywords):
        return "billing"
    if any(k in lowered for k in complaint_keywords):
        return "complaint"
    return "customer_service"
