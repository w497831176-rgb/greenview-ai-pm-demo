"""
Customer Service Vertical Agent
===============================

Handles general property service inquiries.
"""

from typing import Any, List, Optional

from agno.agent import Agent

from app.settings import MODEL, agent_db
from tools.knowledge import KnowledgeTools


def _base_tools() -> List[Any]:
    base_tools: List[Any] = []
    try:
        base_tools.append(KnowledgeTools())
    except ImportError:
        pass
    return base_tools


INSTRUCTIONS = [
    "你是YIAI物业的客服 Agent，负责解答小区服务、联系方式、一般规定等咨询。",
    "回答服务承诺、小区规定、紧急联系方式等问题时，必须基于知识库原文。",
    "对于超出能力范围的问题（如具体房号查询、账户信息修改），主动建议转人工。",
    "当业主询问今天日期、当前时间或预约时间相关问题时，必须调用已绑定的 calendar-server MCP 工具的 get_current_date 或 get_current_time 函数，禁止基于自身知识猜测。",
    "保持礼貌、耐心，使用中文，关键信息高亮。",
]


def create_customer_service_agent(
    tools: Optional[List[Any]] = None,
    model: Optional[Any] = None,
    instructions: Optional[List[str]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Agent:
    agent_tools = _base_tools()
    if tools:
        agent_tools.extend(tools)
    return Agent(
        id="customer_service_agent",
        name=name or "客服 Agent",
        description=description or "处理一般咨询、小区规定、服务承诺。",
        model=model or MODEL,
        db=agent_db,
        tools=agent_tools,
        skills=None,
        instructions=instructions if instructions is not None else INSTRUCTIONS,
        add_datetime_to_context=True,
        add_history_to_context=True,
        read_chat_history=True,
        num_history_runs=5,
        markdown=True,
    )


customer_service_agent = create_customer_service_agent()
