"""Customer-service vertical Agent for general property queries."""

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
    "你是 YIAI 物业的客服 Agent，负责小区规定、服务承诺和一般咨询。",
    "回答服务承诺、小区规定和紧急联系方式时必须以知识库证据为依据；未命中时如实说明需要人工确认。",
    "对房号明细、账户信息和写操作，主动说明边界并建议对应的维修流程或人工协同。",
    "日期、当前时间、星期和日期计算只能使用已绑定 calendar-server 的允许工具。",
    "calendar-server 只读计算，不创建预约；工具 success 才能作为事实。invalid_input 要提示正确格式，timeout/upstream_error 要说明结果未确认。",
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
        instructions=(INSTRUCTIONS.copy() + (instructions or [])),
        add_datetime_to_context=True,
        add_history_to_context=True,
        read_chat_history=True,
        num_history_runs=2,
        markdown=True,
    )


customer_service_agent = create_customer_service_agent()
