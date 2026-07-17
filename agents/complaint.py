"""
Complaint Vertical Agent
========================

Handles owner complaints, disputes, and escalation.
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
    "你是YIAI物业的投诉 Agent，负责处理业主投诉、邻里纠纷、责任争议。",
    "首先要安抚业主情绪，表达理解和重视。",
    "记录投诉要点：时间、地点、涉及人员、事件经过、业主诉求。",
    "不要自行判定责任或给出最终处理结论。",
    "明确告知业主已记录投诉，将转由物业专人跟进处理。",
    "涉及紧急安全、邻里冲突升级时，建议立即联系物业值班电话或报警。",
    "回复要体现同理心，使用中文，关键信息高亮。",
]


def create_complaint_agent(
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
        id="complaint_agent",
        name=name or "投诉 Agent",
        description=description or "处理业主投诉、邻里纠纷、责任争议。",
        model=model or MODEL,
        db=agent_db,
        tools=agent_tools,
        skills=None,
        instructions=(INSTRUCTIONS.copy() + (instructions or [])),
        add_datetime_to_context=True,
        add_history_to_context=True,
        read_chat_history=True,
        num_history_runs=5,
        markdown=True,
    )


complaint_agent = create_complaint_agent()
