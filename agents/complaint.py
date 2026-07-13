"""
Complaint Vertical Agent
========================

Handles owner complaints, disputes, and escalation.
"""

from pathlib import Path
from typing import Any, List, Optional

from agno.agent import Agent

from app.settings import MODEL, agent_db
from tools.knowledge import KnowledgeTools

try:
    from agno.skills import LocalSkills, Skills

    skill_loaders = []
    local_skills_path = Path(__file__).parent / "property" / "skills"
    enterprise_skills_path = Path("/app/enterprise/skills")
    if local_skills_path.exists():
        skill_loaders.append(LocalSkills(str(local_skills_path)))
    if enterprise_skills_path.exists():
        skill_loaders.append(LocalSkills(str(enterprise_skills_path)))
    skills = Skills(loaders=skill_loaders) if skill_loaders else None
except ImportError:
    skills = None


def _base_tools() -> List[Any]:
    base_tools: List[Any] = []
    try:
        base_tools.append(KnowledgeTools())
    except ImportError:
        pass
    return base_tools


INSTRUCTIONS = [
    "你是绿景智服的投诉 Agent，负责处理业主投诉、邻里纠纷、责任争议。",
    "首先要安抚业主情绪，表达理解和重视。",
    "记录投诉要点：时间、地点、涉及人员、事件经过、业主诉求。",
    "不要自行判定责任或给出最终处理结论。",
    "明确告知业主已记录投诉，将转由物业专人跟进处理。",
    "涉及紧急安全、邻里冲突升级时，建议立即联系物业值班电话或报警。",
    "回复要体现同理心，使用中文，关键信息高亮。",
]


def create_complaint_agent(tools: Optional[List[Any]] = None, model: Optional[Any] = None) -> Agent:
    agent_tools = _base_tools()
    if tools:
        agent_tools.extend(tools)
    return Agent(
        id="complaint_agent",
        name="投诉 Agent",
        description="处理业主投诉、邻里纠纷、责任争议。",
        model=model or MODEL,
        db=agent_db,
        tools=agent_tools,
        skills=skills,
        instructions=INSTRUCTIONS,
        add_datetime_to_context=True,
        add_history_to_context=True,
        read_chat_history=True,
        num_history_runs=5,
        markdown=True,
    )


complaint_agent = create_complaint_agent()
