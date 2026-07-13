"""
Customer Service Vertical Agent
===============================

Handles general property service inquiries.
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
    "你是绿景智服的客服 Agent，负责解答小区服务、联系方式、一般规定等咨询。",
    "回答服务承诺、小区规定、紧急联系方式等问题时，必须基于知识库原文。",
    "对于超出能力范围的问题（如具体房号查询、账户信息修改），主动建议转人工。",
    "保持礼貌、耐心，使用中文，关键信息高亮。",
]


def create_customer_service_agent(tools: Optional[List[Any]] = None, model: Optional[Any] = None) -> Agent:
    agent_tools = _base_tools()
    if tools:
        agent_tools.extend(tools)
    return Agent(
        id="customer_service_agent",
        name="客服 Agent",
        description="处理一般咨询、小区规定、服务承诺。",
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


customer_service_agent = create_customer_service_agent()
