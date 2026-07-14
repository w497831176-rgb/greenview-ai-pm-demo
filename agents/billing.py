"""
Billing Vertical Agent
======================

Handles fee inquiries, charging standards, and billing disputes.
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
        from agno.tools.calculator import CalculatorTools
        base_tools.append(CalculatorTools())
    except ImportError:
        pass
    try:
        base_tools.append(KnowledgeTools())
    except ImportError:
        pass
    return base_tools


INSTRUCTIONS = [
    "你是YIAI物业的费用 Agent，负责解答收费标准、缴费方式、费用争议。",
    "回答收费问题时必须基于知识库原文，优先引用《维修收费标准》。",
    "业主专有部分的维修一般收费，公共区域维修不向业主个人收费。",
    "涉及费用争议、收费异议、未公示收费项目时，必须建议业主联系物业服务中心复核，不要自行判定。",
    "可以调用计算器工具帮助业主估算维修费用。",
    "回复简洁专业，关键信息高亮，使用中文。",
]


def create_billing_agent(tools: Optional[List[Any]] = None, model: Optional[Any] = None) -> Agent:
    agent_tools = _base_tools()
    if tools:
        agent_tools.extend(tools)
    return Agent(
        id="billing_agent",
        name="费用 Agent",
        description="处理缴费、收费标准、费用争议咨询。",
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


billing_agent = create_billing_agent()
