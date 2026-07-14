"""
Maintenance Vertical Agent
==========================

Handles repair work orders and maintenance inquiries.
"""

from pathlib import Path
from typing import Any, List, Optional

from agno.agent import Agent

from app.settings import MODEL, agent_db
from tools.bing_search import BingSearchTools
from tools.knowledge import KnowledgeTools
from tools.work_order import WorkOrderTools

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
        base_tools.append(BingSearchTools())
    except ImportError:
        pass
    try:
        from agno.tools.reasoning import ReasoningTools
        base_tools.append(ReasoningTools())
    except ImportError:
        pass
    try:
        base_tools.append(WorkOrderTools())
    except ImportError:
        pass
    try:
        base_tools.append(KnowledgeTools())
    except ImportError:
        pass
    return base_tools


INSTRUCTIONS = [
    "你是YIAI物业的维修 Agent，专门处理业主维修报修、工单创建与进度查询。",
    "核心流程：理解报修意图 → 收集信息（房号、问题类型、紧急程度、联系方式、预约时间） → 检索知识库 → 创建工单 → 查询进度。",
    "必须收集的信息：房号、问题类型（水电/门窗/公区/家户）、问题描述、紧急程度（紧急/高/中/低）、联系人姓名、联系电话、预约上门时间。",
    "信息缺失时，必须主动追问，每次最多追问 1-2 个问题。",
    "回答收费标准、维修责任、服务承诺时，必须基于知识库原文；未命中时明确说'需要人工确认'。",
    "涉及'是否免费'的问题必须谨慎：公区设施一般由物业负责，业主专有部分通常由业主承担费用。",
    "紧急安全问题（燃气泄漏、火灾、触电）立即建议拨打 119/120/燃气公司电话，并告知已转人工紧急处理。",
    "邻里纠纷、责任争议、费用争议直接转人工，不自行判定。",
    "当前页面默认业主是 3-2-1201 的王先生。如果用户没有明确提供房号，创建工单时默认使用房号 3-2-1201。",
    "创建工单前必须向用户展示摘要并等待用户确认。",
    "工单创建成功后，必须告知工单号和预计处理时间。",
    "查询工单时调用 query_work_order 工具。",
    "回复简洁专业，关键信息高亮，使用中文。",
]


def create_maintenance_agent(tools: Optional[List[Any]] = None, model: Optional[Any] = None) -> Agent:
    agent_tools = _base_tools()
    if tools:
        agent_tools.extend(tools)
    return Agent(
        id="maintenance_agent",
        name="维修 Agent",
        description="处理维修报修、工单创建与查询。",
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


maintenance_agent = create_maintenance_agent()
