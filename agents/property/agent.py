"""
Property Agent
==============

AI assistant for property maintenance work orders.
Replaces the generic wechatbot agent in the web chat context.
"""

from pathlib import Path
from typing import Any, List, Optional

from agno.agent import Agent

from agents.property.instructions import PROPERTY_INSTRUCTIONS
from app.settings import MODEL, agent_db
from tools.knowledge import KnowledgeTools
from tools.work_order import WorkOrderTools

try:
    from tools.bing_search import BingSearchTools
except ImportError:
    BingSearchTools = None

# ---------------------------------------------------------------------------
# Skills (lazy-loaded from local + enterprise directories)
# ---------------------------------------------------------------------------
skill_loaders = []
try:
    from agno.skills import LocalSkills, Skills

    local_skills_path = Path(__file__).parent / "skills"
    enterprise_skills_path = Path("/app/enterprise/skills")

    if local_skills_path.exists():
        skill_loaders.append(LocalSkills(str(local_skills_path)))
    if enterprise_skills_path.exists():
        skill_loaders.append(LocalSkills(str(enterprise_skills_path)))

    skills = Skills(loaders=skill_loaders) if skill_loaders else None
except ImportError:
    skills = None


def _base_tools() -> List[Any]:
    """Return the base tool set for the property agent."""
    base_tools: List[Any] = []

    try:
        from agno.tools.calculator import CalculatorTools

        base_tools.append(CalculatorTools())
    except ImportError:
        pass

    try:
        if BingSearchTools is not None:
            base_tools.append(BingSearchTools())
    except ImportError:
        pass

    try:
        from agno.tools.reasoning import ReasoningTools

        base_tools.append(ReasoningTools())
    except ImportError:
        pass

    # Work order tools: create, query, update
    try:
        base_tools.append(WorkOrderTools())
    except ImportError:
        pass

    # Knowledge base tools
    try:
        base_tools.append(KnowledgeTools())
    except ImportError:
        pass

    return base_tools


# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
def create_property_agent(tools: Optional[List[Any]] = None, model: Optional[Any] = None) -> Agent:
    """Create a fresh property agent instance, optionally with extra tools."""
    agent_tools = _base_tools()
    if tools:
        agent_tools.extend(tools)

    return Agent(
        id="property_agent",
        name="YIAI物业 AI 助手",
        description="AI 物业维修助手，帮助业主报修、查询工单、解答维修相关问题。",
        model=model or MODEL,
        db=agent_db,
        tools=agent_tools,
        skills=skills,
        instructions=PROPERTY_INSTRUCTIONS,
        add_datetime_to_context=True,
        add_history_to_context=True,
        read_chat_history=True,
        num_history_runs=2,
        markdown=True,
    )


# Backwards-compatible module-level agent (without dynamic MCP tools).
property_agent = create_property_agent()
