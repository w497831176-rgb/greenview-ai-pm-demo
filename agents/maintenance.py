"""Maintenance vertical Agent for the YIAI property interview demo."""

from typing import Any, List, Optional

from agno.agent import Agent

from app.settings import MODEL, agent_db
from tools.knowledge import KnowledgeTools

try:
    from tools.bing_search import BingSearchTools
except ImportError:
    BingSearchTools = None


def _base_tools() -> List[Any]:
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
    try:
        base_tools.append(KnowledgeTools())
    except ImportError:
        pass
    return base_tools


INSTRUCTIONS = [
    "你是 YIAI 物业的维修 Agent，处理维修报修、工单查询和维修准备建议。",
    "先识别是知识解释、事实查询、工单草稿还是人工协同；不要把不同动作混在一起。",
    "报修草稿需要房号、问题描述、紧急程度、联系电话和预约时间；缺失时每轮最多追问 1-2 项。",
    "正式工单仅由服务端会话工作流创建：没有真实工单号时，绝不能说‘已经创建’或‘已通知师傅’。",
    "用户明确确认创建前，只能说明待确认草稿；确认后只能依据系统返回的真实工单号说明成功。",
    "涉及收费、责任、服务承诺时必须依据知识库证据；未命中时如实说需要人工确认。",
    "紧急安全风险（燃气泄漏、火灾、触电）优先提示 119/120/燃气公司和人工协同，不自行作安全结论。",
    "天气问题只使用已绑定 weather-server 的允许工具。天气 Server 是演示固定样例，不得说成实时互联网天气。",
    "当前演示业主工单明细只使用 get_my_recent_work_orders、count_my_open_work_orders 或 get_my_work_order_by_id。",
    "全小区 count_work_orders 只返回脱敏聚合数；回答时必须说明它不是其他业主工单明细。",
    "工具结果 success 才能作为事实；empty/not_found 只能表示未匹配；invalid_input/unauthorized 要解释范围；timeout/upstream_error 要说结果未确认。",
    "回复中文、简洁、专业；所有工具调用及其结果状态由页面 Trace 保留。",
]


def create_maintenance_agent(
    tools: Optional[List[Any]] = None,
    model: Optional[Any] = None,
    instructions: Optional[List[str]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Agent:
    agent_tools = _base_tools()
    if tools:
        agent_tools.extend(tools)
    final_instructions = INSTRUCTIONS.copy()
    if instructions:
        final_instructions.extend(instructions)
    return Agent(
        id="maintenance_agent",
        name=name or "维修 Agent",
        description=description or "处理维修报修、工单创建与查询。",
        model=model or MODEL,
        db=agent_db,
        tools=agent_tools,
        skills=None,
        instructions=final_instructions,
        add_datetime_to_context=True,
        add_history_to_context=True,
        read_chat_history=True,
        num_history_runs=2,
        markdown=True,
    )


maintenance_agent = create_maintenance_agent()
