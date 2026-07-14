"""
YIAI物业 V1.2｜AI 智能客服与工单协同原型
============================================

物业场景下可运行的最小 AgentOS 入口。
仅挂载 V1.2 需要的 Property Agent、业务 API 与 RAG 能力。
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware

from agno.os import AgentOS
from agno.os.config import AuthorizationConfig

from agents.property import property_agent
from app.agents import router as agents_router
from app.badcases import router as badcases_router
from app.chat import router as chat_router
from app.knowledge import retrieval_router, router as knowledge_router
from app.mcp import discover_all_mcp_tools, router as mcp_router
from app.model_configs import router as model_configs_router
from app.models_compat import router as models_compat_router
from app.skills import router as skills_router
from app.settings import RUNTIME_ENV, agent_db
from app.work_orders import router as work_orders_router
from db.property_db import init_db
from rag_store import init_vector_store


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app):  # type: ignore[no-untyped-def]
    # Initialize property demo database (work orders, knowledge docs, badcases)
    await asyncio.to_thread(init_db)
    # Initialize pgvector table for RAG chunks.
    await asyncio.to_thread(init_vector_store)
    # Seed default Router and Vertical agents for the property demo.
    await asyncio.to_thread(_seed_default_agents)
    # Discover and cache built-in MCP server tools.
    try:
        await discover_all_mcp_tools()
    except Exception:
        import traceback
        traceback.print_exc()
    yield


def _seed_default_agents() -> None:
    """Seed the default Router and Vertical agents if they do not exist."""
    from db.property_db import (
        create_agent as db_create_agent,
        get_agent_by_agent_id,
        set_agent_skills,
        set_agent_tools,
    )

    defaults = [
        {
            "agent_id": "router_agent",
            "name": "路由 Agent",
            "description": "意图分类与智能路由，决定用户请求由哪个垂直 Agent 处理。",
            "instructions": (
                "你是物业智能客服系统的路由 Agent。\n"
                "请分析用户问题，从以下意图中选择最匹配的一个：\n"
                "- maintenance: 维修、报修、设施故障、工单进度\n"
                "- billing: 物业费、缴费、账单、费用查询\n"
                "- complaint: 投诉、不满、建议\n"
                "- customer_service: 一般咨询、问候、小区规则、知识库问答\n"
                "只返回 JSON：{\"intent\": \"maintenance|billing|complaint|customer_service\", \"reason\": \"简短原因\"}"
            ),
            "category": "router",
            "enabled": True,
            "model_id": "deepseek-v4-flash",
        },
        {
            "agent_id": "maintenance_agent",
            "name": "维修 Agent",
            "description": "处理报修、维修工单创建与进度查询。",
            "instructions": (
                "你是维修服务 Agent。帮助业主报修、查询工单进度。"
                "若问题不在维修范围，应主动提出转人工。"
            ),
            "category": "vertical",
            "enabled": True,
            "model_id": "deepseek-v4-flash",
        },
        {
            "agent_id": "billing_agent",
            "name": "费用 Agent",
            "description": "处理物业费、缴费、账单咨询。",
            "instructions": (
                "你是费用咨询 Agent。帮助业主查询物业费、缴费方式、账单明细。"
                "若涉及费用争议，应主动提出转人工。"
            ),
            "category": "vertical",
            "enabled": True,
            "model_id": "deepseek-v4-flash",
        },
        {
            "agent_id": "complaint_agent",
            "name": "投诉 Agent",
            "description": "处理业主投诉与建议。",
            "instructions": (
                "你是投诉处理 Agent。认真倾听业主不满，记录投诉内容，"
                "安抚情绪并承诺人工跟进。"
            ),
            "category": "vertical",
            "enabled": True,
            "model_id": "deepseek-v4-flash",
        },
        {
            "agent_id": "customer_service_agent",
            "name": "客服 Agent",
            "description": "处理一般咨询、小区规则、知识库问答。",
            "instructions": (
                "你是通用客服 Agent。回答业主关于小区规则、公共设施、"
                "知识库等一般性问题。若无法回答，应主动提出转人工。"
            ),
            "category": "vertical",
            "enabled": True,
            "model_id": "deepseek-v4-flash",
        },
    ]

    for payload in defaults:
        if get_agent_by_agent_id(payload["agent_id"]) is None:
            agent = db_create_agent(
                agent_id=payload["agent_id"],
                name=payload["name"],
                description=payload["description"],
                instructions=payload["instructions"],
                category=payload["category"],
                enabled=payload["enabled"],
                model_id=payload["model_id"],
            )
            set_agent_skills(payload["agent_id"], [])
            set_agent_tools(payload["agent_id"], [])


# ---------------------------------------------------------------------------
# Create AgentOS
# ---------------------------------------------------------------------------
agent_os = AgentOS(
    name="YIAI物业 V1.2",
    tracing=True,
    scheduler=False,
    authorization=False,
    lifespan=lifespan,
    db=agent_db,
    agents=[property_agent],
)

app = agent_os.get_app()

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Extra routes
# ---------------------------------------------------------------------------
app.include_router(chat_router)
app.include_router(work_orders_router)
app.include_router(knowledge_router)
app.include_router(skills_router)
app.include_router(mcp_router)
# Compatibility layer for /api/models/* (frontend) and /api/model-configs/{model_id}/* (test cases).
app.include_router(models_compat_router)
app.include_router(model_configs_router)
app.include_router(agents_router)
# Badcase endpoints under both /api/badcases and /api/knowledge/badcases (frontend).
app.include_router(badcases_router, prefix="/api/badcases")
app.include_router(badcases_router, prefix="/api/knowledge/badcases")
# Retrieval endpoints under /api/retrieval (test cases).
app.include_router(retrieval_router, prefix="/api/retrieval")


if __name__ == "__main__":
    agent_os.serve(
        app="app.main:app",
        reload=RUNTIME_ENV == "dev",
    )
