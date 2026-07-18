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
from app.mcp_contracts import router as mcp_contracts_router
from app.multimodal import router as multimodal_router
from app.mcp import discover_all_mcp_tools, router as mcp_router
from app.model_configs import router as model_configs_router
from app.models_compat import router as models_compat_router
from app.observability import router as observability_router
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
    # Canonical agents are already ensured by db.property_db._migrate_runtime_contract.
    # Discover and cache built-in MCP server tools.
    try:
        await discover_all_mcp_tools()
    except Exception:
        import traceback
        traceback.print_exc()
    yield




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
app.include_router(mcp_contracts_router)
app.include_router(multimodal_router)
# Compatibility layer for /api/models/* (frontend) and /api/model-configs/{model_id}/* (test cases).
app.include_router(models_compat_router)
app.include_router(model_configs_router)
app.include_router(agents_router)
app.include_router(observability_router)
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
