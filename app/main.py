"""YIAI物业 V1.8 enterprise runtime convergence demo."""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agno.os import AgentOS

from app.agents import router as agents_router
from app.badcases import router as badcases_router
from app.evaluations import router as evaluations_router
from app.chat import router as chat_router
from app.knowledge import retrieval_router, router as knowledge_router
from app.mcp_contracts import router as mcp_contracts_router
from app.mcp import discover_all_mcp_tools, router as mcp_router
from app.model_configs import router as model_configs_router
from app.models_compat import router as models_compat_router
from app.observability import router as observability_router
from app.runtime.agent_factory import runtime_agent_factory
from app.runtime.api import router as runtime_router
from app.runtime.release_compiler import ensure_bootstrap_release
from app.runtime.workflow_factory import runtime_workflow_factory
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
    # Compile and publish exactly one bootstrap release only when no published
    # release exists.  Existing platform configuration is never overwritten.
    await asyncio.to_thread(ensure_bootstrap_release)
    yield




# ---------------------------------------------------------------------------
# Create AgentOS
# ---------------------------------------------------------------------------
agent_os = AgentOS(
    name="YIAI物业 V1.8",
    tracing=True,
    scheduler=False,
    authorization=False,
    lifespan=lifespan,
    db=agent_db,
    agents=[runtime_agent_factory],
    workflows=[runtime_workflow_factory],
)

app = agent_os.get_app()


def _disabled_agentos_direct_surface(method: str, path: str) -> Optional[str]:
    """Identify unused direct AgentOS execution surfaces.

    The product frontend and internal services use the audited ``/api``
    capabilities.  Direct AgentOS model execution therefore stays disabled,
    while read-only GET history routes remain available.
    """
    if str(method).upper() != "POST":
        return None
    segments = [segment for segment in str(path).split("/") if segment]
    if segments == ["eval-runs"]:
        return "agentos_builtin_eval"
    if segments == ["optimize-memories"]:
        return "agentos_memory_optimization"
    if segments == ["agents", "runtime-agent", "runs"] or (
        len(segments) == 5
        and segments[:3] == ["agents", "runtime-agent", "runs"]
        and segments[-1] in {"continue", "resume"}
    ):
        return "agentos_direct_agent_run"
    if segments == ["workflows", "yiai-runtime", "runs"] or (
        len(segments) == 5
        and segments[:3] == ["workflows", "yiai-runtime", "runs"]
        and segments[-1] in {"continue", "resume"}
    ):
        return "agentos_direct_workflow_run"
    return None


@app.middleware("http")
async def _disabled_agentos_direct_surface_middleware(request: Request, call_next):
    """Return a stable 410 before AgentOS can dispatch an unused model run."""
    disabled_surface = _disabled_agentos_direct_surface(
        request.method, request.url.path
    )
    if disabled_surface:
        return JSONResponse(
            status_code=410,
            content={
                "detail": {
                    "code": "agentos_builtin_model_surface_disabled",
                    "surface": disabled_surface,
                    "message": "该框架模型入口未在本演示启用，请使用平台受审计的对应能力",
                }
            },
        )
    return await call_next(request)

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
# Compatibility layer for /api/models/* (frontend) and /api/model-configs/{model_id}/* (test cases).
app.include_router(models_compat_router)
app.include_router(model_configs_router)
app.include_router(agents_router)
app.include_router(observability_router)
app.include_router(evaluations_router)
app.include_router(runtime_router)
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
