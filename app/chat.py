"""Thin owner-chat transport adapter.

All compatibility endpoints and the SSE wire format remain in the isolated
legacy adapter during the V1.8 migration.  Runtime authority is selected once
there and delegated to :class:`app.runtime.coordinator.RuntimeCoordinator`.
No routing, RAG, Skill, Tool, business-write, citation or cost decisions live
in this transport module.
"""

from app.runtime.legacy_chat import _policy_mcp_args, _unique_rag_results, router

__all__ = ["router", "_policy_mcp_args", "_unique_rag_results"]
