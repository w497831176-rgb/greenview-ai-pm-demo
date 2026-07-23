"""Thin owner-chat transport adapter.

All compatibility endpoints and the SSE wire format remain in the isolated
legacy adapter during the V1.8 migration.  Runtime authority is selected once
there and delegated to :class:`app.runtime.coordinator.RuntimeCoordinator`.
No routing, RAG, Skill, Tool, business-write, citation or cost decisions live
in this transport module.
"""

from typing import AsyncIterator

from app.runtime.legacy_chat import (
    _policy_mcp_args,
    _stream_agent_response,
    _unique_rag_results,
    router,
)


async def stream_chat_response(
    message: str,
    session_id: str,
    user_id: str,
) -> AsyncIterator[str]:
    """Public in-process adapter for chat retest and evaluation consumers."""
    async for chunk in _stream_agent_response(
        message,
        session_id,
        user_id,
    ):
        yield chunk


__all__ = [
    "router",
    "stream_chat_response",
    "_policy_mcp_args",
    "_unique_rag_results",
]
