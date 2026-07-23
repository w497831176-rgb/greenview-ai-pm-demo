"""Thin owner-chat transport adapter.

The SSE wire format and compatibility endpoints remain stable while all
business execution delegates to :class:`app.runtime.coordinator.RuntimeCoordinator`.
No routing, RAG, Skill, Tool, business-write, citation or cost decisions live
in this transport module.
"""

from typing import AsyncIterator

from app.runtime.legacy_chat import _stream_agent_response, router


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


__all__ = ["router", "stream_chat_response"]
