"""Owner chat HTTP/SSE transport and human-collaboration endpoints.

Runtime business authority belongs exclusively to
:class:`app.runtime.coordinator.RuntimeCoordinator`. This module preserves the
public wire protocol, history/feedback APIs and deterministic handoff APIs; it
contains no Router, Agent, Skill, RAG, MCP, action or cost execution path.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.handoff_policy import evaluate_handoff_policy
from db.property_db import (
    add_badcase_action,
    cancel_handoff,
    claim_handoff,
    close_handoff,
    create_badcase,
    create_chat_session,
    get_chat_message,
    get_chat_session,
    get_handoff_package,
    get_previous_user_message,
    list_chat_messages,
    list_handoff_sessions,
    list_user_chat_sessions,
    now_cn,
    request_handoff,
    resolve_handoff,
    save_chat_message,
    wait_for_handoff_user,
)


router = APIRouter(prefix="/api/chat", tags=["chat"])


def _latest_ai_evidence(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract the last AI turn's verifiable evidence for a staff handoff."""
    for item in reversed(messages):
        if item.get("role") != "assistant":
            continue
        return {
            "message_id": item.get("id"),
            "trace_id": item.get("trace_id"),
            "route": {
                "intent": item.get("route_intent"),
                "reason": item.get("route_reason"),
                "agent": item.get("current_agent"),
                "agent_id": item.get("current_agent_id"),
            },
            "skills": item.get("activated_skills") or [],
            "tools": item.get("tool_calls") or [],
            "mcp_calls": item.get("mcp_calls") or [],
            "citations": item.get("citations") or [],
            "model": {
                "model_id": item.get("model_id"),
                "token_count": item.get("token_count"),
                "token_detail": item.get("token_detail"),
                "usage_source": item.get("usage_source"),
            },
        }
    return {"skills": [], "tools": [], "mcp_calls": [], "citations": []}


def _build_handoff_package(
    session_id: str,
    policy: Dict[str, Any],
    *,
    trace_id: Optional[str] = None,
    trigger_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the compact, inspectable context package shown to staff.

    No LLM summary is generated here.  The package preserves the original
    wording and the facts/evidence that were actually present in the session.
    """
    messages = list_chat_messages(session_id)
    latest_owner = next((m for m in reversed(messages) if m.get("role") in {"user", "owner"}), None)
    evidence = _latest_ai_evidence(messages)
    context = [
        {"role": m.get("role"), "content": m.get("content"), "created_at": m.get("created_at")}
        for m in messages[-8:]
    ]
    verified = []
    for call in evidence.get("mcp_calls") or []:
        if str(call.get("status") or "").lower() == "success":
            verified.append({"type": "mcp", "name": call.get("tool_name") or call.get("server_name"), "summary": call.get("result_summary")})
    for citation in evidence.get("citations") or []:
        verified.append({"type": "rag", "name": citation.get("doc_title"), "chunk_index": citation.get("chunk_index")})
    return {
        "version": "v1.5.8",
        "generated_at": now_cn(),
        "session_id": session_id,
        "owner_request": {
            "content": trigger_message or (latest_owner or {}).get("content") or "",
            "message_id": (latest_owner or {}).get("id"),
        },
        "recent_context": context,
        "ai_evidence": evidence,
        "verified_facts": verified,
        "risk": policy,
        "human_task": policy.get("human_task"),
        "trigger_trace_id": trace_id or evidence.get("trace_id"),
    }


def _request_handoff_with_context(
    session_id: str,
    policy: Dict[str, Any],
    *,
    trace_id: Optional[str] = None,
    trigger_message: Optional[str] = None,
    actor: str = "owner",
) -> Dict[str, Any]:
    package = _build_handoff_package(session_id, policy, trace_id=trace_id, trigger_message=trigger_message)
    return request_handoff(
        session_id,
        policy.get("reason") or "需要人工处理",
        risk_level=policy.get("level") or "L3",
        reason_code=policy.get("reason_code") or "owner_requested",
        queue=policy.get("queue") or "property_service",
        handoff_package=package,
        actor=actor,
    )


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: Optional[int] = None
    reason: str
    type: Optional[str] = "thumb_down"  # thumb_up / thumb_down


class HandoffRequest(BaseModel):
    session_id: str
    reason: Optional[str] = None


class HandoffReplyRequest(BaseModel):
    session_id: str
    staff_name: str
    message: str


class HandoffClaimRequest(BaseModel):
    session_id: str
    staff_name: str


class HandoffWaitForOwnerRequest(BaseModel):
    session_id: str
    staff_name: str
    message: str


class HandoffResolveRequest(BaseModel):
    session_id: str
    resolution: Optional[str] = None
    staff_name: Optional[str] = None
    create_badcase: bool = False


class HandoffCloseRequest(BaseModel):
    session_id: str
    staff_name: Optional[str] = None


class HandoffCancelRequest(BaseModel):
    session_id: str
    reason: Optional[str] = None


class HandoffPolicyDiagnosticRequest(BaseModel):
    message: str
    mcp_calls: Optional[List[Dict[str, Any]]] = None


async def _stream_agent_response(
    message: str,
    session_id: str,
    user_id: str,
) -> AsyncIterator[str]:
    """Delegate the public SSE stream to the single V1.8 runtime authority."""
    from app.runtime.coordinator import RuntimeCoordinator

    async for event in RuntimeCoordinator().stream(message, session_id, user_id):
        yield event
    # Keep the semantic terminal event away from the physical end of the HTTP
    # response. Synology's TLS reverse proxy has been observed to discard the
    # final upstream chunk even with proxy_buffering disabled. SSE comments are
    # ignored by clients, so this padding absorbs that transport quirk without
    # inventing another business event.
    yield ": transport-flush " + (" " * 4096) + "\n\n"


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """Stream an agent response via Server-Sent Events."""

    session_id = request.session_id or f"web-{uuid.uuid4().hex[:12]}"
    user_id = request.user_id or "web-user"

    return StreamingResponse(
        _stream_agent_response(request.message, session_id, user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/stream")
async def chat_stream_get(
    message: str,
    session_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
):
    """GET variant of the stream endpoint (useful for quick testing with curl)."""

    session_id = session_id or f"web-{uuid.uuid4().hex[:12]}"
    user_id = user_id or "web-user"

    return StreamingResponse(
        _stream_agent_response(message, session_id, user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history")
async def chat_history(
    session_id: str = Query(..., description="Chat session id"),
):
    """Return persisted chat messages for a session."""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    messages = list_chat_messages(session_id)
    session = get_chat_session(session_id)
    return {"messages": messages, "session": session}


@router.get("/sessions")
async def chat_sessions(
    user_id: Optional[str] = Query(None, description="Optional user id"),
    limit: int = Query(100, ge=1, le=500),
):
    """Return recent chat sessions with last activity metadata."""
    sessions = list_user_chat_sessions(user_id=user_id, limit=limit)
    return {"sessions": sessions}


@router.post("/sessions")
async def create_new_session(
    user_id: Optional[str] = Query(None, description="Optional user id"),
):
    """Create a new chat session and return its session_id."""
    session = create_chat_session(user_id=user_id)
    return {"session": session}


@router.get("/sessions/{session_id}")
async def chat_session_detail(session_id: str):
    """Return a single chat session by id."""
    session = get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session}


@router.post("/feedback")
async def chat_feedback(request: FeedbackRequest):
    """Create a badcase from user feedback on an AI response."""
    if not request.reason or not request.reason.strip():
        raise HTTPException(status_code=400, detail="反馈描述不能为空")

    reason = request.reason.strip()
    msg = None
    user_msg = None
    if request.message_id:
        msg = get_chat_message(request.message_id)
        if msg:
            user_msg = get_previous_user_message(request.session_id, request.message_id)

    original_query = ""
    ai_response = ""
    trace_id = None
    context_json: Dict[str, Any] = {
        "session_id": request.session_id,
        "message_id": request.message_id,
        "feedback_type": request.type,
    }
    if msg:
        ai_response = msg.get("content") or ""
        trace_id = msg.get("trace_id")
        context_json.update({
            "route_intent": msg.get("route_intent"),
            "route_reason": msg.get("route_reason"),
            "current_agent": msg.get("current_agent"),
            "activated_skills": msg.get("activated_skills"),
            "citations": msg.get("citations"),
            "tool_calls": msg.get("tool_calls"),
            "mcp_calls": msg.get("mcp_calls"),
            "model_id": msg.get("model_id"),
            "model_selection_reason": msg.get("model_selection_reason"),
            "token_count": msg.get("token_count"),
            "token_detail": msg.get("token_detail"),
            "usage_source": msg.get("usage_source"),
            "latency_ms": msg.get("latency_ms"),
            "thinking_enabled": msg.get("thinking_enabled"),
            "trace_id": trace_id,
        })
    if user_msg:
        original_query = user_msg.get("content") or ""
        context_json["user_message_id"] = user_msg.get("id")

    source = "user_feedback" if request.type == "thumb_down" else "manual"
    action_type = "user_feedback" if request.type == "thumb_down" else "manual_feedback"
    badcase = create_badcase(
        title=f"人工反馈：{reason[:40]}",
        description=reason,
        category="pending",
        status="pending",
        created_at=now_cn(),
        evidence=reason,
        source_message_id=request.message_id,
        session_id=request.session_id,
        source=source,
        original_query=original_query,
        ai_response=ai_response,
        feedback_reason=reason,
        context_json=json.dumps(context_json, ensure_ascii=False, default=str),
        trace_id=trace_id,
        priority="high" if request.type == "thumb_down" else "medium",
        message_id=request.message_id,
    )
    add_badcase_action(
        badcase_id=badcase["id"],
        action_type=action_type,
        action_detail=json.dumps(
            {
                "reason": reason,
                "type": request.type,
                "query": original_query,
                "response": ai_response,
            },
            ensure_ascii=False,
            default=str,
        ),
        status_before="pending",
        status_after="pending",
        created_by="owner",
    )
    return {"status": "ok", "badcase": badcase, "source": source}


@router.post("/handoff")
async def chat_handoff(request: HandoffRequest):
    """Owner-facing explicit transfer request with an inspectable context package."""
    reason = (request.reason or "业主主动请求人工服务").strip()
    policy = evaluate_handoff_policy("", explicit_reason=reason)
    session = _request_handoff_with_context(request.session_id, policy, actor="owner")
    return {"status": "ok", "session": session, "policy": policy, "package_available": True}


@router.post("/handoff-policy")
async def chat_handoff_policy(request: HandoffPolicyDiagnosticRequest):
    """Explain the deterministic collaboration boundary without calling a model."""
    return {"policy": evaluate_handoff_policy(request.message, mcp_calls=request.mcp_calls)}


@router.get("/handoffs")
async def chat_handoffs(
    status: Optional[str] = Query(None, description="Filter by handoff status"),
    include_completed: bool = Query(False, description="Include closed/cancelled sessions"),
):
    """List actionable human-copilot sessions and their responsibility state."""
    sessions = list_handoff_sessions(status=status, include_completed=include_completed)
    return {"sessions": sessions}


@router.get("/handoff/{session_id}/package")
async def chat_handoff_package(session_id: str):
    try:
        return get_handoff_package(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/handoff-claim")
async def chat_handoff_claim(request: HandoffClaimRequest):
    if not request.staff_name or not request.staff_name.strip():
        raise HTTPException(status_code=400, detail="工作人员姓名不能为空")
    try:
        session = claim_handoff(request.session_id, request.staff_name.strip())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"status": "ok", "session": session}


@router.post("/handoff-reply")
async def chat_handoff_reply(request: HandoffReplyRequest):
    """Staff sends a human reply into a chat session."""
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="回复内容不能为空")

    current = get_chat_session(request.session_id)
    if current is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if current.get("handoff_status") in {"requested", "waiting_user"}:
        try:
            claim_handoff(request.session_id, request.staff_name)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
    elif current.get("handoff_status") != "active":
        raise HTTPException(status_code=409, detail="当前状态不允许人工回复")

    save_chat_message(
        session_id=request.session_id,
        role="staff",
        content=request.message.strip(),
    )
    messages = list_chat_messages(request.session_id)
    session = get_chat_session(request.session_id)
    return {"status": "ok", "messages": messages, "session": session}


@router.post("/handoff-waiting-user")
async def chat_handoff_waiting_user(request: HandoffWaitForOwnerRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="请说明需要业主补充的信息")
    try:
        session = wait_for_handoff_user(request.session_id, request.staff_name.strip() or "物业工作人员", request.message.strip())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    save_chat_message(session_id=request.session_id, role="staff", content=request.message.strip())
    return {"status": "ok", "session": session}


@router.post("/handoff-resolve")
async def chat_handoff_resolve(request: HandoffResolveRequest):
    """Record a human result; it remains reviewable until explicitly closed."""
    staff_name = (request.staff_name or "物业工作人员").strip()
    try:
        session = resolve_handoff(request.session_id, request.resolution, staff_name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if request.resolution:
        save_chat_message(session_id=request.session_id, role="staff", content=request.resolution.strip())
    badcase = None
    if request.create_badcase:
        package = get_handoff_package(request.session_id)
        badcase = create_badcase(
            title="人工协同需复盘",
            description=request.resolution or "工作人员标记该人工协同需要沉淀为 Badcase。",
            category="response_quality",
            status="pending",
            created_at=now_cn(),
            evidence=json.dumps(package.get("package") or {}, ensure_ascii=False, default=str),
            session_id=request.session_id,
            source="human_handoff",
            original_query=((package.get("package") or {}).get("owner_request") or {}).get("content") or "",
            feedback_reason=request.resolution or "人工协同复盘",
            context_json=json.dumps(package, ensure_ascii=False, default=str),
            priority="medium",
        )
        add_badcase_action(
            badcase_id=badcase["id"], action_type="human_handoff_outcome", action_detail=json.dumps({"session_id": request.session_id}, ensure_ascii=False),
            status_before="pending", status_after="pending", created_by=staff_name,
        )
    return {"status": "ok", "session": session, "badcase": badcase}


@router.post("/handoff-close")
async def chat_handoff_close(request: HandoffCloseRequest):
    try:
        session = close_handoff(request.session_id, (request.staff_name or "物业工作人员").strip())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"status": "ok", "session": session}


@router.post("/handoff-cancel")
async def chat_handoff_cancel(request: HandoffCancelRequest):
    try:
        session = cancel_handoff(request.session_id, "owner", request.reason)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"status": "ok", "session": session}
