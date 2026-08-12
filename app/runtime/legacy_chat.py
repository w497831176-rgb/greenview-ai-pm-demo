"""Owner chat HTTP/SSE transport and human-collaboration endpoints.

Runtime business authority belongs exclusively to
:class:`app.runtime.coordinator.RuntimeCoordinator`. This module preserves the
public wire protocol, history/feedback APIs and deterministic handoff APIs; it
contains no Router, Agent, Skill, RAG, MCP, action or cost execution path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.work_order_workflow import decide_work_order_proposal

from app.handoff_policy import evaluate_handoff_policy
from db.property_db import (
    add_badcase_action,
    cancel_handoff,
    claim_handoff,
    close_handoff,
    create_badcase,
    create_chat_feedback,
    create_chat_session,
    get_chat_feedback,
    get_chat_message,
    get_chat_session,
    get_handoff_package,
    get_previous_user_message,
    get_user_feedback_badcase,
    link_chat_feedback_badcase,
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
logger = logging.getLogger(__name__)


@dataclass
class _BackgroundStreamRun:
    """One accepted owner bubble whose execution outlives its SSE consumer."""

    session_id: str
    queue: asyncio.Queue[object]
    task: Optional[asyncio.Task[None]] = None
    consumer_attached: bool = True


@dataclass(frozen=True)
class _StreamFailure:
    error: Exception


_STREAM_END = object()
_ACTIVE_STREAM_RUNS: Dict[str, _BackgroundStreamRun] = {}


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


class WorkOrderProposalDecisionRequest(BaseModel):
    session_id: str
    proposal_id: str
    decision: str


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: int
    reason: Optional[str] = None
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


def _observe_stream_producer(
    run: _BackgroundStreamRun,
    task: asyncio.Task[None],
) -> None:
    """Release the strong reference and retrieve every producer outcome."""

    try:
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error(
                    "Background chat producer failed for session %s",
                    run.session_id,
                    exc_info=(type(error), error, error.__traceback__),
                )
    finally:
        if _ACTIVE_STREAM_RUNS.get(run.session_id) is run:
            _ACTIVE_STREAM_RUNS.pop(run.session_id, None)


async def _produce_stream(
    run: _BackgroundStreamRun,
    message: str,
    user_id: str,
) -> None:
    """Own and exhaust the governed runtime independently of HTTP delivery."""

    try:
        async for frame in _stream_agent_response(message, run.session_id, user_id):
            if run.consumer_attached:
                run.queue.put_nowait(frame)
    except asyncio.CancelledError:
        # Only process shutdown should cancel this task. A disconnected SSE
        # consumer never calls cancel() or aclose() on the runtime producer.
        raise
    except Exception as exc:
        if run.consumer_attached:
            run.queue.put_nowait(_StreamFailure(exc))
        raise
    finally:
        if run.consumer_attached:
            run.queue.put_nowait(_STREAM_END)


def _start_stream_run(
    message: str,
    session_id: str,
    user_id: str,
) -> _BackgroundStreamRun:
    """Atomically accept one active producer per Session."""

    if _ACTIVE_STREAM_RUNS.get(session_id) is not None:
        raise HTTPException(
            status_code=409,
            detail="该会话已有回答正在后台生成，请等待完成后再发送。",
        )

    run = _BackgroundStreamRun(session_id=session_id, queue=asyncio.Queue())
    _ACTIVE_STREAM_RUNS[session_id] = run
    task = asyncio.create_task(
        _produce_stream(run, message, user_id),
        name=f"chat-producer:{session_id}",
    )
    run.task = task
    task.add_done_callback(lambda done: _observe_stream_producer(run, done))
    return run


async def _consume_stream_run(run: _BackgroundStreamRun) -> AsyncIterator[str]:
    """Forward queued frames; detaching never cancels the accepted run."""

    try:
        while True:
            item = await run.queue.get()
            if item is _STREAM_END:
                return
            if isinstance(item, _StreamFailure):
                raise item.error
            yield str(item)
    finally:
        run.consumer_attached = False


class _DetachedStreamingResponse(StreamingResponse):
    """Detach delivery on every ASGI exit without touching the producer."""

    def __init__(self, run: _BackgroundStreamRun) -> None:
        self._run = run
        super().__init__(
            _consume_stream_run(run),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def stream_response(self, send: Any) -> None:
        try:
            await super().stream_response(send)
        finally:
            # ASGI 2.4 reports disconnect at send(), outside body_iterator.
            # Mark only the delivery side detached; never cancel/close runtime.
            self._run.consumer_attached = False


def _streaming_response(run: _BackgroundStreamRun) -> StreamingResponse:
    return _DetachedStreamingResponse(run)


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """Stream an agent response via Server-Sent Events."""

    session_id = request.session_id or f"web-{uuid.uuid4().hex[:12]}"
    user_id = request.user_id or "web-user"

    return _streaming_response(
        _start_stream_run(request.message, session_id, user_id)
    )


@router.post("/work-order-proposal/decision")
async def work_order_proposal_decision(request: WorkOrderProposalDecisionRequest):
    """Apply the existing card's explicit confirm/cancel button without a model."""

    try:
        result = decide_work_order_proposal(
            session_id=request.session_id,
            proposal_id=request.proposal_id,
            decision=request.decision,
            actor=f"owner:{request.session_id}",
        )
        return {"status": "ok", **result, "model_invoked": False}
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/stream")
async def chat_stream_get(
    message: str,
    session_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
):
    """GET variant of the stream endpoint (useful for quick testing with curl)."""

    session_id = session_id or f"web-{uuid.uuid4().hex[:12]}"
    user_id = user_id or "web-user"

    return _streaming_response(_start_stream_run(message, session_id, user_id))


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
    """Persist message feedback; only negative feedback enters Badcase."""
    feedback_type = str(request.type or "thumb_down").strip()
    if feedback_type not in {"thumb_up", "thumb_down"}:
        raise HTTPException(status_code=400, detail="反馈类型仅支持 thumb_up 或 thumb_down")
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    reason = (request.reason or "").strip()
    if feedback_type == "thumb_up":
        reason = reason or "回答有帮助"
    else:
        reason = reason or "未提供具体原因"

    msg = get_chat_message(request.message_id)
    if not msg:
        raise HTTPException(status_code=404, detail="反馈目标消息不存在")
    if msg.get("session_id") != request.session_id:
        raise HTTPException(status_code=409, detail="反馈消息不属于当前会话")
    if msg.get("role") != "assistant":
        raise HTTPException(status_code=409, detail="只能反馈 AI 回答")
    if feedback_type == "thumb_down" and not msg.get("trace_id"):
        raise HTTPException(status_code=409, detail="该历史回答缺少 Trace，无法形成可验证 Badcase")

    existing = get_chat_feedback(request.session_id, request.message_id)
    if existing:
        if existing.get("feedback_type") != feedback_type:
            raise HTTPException(status_code=409, detail="该回答已记录另一种反馈，不能重复改写")
        badcase = (
            get_user_feedback_badcase(request.session_id, request.message_id)
            if existing.get("badcase_id")
            else None
        )
        return {
            "status": "ok",
            "feedback": existing,
            "badcase": badcase,
            "already_recorded": True,
            "message": "该反馈已记录，未重复创建 Badcase",
        }

    user_msg = get_previous_user_message(request.session_id, request.message_id)

    original_query = ""
    ai_response = ""
    trace_id = None
    context_json: Dict[str, Any] = {
        "session_id": request.session_id,
        "message_id": request.message_id,
        "feedback_type": feedback_type,
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

    legacy_badcase = get_user_feedback_badcase(request.session_id, request.message_id)
    if feedback_type == "thumb_up" and legacy_badcase:
        raise HTTPException(status_code=409, detail="该回答已有历史负向反馈，不能改写为正向反馈")

    if feedback_type == "thumb_up":
        feedback = create_chat_feedback(
            session_id=request.session_id,
            message_id=request.message_id,
            feedback_type=feedback_type,
            reason=reason,
            trace_id=trace_id,
        )
        return {
            "status": "ok",
            "feedback": feedback,
            "badcase": None,
            "already_recorded": False,
            "message": "感谢反馈；点赞已记录，不会创建 Badcase",
        }

    if legacy_badcase:
        feedback = create_chat_feedback(
            session_id=request.session_id,
            message_id=request.message_id,
            feedback_type=feedback_type,
            reason=reason,
            trace_id=trace_id,
            badcase_id=legacy_badcase["id"],
        )
        return {
            "status": "ok",
            "feedback": feedback,
            "badcase": legacy_badcase,
            "already_recorded": True,
            "message": "已关联原有 Badcase，未重复创建",
        }

    badcase = create_badcase(
        title=f"人工反馈：{reason[:40]}",
        description=reason,
        category="pending",
        status="pending",
        created_at=now_cn(),
        evidence=reason,
        source_message_id=request.message_id,
        session_id=request.session_id,
        source="user_feedback",
        original_query=original_query,
        ai_response=ai_response,
        feedback_reason=reason,
        context_json=json.dumps(context_json, ensure_ascii=False, default=str),
        trace_id=trace_id,
        priority="high",
        message_id=request.message_id,
    )
    add_badcase_action(
        badcase_id=badcase["id"],
        action_type="user_feedback",
        action_detail=json.dumps(
            {
                "reason": reason,
                "type": feedback_type,
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
    feedback = create_chat_feedback(
        session_id=request.session_id,
        message_id=request.message_id,
        feedback_type=feedback_type,
        reason=reason,
        trace_id=trace_id,
        badcase_id=badcase["id"],
    )
    if not feedback.get("badcase_id"):
        feedback = link_chat_feedback_badcase(
            request.session_id, request.message_id, badcase["id"]
        ) or feedback
    return {
        "status": "ok",
        "feedback": feedback,
        "badcase": badcase,
        "already_recorded": False,
        "message": "负向反馈已记录并关联一个 Badcase",
    }


@router.post("/handoff")
async def chat_handoff(request: HandoffRequest):
    """Retired owner shortcut; owner bubbles must pass the one Router."""

    raise HTTPException(
        status_code=410,
        detail="owner Handoff requests must be sent through /api/chat/stream",
    )


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
