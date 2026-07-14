"""
Web Chat API
============

A full-power chat endpoint for web clients.

- Uses Server-Sent Events (SSE) so long responses can stream in real time.
- No artificial length limits (the model still has its own context limits).
- Supports multi-turn sessions via session_id.
- Uses the property_agent for maintenance work order scenarios.
- Supports dynamic Skill / MCP activation from platform database.
- Supports semantic RAG retrieval with citations.
- Supports human handoff for owner escalation.
"""

import asyncio
import json
import os
import re
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.settings import MODEL_ID, USE_THINKING
from agents.billing import create_billing_agent
from agents.complaint import create_complaint_agent
from agents.customer_service import create_customer_service_agent
from agents.maintenance import create_maintenance_agent
from agents.property import create_property_agent
from agents.router import classify_intent
from db.property_db import (
    activate_handoff,
    create_badcase,
    ensure_chat_session,
    get_chat_session,
    is_handoff_active,
    is_handoff_requested,
    list_chat_messages,
    list_handoff_sessions,
    list_mcp_servers,
    list_skills,
    request_handoff,
    resolve_handoff,
    save_chat_message,
    now_cn,
)
import rag_indexer
import rag_retrieval
import skill_storage

# Optional MCP toolkit.  Import at module level so ObservableMCPTools can subclass it;
# if extras are missing the fallback disables MCP loading gracefully.
try:
    from agno.tools.mcp import MCPTools
except Exception:  # pragma: no cover
    MCPTools = None  # type: ignore


if MCPTools is not None:
    class ObservableMCPTools(MCPTools):
        """MCPTools subclass that records every tool invocation.

        Agno's streaming chunks do not surface MCP tool-call metadata, so we
        wrap each registered function's entrypoint to capture the tool name and
        arguments actually passed to the MCP server.  The recorded calls are
        then included in the SSE `done` event and persisted with the chat
        message.
        """

        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.recorded_calls: List[Dict[str, Any]] = []

        async def build_tools(self) -> None:
            await super().build_tools()
            for fn_name, fn in (getattr(self, "functions", None) or {}).items():
                original = getattr(fn, "entrypoint", None)
                if original is None:
                    continue
                wrapped = self._wrap(original, fn_name)
                fn.entrypoint = wrapped

        def _wrap(self, original: Any, fn_name: str):
            if asyncio.iscoroutinefunction(original):
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    args_dict = kwargs if kwargs else (args[0] if args else {})
                    self.recorded_calls.append({"tool_name": fn_name, "arguments": args_dict})
                    return await original(*args, **kwargs)
                return async_wrapper

            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                args_dict = kwargs if kwargs else (args[0] if args else {})
                self.recorded_calls.append({"tool_name": fn_name, "arguments": args_dict})
                return original(*args, **kwargs)
            return sync_wrapper
else:
    ObservableMCPTools = None  # type: ignore


router = APIRouter(prefix="/api/chat", tags=["chat"])

# Tokenizer fallback when the model does not report token metrics in streaming chunks.
try:
    import tiktoken

    _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    _tiktoken_encoding = None


def _estimate_tokens(text: str) -> int:
    if _tiktoken_encoding is None:
        return 0
    try:
        return len(_tiktoken_encoding.encode(text))
    except Exception:
        return 0


DEFAULT_ROOM_ID = "3-2-1201"
DEFAULT_OWNER_NAME = "王先生"


def _skill_matches_trigger(skill: Dict[str, Any], message: str) -> bool:
    """Return True when a skill's trigger condition matches the user message.

    A skill with no trigger_condition is considered globally active.

    Trigger conditions are expected to be comma/、 separated keywords or short
    phrases (e.g. "报修、查询工单、维修进度").  We clean stop words, split
    into keywords, and match with a combination of substring containment and
    character-bigram Jaccard similarity.  This handles both normal word order
    ("宠物托管") and reversed/colloquial expressions ("托管宠物").
    """
    trigger = (skill.get("trigger_condition") or "").strip()
    if not trigger:
        return True

    # Stop words commonly used in trigger descriptions but not meaningful for matching.
    stop_words = {"用户", "提到", "说到", "询问", "问题", "关于", "相关", "的", "时", "如果", "当", "要", "等"}

    def _clean(text: str) -> str:
        text = text.lower().strip()
        for sw in stop_words:
            text = text.replace(sw, "")
        return text

    def _bigrams(text: str):
        chars = [c for c in text if c.strip()]
        return set(chars[i] + chars[i + 1] for i in range(len(chars) - 1)) if len(chars) >= 2 else set(chars)

    cleaned_trigger = _clean(trigger)
    cleaned_message = _clean(message)

    # Split trigger into keywords by common separators.
    raw_keywords = [k.strip() for k in re.split(r"[,，、；;｜|\\/]+", cleaned_trigger) if k.strip()]
    # If splitting produced nothing, treat the whole trigger as one keyword.
    keywords = raw_keywords if raw_keywords else [cleaned_trigger]

    msg_bg = _bigrams(cleaned_message)

    for keyword in keywords:
        if len(keyword) < 2:
            continue
        # Exact substring match after cleaning (handles "报修" in "帮我报修...").
        if keyword in cleaned_message:
            return True
        # Bigram Jaccard similarity for reversed/variant expressions.
        key_bg = _bigrams(keyword)
        if not key_bg or not msg_bg:
            continue
        overlap = len(key_bg & msg_bg)
        union = len(key_bg | msg_bg)
        if union > 0 and overlap / union >= 0.45:
            return True

    return False


def _build_skill_context(message: str, agent_id: Optional[str] = None) -> tuple:
    """Load enabled skills bound to the current agent, filter by trigger.

    Returns (skill_context_string, activated_skill_names, skill_model_id).

    Only skills explicitly bound to the current agent are considered. The
    router agent is treated specially: it sees all enabled skills so that it
    can reason about available capabilities when classifying intent.
    """
    try:
        is_router = agent_id == "router"
        if is_router:
            candidate_skills = [s for s in list_skills() if s.get("enabled")]
        else:
            from db.property_db import get_agent_skills, get_skill

            bound_skill_ids = get_agent_skills(agent_id) if agent_id else []
            candidate_skills = [
                get_skill(int(skill_id)) for skill_id in bound_skill_ids
            ]
            candidate_skills = [s for s in candidate_skills if s and s.get("enabled")]
        if not candidate_skills:
            return "", [], None
        parts = []
        activated = []
        skill_model_id = None
        for skill in candidate_skills:
            name = skill.get("name", "")
            instructions = skill_storage.build_instructions(skill.get("id"), skill)
            trigger = skill.get("trigger_condition", "")
            if not name or not instructions:
                continue
            triggered = _skill_matches_trigger(skill, message)
            # Only skills with an explicit trigger condition that matches are
            # shown as "activated". Skills without a trigger behave as default
            # platform capabilities and stay in context but are not listed.
            if trigger and triggered:
                activated.append(name)
                # Owner-facing chat ignores any Skill model_id override.
                if skill_model_id is None and skill.get("model_id"):
                    skill_model_id = None
            if trigger and not triggered:
                continue
            header = f"【Skill：{name}】"
            if trigger:
                header += f"（触发条件：{trigger}）"
            parts.append(f"{header}\n{instructions}")
        if not parts:
            return "", [], None
        return (
            "\n\n[已启用的平台 Skill（当用户问题命中 Skill 名称或相关场景时，必须按对应 Skill 的指令回答）：\n"
            + "\n".join(parts)
            + "]"
        ), activated, skill_model_id
    except Exception:
        return "", [], None


def _build_rag_context(message: str, top_k: int = 3, threshold: Optional[float] = None) -> tuple:
    """Run advanced RAG and format retrieved chunks as context.

    Returns (rag_context_string, citations).
    """
    try:
        from db.property_db import get_retrieval_settings
        settings = get_retrieval_settings("default") or {}
        settings_payload = {
            "top_k": top_k,
            "keyword_weight": settings.get("keyword_weight", 0.3),
            "semantic_weight": settings.get("semantic_weight", 0.7),
            "rrf_k": settings.get("rrf_k", 60),
            "enable_rerank": settings.get("enable_rerank", False),
            "rerank_model": settings.get("rerank_model"),
            "score_threshold": threshold if threshold is not None else settings.get("score_threshold", 0.0),
            "context_threshold": settings.get("context_threshold", 0.2),
        }
        result = rag_retrieval.advanced_search(message, settings=settings_payload)
        results = result.get("results", [])
        if not results:
            return "", []
        parts = ["\n\n[相关知识库片段（回答时请引用出处，每条引用必须对应下方确切分片）："]
        citations = []
        for i, r in enumerate(results, 1):
            title = r.get("doc_title") or "未知文档"
            content = r.get("content", "")
            score = r.get("score", 0)
            chunk_index = r.get("chunk_index")
            parts.append(f"[引用{i}]《{title}》（分片 {chunk_index}）：{content}")
            citations.append({
                "index": i,
                "doc_title": title,
                "doc_id": r.get("doc_id"),
                "chunk_index": chunk_index,
                "content": content,
                "score": score,
            })
        parts.append("]")
        return "\n".join(parts), citations
    except Exception:
        return "", []


def _build_mcp_tools(agent_id: Optional[str] = None) -> List[Any]:
    """Load MCP servers bound to the current agent and return MCPTools instances."""
    tools = []
    try:
        is_router = agent_id == "router"
        if is_router:
            candidate_servers = [s for s in list_mcp_servers() if s.get("enabled")]
        else:
            from db.property_db import get_agent_tools, list_mcp_servers

            bound_tools = get_agent_tools(agent_id) if agent_id else []
            bound_names = {t.get("tool_name") for t in bound_tools if t.get("tool_name")}
            all_servers = {s.get("name"): s for s in list_mcp_servers() if s.get("enabled")}
            candidate_servers = [all_servers[name] for name in bound_names if name in all_servers]
        if not candidate_servers:
            return tools
        if ObservableMCPTools is None:
            return tools

        for server in candidate_servers:
            command = server.get("command")
            args = server.get("args") or []
            env = server.get("env") or {}
            name = server.get("name", "mcp-server")
            if not command:
                continue
            # Merge current process env so PATH and other vars are available.
            merged_env = {**dict(os.environ), **env}
            try:
                # Agno MCPTools stdio transport expects the full command as a single
                # string (it uses shlex.split internally); passing args separately is
                # not supported by the Toolkit constructor.
                import shlex

                full_command = command
                if args:
                    full_command = shlex.join([command] + list(args))
                tools.append(ObservableMCPTools(
                    command=full_command,
                    env=merged_env,
                    name=name,
                    transport="stdio",
                    timeout_seconds=15,
                ))
            except Exception:
                # If a single MCP server fails to initialize, log and continue.
                import traceback
                traceback.print_exc()
                continue
    except Exception:
        import traceback
        traceback.print_exc()
    return tools


def _format_mcp_context(agent_id: Optional[str] = None) -> str:
    """Format MCP servers bound to the current agent for the agent prompt."""
    try:
        is_router = agent_id == "router"
        if is_router:
            servers = [s for s in list_mcp_servers() if s.get("enabled")]
        else:
            from db.property_db import get_agent_tools, list_mcp_servers

            bound_tools = get_agent_tools(agent_id) if agent_id else []
            bound_names = {t.get("tool_name") for t in bound_tools if t.get("tool_name")}
            servers = [s for s in list_mcp_servers() if s.get("enabled") and s.get("name") in bound_names]
        if not servers:
            return ""
        parts = []
        for server in servers:
            name = server.get("name", "")
            description = server.get("description", "")
            if name:
                parts.append(f"- {name}：{description or '无描述'}")
        if not parts:
            return ""
        return (
            "\n\n[已启用的 MCP Server 工具（当用户问题涉及以下能力时，必须调用对应工具；"
            "禁止基于自身知识猜测，必须实际调用工具获取结果）：\n"
            + "\n".join(parts)
            + "]"
        )
    except Exception:
        return ""


def _detect_handoff_intent(message: str) -> Optional[str]:
    """Detect whether the owner explicitly asks for human support."""
    triggers = ["人工", "客服", "找物业", "找管家", "我要人"]
    lowered = message.lower()
    if any(t in lowered for t in triggers):
        return "业主主动要求人工服务"
    return None


def _select_agent(intent: str, tools: Optional[List[Any]] = None):
    """Return the vertical agent factory and display name for a classified intent."""
    agents = {
        "billing": (create_billing_agent, "费用 Agent"),
        "complaint": (create_complaint_agent, "投诉 Agent"),
        "customer_service": (create_customer_service_agent, "客服 Agent"),
        "maintenance": (create_maintenance_agent, "维修 Agent"),
    }
    if intent in agents:
        return agents[intent]
    # Default to the original property agent for backward compatibility.
    return create_property_agent, "物业 Agent"


def _agent_id_for_intent(intent: str) -> str:
    """Return the canonical agent_id used for Skill/MCP binding lookups."""
    return {
        "maintenance": "maintenance",
        "billing": "billing",
        "complaint": "complaint",
        "customer_service": "customer_service",
        "other": "customer_service",
    }.get(intent, "customer_service")


def _extract_tool_calls(chunk: Any) -> List[Dict[str, Any]]:
    """Extract tool call metadata from an Agno streaming chunk if available."""
    tool_calls: List[Dict[str, Any]] = []
    try:
        # Agno may expose tool calls on the chunk or its run_response.
        candidate = chunk
        if hasattr(chunk, "run_response") and chunk.run_response:
            candidate = chunk.run_response
        if hasattr(candidate, "tool_calls") and candidate.tool_calls:
            for tc in candidate.tool_calls:
                if isinstance(tc, dict):
                    tool_calls.append(_normalize_tool_call(tc))
                else:
                    tool_calls.append({
                        "tool_name": getattr(tc, "tool", getattr(tc, "name", "")),
                        "arguments": getattr(tc, "arguments", getattr(tc, "args", {})),
                    })
        elif hasattr(candidate, "tools") and candidate.tools:
            for t in candidate.tools:
                if isinstance(t, dict):
                    tool_calls.append(_normalize_tool_call(t))
                else:
                    tool_calls.append({
                        "tool_name": getattr(t, "name", getattr(t, "tool", "")),
                        "arguments": getattr(t, "arguments", getattr(t, "args", {})),
                    })
    except Exception:
        pass
    return tool_calls


def _normalize_tool_call(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a tool-call dict so the frontend can render a stable tool name."""
    name = tc.get("tool_name") or tc.get("name") or tc.get("tool") or ""
    return {
        "tool_name": name,
        "arguments": tc.get("arguments") or tc.get("args") or {},
    }


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: Optional[int] = None
    reason: str


class HandoffRequest(BaseModel):
    session_id: str
    reason: str


class HandoffReplyRequest(BaseModel):
    session_id: str
    staff_name: str
    message: str


class HandoffResolveRequest(BaseModel):
    session_id: str
    resolution: Optional[str] = None


async def _stream_agent_response(
    message: str,
    session_id: str,
    user_id: str,
) -> AsyncIterator[str]:
    """Run the agent with streaming and yield SSE events."""

    try:
        # Ensure session exists and check handoff state.
        ensure_chat_session(session_id)

        # First send a "start" event
        yield f"event: start\ndata: {json.dumps({'session_id': session_id})}\n\n"

        # Persist the user message before invoking the agent.
        save_chat_message(session_id=session_id, role="user", content=message)

        # Owner-initiated handoff detection.
        handoff_reason = _detect_handoff_intent(message)
        if handoff_reason:
            request_handoff(session_id, handoff_reason)
            reply = (
                "已收到您的请求，已为您转接人工服务。"
                "物业工作人员会尽快在对话中回复您，请稍候。"
            )
            save_chat_message(session_id=session_id, role="assistant", content=reply)
            yield f"event: delta\ndata: {json.dumps({'content': reply})}\n\n"
            yield f"event: done\ndata: {json.dumps({'status': 'complete', 'token_count': 0, 'message_id': None, 'handoff': True})}\n\n"
            return

        # If handoff is active, do not run the agent; tell user to wait.
        if is_handoff_active(session_id):
            reply = "当前会话已由人工接管，工作人员会尽快回复您，请稍候。"
            save_chat_message(session_id=session_id, role="assistant", content=reply)
            yield f"event: delta\ndata: {json.dumps({'content': reply})}\n\n"
            yield f"event: done\ndata: {json.dumps({'status': 'complete', 'token_count': 0, 'message_id': None, 'handoff': True})}\n\n"
            return

        # Classify intent and dispatch to the appropriate vertical agent.
        intent_result = await classify_intent(message, user_id=user_id, session_id=session_id)
        intent = intent_result.get("intent", "other")
        create_agent_fn, agent_name = _select_agent(intent)
        current_agent = agent_name
        current_agent_id = _agent_id_for_intent(intent)

        # Yield routing event so the UI can show which agent is handling the request.
        yield f"event: route\ndata: {json.dumps({'intent': intent, 'reason': intent_result.get('reason', ''), 'current_agent': current_agent})}\n\n"

        # Build dynamic context and tools scoped to the current vertical agent.
        skill_context, activated_skills, skill_model_id = _build_skill_context(message, agent_id=current_agent_id)
        rag_context, citations = _build_rag_context(message)
        mcp_context = _format_mcp_context(agent_id=current_agent_id)
        mcp_tools = _build_mcp_tools(agent_id=current_agent_id)

        # If no relevant knowledge was retrieved, record a badcase for the gap
        # and instruct the agent to admit the missing knowledge.
        auto_badcase_id: Optional[int] = None
        if not citations:
            try:
                bc = create_badcase(
                    title=(message[:60] + "...") if len(message) > 60 else message,
                    description="检索阶段未命中知识库，可能缺少相关文档。",
                    category="knowledge",
                    evidence=f"user: {message}",
                    source_message_id=None,
                    session_id=session_id,
                )
                auto_badcase_id = bc.get("id") if bc else None
            except Exception:
                pass

        # Owner-facing chat ignores any model_id declared by a Skill. Skills may
        # inject prompt context, tools, and knowledge, but the runtime model is
        # always the owner-facing default (deepseek-v4-flash). Pro and any other
        # alternative model are reserved for backend-only workflows.
        skill_model_id = None
        turn_model = None

        # Provide owner context so the agent defaults to the current room_id
        # when the user does not explicitly mention one.
        knowledge_gap_note = (
            "[注意：未从知识库检索到相关内容，请明确告知用户未找到相关知识，"
            "并建议转人工或等待补充资料。]"
            if not citations
            else ""
        )
        contextual_message = (
            f"[系统上下文：当前业主是 {DEFAULT_ROOM_ID} 的{DEFAULT_OWNER_NAME}，"
            f"如果用户没有提供房号，创建工单时默认使用 {DEFAULT_ROOM_ID}。"
            f"当用户明确要求人工、表达强烈不满、或问题超出物业维修/收费/知识库范围时，"
            f"你必须主动提出转人工处理，不要强行回答。]"
            f"{knowledge_gap_note}"
            f"{rag_context}"
            f"{skill_context}"
            f"{mcp_context}\n{message}"
        )

        full_content = ""
        token_count = 0
        tool_calls: List[Dict[str, Any]] = []
        token_detail: Dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }

        # Create a fresh agent instance with dynamic MCP tools.
        agent = create_agent_fn(tools=mcp_tools, model=turn_model)

        # Run agent in streaming mode (returns an async generator)
        async for chunk in agent.arun(
            contextual_message,
            user_id=user_id,
            session_id=session_id,
            stream=True,
        ):
            content = ""
            if hasattr(chunk, "content") and chunk.content:
                content = str(chunk.content)
            elif hasattr(chunk, "delta") and chunk.delta:
                content = str(chunk.delta)

            if content:
                full_content += content
                yield f"event: delta\ndata: {json.dumps({'content': content, 'current_agent': current_agent})}\n\n"

            # Collect tool call metadata.
            chunk_tools = _extract_tool_calls(chunk)
            if chunk_tools:
                tool_calls.extend(chunk_tools)
                yield f"event: tool_calls\ndata: {json.dumps({'tool_calls': chunk_tools, 'current_agent': current_agent})}\n\n"

            # Try to capture token usage if the chunk exposes it.
            if hasattr(chunk, "metrics") and chunk.metrics:
                try:
                    metrics = chunk.metrics
                    if hasattr(metrics, "input_tokens") and metrics.input_tokens is not None:
                        token_detail["input_tokens"] = int(metrics.input_tokens)
                    if hasattr(metrics, "output_tokens") and metrics.output_tokens is not None:
                        token_detail["output_tokens"] = int(metrics.output_tokens)
                    if hasattr(metrics, "total_tokens") and metrics.total_tokens is not None:
                        token_detail["total_tokens"] = int(metrics.total_tokens)
                    if hasattr(metrics, "reasoning_tokens") and metrics.reasoning_tokens is not None:
                        token_detail["reasoning_tokens"] = int(metrics.reasoning_tokens)
                    if hasattr(metrics, "cached_tokens") and metrics.cached_tokens is not None:
                        token_detail["cached_tokens"] = int(metrics.cached_tokens)
                except Exception:
                    pass

        # Streaming chunks do not expose MCP tool-call metadata.  Collect any
        # invocations recorded by our ObservableMCPTools wrappers so the front
        # end and acceptance tests can see real tool usage.
        for toolkit in mcp_tools:
            if hasattr(toolkit, "recorded_calls") and toolkit.recorded_calls:
                for call in toolkit.recorded_calls:
                    if call not in tool_calls:
                        tool_calls.append(call)
                        yield f"event: tool_calls\ndata: {json.dumps({'tool_calls': [call], 'current_agent': current_agent})}\n\n"

        # Derive token_count from total_tokens or input+output when possible.
        if token_detail["total_tokens"]:
            token_count = token_detail["total_tokens"]
        elif token_detail["input_tokens"] or token_detail["output_tokens"]:
            token_count = token_detail["input_tokens"] + token_detail["output_tokens"]

        # Fall back to tiktoken estimate if the model did not report metrics.
        if not token_count and full_content:
            output_tokens = _estimate_tokens(full_content)
            input_tokens = _estimate_tokens(contextual_message)
            token_count = input_tokens + output_tokens
            token_detail["input_tokens"] = input_tokens
            token_detail["output_tokens"] = output_tokens
            token_detail["total_tokens"] = token_count

        # AI-initiated handoff: if the agent explicitly asks to transfer.
        ai_handoff = False
        handoff_phrases = ["已为您转接人工", "已转人工", "已转接人工", "转接人工服务"]
        if full_content and any(p in full_content for p in handoff_phrases):
            ai_handoff = True
            request_handoff(session_id, "AI 判断需要人工处理")

        # Determine the model actually used for this turn.
        runtime_model_id = skill_model_id if skill_model_id else MODEL_ID
        model_selection_reason = (
            f"skill_model_override:{skill_model_id}"
            if skill_model_id
            else "owner-facing default"
        )

        # Persist the assistant message.
        saved = save_chat_message(
            session_id=session_id,
            role="assistant",
            content=full_content,
            token_count=token_count,
            token_detail=token_detail,
            citations=citations,
            activated_skills=activated_skills,
            route_intent=intent,
            route_reason=intent_result.get("reason", ""),
            current_agent=current_agent,
            tool_calls=tool_calls or None,
            model_id=runtime_model_id,
            thinking_enabled=USE_THINKING,
            model_selection_reason=model_selection_reason,
        )

        # Send completion event including token metrics, citations, activated skills and agent info.
        done_payload = {
            'status': 'complete',
            'token_count': token_count,
            'token_detail': token_detail,
            'message_id': saved.get('id'),
            'handoff': ai_handoff,
            'citations': citations,
            'activated_skills': activated_skills,
            'current_agent': current_agent,
            'route_intent': intent,
            'route_reason': intent_result.get("reason", ""),
            'tool_calls': tool_calls,
            'auto_badcase_id': auto_badcase_id,
            'model_id': runtime_model_id,
            'thinking_enabled': USE_THINKING,
            'model_selection_reason': model_selection_reason,
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


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


@router.post("/feedback")
async def chat_feedback(request: FeedbackRequest):
    """Create a badcase from user feedback on an AI response."""
    if not request.reason or not request.reason.strip():
        raise HTTPException(status_code=400, detail="反馈描述不能为空")

    title = "业主不满意 Agent 回答"
    description = request.reason.strip()
    if request.message_id:
        description = f"消息 ID: {request.message_id}\n{description}"

    badcase = create_badcase(
        title=title,
        description=description,
        category="model",
        status="pending",
        created_at=now_cn(),
        evidence=request.reason.strip(),
        source_message_id=request.message_id,
        session_id=request.session_id,
    )
    return {"status": "ok", "badcase": badcase}


@router.post("/handoff")
async def chat_handoff(request: HandoffRequest):
    """Request human handoff for a chat session."""
    if not request.reason or not request.reason.strip():
        raise HTTPException(status_code=400, detail="转人工原因不能为空")
    session = request_handoff(request.session_id, request.reason.strip())
    return {"status": "ok", "session": session}


@router.get("/handoffs")
async def chat_handoffs(
    status: Optional[str] = Query(None, description="Filter by handoff status"),
):
    """List chat sessions awaiting or under human takeover."""
    sessions = list_handoff_sessions(status=status)
    return {"sessions": sessions}


@router.post("/handoff-reply")
async def chat_handoff_reply(request: HandoffReplyRequest):
    """Staff sends a human reply into a chat session."""
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="回复内容不能为空")

    # Activate handoff if this is the first staff reply.
    current = get_chat_session(request.session_id)
    if current is None or current.get("handoff_status") != "active":
        activate_handoff(request.session_id, request.staff_name)

    save_chat_message(
        session_id=request.session_id,
        role="staff",
        content=request.message.strip(),
    )
    messages = list_chat_messages(request.session_id)
    session = get_chat_session(request.session_id)
    return {"status": "ok", "messages": messages, "session": session}


@router.post("/handoff-resolve")
async def chat_handoff_resolve(request: HandoffResolveRequest):
    """Resolve a handoff request."""
    session = resolve_handoff(request.session_id)
    return {"status": "ok", "session": session}
