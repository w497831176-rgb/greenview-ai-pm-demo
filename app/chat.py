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
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.observability import _check_budget
from app.settings import MODEL_ID, USE_THINKING
from agents.billing import create_billing_agent
from agents.complaint import create_complaint_agent
from agents.customer_service import create_customer_service_agent
from agents.maintenance import create_maintenance_agent
from agents.property import create_property_agent
from agents.router import classify_intent
from db.property_db import (
    activate_handoff,
    add_badcase_action,
    create_badcase,
    create_chat_session,
    create_chat_trace,
    ensure_chat_session,
    get_budget_thresholds,
    get_chat_message,
    get_chat_session,
    get_enabled_price_for_model,
    get_model_calls_for_trace,
    get_previous_user_message,
    is_handoff_active,
    is_handoff_requested,
    list_chat_messages,
    list_handoff_sessions,
    list_mcp_servers,
    list_skills,
    list_user_chat_sessions,
    record_mcp_call_audit,
    record_model_call,
    request_handoff,
    resolve_handoff,
    save_chat_message,
    update_budget_thresholds,
    update_chat_trace,
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

        In V1.3 each call is also written to `mcp_call_audits` after the tool
        actually executes, with status, result summary, and latency.
        """

        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.recorded_calls: List[Dict[str, Any]] = []
            self.trace_id: Optional[str] = None
            self.server_name: str = "unknown"

        async def build_tools(self) -> None:
            # Build tools in the current event loop. MCPTools uses asyncio stdio
            # subprocess, so it is non-blocking and safe to await directly.
            await super(ObservableMCPTools, self).build_tools()
            functions = getattr(self, "functions", None) or {}
            for fn_name, fn in functions.items():
                original = getattr(fn, "entrypoint", None)
                if original is None:
                    continue
                wrapped = self._wrap(original, fn_name)
                fn.entrypoint = wrapped

        def _record_audit(
            self,
            fn_name: str,
            args_dict: Any,
            status: str,
            result_summary: str,
            error_summary: Optional[str],
            latency_ms: int,
        ) -> None:
            call_record = {
                "tool_name": fn_name,
                "arguments": args_dict,
                "status": status,
                "result_summary": result_summary,
                "error_summary": error_summary,
                "latency_ms": latency_ms,
            }
            self.recorded_calls.append(call_record)
            try:
                record_mcp_call_audit(
                    trace_id=self.trace_id or "unknown",
                    server_name=self.server_name,
                    tool_name=fn_name,
                    arguments=args_dict if isinstance(args_dict, dict) else {"value": str(args_dict)},
                    status=status,
                    result_summary=result_summary,
                    error_summary=error_summary,
                    latency_ms=latency_ms,
                )
            except Exception:
                # Audit failures must not break the chat flow.
                pass

            # Auto-capture MCP tool failures as badcases for ops governance.
            if status == "failed":
                try:
                    sanitized_args = {
                        k: f"<{type(v).__name__}>" if not isinstance(v, (str, int, float, bool, type(None))) else v
                        for k, v in (args_dict.items() if isinstance(args_dict, dict) else {}.items())
                    }
                    # Remove any potential secret values from the summary.
                    sanitized_summary = {k: v for k, v in sanitized_args.items() if isinstance(v, (str, int, float, bool))}
                    create_badcase(
                        title=f"MCP 工具失败：{fn_name}",
                        description=error_summary or result_summary or "MCP 工具调用失败，待排查能力缺口或配置问题。",
                        category="mcp_capability" if "mcp" in self.server_name.lower() or "mcp" in fn_name.lower() else "tool_failure",
                        evidence=f"server={self.server_name}, tool={fn_name}",
                        source_message_id=None,
                        session_id=None,
                        source="auto",
                        original_query=fn_name,
                        ai_response=error_summary or result_summary,
                        context_json=json.dumps(
                            {
                                "server_name": self.server_name,
                                "tool_name": fn_name,
                                "sanitized_params_summary": sanitized_summary,
                                "error_message": error_summary,
                                "result_summary": result_summary,
                                "trace_id": self.trace_id,
                            },
                            ensure_ascii=False,
                            default=str,
                        ),
                        trace_id=self.trace_id or "unknown",
                    )
                except Exception:
                    # Badcase auto-capture failures must not break the chat flow.
                    pass

        def _wrap(self, original: Any, fn_name: str):
            if asyncio.iscoroutinefunction(original):
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    args_dict = kwargs if kwargs else (args[0] if args else {})
                    start = __import__('time').time()
                    status = "success"
                    result_summary = ""
                    error_summary = None
                    try:
                        result = await original(*args, **kwargs)
                        result_summary = _summarize_tool_result(result)
                        if "Error from MCP tool" in result_summary:
                            status = "failed"
                            error_summary = result_summary[:300]
                        return result
                    except Exception as exc:
                        status = "failed"
                        error_summary = str(exc)[:300]
                        raise
                    finally:
                        latency_ms = int((__import__('time').time() - start) * 1000)
                        self._record_audit(fn_name, args_dict, status, result_summary, error_summary, latency_ms)
                return async_wrapper

            async def async_sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                args_dict = kwargs if kwargs else (args[0] if args else {})
                start = __import__('time').time()
                status = "success"
                result_summary = ""
                error_summary = None
                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: original(*args, **kwargs))
                    result_summary = _summarize_tool_result(result)
                    if "Error from MCP tool" in result_summary:
                        status = "failed"
                        error_summary = result_summary[:300]
                    return result
                except Exception as exc:
                    status = "failed"
                    error_summary = str(exc)[:300]
                    raise
                finally:
                    latency_ms = int((__import__('time').time() - start) * 1000)
                    self._record_audit(fn_name, args_dict, status, result_summary, error_summary, latency_ms)
            return async_sync_wrapper
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


def _is_pro_model(model_id: Optional[str]) -> bool:
    """Return True for models classified as Pro (higher-cost) models."""
    return (model_id or "").lower() in {"deepseek-v4-pro"}


DEFAULT_ROOM_ID = "3-2-1201"
DEFAULT_OWNER_NAME = "王先生"


def _build_price_snapshot(model_id: str) -> Optional[Dict[str, Any]]:
    price = get_enabled_price_for_model(model_id)
    if not price:
        return None
    return {
        "model_id": price.get("model_id"),
        "currency": price.get("currency"),
        "effective_date": price.get("effective_date"),
        "input_price_per_1m": price.get("input_price_per_1m"),
        "cached_input_price_per_1m": price.get("cached_input_price_per_1m"),
        "output_price_per_1m": price.get("output_price_per_1m"),
        "reasoning_price_per_1m": price.get("reasoning_price_per_1m"),
        "source_note": price.get("source_note"),
    }


def _calculate_cost(model_id: str, usage: Dict[str, Optional[int]]) -> tuple:
    """Return (cost_cny, price_snapshot) for a model call.

    Cost is always an estimate based on the configured price table at the time
    of the call. If no price is configured, return (None, None).
    """
    snapshot = _build_price_snapshot(model_id)
    if not snapshot:
        return None, None

    input_tk = usage.get("input_tokens") or 0
    output_tk = usage.get("output_tokens") or 0
    reasoning_tk = usage.get("reasoning_tokens") or 0
    cached_tk = usage.get("cached_tokens") or 0

    cost = 0.0
    if snapshot.get("input_price_per_1m") is not None:
        cost += (input_tk - cached_tk) * (snapshot["input_price_per_1m"] / 1_000_000)
    if snapshot.get("cached_input_price_per_1m") is not None:
        cost += cached_tk * (snapshot["cached_input_price_per_1m"] / 1_000_000)
    if snapshot.get("output_price_per_1m") is not None:
        cost += output_tk * (snapshot["output_price_per_1m"] / 1_000_000)
    if snapshot.get("reasoning_price_per_1m") is not None:
        cost += reasoning_tk * (snapshot["reasoning_price_per_1m"] / 1_000_000)

    return round(cost, 8), snapshot


def _summarize_tool_result(result: Any, max_len: int = 300) -> str:
    """Build a short, safe summary of an MCP tool result."""
    try:
        if result is None:
            return ""
        if isinstance(result, (dict, list)):
            text = json.dumps(result, ensure_ascii=False, default=str)
        else:
            text = str(result)
        if len(text) > max_len:
            text = text[:max_len] + "…"
        return text
    except Exception:
        return ""


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


def _build_rag_context(message: str, top_k: Optional[int] = None, threshold: Optional[float] = None) -> tuple:
    """Run advanced RAG and format retrieved chunks as context.

    Returns (rag_context_string, citations).

    The number of retrieved chunks is read from retrieval_settings.top_k unless
    explicitly overridden.  It is clamped to a sensible 1-10 range.
    """
    try:
        from db.property_db import get_retrieval_settings
        settings = get_retrieval_settings("default") or {}
        effective_top_k = top_k if top_k is not None else settings.get("top_k", 3)
        effective_top_k = max(1, min(10, int(effective_top_k)))
        settings_payload = {
            "top_k": effective_top_k,
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


def _build_mcp_tools(
    agent_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> List[Any]:
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
                tool = ObservableMCPTools(
                    command=full_command,
                    env=merged_env,
                    name=name,
                    transport="stdio",
                    timeout_seconds=15,
                )
                tool.trace_id = trace_id
                tool.server_name = name
                tools.append(tool)
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
    args = tc.get("arguments") or tc.get("args") or {}
    # Ensure arguments are JSON-serializable (convert Pydantic models, etc.)
    if hasattr(args, "model_dump"):
        args = args.model_dump()
    elif hasattr(args, "dict"):
        args = args.dict()
    elif not isinstance(args, dict):
        args = {"value": str(args)}
    return {
        "tool_name": name,
        "arguments": args,
    }


def _safe_json_dumps(obj: Any) -> str:
    """Serialize to JSON, converting non-serializable objects to strings."""
    def _default(o: Any):
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if hasattr(o, "dict"):
            return o.dict()
        return str(o)
    return json.dumps(obj, ensure_ascii=False, default=_default)


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

    done_yielded = False
    error_yielded = False
    trace_id = uuid.uuid4().hex[:16]
    trace_start = time.time()
    turn_model_id = MODEL_ID
    turn_selection_reason = "owner-facing default"
    vertical_latency_ms: Optional[int] = None
    router_latency_ms: Optional[int] = None

    try:
        # Ensure session exists and check handoff state.
        ensure_chat_session(session_id)

        # Create the trace record for this turn.
        create_chat_trace(trace_id=trace_id, session_id=session_id, user_message=message)

        # First send a "start" event
        yield f"event: start\ndata: {json.dumps({'session_id': session_id, 'trace_id': trace_id})}\n\n"

        # Persist the user message before invoking the agent.
        save_chat_message(session_id=session_id, role="user", content=message, trace_id=trace_id)

        # Owner-initiated handoff detection.
        handoff_reason = _detect_handoff_intent(message)
        if handoff_reason:
            request_handoff(session_id, handoff_reason)
            reply = (
                "已收到您的请求，已为您转接人工服务。"
                "物业工作人员会尽快在对话中回复您，请稍候。"
            )
            save_chat_message(session_id=session_id, role="assistant", content=reply, trace_id=trace_id)
            update_chat_trace(trace_id=trace_id, intent="handoff", agent_name="人工", status="complete")
            yield f"event: delta\ndata: {json.dumps({'content': reply})}\n\n"
            yield f"event: done\ndata: {_safe_json_dumps({'status': 'complete', 'token_count': 0, 'message_id': None, 'handoff': True, 'trace_id': trace_id})}\n\n"
            done_yielded = True
            return

        # If handoff is active, do not run the agent; tell user to wait.
        if is_handoff_active(session_id):
            reply = "当前会话已由人工接管，工作人员会尽快回复您，请稍候。"
            save_chat_message(session_id=session_id, role="assistant", content=reply, trace_id=trace_id)
            update_chat_trace(trace_id=trace_id, intent="handoff", agent_name="人工", status="complete")
            yield f"event: delta\ndata: {json.dumps({'content': reply})}\n\n"
            yield f"event: done\ndata: {_safe_json_dumps({'status': 'complete', 'token_count': 0, 'message_id': None, 'handoff': True, 'trace_id': trace_id})}\n\n"
            done_yielded = True
            return

        # Classify intent and dispatch to the appropriate vertical agent.
        router_start = time.time()
        intent_result = await classify_intent(message, user_id=user_id, session_id=session_id)
        router_latency_ms = int((time.time() - router_start) * 1000)
        intent = intent_result.get("intent", "other")

        # Weather queries (even for unsupported cities) must be dispatched to the
        # maintenance agent, which owns the weather MCP tools.
        if intent in {"other", "customer_service"} and any(k in message for k in ("天气", "气温", "下雨")):
            intent = "maintenance"
            intent_result["intent"] = intent
            intent_result["reason"] = "天气查询属于维修/工单场景（含工具支持）"

        create_agent_fn, agent_name = _select_agent(intent)
        current_agent = agent_name
        current_agent_id = _agent_id_for_intent(intent)

        # Record the router model call (metrics are not returned by the non-streaming router).
        try:
            router_cost, router_price = _calculate_cost(
                MODEL_ID,
                {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0},
            )
            record_model_call(
                trace_id=trace_id,
                stage="router",
                model_id=MODEL_ID,
                model_selection_reason="owner-facing default",
                latency_ms=router_latency_ms,
                usage_source="unavailable",
                status="success",
                estimated_cost_cny=router_cost,
                price_snapshot=router_price,
            )
        except Exception:
            pass

        # Yield routing event so the UI can show which agent is handling the request.
        yield f"event: route\ndata: {json.dumps({'intent': intent, 'reason': intent_result.get('reason', ''), 'current_agent': current_agent, 'trace_id': trace_id})}\n\n"

        # Build dynamic context and tools scoped to the current vertical agent.
        skill_context, activated_skills, skill_model_id = _build_skill_context(message, agent_id=current_agent_id)
        rag_context, citations = _build_rag_context(message)
        mcp_context = _format_mcp_context(agent_id=current_agent_id)
        mcp_tools = _build_mcp_tools(agent_id=current_agent_id, trace_id=trace_id)

        # If no relevant knowledge was retrieved, record a badcase for the gap
        # and instruct the agent to admit the missing knowledge.
        auto_badcase_id: Optional[int] = None
        rag_retrieval_summary = {
            "query": message,
            "top_k": effective_top_k if 'effective_top_k' in locals() else None,
            "results_count": len(citations),
            "citation_titles": [c.get("doc_title") for c in citations],
        }
        if not citations:
            try:
                bc = create_badcase(
                    title=(message[:60] + "...") if len(message) > 60 else message,
                    description="检索阶段未命中知识库，可能缺少相关文档。",
                    category="knowledge_gap",
                    evidence=f"user: {message}",
                    source_message_id=None,
                    session_id=session_id,
                    source="auto",
                    original_query=message,
                    ai_response="",
                    context_json=json.dumps(
                        {
                            "retrieval_summary": rag_retrieval_summary,
                            "route_intent": intent if 'intent' in locals() else None,
                            "current_agent": current_agent if 'current_agent' in locals() else None,
                            "trace_id": trace_id,
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                    trace_id=trace_id,
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

        # Owner-facing chat must never silently downgrade or fake execution when
        # a Pro model is requested. If the budget is exhausted, block the Pro
        # call and record it as blocked.
        runtime_model_id = skill_model_id if skill_model_id else MODEL_ID
        pro_selection_reason = (
            f"skill_model_override:{skill_model_id}"
            if skill_model_id
            else "owner-facing default"
        )
        if _is_pro_model(runtime_model_id):
            budget = _check_budget("pro_call")
            if budget.get("alert_level") == "blocked":
                try:
                    record_model_call(
                        trace_id=trace_id,
                        stage="vertical_agent",
                        model_id=runtime_model_id,
                        status="blocked",
                        latency_ms=0,
                        usage_source="unavailable",
                        model_selection_reason=pro_selection_reason,
                        error_summary="预算已达上限，Pro 调用被阻止",
                        estimated_cost_cny=None,
                        price_snapshot=None,
                    )
                    update_chat_trace(trace_id=trace_id, status="failed")
                except Exception:
                    pass
                yield f"event: error\ndata: {_safe_json_dumps({'error': '预算已达上限，Pro 调用被阻止', 'trace_id': trace_id})}\n\n"
                error_yielded = True
                return

        # Create a fresh agent instance with dynamic MCP tools.
        agent = create_agent_fn(tools=mcp_tools, model=turn_model)
        vertical_start = time.time()

        # Run agent in streaming mode.  We decouple Agno's async generator from
        # the HTTP SSE generator via an asyncio.Queue so that any quirks in the
        # Agno generator lifecycle (especially after MCP tool-call turns) do not
        # abort the SSE response before we can send the done event.
        sse_queue: asyncio.Queue[Tuple[str, Any]] = asyncio.Queue()

        async def _produce_chunks() -> None:
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
                    sse_queue.put_nowait(("delta", content))

                chunk_tools = _extract_tool_calls(chunk)
                if chunk_tools:
                    sse_queue.put_nowait(("tool_calls", chunk_tools))

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
            sse_queue.put_nowait(("done", None))

        producer_task = asyncio.create_task(_produce_chunks())

        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(sse_queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield "event: keepalive\ndata: {}\n\n"
                    continue

                if kind == "done":
                    break
                if kind == "delta":
                    full_content += payload
                    yield f"event: delta\ndata: {json.dumps({'content': payload, 'current_agent': current_agent})}\n\n"
                elif kind == "tool_calls":
                    tool_calls.extend(payload)
                    yield f"event: tool_calls\ndata: {_safe_json_dumps({'tool_calls': payload, 'current_agent': current_agent})}\n\n"
        finally:
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

        # Streaming chunks do not expose MCP tool-call metadata.  Collect any
        # invocations recorded by our ObservableMCPTools wrappers so the front
        # end and acceptance tests can see real tool usage.
        for toolkit in mcp_tools:
            if hasattr(toolkit, "recorded_calls") and toolkit.recorded_calls:
                for call in toolkit.recorded_calls:
                    normalized = _normalize_tool_call(call)
                    if normalized not in tool_calls:
                        tool_calls.append(normalized)
                        yield f"event: tool_calls\ndata: {_safe_json_dumps({'tool_calls': [normalized], 'current_agent': current_agent})}\n\n"

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
        turn_model_id = runtime_model_id
        turn_selection_reason = model_selection_reason

        # Record the vertical agent model call.
        try:
            vertical_latency_ms = int((time.time() - vertical_start) * 1000) if 'vertical_start' in locals() else None
        except Exception:
            vertical_latency_ms = None
        usage_source = "provider_reported" if token_detail.get("total_tokens") else "estimated_tokenization" if token_count else "unavailable"
        try:
            vertical_cost, vertical_price = _calculate_cost(runtime_model_id, token_detail)
            record_model_call(
                trace_id=trace_id,
                stage="vertical_agent",
                model_id=runtime_model_id,
                model_selection_reason=model_selection_reason,
                latency_ms=vertical_latency_ms,
                input_tokens=token_detail.get("input_tokens"),
                output_tokens=token_detail.get("output_tokens"),
                reasoning_tokens=token_detail.get("reasoning_tokens"),
                cached_tokens=token_detail.get("cached_tokens"),
                total_tokens=token_detail.get("total_tokens") or token_count,
                usage_source=usage_source,
                status="success",
                estimated_cost_cny=vertical_cost,
                price_snapshot=vertical_price,
            )
        except Exception:
            pass

        # Update the trace record with final intent/agent/status.
        try:
            update_chat_trace(
                trace_id=trace_id,
                intent=intent,
                agent_name=current_agent,
                status="failed" if error_yielded else "complete",
            )
        except Exception:
            pass

        # Build MCP audit list for the done event and persistence.
        mcp_calls_for_done: List[Dict[str, Any]] = []
        for toolkit in mcp_tools:
            if hasattr(toolkit, "recorded_calls") and toolkit.recorded_calls:
                for call in toolkit.recorded_calls:
                    mcp_calls_for_done.append({
                        "server_name": getattr(toolkit, "server_name", "unknown"),
                        "tool_name": call.get("tool_name", ""),
                        "arguments": call.get("arguments", {}),
                        "status": call.get("status", "success"),
                        "result_summary": call.get("result_summary", ""),
                        "error_summary": call.get("error_summary"),
                        "latency_ms": call.get("latency_ms"),
                    })

        # Persist the assistant message synchronously. SQLite writes against the
        # local demo DB are fast enough that the brief event-loop block is
        # acceptable for this demo; the previous thread-pool + keepalive path
        # hung under the current Agno/FastAPI runtime, so we keep it simple.
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
            trace_id=trace_id,
            status="failed" if error_yielded else "success",
            latency_ms=vertical_latency_ms,
            error_summary=str(e) if error_yielded else None,
            mcp_calls=mcp_calls_for_done or None,
            usage_source=usage_source,
        )

        # Back-fill the auto-captured knowledge-gap badcase with the actual AI response.
        if auto_badcase_id:
            try:
                from db.property_db import update_badcase
                update_badcase(
                    auto_badcase_id,
                    ai_response=full_content,
                    context_json=json.dumps(
                        {
                            "retrieval_summary": rag_retrieval_summary,
                            "route_intent": intent,
                            "current_agent": current_agent,
                            "activated_skills": activated_skills,
                            "trace_id": trace_id,
                            "model_id": runtime_model_id,
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                )
            except Exception:
                pass

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
            'mcp_calls': mcp_calls_for_done,
            'auto_badcase_id': auto_badcase_id,
            'model_id': runtime_model_id,
            'thinking_enabled': USE_THINKING,
            'model_selection_reason': model_selection_reason,
            'trace_id': trace_id,
            'usage_source': usage_source,
        }
        yield f"event: done\ndata: {_safe_json_dumps(done_payload)}\n\n"
        done_yielded = True

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"event: error\ndata: {_safe_json_dumps({'error': str(e), 'trace_id': trace_id})}\n\n"
        error_yielded = True
        try:
            update_chat_trace(trace_id=trace_id, status="failed")
        except Exception:
            pass
    finally:
        # If the generator exits for any reason (including client disconnect)
        # without having sent done or error, send a minimal done so the client
        # can close gracefully rather than hang.
        if not done_yielded and not error_yielded:
            try:
                yield "event: done\ndata: {\"status\":\"complete\"}\n\n"
            except BaseException:
                pass


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
