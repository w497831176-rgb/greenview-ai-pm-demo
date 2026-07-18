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
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.observability import _check_budget
from app.handoff_policy import HANDOFF_STATUS_LABELS, evaluate_handoff_policy, handoff_policy_summary
from app.mcp_policy import (
    allowed_tools_for_agent,
    extract_tool_outcome,
    is_builtin_policy_server,
    outcome_instruction,
)
from app.multimodal import get_analysis_context, normalise_analysis_ids
from app.settings import MODEL_ID, USE_THINKING, build_model
from app.skill_runtime import activation_evidence, select_skills, skill_contract
from app.utils.cost_utils import build_price_snapshot, compute_cost_cny, normalize_usage
from agents.billing import create_billing_agent
from agents.complaint import create_complaint_agent
from agents.customer_service import create_customer_service_agent
from agents.maintenance import create_maintenance_agent
from agents.router import classify_intent
from app.work_order_workflow import advance_work_order_workflow
from tools.work_order import set_work_order_context
from db.property_db import (
    activate_handoff,
    add_badcase_action,
    cancel_handoff,
    claim_handoff,
    close_handoff,
    create_badcase,
    create_chat_session,
    create_chat_trace,
    ensure_chat_session,
    get_agent_by_agent_id,
    get_agent_skills,
    get_agent_tools,
    get_skill,
    get_budget_thresholds,
    get_chat_message,
    get_chat_session,
    get_handoff_package,
    get_enabled_price_for_model,
    get_model_calls_for_trace,
    get_previous_user_message,
    is_handoff_active,
    is_handoff_requested,
    list_agents,
    list_chat_messages,
    list_handoff_sessions,
    list_mcp_servers,
    list_mcp_tools,
    list_knowledge_docs,
    list_skills,
    list_user_chat_sessions,
    record_mcp_call_audit,
    record_model_call,
    record_trace_event,
    request_handoff,
    resume_handoff_after_owner_message,
    resolve_handoff,
    save_chat_message,
    update_budget_thresholds,
    update_chat_trace,
    wait_for_handoff_user,
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
            # These are Host-side controls, not MCPTools constructor options.
            # Pop them before calling the library so they cannot silently leak
            # into a future MCPTools implementation.
            self.invocation_mode: str = kwargs.pop("invocation_mode", "model_native")
            allowed_names = kwargs.pop("allowed_function_names", None)
            self.allowed_function_names = set(allowed_names or [])
            super().__init__(*args, **kwargs)
            self.recorded_calls: List[Dict[str, Any]] = []
            self.trace_id: Optional[str] = None
            self.server_name: str = "unknown"

        async def build_tools(self) -> None:
            # Build tools in the current event loop. MCPTools uses asyncio stdio
            # subprocess, so it is non-blocking and safe to await directly.
            await super(ObservableMCPTools, self).build_tools()
            functions = getattr(self, "functions", None) or {}
            if self.allowed_function_names:
                # Server discovery is not a permission grant.  The Host only
                # exposes functions explicitly allowed for the current Agent.
                functions = {
                    name: function
                    for name, function in functions.items()
                    if name in self.allowed_function_names
                }
                self.functions = functions
            for fn_name, fn in functions.items():
                original = getattr(fn, "entrypoint", None)
                if original is None or getattr(original, "_observable_wrapped", False):
                    continue
                wrapped = self._wrap(original, fn_name)
                wrapped._observable_wrapped = True  # type: ignore[attr-defined]
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
                "invocation_mode": self.invocation_mode,
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
                    invocation_mode=self.invocation_mode,
                )
            except Exception:
                # Audit failures must not break the chat flow.
                pass

            # Auto-capture genuine service failures only.  An empty result,
            # input validation failure, or a permission denial is an expected
            # product outcome, not automatically a capability-gap Badcase.
            if status in {"timeout", "upstream_error"}:
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
                        category="mcp_capability",
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
                        status = extract_tool_outcome(result_summary)
                        if status in {"timeout", "upstream_error"}:
                            error_summary = result_summary[:300]
                        return result
                    except Exception as exc:
                        status = "upstream_error"
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
                    status = extract_tool_outcome(result_summary)
                    if status in {"timeout", "upstream_error"}:
                        error_summary = result_summary[:300]
                    return result
                except Exception as exc:
                    status = "upstream_error"
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


def _estimate_tokens(text: str) -> Optional[int]:
    """Estimate token count for a text string.

    Returns None when tiktoken is unavailable so the UI can show "不可得"
    instead of 0.
    """
    if _tiktoken_encoding is None:
        return None
    try:
        return len(_tiktoken_encoding.encode(text or ""))
    except Exception:
        return None


def _estimate_context_breakdown(
    system_prompt: Optional[List[str]] = None,
    history_messages: Optional[List[Dict[str, Any]]] = None,
    skill_context: str = "",
    rag_context: str = "",
    tool_results: Optional[List[str]] = None,
    user_message: str = "",
    system_context_prefix: str = "",
) -> Dict[str, Any]:
    """Build an honest local estimate of the prompt context composition.

    Any component that cannot be estimated is returned as null, which the
    frontend renders as "不可得". The total is not forced to match provider
    usage; a note makes this explicit.
    """
    breakdown: Dict[str, Any] = {
        "system_prompt_tokens": None,
        "history_tokens": None,
        "skill_tokens": None,
        "rag_tokens": None,
        "tool_result_tokens": None,
        "user_message_tokens": None,
        "note": "本地上下文估算，不等于 Provider 原始账单",
    }

    if system_prompt:
        breakdown["system_prompt_tokens"] = _estimate_tokens("\n".join(str(p) for p in system_prompt))

    if history_messages:
        history_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history_messages
        )
        breakdown["history_tokens"] = _estimate_tokens(history_text)

    breakdown["skill_tokens"] = _estimate_tokens(skill_context)
    breakdown["rag_tokens"] = _estimate_tokens(rag_context)

    if tool_results:
        breakdown["tool_result_tokens"] = _estimate_tokens("\n".join(tool_results))

    # User-facing message includes the explicit system context prefix we prepend
    # to the question so the model knows the current owner / room defaults.
    breakdown["user_message_tokens"] = _estimate_tokens(f"{system_context_prefix}\n{user_message}".strip())

    return breakdown


def _is_pro_model(model_id: Optional[str]) -> bool:
    """Return True for models classified as Pro (higher-cost) models."""
    return (model_id or "").lower() in {"deepseek-v4-pro"}


DEFAULT_ROOM_ID = "3-2-1201"
DEFAULT_OWNER_NAME = "王先生"


def _get_price_snapshot(model_id: str) -> Optional[Dict[str, Any]]:
    """Return a serializable price snapshot for a model_id."""
    price = get_enabled_price_for_model(model_id)
    return build_price_snapshot(price)


def _calculate_cost(model_id: str, usage: Dict[str, Optional[int]]) -> tuple:
    """Return (cost_cny, price_snapshot) for a model call.

    Uses the centralized cost helper. If the provider did not return a usable
    cached/uncached split, cost is None but the price snapshot is still returned.
    """
    snapshot = _get_price_snapshot(model_id)
    cost, _status = compute_cost_cny(snapshot, usage)
    return cost, snapshot


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


def _build_skill_context(message: str, agent_id: Optional[str] = None) -> tuple:
    """Build governed Skill context and explain every runtime selection.

    Bound Skills are evaluated by one deterministic policy: positive/negative
    triggers, priority and optional conflict groups.  A missing trigger no
    longer silently injects legacy instructions; an operator must explicitly
    set ``always_on`` for that exceptional case.
    """
    try:
        is_router = agent_id == "router"
        if is_router:
            candidate_skills = [s for s in list_skills() if s.get("enabled")]
            if not candidate_skills:
                return "", [], None
            summary_lines = []
            for skill in candidate_skills:
                name = skill.get("name", "")
                contract = skill_contract(skill)
                if not name:
                    continue
                trigger = "、".join(contract["positive_triggers"])
                line = f"- {name}" + (f"（适用：{trigger}）" if trigger else "（未配置触发，不作为默认能力）")
                summary_lines.append(line)
            if not summary_lines:
                return "", [], None
            context = (
                "\n\n[可用 Skill 清单（仅用于路由参考，不要注入完整 Skill 指令）：\n"
                + "\n".join(summary_lines)
                + "]"
            )
            return context, [], None

        from db.property_db import get_agent_skills, get_skill

        bound_skill_ids = get_agent_skills(agent_id) if agent_id else []
        candidate_skills = [
            get_skill(int(skill_id)) for skill_id in bound_skill_ids
        ]
        candidate_skills = [s for s in candidate_skills if s and s.get("enabled")]
        if not candidate_skills:
            return "", [], None
        selected, _decisions = select_skills(candidate_skills, message)
        if not selected:
            return "", [], None
        by_id = {skill.get("id"): skill for skill in candidate_skills}
        parts, activated = [], []
        for decision in selected:
            skill = by_id.get(decision.get("skill_id"))
            if not skill:
                continue
            name = skill.get("name", "")
            instructions = skill_storage.build_instructions(skill.get("id"), skill)
            if not name or not instructions:
                continue
            contract = decision["contract"]
            header = (
                f"【Skill：{name}｜版本 {contract['version']}｜{decision['match_reason']}】\n"
                "权限边界：Skill 提供业务 SOP，不新增工具权限；仅可使用当前 Agent 已绑定的 MCP 工具。"
            )
            parts.append(f"{header}\n{instructions}")
            activated.append(activation_evidence(decision, len(instructions)))
        if not parts:
            return "", [], None
        return (
            "\n\n[已命中的平台 Skill（仅以下经过触发与冲突策略选择的能力可注入；必须遵守其 SOP 和边界）：\n"
            + "\n".join(parts)
            + "]"
        ), activated, None
    except Exception:
        return "", [], None


_CITATION_MARKER_RE = re.compile(
    r"【(?:参考)?引用\s*(\d+)】|\[(?:参考)?引用\s*(\d+)\]|[（(](?:参考)?引用\s*(\d+)[）)]"
)


def _canonicalize_citation_markers(content: str) -> str:
    """Normalize accepted citation variants to one clickable UI contract."""
    if not content:
        return content
    return _CITATION_MARKER_RE.sub(
        lambda match: f"【引用{next(group for group in match.groups() if group)}】",
        content,
    )


def _annotate_citation_usage(content: str, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Persist both retrieved candidates and whether each one supports the answer."""
    normalized = _canonicalize_citation_markers(content)
    used = {int(index) for index in re.findall(r"【引用(\d+)】", normalized)}
    annotated: List[Dict[str, Any]] = []
    for index, citation in enumerate(citations, 1):
        item = dict(citation)
        item["index"] = index
        item["used_in_answer"] = index in used
        annotated.append(item)
    return annotated


def _explicit_knowledge_docs(message: str) -> tuple[List[Dict[str, Any]], bool]:
    """Find indexed documents explicitly named by the owner in this turn.

    A user can add a document at runtime and then ask “only rely on 《X》”.
    Retrieval scores alone must not replace that explicit product instruction.
    """
    text = (message or "").replace("《", "").replace("》", "").strip().lower()
    if not text:
        return [], False
    try:
        matched = []
        for doc in list_knowledge_docs() or []:
            title = str(doc.get("title") or "").replace("《", "").replace("》", "").strip()
            if len(title) >= 3 and title.lower() in text and doc.get("is_indexed"):
                matched.append(dict(doc))
        only_requested = any(phrase in (message or "") for phrase in ("只依据", "仅依据", "只根据", "仅根据"))
        return matched, only_requested
    except Exception:
        return [], False


def _unique_rag_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for result in results:
        key = (result.get("doc_id"), result.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def _apply_explicit_document_coverage(
    message: str,
    results: List[Dict[str, Any]],
    settings: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Honor explicit doc-title constraints and cover every named document.

    This is not a title-hardcoded business rule.  It is a generic retrieval
    control: when an owner names one or more currently indexed documents, the
    final evidence set contains the relevant named sources (or, for “only
    rely”, contains no unrelated sources).  Missing named documents simply do
    not fabricate a citation.
    """
    named_docs, only_requested = _explicit_knowledge_docs(message)
    if not named_docs:
        return _unique_rag_results(results)

    named_ids = {doc.get("id") for doc in named_docs}
    selected = [item for item in results if item.get("doc_id") in named_ids]
    if not only_requested:
        selected.extend(item for item in results if item.get("doc_id") not in named_ids)

    existing_ids = {item.get("doc_id") for item in selected}
    # Ask the same local retrieval stack a title-focused query only for a
    # named document that was absent from the first mixed query.  This repairs
    # multi-source questions such as FAQ + service promise without invoking a
    # model or silently lowering the evidence threshold.
    for doc in named_docs:
        doc_id = doc.get("id")
        if doc_id in existing_ids:
            continue
        try:
            focused = rag_retrieval.advanced_search(
                f"{doc.get('title', '')}\n{message}", settings=settings
            ).get("results", [])
            matched = next((item for item in focused if item.get("doc_id") == doc_id), None)
            if matched:
                selected.insert(0, matched)
                existing_ids.add(doc_id)
        except Exception:
            continue

    selected = _unique_rag_results(selected)
    # Named sources come first so citation indexes are stable and visibly
    # correspond to the documents the user asked us to use.
    selected.sort(key=lambda item: (0 if item.get("doc_id") in named_ids else 1, -(float(item.get("score") or 0))))
    top_k = max(1, min(10, int(settings.get("top_k") or 5)))
    return selected[:top_k]


def _build_rag_context(message: str, top_k: Optional[int] = None, threshold: Optional[float] = None) -> tuple:
    """Run advanced RAG and format retrieved chunks as context.

    Returns (rag_context_string, citations).

    The number of retrieved chunks is read from retrieval_settings.top_k unless
    explicitly overridden.  It is clamped to a sensible 1-10 range.
    """
    try:
        from db.property_db import get_retrieval_settings
        settings = get_retrieval_settings("default") or {}
        effective_top_k = top_k if top_k is not None else settings.get("top_k", 5)
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
        results = _apply_explicit_document_coverage(
            message,
            result.get("results", []),
            settings_payload,
        )
        if not results:
            return "", []
        parts = ["\n\n[相关知识库证据（仅对确有证据支持的结论使用【引用n】；不得把未引用候选、常识或实时数据伪装成知识库结论。每个引用必须对应下方确切分片）："]
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
                "context_score": r.get("context_score"),
                "retrieval_sources": r.get("retrieval_sources", []),
                "retrieval_paths": r.get("retrieval_paths", []),
                "evidence_status": r.get("evidence_status", "accepted"),
            })
        parts.append("]")
        return "\n".join(parts), citations
    except Exception:
        return "", []


def _mcp_server_relevant(server_name: str, message: str) -> bool:
    """Return True only when the user message explicitly needs live external data.

    Knowledge-base questions must not trigger MCP tools. Each server is gated
    by narrow intent keywords so that, for example, a repair request does not
    accidentally call a work-order query tool.
    """
    lowered = message.lower()
    relevance = {
        "weather-server": [
            "天气", "气温", "下雨", "雨天", "暴雨", "降雨", "湿度", "天气变化",
            "晴天", "阴天", "多云", "下雪", "刮风", "温度",
        ],
        # Work-order server is for querying existing work orders / progress, not
        # for creating a new repair request.
        "workorder-server": [
            "工单进度", "查询工单", "我的工单", "工单状态", "查看工单",
            "最近工单", "待处理工单", "工单数量", "多少工单", "维修进度",
            "查询房号工单", "工单统计",
        ],
        # Calendar server is for explicit date/time/appointment questions.
        "calendar-server": [
            "今天日期", "现在几点", "当前时间", "今天星期", "今天周几", "今天几号",
            "预约", "几点了", "什么日期", "现在时间", "日期加减", "几号", "星期几",
        ],
    }
    for key, keywords in relevance.items():
        if server_name == key or key in server_name:
            return any(k in lowered for k in keywords)
    # Unknown servers: attach only if explicitly mentioned by name.
    return server_name.lower() in lowered


def _build_mcp_tools(
    agent_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    message: str = "",
    excluded_servers: Optional[set] = None,
) -> List[Any]:
    """Load MCP servers bound to the current agent, filtered by user message."""
    tools = []
    try:
        if agent_id == "router":
            return tools
        from db.property_db import get_agent_tools, list_mcp_servers, list_mcp_tools

        bound_tools = get_agent_tools(agent_id) if agent_id else []
        bound_names = {t.get("tool_name") for t in bound_tools if t.get("tool_name")}
        all_servers = {s.get("name"): s for s in list_mcp_servers() if s.get("enabled")}

        candidate_servers = [all_servers[name] for name in bound_names if name in all_servers]
        if not candidate_servers:
            return tools
        if ObservableMCPTools is None:
            return tools

        for server in candidate_servers:
            name = server.get("name", "mcp-server")
            if excluded_servers and name in excluded_servers:
                continue
            # Formal servers have a narrow deterministic relevance gate.  A
            # user-added server is already an explicit Agent binding, so it is
            # injected as a model-native capability instead of being silently
            # blocked by the old three-server keyword list.
            if is_builtin_policy_server(name) and not _mcp_server_relevant(name, message):
                continue
            discovered_names = [
                str(item.get("name"))
                for item in (list_mcp_tools(server_id=server.get("id")) or [])
                if item.get("name")
            ]
            allowed_function_names = allowed_tools_for_agent(
                agent_id,
                name,
                bound_server_names=bound_names,
                discovered_tool_names=discovered_names,
            )
            if not allowed_function_names:
                # A bound server with no discovered tool is not callable yet.
                # The console should run discovery before it becomes live.
                continue
            command = server.get("command")
            args = server.get("args") or []
            env = server.get("env") or {}
            if not command:
                continue
            merged_env = {**dict(os.environ), **env}
            try:
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
                    invocation_mode="model_native",
                    allowed_function_names=allowed_function_names,
                )
                tool.trace_id = trace_id
                tool.server_name = name
                tools.append(tool)
            except Exception:
                import traceback
                traceback.print_exc()
                continue
    except Exception:
        import traceback
        traceback.print_exc()
    return tools


def _format_mcp_context(
    agent_id: Optional[str] = None,
    message: str = "",
    excluded_servers: Optional[set] = None,
) -> str:
    """Format MCP servers bound to the current agent, filtered by user message."""
    try:
        if agent_id == "router":
            return ""
        from db.property_db import get_agent_tools, list_mcp_servers, list_mcp_tools

        bound_tools = get_agent_tools(agent_id) if agent_id else []
        bound_names = {t.get("tool_name") for t in bound_tools if t.get("tool_name")}
        excluded_servers = excluded_servers or set()
        servers = [
            s for s in list_mcp_servers()
            if s.get("enabled")
            and s.get("name") in bound_names
            and s.get("name") not in excluded_servers
            and (not is_builtin_policy_server(str(s.get("name") or "")) or _mcp_server_relevant(s.get("name", ""), message))
        ]
        if not servers:
            return ""
        parts = []
        for server in servers:
            name = server.get("name", "")
            description = server.get("description", "")
            discovered_names = [
                str(item.get("name"))
                for item in (list_mcp_tools(server_id=server.get("id")) or [])
                if item.get("name")
            ]
            allowed_function_names = sorted(allowed_tools_for_agent(
                agent_id,
                name,
                bound_server_names=bound_names,
                discovered_tool_names=discovered_names,
            ))
            if name and not allowed_function_names:
                continue
            if name:
                mode = "内置只读策略" if is_builtin_policy_server(name) else "动态绑定（模型自主调用）"
                parts.append(f"- {mode}: {name} -> {', '.join(allowed_function_names)}")
                parts.append(f"- {name}：{description or '无描述'}")
        if not parts:
            return ""
        return (
            "\n\n[已启用的 MCP Server 工具（当用户问题涉及以下能力时，你必须在回答前先调用对应工具；"
            "禁止基于自身知识猜测，必须实际调用工具获取结果后再回复用户）：\n"
            + "\n".join(parts)
            + "]"
        )
    except Exception:
        return ""


# V1.4.3: narrow policy pre-invocation for the three formal readonly MCP servers.
# Only runs when the current agent is explicitly bound to the server and the user
# message explicitly requests that live capability. Results are injected into the
# agent context and recorded in the trace as invocation_mode=policy_preinvoke.
_READONLY_MCP_DEFAULT_TOOLS = {
    "weather-server": ["get_current_weather", "get_weather_advice"],
    "workorder-server": ["get_my_recent_work_orders", "count_my_open_work_orders", "count_work_orders"],
    "calendar-server": ["get_current_datetime"],
}


def _policy_preinvoke_plan(server_name: str, message: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Choose the smallest deterministic read-only MCP call set for one turn.

    DeepSeek thinking mode cannot safely receive a forced ``tool_choice``.  A
    narrow policy pre-invocation therefore covers only stable owner-facing
    reads, while the Agent keeps its allowlisted tools for any extra work.
    """
    if server_name == "weather-server":
        cities = ("北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安")
        city = next((item for item in cities if item in message), None)
        if not city:
            return []
        plan = [("get_current_weather", {"city": city})]
        if any(word in message for word in ("建议", "风险", "上门", "户外", "暴雨", "台风", "下雨", "降雨")):
            plan.append(("get_weather_advice", {"city": city}))
        return plan

    if server_name == "workorder-server":
        plan: List[Tuple[str, Dict[str, Any]]] = []
        asks_recent = any(word in message for word in (
            "最近工单", "房号最近", "我的工单", "我家工单", "相似工单", "维修记录", "工单进度",
        ))
        asks_open = any(word in message for word in ("待处理", "待派单", "未处理", "还有多少", "未关闭"))
        asks_system_total = any(word in message for word in (
            "系统当前", "全小区", "系统中", "系统当前待处理", "待处理工单数量", "总量", "总数",
        ))
        asks_recent = asks_recent or ("\u6700\u8fd1" in message and ("\u7ef4\u4fee\u5de5\u5355" in message or "\u623f\u53f7" in message))
        if asks_recent:
            plan.append(("get_my_recent_work_orders", {"limit": 5}))
        if asks_open:
            plan.append(("count_my_open_work_orders", {}))
        if asks_system_total:
            plan.append(("count_work_orders", {"status": "待处理"} if "待处理" in message else {}))
        return plan

    if server_name == "calendar-server":
        if any(word in message for word in ("今天", "当前时间", "现在几点", "星期几", "几号", "日期")):
            return [("get_current_datetime", {})]
    return []


def _policy_mcp_args(server_name: str, tool_name: str, message: str) -> Optional[Dict[str, Any]]:
    """Build explicit arguments for a policy pre-invocation, or skip it safely."""
    cities = ("北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安")
    if server_name == "weather-server":
        city = next((item for item in cities if item in message), None)
        return {"city": city} if city else None
    if server_name == "workorder-server":
        room_match = re.search(r"(\d{1,2})\s*[-#—]\s*(\d{1,2})\s*[-#—]\s*(\d{3,4})", message)
        room = (
            f"{room_match.group(1)}-{room_match.group(2)}-{room_match.group(3)}"
            if room_match else (DEFAULT_ROOM_ID if any(word in message for word in ("我的", "我家", "本房号")) else None)
        )
        if tool_name == "list_recent_work_orders":
            return {"room_id": room, "limit": 5}
        if tool_name == "count_work_orders":
            return {"status": "pending"} if any(word in message for word in ("待处理", "待派单", "未处理")) else {}
    if server_name == "calendar-server":
        return {}
    return {}


async def _preinvoke_readonly_mcp_tools(
    agent_id: Optional[str],
    message: str,
    trace_id: Optional[str],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Pre-invoke bound readonly MCP servers when the user explicitly needs them.

    Returns (context_string, call_records). The context string tells the agent
    the real results so it does not need to call the same tools again.
    """
    if not agent_id or agent_id == "router" or ObservableMCPTools is None:
        return "", []

    try:
        from db.property_db import get_agent_tools, list_mcp_servers

        bound_tools = get_agent_tools(agent_id) or []
        bound_names = {t.get("tool_name") for t in bound_tools if t.get("tool_name")}
        servers = {
            s.get("name"): s
            for s in list_mcp_servers()
            if s.get("enabled") and s.get("name") in bound_names and _mcp_server_relevant(s.get("name", ""), message)
        }
    except Exception:
        return "", []

    if not servers:
        return "", []

    context_parts: List[str] = []
    call_records: List[Dict[str, Any]] = []

    for server_name, server in servers.items():
        default_tools = _READONLY_MCP_DEFAULT_TOOLS.get(server_name, [])
        allowed_function_names = allowed_tools_for_agent(agent_id, server_name)
        planned_calls = _policy_preinvoke_plan(server_name, message)
        if not default_tools or not allowed_function_names or not planned_calls:
            continue

        command = server.get("command")
        args = server.get("args") or []
        env = server.get("env") or {}
        if not command:
            continue

        import shlex

        merged_env = {**dict(os.environ), **env}
        full_command = shlex.join([command] + list(args))

        tool: Any = None
        try:
            tool = ObservableMCPTools(
                command=full_command,
                env=merged_env,
                name=server_name,
                transport="stdio",
                timeout_seconds=15,
                invocation_mode="policy_preinvoke",
                allowed_function_names=allowed_function_names,
            )
            tool.trace_id = trace_id
            tool.server_name = server_name
            if hasattr(tool, "connect"):
                await asyncio.wait_for(tool.connect(), timeout=5)
            await asyncio.wait_for(tool.build_tools(), timeout=8)
        except Exception as exc:
            error_summary = str(exc)[:300]
            # A failed discovery is still evidence.  Do not silently fall back
            # to a second model-native connection that can hold the chat stream
            # open until the reverse-proxy timeout.
            call_records.append({
                "server_name": server_name,
                "tool_name": "discovery",
                "arguments": {},
                "status": "upstream_error",
                "result_summary": error_summary,
                "error_summary": error_summary,
                "latency_ms": None,
                "invocation_mode": "policy_preinvoke",
            })
            context_parts.append(
                f"[{server_name}:discovery] policy_preinvoke outcome=upstream_error; "
                f"{outcome_instruction('upstream_error')}; result={error_summary}"
            )
            if tool and hasattr(tool, "close"):
                try:
                    await asyncio.wait_for(tool.close(), timeout=3)
                except Exception:
                    pass
            continue

        functions = getattr(tool, "functions", None) or {}
        for fn_name, arguments in planned_calls:
            if fn_name not in default_tools or fn_name not in allowed_function_names:
                continue
            fn = functions.get(fn_name)
            if fn is None:
                continue
            entrypoint = getattr(fn, "entrypoint", None)
            if not entrypoint:
                continue
            start = time.time()
            status = "success"
            result_summary = ""
            error_summary = None
            try:
                result = await asyncio.wait_for(entrypoint(**arguments), timeout=8)
                result_summary = _summarize_tool_result(result)
                status = extract_tool_outcome(result_summary)
                if status in {"timeout", "upstream_error"}:
                    error_summary = result_summary[:300]
            except Exception as exc:
                status = "upstream_error"
                error_summary = str(exc)[:300]
                result_summary = error_summary
            finally:
                latency_ms = int((time.time() - start) * 1000)
                call_records.append({
                    "server_name": server_name,
                    "tool_name": fn_name,
                    "arguments": arguments,
                    "status": status,
                    "result_summary": result_summary,
                    "error_summary": error_summary,
                    "latency_ms": latency_ms,
                    "invocation_mode": "policy_preinvoke",
                })
                context_parts.append(
                    f"[{server_name}:{fn_name}] policy_preinvoke "
                    f"outcome={status}; {outcome_instruction(status)}; result={result_summary}"
                )
                # ObservableMCPTools wraps entrypoint() and writes the one authoritative
                # audit row.  call_records above is only the in-memory done-event list.

        if hasattr(tool, "close"):
            try:
                await asyncio.wait_for(tool.close(), timeout=3)
            except Exception:
                pass

    if not context_parts:
        return "", call_records

    context_str = (
        "\n\n[以下 MCP 工具已由系统策略预先真实调用，结果已注入上下文；"
        "你在后续回答中不要再次重复调用这些工具：\n"
        + "\n".join(context_parts)
        + "]"
    )
    return context_str, call_records


def _detect_handoff_intent(message: str) -> Optional[str]:
    """Backward-compatible explicit-request detector.

    Responsibility changes are decided by :mod:`app.handoff_policy`, never by
    a free-form model sentence.  This helper remains for the stream's compact
    early branch only.
    """
    policy = evaluate_handoff_policy(message)
    return policy["reason"] if policy.get("reason_code") == "owner_requested" else None


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


def _get_vertical_agents() -> List[Dict[str, Any]]:
    """Load the live vertical-Agent registry with its bound capabilities.

    This is intentionally rebuilt for every owner turn.  Creating/enabling an
    Agent or binding a Skill/MCP server in the platform console therefore takes
    effect on the *next message* without a restart, seed migration or a hidden
    five-Agent allowlist.
    """
    try:
        servers_by_name = {
            str(server.get("name")): server
            for server in (list_mcp_servers() or [])
            if server.get("enabled") and server.get("name")
        }
        result: List[Dict[str, Any]] = []
        for source in list_agents(category="vertical") or []:
            agent = dict(source)
            agent_id = str(agent.get("agent_id") or "")
            skills: List[Dict[str, Any]] = []
            for skill_id in get_agent_skills(agent_id) or []:
                skill = get_skill(int(skill_id))
                if not skill or not skill.get("enabled"):
                    continue
                contract = skill_contract(skill)
                skills.append({
                    "id": skill.get("id"),
                    "name": skill.get("name"),
                    "positive_triggers": contract.get("positive_triggers", []),
                    "negative_triggers": contract.get("negative_triggers", []),
                    "tool_hints": contract.get("tool_hints", []),
                })

            mcp_servers: List[Dict[str, Any]] = []
            for binding in get_agent_tools(agent_id) or []:
                server_name = str(binding.get("tool_name") or "")
                server = servers_by_name.get(server_name)
                if not server:
                    continue
                discovered = [
                    str(tool.get("name"))
                    for tool in (list_mcp_tools(server_id=server.get("id")) or [])
                    if tool.get("name")
                ]
                mcp_servers.append({
                    "name": server_name,
                    "description": server.get("description") or "",
                    "tools": discovered,
                    "runtime_mode": "policy_preinvoke" if is_builtin_policy_server(server_name) else "model_native_dynamic",
                })

            # Router only needs a compact, inspectable capability card.  The
            # full prompt remains private to the selected runtime Agent.
            agent["capability_card"] = {
                "service_scope": agent.get("description") or "",
                "routing_hints": (agent.get("instructions") or "")[:480],
                "skills": skills,
                "mcp_servers": mcp_servers,
                "effective_on_next_message": bool(agent.get("enabled")),
            }
            result.append(agent)
        return result
    except Exception:
        return []


def _select_agent(agent_id: str, tools: Optional[List[Any]] = None, mcp_context: str = ""):
    """Return a factory that creates the vertical agent for target_agent_id.

    A factory is returned (instead of an Agent instance) so the chat runtime
    can instantiate the agent with the model selected for this turn.
    """
    from functools import partial

    factory = partial(_create_vertical_agent_for_id, agent_id, mcp_context=mcp_context)
    # Determine the display name without constructing the agent.
    db_agent = get_agent_by_agent_id(agent_id)
    agent_name = db_agent.get("name") if db_agent else agent_id
    canonical_labels = {
        "maintenance": "维修 Agent",
        "billing": "费用 Agent",
        "complaint": "投诉 Agent",
        "customer_service": "客服 Agent",
    }
    # Preserve an operator-created Agent's own name.  The mapping only repairs
    # old canonical rows that were accidentally displayed as an English id.
    if not agent_name or str(agent_name).strip() == agent_id:
        agent_name = canonical_labels.get(agent_id, agent_id)
    return factory, agent_name


def _agent_id_for_intent(intent: str) -> str:
    """Return the canonical agent_id used for Skill/MCP binding lookups."""
    # Intent is already an agent_id when using the dynamic router.
    return intent if intent else "customer_service"


def _create_vertical_agent_for_id(
    agent_id: str,
    tools: Optional[List[Any]] = None,
    model: Optional[Any] = None,
    mcp_context: str = "",
):
    """Create a vertical Agent instance from DB configuration, falling back to code defaults."""
    db_agent = get_agent_by_agent_id(agent_id)
    instructions = None
    name = None
    description = None
    if db_agent:
        raw_instructions = db_agent.get("instructions") or ""
        if raw_instructions.strip():
            # Allow either newline-separated or list JSON.
            try:
                parsed = json.loads(raw_instructions)
                if isinstance(parsed, list):
                    instructions = [str(x) for x in parsed]
            except Exception:
                instructions = [line.strip() for line in raw_instructions.split("\n") if line.strip()]
        name = db_agent.get("name")
        description = db_agent.get("description")

    # Inject MCP admission context into instructions so the model knows it must
    # call the admitted tools rather than hallucinate answers.
    if mcp_context:
        if instructions is None:
            instructions = []
        elif isinstance(instructions, str):
            instructions = [instructions]
        instructions = list(instructions) + [mcp_context]

    # Map canonical agent IDs to their factory functions so we keep base tools.
    factories = {
        "maintenance": create_maintenance_agent,
        "billing": create_billing_agent,
        "complaint": create_complaint_agent,
        "customer_service": create_customer_service_agent,
    }
    factory = factories.get(agent_id)
    if factory:
        return factory(tools=tools, instructions=instructions, name=name, description=description, model=model)

    # For dynamically created vertical agents without a dedicated factory, build a generic Agent.
    from agno.agent import Agent
    from app.settings import MODEL, agent_db
    from tools.knowledge import KnowledgeTools

    base_tools = []
    try:
        base_tools.append(KnowledgeTools())
    except Exception:
        pass
    if tools:
        base_tools.extend(tools)
    return Agent(
        id=f"{agent_id}_agent",
        name=name or agent_id,
        description=description or "",
        model=model or MODEL,
        db=agent_db,
        tools=base_tools,
        skills=None,
        instructions=instructions or ["你是YIAI物业的专属 Agent，请基于知识库与绑定工具回答业主问题。"],
        add_datetime_to_context=True,
        add_history_to_context=True,
        read_chat_history=True,
        num_history_runs=2,
        markdown=True,
    )


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
    # Some Agno response metadata objects describe the tool schema rather
    # than an invocation.  They have no callable name and must not be sent
    # to the UI or persisted as a phantom tool call.
    tool_calls = [call for call in tool_calls if str(call.get("tool_name") or "").strip()]
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
    image_analysis_ids: List[str] = []


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
    image_analysis_ids: Optional[List[str]] = None,
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
    # Early confirmation returns must not leave final cleanup with an undefined local.
    mcp_tools: List[Any] = []

    try:
        # Ensure session exists and check handoff state.
        ensure_chat_session(session_id)

        # Create the trace record for this turn.
        create_chat_trace(trace_id=trace_id, session_id=session_id, user_message=message)
        record_trace_event(
            trace_id,
            "request_received",
            input_summary=message[:240],
            metadata={"session_id": session_id, "channel": "owner_chat"},
        )

        # First send a "start" event
        yield f"event: start\ndata: {json.dumps({'session_id': session_id, 'trace_id': trace_id})}\n\n"

        # Persist the user message before invoking the agent.
        save_chat_message(session_id=session_id, role="user", content=message, trace_id=trace_id)

        # Kimi is invoked by the dedicated image endpoint before this text chain.
        # Only its compact structured output is injected here; the Flash router
        # and vertical Agent never receive raw image bytes or get to treat OCR as
        # a system instruction.  A missing/invalid id simply contributes no
        # image context, rather than inventing visual evidence.
        image_context = get_analysis_context(normalise_analysis_ids(image_analysis_ids))
        message_for_agent = f"{message}{image_context}"

        # Human collaboration is controlled by deterministic policy, not by a
        # model sentence.  It makes the accountability boundary explainable and
        # prevents a casual mention of customer service from taking over a chat.
        handoff_policy = evaluate_handoff_policy(message)
        if handoff_policy.get("should_request_handoff"):
            handoff_session = _request_handoff_with_context(
                session_id, handoff_policy, trace_id=trace_id, trigger_message=message, actor="owner"
            )
            reply = (
                "已为您发起人工协同处理。当前状态：等待工作人员领取。\n\n"
                f"原因：{handoff_policy.get('reason')}\n"
                "AI 将不再代替工作人员作出后续处理决定；您可以继续补充信息，补充内容会同步到接管包。"
            )
            saved = save_chat_message(
                session_id=session_id,
                role="assistant",
                content=reply,
                trace_id=trace_id,
                current_agent="人工协同控制器",
                current_agent_id="human_copilot",
                status="complete",
                usage_source="not_applicable",
            )
            update_chat_trace(trace_id=trace_id, intent="handoff", agent_name="人工协同控制器", agent_id="human_copilot", status="complete")
            record_trace_event(
                trace_id, "handoff_policy", "success",
                latency_ms=int((time.time() - trace_start) * 1000),
                output_summary=handoff_policy.get("reason", ""),
                metadata={"handoff": True, "policy": handoff_policy},
            )
            yield f"event: delta\ndata: {json.dumps({'content': reply}, ensure_ascii=False)}\n\n"
            yield f"event: done\ndata: {_safe_json_dumps({'status': 'complete', 'token_count': 0, 'message_id': saved.get('id') if saved else None, 'handoff': True, 'handoff_state': handoff_session.get('handoff_status'), 'handoff_policy': handoff_policy, 'handoff_package_available': True, 'trace_id': trace_id, 'usage_source': 'not_applicable'})}\n\n"
            done_yielded = True
            return

        current_handoff = get_chat_session(session_id) or {}
        handoff_status = current_handoff.get("handoff_status") or "none"
        if handoff_status == "waiting_user":
            handoff_session = resume_handoff_after_owner_message(session_id)
            reply = "已将您补充的信息同步给接管工作人员，人工处理已恢复。"
        elif handoff_status in {"requested", "active"}:
            handoff_session = current_handoff
            reply = "当前会话正在人工协同处理中。您的补充信息已保存并会同步给工作人员，AI 不会重复作出处理决定。"
        else:
            handoff_session = None
            reply = ""
        if handoff_session is not None:
            saved = save_chat_message(
                session_id=session_id,
                role="assistant",
                content=reply,
                trace_id=trace_id,
                current_agent="人工协同控制器",
                current_agent_id="human_copilot",
                status="complete",
                usage_source="not_applicable",
            )
            update_chat_trace(trace_id=trace_id, intent="handoff", agent_name="人工协同控制器", agent_id="human_copilot", status="complete")
            record_trace_event(
                trace_id, "handoff_state", "success",
                latency_ms=int((time.time() - trace_start) * 1000),
                output_summary=reply,
                metadata={"handoff_status": handoff_session.get("handoff_status")},
            )
            yield f"event: delta\ndata: {json.dumps({'content': reply}, ensure_ascii=False)}\n\n"
            yield f"event: done\ndata: {_safe_json_dumps({'status': 'complete', 'token_count': 0, 'message_id': saved.get('id') if saved else None, 'handoff': True, 'handoff_state': handoff_session.get('handoff_status'), 'handoff_package_available': True, 'trace_id': trace_id, 'usage_source': 'not_applicable'})}\n\n"
            done_yielded = True
            return

        # Repair work orders are a stateful command workflow, not a best-effort
        # model tool call.  A pending draft pins follow-up turns to Maintenance;
        # a formal work order is created exactly once only on explicit confirmation.
        workflow = advance_work_order_workflow(session_id, message)
        if workflow is not None:
            current_agent_id = "maintenance"
            _, current_agent = _select_agent(current_agent_id)
            route_reason = workflow.get("route_reason") or "已进入维修工单流程。"
            workflow_call = {
                "tool_name": "work_order_workflow",
                "arguments": {
                    "action": workflow.get("action"),
                    "missing_fields": workflow.get("missing_fields", []),
                },
                "status": "success",
                "result_summary": workflow.get("reply", ""),
                "work_order_id": workflow.get("work_order_id"),
            }
            update_chat_trace(
                trace_id=trace_id,
                intent="maintenance",
                agent_name=current_agent,
                agent_id=current_agent_id,
                status="complete",
            )
            yield f"event: route\ndata: {json.dumps({'intent': 'maintenance', 'reason': route_reason, 'current_agent': current_agent, 'current_agent_id': current_agent_id, 'trace_id': trace_id})}\n\n"
            reply = workflow.get("reply", "")
            record_trace_event(
                trace_id, "work_order_workflow", "success",
                latency_ms=int((time.time() - trace_start) * 1000),
                output_summary=reply[:240],
                metadata={"action": workflow.get("action"), "work_order_id": workflow.get("work_order_id")},
            )
            yield f"event: delta\ndata: {_safe_json_dumps({'content': reply, 'current_agent': current_agent, 'current_agent_id': current_agent_id})}\n\n"
            yield f"event: tool_calls\ndata: {_safe_json_dumps({'tool_calls': [workflow_call], 'current_agent': current_agent, 'current_agent_id': current_agent_id})}\n\n"
            saved = save_chat_message(
                session_id=session_id,
                role="assistant",
                content=reply,
                token_count=0,
                token_detail={"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0, "total_tokens": 0},
                citations=[],
                activated_skills=[],
                route_intent="maintenance",
                route_reason=route_reason,
                current_agent=current_agent,
                current_agent_id=current_agent_id,
                tool_calls=[workflow_call],
                model_id=None,
                thinking_enabled=False,
                model_selection_reason="workflow_controller",
                trace_id=trace_id,
                status="success",
                latency_ms=int((time.time() - trace_start) * 1000),
                error_summary=None,
                mcp_calls=None,
                usage_source="not_applicable",
            )
            yield f"event: done\ndata: {_safe_json_dumps({'status': 'complete', 'token_count': 0, 'token_detail': {'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0, 'total_tokens': 0}, 'message_id': saved.get('id') if saved else None, 'handoff': False, 'citations': [], 'activated_skills': [], 'current_agent': current_agent, 'current_agent_id': current_agent_id, 'route_intent': 'maintenance', 'route_reason': route_reason, 'tool_calls': [workflow_call], 'mcp_calls': [], 'auto_badcase_id': None, 'model_id': None, 'thinking_enabled': False, 'model_selection_reason': 'workflow_controller', 'trace_id': trace_id, 'usage_source': 'not_applicable'})}\n\n"
            done_yielded = True
            return

        # Load enabled vertical agents from the database for dynamic routing.
        vertical_agents = _get_vertical_agents()

        # Classify intent and dispatch to the appropriate vertical agent.
        router_start = time.time()
        intent_result = await classify_intent(
            message_for_agent, vertical_agents=vertical_agents, user_id=user_id, session_id=session_id
        )
        router_latency_ms = int((time.time() - router_start) * 1000)
        target_agent_id = intent_result.get("target_agent_id") or intent_result.get("intent") or "customer_service"
        intent = target_agent_id

        # Explicit repair creation and any repair follow-up were handled above
        # by the session workflow controller.  Ordinary repair consultation still
        # uses the normal Router -> vertical Agent path below.

        create_agent_fn, agent_name = _select_agent(target_agent_id)
        current_agent_id = _agent_id_for_intent(target_agent_id)
        current_agent = agent_name  # human-readable Chinese name for UI/Trace

        router_token_count = 0
        # Record the router model call. Preserve the configured price snapshot so the
        # UI can show "单价已配置，但 Provider 未返回本次 Router usage".
        try:
            router_price = _get_price_snapshot(MODEL_ID)
            router_metrics = intent_result.get("metrics") or {}
            router_token_count = int(router_metrics.get("total_tokens") or 0)
            has_router_usage = bool(router_metrics.get("total_tokens"))
            record_model_call(
                trace_id=trace_id,
                stage="router",
                model_id=MODEL_ID,
                model_selection_reason=f"router selected {current_agent_id} ({current_agent})",
                latency_ms=router_latency_ms,
                input_tokens=router_metrics.get("input_tokens"),
                output_tokens=router_metrics.get("output_tokens"),
                reasoning_tokens=router_metrics.get("reasoning_tokens"),
                cached_tokens=router_metrics.get("cached_tokens"),
                total_tokens=router_metrics.get("total_tokens"),
                usage_source="provider_reported" if has_router_usage else "unavailable",
                status="success",
                estimated_cost_cny=None,
                price_snapshot=router_price,
                context_breakdown={
                    "route_mode": intent_result.get("route_mode", "unknown"),
                    "reason": intent_result.get("reason", ""),
                    "capability_candidates": intent_result.get("fallback_scores", []),
                },
                usage_normalized=normalize_usage(router_metrics) if has_router_usage else None,
            )
        except Exception:
            pass

        record_trace_event(
            trace_id,
            "router",
            "success",
            latency_ms=router_latency_ms,
            input_summary=message[:240],
            output_summary=intent_result.get("reason", "")[:240],
            metadata={
                "intent": intent,
                "agent_id": current_agent_id,
                "agent_name": current_agent,
                "route_mode": intent_result.get("route_mode", "unknown"),
                "capability_candidates": intent_result.get("fallback_scores", []),
            },
        )

        # Yield routing event so the UI can show which agent is handling the request.
        yield f"event: route\ndata: {json.dumps({'intent': intent, 'reason': intent_result.get('reason', ''), 'current_agent': current_agent, 'current_agent_id': current_agent_id, 'trace_id': trace_id})}\n\n"

        # Build dynamic context and tools scoped to the current vertical agent.
        skill_context, activated_skills, skill_model_id = _build_skill_context(message, agent_id=current_agent_id)
        rag_started_at = time.time()
        rag_context, citations = _build_rag_context(message_for_agent)
        rag_latency_ms = int((time.time() - rag_started_at) * 1000)
        record_trace_event(
            trace_id,
            "rag_retrieval",
            "success" if citations else "empty",
            latency_ms=rag_latency_ms,
            input_summary=message_for_agent[:240],
            output_summary=f"retrieved {len(citations)} candidate chunks",
            metadata={
                "candidate_count": len(citations),
                "citation_docs": [c.get("doc_title") for c in citations],
                "citation_chunks": [c.get("chunk_index") for c in citations],
            },
        )
        # V1.4.3: narrow policy pre-invocation for readonly MCP servers. Real
        # tool results are injected into context and recorded as policy_preinvoke.
        preinvoke_context, preinvoke_calls = await _preinvoke_readonly_mcp_tools(
            agent_id=current_agent_id,
            message=message,
            trace_id=trace_id,
        )
        # A formal server is executed through exactly one path per turn. Even
        # a discovery failure remains visible evidence rather than triggering
        # a silent second model-native connection.
        policy_managed_servers = {
            str(call.get("server_name"))
            for call in preinvoke_calls
            if call.get("server_name") and is_builtin_policy_server(str(call.get("server_name")))
        }
        mcp_context = _format_mcp_context(
            agent_id=current_agent_id,
            message=message,
            excluded_servers=policy_managed_servers,
        )
        if preinvoke_context:
            mcp_context = f"{mcp_context}{preinvoke_context}"

        # Formal work-order writes have already been guarded by the workflow controller.
        # This model path is for consultation, RAG and read-only MCP only.
        # A successful (or auditable failed) policy read is already injected in
        # context.  Do not start the same builtin stdio server a second time for
        # model-native tools: that duplicate discovery was serial and could use
        # up the whole reverse-proxy timeout before the vertical answer began.
        # User-added dynamic MCP servers remain model-native and extensible.
        mcp_tools = _build_mcp_tools(
            agent_id=current_agent_id,
            trace_id=trace_id,
            message=message,
            excluded_servers=policy_managed_servers,
        )
        for tool in mcp_tools:
            if hasattr(tool, "connect"):
                try:
                    await tool.connect()
                except Exception:
                    pass

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
        system_context_prefix = (
            f"[系统上下文：当前业主是 {DEFAULT_ROOM_ID} 的{DEFAULT_OWNER_NAME}，"
            f"如果用户没有提供房号，创建工单时默认使用 {DEFAULT_ROOM_ID}。"
            f"当用户明确要求人工、表达强烈不满、或问题超出物业维修/收费/知识库范围时，"
            f"你必须主动提出转人工处理，不要强行回答。]"
            f"{knowledge_gap_note}"
        )
        response_style = (
            "\n\n[Answer style: lead with the conclusion; do not repeat the question or narrate progress. "
            "Keep the default answer under 900 Chinese characters and retain only directly relevant facts.]"
        )
        contextual_message = f"{system_context_prefix}{rag_context}{skill_context}{mcp_context}{response_style}\n{message_for_agent}"

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
        agent = create_agent_fn(tools=mcp_tools, model=turn_model, mcp_context=mcp_context)
        # V1.4.3: never force tool_choice="required" when thinking is enabled,
        # because DeepSeek's thinking mode rejects that value. Leave tool_choice
        # unset for the provider default.
        if mcp_tools and not USE_THINKING:
            agent.tool_choice = "required"
        elif not mcp_tools:
            agent.tool_choice = "auto"
        vertical_start = time.time()

        # Run agent in streaming mode.  We decouple Agno's async generator from
        # the HTTP SSE generator via an asyncio.Queue so that any quirks in the
        # Agno generator lifecycle (especially after MCP tool-call turns) do not
        # abort the SSE response before we can send the done event.
        sse_queue: asyncio.Queue[Tuple[str, Any]] = asyncio.Queue()

        async def _produce_chunks() -> None:
            # Work around Agno streaming + async MCP tool interactions by using a
            # synchronous run whenever MCP tools are admitted for this turn.
            if mcp_tools:
                response = await agent.arun(
                    contextual_message,
                    user_id=user_id,
                    session_id=session_id,
                    stream=False,
                )
                content = ""
                if hasattr(response, "content") and response.content:
                    content = str(response.content)
                elif isinstance(response, str):
                    content = response
                if content:
                    sse_queue.put_nowait(("delta", content))
                resp_tools = _extract_tool_calls(response)
                if resp_tools:
                    sse_queue.put_nowait(("tool_calls", resp_tools))
                if hasattr(response, "metrics") and response.metrics:
                    try:
                        metrics = response.metrics
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
                return

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
                    yield f"event: delta\ndata: {json.dumps({'content': payload, 'current_agent': current_agent, 'current_agent_id': current_agent_id})}\n\n"
                elif kind == "tool_calls":
                    tool_calls.extend(payload)
                    yield f"event: tool_calls\ndata: {_safe_json_dumps({'tool_calls': payload, 'current_agent': current_agent, 'current_agent_id': current_agent_id})}\n\n"
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
                        yield f"event: tool_calls\ndata: {_safe_json_dumps({'tool_calls': [normalized], 'current_agent': current_agent, 'current_agent_id': current_agent_id})}\n\n"

        # Derive token_count from total_tokens or input+output when possible.
        if token_detail["total_tokens"]:
            token_count = token_detail["total_tokens"]
        elif token_detail["input_tokens"] or token_detail["output_tokens"]:
            token_count = (token_detail["input_tokens"] or 0) + (token_detail["output_tokens"] or 0)

        # Fall back to tiktoken estimate if the model did not report metrics.
        if not token_count and full_content:
            output_tokens = _estimate_tokens(full_content) or 0
            input_tokens = _estimate_tokens(contextual_message) or 0
            token_count = input_tokens + output_tokens
            token_detail["input_tokens"] = input_tokens
            token_detail["output_tokens"] = output_tokens
            token_detail["total_tokens"] = token_count

        # Router and vertical Agent are separate model calls. Keep the business
        # answer's token count readable, while exposing the actual full-turn total.
        round_token_count = token_count + router_token_count

        # Normalize model citation variants before persistence so citations are
        # always clickable and retrieved candidates can be distinguished from evidence.
        full_content = _canonicalize_citation_markers(full_content)
        citations = _annotate_citation_usage(full_content, citations)

        # A natural-language phrase such as “建议转人工” must not mutate the
        # responsibility state.  The model may recommend escalation, but only
        # deterministic policy or an explicit owner/staff action creates a
        # handoff record.
        ai_handoff = False

        # Determine the model actually used for this turn.
        runtime_model_id = skill_model_id if skill_model_id else MODEL_ID
        model_selection_reason = (
            f"skill_model_override:{skill_model_id}"
            if skill_model_id
            else "owner-facing default"
        )
        turn_model_id = runtime_model_id
        turn_selection_reason = model_selection_reason

        # Build an honest local estimate of the prompt context composition.
        # Tool-result summaries are only available after the MCP wrappers have
        # recorded their calls.
        tool_result_summaries = []
        for toolkit in mcp_tools:
            if hasattr(toolkit, "recorded_calls") and toolkit.recorded_calls:
                for call in toolkit.recorded_calls:
                    summary = call.get("result_summary") or call.get("error_summary") or ""
                    if summary:
                        tool_result_summaries.append(summary)
        try:
            history_messages = list_chat_messages(session_id)
        except Exception:
            history_messages = []
        try:
            agent_instructions = getattr(agent, "instructions", None) or []
        except Exception:
            agent_instructions = []
        context_breakdown = _estimate_context_breakdown(
            system_prompt=agent_instructions,
            history_messages=history_messages,
            skill_context=skill_context,
            rag_context=rag_context,
            tool_results=tool_result_summaries or None,
            user_message=message,
            system_context_prefix=system_context_prefix,
        )

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
                context_breakdown=context_breakdown,
                usage_normalized=normalize_usage(token_detail),
            )
        except Exception:
            pass

        # Update the trace record with final intent/agent/status.
        try:
            update_chat_trace(
                trace_id=trace_id,
                intent=intent,
                agent_name=current_agent,
                agent_id=current_agent_id,
                status="failed" if error_yielded else "complete",
            )
        except Exception:
            pass

        record_trace_event(
            trace_id,
            "final_response",
            "success",
            latency_ms=int((time.time() - trace_start) * 1000),
            output_summary=full_content[:240],
            metadata={
                "agent_id": current_agent_id,
                "skills": activated_skills,
                "citations_used": [c.get("doc_title") for c in citations if c.get("used_in_answer")],
                "mcp_call_count": len(preinvoke_calls) + sum(len(getattr(toolkit, "recorded_calls", []) or []) for toolkit in mcp_tools),
                "handoff": ai_handoff,
            },
        )

        # Build MCP audit list for the done event and persistence.
        mcp_calls_for_done: List[Dict[str, Any]] = []
        # Pre-invoked calls first (policy_preinvoke).
        for call in preinvoke_calls:
            mcp_calls_for_done.append({
                "server_name": call.get("server_name", "unknown"),
                "tool_name": call.get("tool_name", ""),
                "arguments": call.get("arguments", {}),
                "status": call.get("status", "success"),
                "result_summary": call.get("result_summary", ""),
                "error_summary": call.get("error_summary"),
                "latency_ms": call.get("latency_ms"),
                "invocation_mode": call.get("invocation_mode", "policy_preinvoke"),
            })
        # Model-native calls recorded by ObservableMCPTools wrappers.
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
                        "invocation_mode": "model_native",
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
            round_token_count=round_token_count,
            token_detail=token_detail,
            citations=citations,
            activated_skills=activated_skills,
            route_intent=intent,
            route_reason=intent_result.get("reason", ""),
            current_agent=current_agent,
            current_agent_id=current_agent_id,
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
            'round_token_count': round_token_count,
            'router_token_count': router_token_count,
            'token_detail': token_detail,
            'message_id': saved.get('id'),
            'handoff': ai_handoff,
            'citations': citations,
            'activated_skills': activated_skills,
            'current_agent': current_agent,
            'current_agent_id': current_agent_id,
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
            record_trace_event(trace_id, "final_response", "failed", output_summary=str(e)[:240])
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
        # Clean up MCP server stdio sessions so we don't leak subprocesses.
        for toolkit in mcp_tools:
            if hasattr(toolkit, "close"):
                try:
                    await toolkit.close()
                except Exception:
                    pass


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """Stream an agent response via Server-Sent Events."""

    session_id = request.session_id or f"web-{uuid.uuid4().hex[:12]}"
    user_id = request.user_id or "web-user"

    return StreamingResponse(
        _stream_agent_response(request.message, session_id, user_id, request.image_analysis_ids),
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
        _stream_agent_response(message, session_id, user_id, None),
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
