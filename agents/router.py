"""
Router Agent
============

Classifies owner intent and dispatches to the appropriate vertical agent.
"""

import json
import inspect
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from agno.agent import Agent

from app.settings import MODEL, agent_db
from db.property_db import get_agent_by_agent_id


def _base_router_instructions(
    vertical_agents: List[Dict[str, Any]],
    published_instructions: Optional[str] = None,
) -> List[str]:
    """Build router instructions from DB Router config + current enabled vertical agents."""
    valid_targets = []
    for agent in vertical_agents:
        if not agent.get("agent_id") or not agent.get("enabled"):
            continue
        card = agent.get("capability_card") or {}
        valid_targets.append({
            "agent_id": agent.get("agent_id"),
            "name": agent.get("name"),
            "description": agent.get("description", ""),
            "skills": [str(item.get("name")) for item in card.get("skills") or [] if item.get("name")],
            "mcp_servers": [str(item.get("name")) for item in card.get("mcp_servers") or [] if item.get("name")],
            "mcp_intents": [
                str(intent)
                for server in card.get("mcp_servers") or []
                for intent in server.get("natural_language_intents") or []
                if str(intent).strip()
            ],
            "knowledge_docs": [
                str(item.get("title"))
                for item in card.get("knowledge_docs") or []
                if item.get("title")
            ],
        })
    if not valid_targets:
        valid_targets = [
            {"agent_id": "maintenance", "name": "维修 Agent", "description": "维修报修"},
            {"agent_id": "billing", "name": "费用 Agent", "description": "费用缴费"},
            {"agent_id": "complaint", "name": "投诉 Agent", "description": "投诉纠纷"},
            {"agent_id": "customer_service", "name": "客服 Agent", "description": "一般咨询"},
        ]
    target_lines = "\n".join(
        f'- {t["agent_id"]}（{t["name"]}）：{t["description"] or "无描述"}'
        + (f'；Skill={"、".join(t["skills"])}' if t.get("skills") else "")
        + (f'；MCP={"、".join(t["mcp_servers"])}' if t.get("mcp_servers") else "")
        + (f'；MCP意图={"、".join(t["mcp_intents"])}' if t.get("mcp_intents") else "")
        + (f'；RAG={"、".join(t["knowledge_docs"])}' if t.get("knowledge_docs") else "")
        for t in valid_targets
    )

    if published_instructions is None:
        router = get_agent_by_agent_id("router")
        user_instructions = (router.get("instructions") or "").strip() if router else ""
    else:
        user_instructions = published_instructions.strip()

    base = [
        "你是YIAI物业的路由 Agent，负责识别业主意图并分发给合适的垂直 Agent。",
        f"你只能从以下启用的垂直 Agent 中选择目标，输出其 agent_id：\n{target_lines}",
        '输出格式必须严格为 JSON：{"target_agent_id": "<agent_id>", "reason": "<一句话理由>"}',
        "如果用户问题同时涉及多个 Agent，选择最核心、最紧急的意图对应的 Agent。",
        "优先选择描述与用户问题关键词最匹配的垂直 Agent；若用户问题明确指向某个 Agent 的描述，必须选择该 Agent。",
        "如果无法判断，选择 customer_service 或其他最接近的垂直 Agent；不要编造不存在的 agent_id。",
    ]
    if user_instructions:
        base.insert(0, f"[路由策略：{user_instructions}]")
    return base


def create_router_agent(
    vertical_agents: Optional[List[Dict[str, Any]]] = None,
    published_instructions: Optional[str] = None,
    model: Any = None,
) -> Agent:
    return Agent(
        id="router_agent",
        name="路由 Agent",
        description="识别业主意图并分发给垂直 Agent。",
        model=model or MODEL,
        db=agent_db,
        instructions=_base_router_instructions(
            vertical_agents or [],
            published_instructions=published_instructions,
        ),
        add_datetime_to_context=True,
        add_history_to_context=True,
        read_chat_history=True,
        num_history_runs=3,
        markdown=False,
    )


async def _collect_response(generator) -> str:
    """Collect text from an Agno async generator or return a single response."""
    response = ""
    try:
        if isinstance(generator, str):
            return generator
        if hasattr(generator, "__aiter__"):
            async for chunk in generator:
                if hasattr(chunk, "content") and chunk.content:
                    response += str(chunk.content)
                elif hasattr(chunk, "delta") and chunk.delta:
                    response += str(chunk.delta)
                elif isinstance(chunk, str):
                    response += chunk
            return response.strip()
        # Agno may return either an awaitable run or an already materialised
        # RunOutput, depending on the SDK execution path.  Only await the
        # former; awaiting a completed RunOutput aborts the whole SSE stream.
        result = await generator if inspect.isawaitable(generator) else generator
        if hasattr(result, "content"):
            return str(result.content).strip()
        if isinstance(result, str):
            return result.strip()
        return ""
    except Exception:
        import traceback
        traceback.print_exc()
        return ""


def _fallback_reason(message: str, target_agent_id: str, vertical_agents: Optional[List[Dict[str, Any]]] = None) -> str:
    """Business-readable explanation for deterministic route fallback."""
    names = {item.get("agent_id"): item.get("name") for item in (vertical_agents or [])}
    target_name = names.get(target_agent_id) or {
        "maintenance": "维修 Agent",
        "billing": "费用 Agent",
        "complaint": "投诉 Agent",
        "customer_service": "客服 Agent",
    }.get(target_agent_id, target_agent_id)
    keyword_groups = {
        "maintenance": ("报修", "漏水", "维修", "工单", "电梯", "天气"),
        "billing": ("缴费", "费用", "物业费", "收费", "账单"),
        "complaint": ("投诉", "扰民", "纠纷", "赔偿", "不满意"),
    }
    matched = next((word for word in keyword_groups.get(target_agent_id, ()) if word in message), None)
    if matched:
        return f"用户提及“{matched}”，与{target_name}的服务范围匹配，由{target_name}处理。"
    return f"根据问题内容与{target_name}的服务范围匹配，由{target_name}处理。"


_ROUTING_STOP_TERMS = {
    "负责", "处理", "服务", "咨询", "问题", "用户", "业主", "相关", "当前", "需要", "可以",
    "系统", "平台", "能力", "帮助", "提供", "进行", "通过", "以及", "一般", "工作",
}

# These words describe the orchestration mechanism rather than a business
# capability.  They must never influence dynamic routing: a user asking
# "which Agent handles this" would otherwise match every Agent whose display
# name ends with "Agent" and make the fallback explanation meaningless.
_ROUTING_GENERIC_TERMS = {
    "agent", "agents", "assistant", "assistants", "skill", "skills",
    "tool", "tools", "mcp", "server", "servers",
    "智能体", "助手", "服务", "咨询", "问题", "处理", "能力", "系统",
    "平台", "管理", "用户", "业主", "哪个", "什么", "如何", "请问",
    "相关", "当前", "需要", "可以", "一个", "进行", "提供",
}


_CANONICAL_FALLBACK_TERMS = {
    "maintenance": ("报修", "漏水", "维修", "工单", "电梯", "下水道", "上门", "师傅"),
    "billing": ("缴费", "物业费", "账单", "收费", "费用", "停车费", "价格"),
    "complaint": ("投诉", "扰民", "纠纷", "噪音", "赔偿", "不满意", "举报"),
}


def _routing_terms(agent: Dict[str, Any]) -> List[tuple[str, str]]:
    """Extract compact routing signals from an Agent's live capability card."""
    card = agent.get("capability_card") or {}
    sources: List[tuple[str, str]] = [
        ("Agent 名称", str(agent.get("name") or "")),
        ("服务范围", str(agent.get("description") or card.get("service_scope") or "")),
        ("路由提示", str(card.get("routing_hints") or "")),
    ]
    for skill in card.get("skills") or []:
        sources.append(("绑定 Skill", str(skill.get("name") or "")))
        for trigger in skill.get("positive_triggers") or []:
            sources.append(("Skill 触发词", str(trigger)))
        for hint in skill.get("tool_hints") or []:
            sources.append(("Skill 工具提示", str(hint)))
    for server in card.get("mcp_servers") or []:
        sources.append(("绑定 MCP", str(server.get("name") or "")))
        sources.append(("MCP 说明", str(server.get("description") or "")))
        for tool_name in server.get("tools") or []:
            sources.append(("MCP 工具", str(tool_name)))
        for intent in server.get("natural_language_intents") or []:
            sources.append(("MCP 意图", str(intent)))
        for trigger in server.get("trigger_keywords") or []:
            sources.append(("MCP 触发词", str(trigger)))
    for document in card.get("knowledge_docs") or []:
        sources.append(("RAG 文档", str(document.get("title") or "")))
        sources.append(("RAG 分类", str(document.get("category") or "")))

    terms: List[tuple[str, str]] = []
    seen: set[str] = set()
    for source, text in sources:
        for phrase in re.findall(r"[\u4e00-\u9fff]{2,16}|[a-zA-Z][a-zA-Z0-9_-]{2,}", text or ""):
            phrase = phrase.strip().lower()
            if phrase in _ROUTING_STOP_TERMS or phrase in _ROUTING_GENERIC_TERMS or phrase in seen:
                continue
            seen.add(phrase)
            terms.append((phrase, source))
            # Long Chinese capability phrases are useful both as a whole and
            # via short meaningful windows, e.g. “老年关怀” -> “老年”.
            if len(phrase) >= 4 and re.fullmatch(r"[\u4e00-\u9fff]+", phrase):
                for size in (2, 3, 4):
                    for start in range(0, len(phrase) - size + 1):
                        window = phrase[start : start + size]
                        if (
                            window not in _ROUTING_STOP_TERMS
                            and window not in _ROUTING_GENERIC_TERMS
                            and window not in seen
                        ):
                            seen.add(window)
                            terms.append((window, source))
    return terms


def _capability_fallback(message: str, vertical_agents: Optional[List[Dict[str, Any]]]) -> tuple[str, str, List[Dict[str, Any]]]:
    """Choose a live Agent when Router JSON is absent or invalid.

    This is deliberately capability-driven: all currently enabled vertical
    Agents participate, including an Agent created seconds ago in the console.
    Canonical property terms give reliable emergency/business fallbacks, while
    names, descriptions, Skill triggers and bound MCP tools make extension
    domains such as child education or elderly care routable without code edits.
    """
    agents = [dict(agent) for agent in (vertical_agents or []) if agent.get("enabled") and agent.get("agent_id")]
    if not agents:
        return "customer_service", "能力匹配路由：没有可用垂直 Agent，由客服承接通用咨询。", []

    lowered = (message or "").lower()
    scored: List[Dict[str, Any]] = []
    for order, agent in enumerate(agents):
        agent_id = str(agent.get("agent_id"))
        score = 0
        matches: List[Dict[str, Any]] = []
        for term in _CANONICAL_FALLBACK_TERMS.get(agent_id, ()):
            if term in lowered:
                weight = 100 + min(len(term), 6)
                score += weight
                matches.append({"term": term, "source": "基础业务规则", "weight": weight})
        for term, source in _routing_terms(agent):
            if len(term) < 2 or term not in lowered:
                continue
            # Descriptions / Skill triggers are stronger than a bare tool name.
            weight = 35 + min(len(term), 8) * 4
            if source in {
                "服务范围",
                "绑定 Skill",
                "Skill 触发词",
                "MCP 意图",
                "MCP 触发词",
                "RAG 文档",
            }:
                weight += 15
            score += weight
            matches.append({"term": term, "source": source, "weight": weight})
        scored.append({"agent_id": agent_id, "name": agent.get("name") or agent_id, "score": score, "matches": matches, "order": order})

    scored.sort(key=lambda item: (-item["score"], item["order"]))
    winner = scored[0]
    if winner["score"] <= 0:
        customer = next((item for item in scored if item["agent_id"] == "customer_service"), winner)
        return (
            customer["agent_id"],
            f"能力匹配路由：未命中特定业务能力，转由{customer['name']}承接通用咨询。",
            scored,
        )
    # Explain the winning business capability, not the first textual match.
    # Agent display names are deliberately low-signal; Skill triggers and
    # service-scope terms should be visible in the user-facing route reason.
    strongest = max(
        winner["matches"],
        key=lambda item: (item["weight"], len(item["term"])),
    )
    return (
        winner["agent_id"],
        f"能力匹配路由：命中“{strongest['term']}”（{strongest['source']}），与{winner['name']}的已配置能力匹配。",
        scored,
    )


async def classify_intent(
    message: str,
    vertical_agents: Optional[List[Dict[str, Any]]] = None,
    user_id: str = "web-user",
    session_id: str = "",
    published_instructions: Optional[str] = None,
    model: Any = None,
) -> Dict[str, Any]:
    """Use the router agent to classify the user message intent.

    Returns route_mode to distinguish true model success from fallback paths.
    """
    vertical_agents = vertical_agents or []
    enabled_ids = {a.get("agent_id") for a in vertical_agents if a.get("enabled") and a.get("agent_id")}
    if not enabled_ids:
        enabled_ids = {"maintenance", "billing", "complaint", "customer_service"}
    valid_entries = []
    for agent in vertical_agents:
        if not agent.get("enabled") or not agent.get("agent_id"):
            continue
        card = agent.get("capability_card") or {}
        skills = [str(item.get("name")) for item in card.get("skills") or [] if item.get("name")]
        mcp_servers = [str(item.get("name")) for item in card.get("mcp_servers") or [] if item.get("name")]
        mcp_intents = [
            str(intent)
            for server in card.get("mcp_servers") or []
            for intent in server.get("natural_language_intents") or []
            if str(intent).strip()
        ]
        knowledge_docs = [
            str(item.get("title"))
            for item in card.get("knowledge_docs") or []
            if item.get("title")
        ]
        valid_entries.append(
            f'- {agent.get("agent_id")}（{agent.get("name")}）：{agent.get("description") or "无描述"}'
            + (f'；绑定 Skill={"、".join(skills)}' if skills else "")
            + (f'；绑定 MCP={"、".join(mcp_servers)}' if mcp_servers else "")
            + (f'；MCP 意图={"、".join(mcp_intents)}' if mcp_intents else "")
            + (f'；绑定 RAG={"、".join(knowledge_docs)}' if knowledge_docs else "")
        )
    valid_lines = "\n".join(valid_entries) or "- maintenance（维修 Agent）\n- billing（费用 Agent）\n- complaint（投诉 Agent）\n- customer_service（客服 Agent）"
    prompt = (
        "请判断以下业主问题的意图，并从当前启用的垂直 Agent 中选择一个目标。只输出 JSON，不要添加其他解释。\n"
        "选择规则：优先选择描述与用户问题关键词最匹配的垂直 Agent；"
        "如果某个 Agent 的描述明确包含用户问题的主题词，则必须选择该 Agent。\n\n"
        f"用户问题：{message}\n\n"
        "可选目标：\n" + valid_lines + "\n\n"
        '输出格式：{"target_agent_id": "<agent_id>", "reason": "<简要理由>"}'
    )
    route_mode = "model_success"
    raw_response = ""
    metrics: Dict[str, Any] = {}
    try:
        router_agent = create_router_agent(
            vertical_agents,
            published_instructions=published_instructions,
            model=model,
        )
        response_obj = await router_agent.arun(
            prompt,
            user_id=user_id,
            session_id=session_id or f"router-{id(message)}",
            stream=False,
        )
        response = await _collect_response(response_obj)
        raw_response = response

        # Collect provider metrics if available.
        if hasattr(response_obj, "metrics") and response_obj.metrics:
            m = response_obj.metrics
            value = lambda key: m.get(key) if isinstance(m, dict) else getattr(m, key, None)
            metrics = {
                "input_tokens": value("input_tokens"),
                "output_tokens": value("output_tokens"),
                "total_tokens": value("total_tokens"),
                "reasoning_tokens": value("reasoning_tokens"),
                "cached_tokens": value("cached_tokens"),
            }

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            target = parsed.get("target_agent_id", parsed.get("intent", "customer_service"))
            reason = parsed.get("reason", "")
        else:
            target, reason, fallback_scores = _capability_fallback(message, vertical_agents)
            route_mode = "capability_fallback"

        capability_target, capability_reason, capability_scores = _capability_fallback(message, vertical_agents)
        if target not in enabled_ids:
            target, reason, fallback_scores = capability_target, capability_reason, capability_scores
            route_mode = "capability_fallback"
        elif route_mode == "model_success" and capability_target != target:
            # A valid JSON response can still be an obviously weaker choice.
            # Only correct it when the live capability evidence is decisive;
            # this keeps fuzzy new domains under model control while ensuring
            # a repair/work-order composite does not fall through to customer
            # service merely because the Router returned syntactically valid JSON.
            score_by_id = {item["agent_id"]: item["score"] for item in capability_scores}
            best_score = score_by_id.get(capability_target, 0)
            selected_score = score_by_id.get(target, 0)
            if best_score >= 100 and best_score >= selected_score + 45:
                target = capability_target
                reason = f"能力策略校正：{capability_reason}"
                route_mode = "capability_policy_override"

        return {
            "target_agent_id": target,
            "reason": reason,
            "raw": raw_response,
            "route_mode": route_mode,
            "metrics": metrics,
            "fallback_scores": (
                fallback_scores if route_mode == "capability_fallback"
                else capability_scores if route_mode == "capability_policy_override"
                else []
            ),
        }
    except Exception:
        import traceback
        traceback.print_exc()
        target, reason, fallback_scores = _capability_fallback(message, vertical_agents)
        return {
            "target_agent_id": target,
            "reason": reason,
            "raw": raw_response,
            "route_mode": "capability_fallback",
            "metrics": metrics,
            "fallback_scores": fallback_scores,
        }


def _keyword_intent(message: str, vertical_agents: Optional[List[Dict[str, Any]]] = None) -> str:
    """Fallback keyword-based intent classification."""
    lowered = message.lower()
    # Prefer vertical agents whose description keywords directly appear in the message.
    if vertical_agents:
        for agent in vertical_agents:
            if not agent.get("enabled") or not agent.get("agent_id"):
                continue
            desc = (agent.get("description") or "").lower()
            # Use 2+ character descriptive keywords from the description.
            desc_keywords = {w for w in re.findall(r"[\u4e00-\u9fa5]{2,}|\b[a-z_]{3,}\b", desc)}
            if any(k in lowered for k in desc_keywords):
                return agent["agent_id"]
    # Canonical category keywords preserved for backward compatibility.
    maintenance_keywords = ["报修", "漏水", "跳闸", "灯不亮", "门锁", "窗户", "电梯", "下水道", "维修", "工单", "师傅", "上门", "天气", "气温", "下雨"]
    billing_keywords = ["收费", "缴费", "多少钱", "费用", "物业费", "停车费", "账单", "价格", "收费标准"]
    complaint_keywords = ["投诉", "不满意", "纠纷", "邻居", "噪音", "责任", "赔偿", "物业不作为", "举报"]
    if any(k in lowered for k in maintenance_keywords):
        return "maintenance" if _agent_enabled("maintenance", vertical_agents) else _first_enabled(vertical_agents)
    if any(k in lowered for k in billing_keywords):
        return "billing" if _agent_enabled("billing", vertical_agents) else _first_enabled(vertical_agents)
    if any(k in lowered for k in complaint_keywords):
        return "complaint" if _agent_enabled("complaint", vertical_agents) else _first_enabled(vertical_agents)
    return "customer_service" if _agent_enabled("customer_service", vertical_agents) else _first_enabled(vertical_agents)


def _agent_enabled(agent_id: str, vertical_agents: Optional[List[Dict[str, Any]]]) -> bool:
    if not vertical_agents:
        return True
    return any(a.get("agent_id") == agent_id and a.get("enabled") for a in vertical_agents)


def _first_enabled(vertical_agents: Optional[List[Dict[str, Any]]]) -> str:
    if vertical_agents:
        for a in vertical_agents:
            if a.get("enabled") and a.get("agent_id"):
                return a["agent_id"]
    return "customer_service"
