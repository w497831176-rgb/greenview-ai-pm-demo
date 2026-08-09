"""Build Agno Agents from a pinned RunConfigSnapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agno.agent import Agent, AgentFactory
from agno.factory import RequestContext

from app.runtime.contracts import AgentTurnResult, RunConfigSnapshot, SkillActivation
from app.runtime.skill_projector import project_skills
from app.settings import MODEL_ID, agent_db, build_model

try:
    from agno.skills import LocalSkills, Skills
except Exception:  # pragma: no cover - guarded for an older emergency image
    LocalSkills = None  # type: ignore
    Skills = None  # type: ignore


@dataclass
class AgentBuild:
    agent: Agent
    agent_config: Dict[str, Any]
    activated_skills: List[SkillActivation]
    skill_decisions: List[Dict[str, Any]]
    skill_tool_calls: List[Dict[str, Any]]
    skill_evidence_sources: List[Dict[str, Any]]
    bound_skills: List[SkillActivation] = field(default_factory=list)
    bound_skill_evidence_sources: List[Dict[str, Any]] = field(default_factory=list)


def resolve_model_used_skills(
    build: AgentBuild,
    tool_calls: List[Dict[str, Any]],
) -> tuple[List[SkillActivation], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Resolve actual Skill use from the frozen Agent's native tool calls.

    Bound Skills are only a progressive-discovery candidate set. A Skill is
    considered used only when this same Agent called get_skill_instructions
    for its published package in this physical model run.
    """

    bound_by_id = {item.skill_id: item for item in build.bound_skills}
    evidence_by_id = {
        int(item["skill_id"]): item
        for item in build.bound_skill_evidence_sources
        if item.get("skill_id") is not None
    }
    used_ids: List[int] = []
    observed_calls: List[Dict[str, Any]] = []
    for call in tool_calls:
        if str(call.get("tool_name") or "") != "get_skill_instructions":
            continue
        arguments = call.get("arguments") or {}
        skill_name = str(
            arguments.get("skill_name")
            or arguments.get("name")
            or ""
        )
        if not skill_name.startswith("skill-"):
            continue
        try:
            skill_id = int(skill_name.removeprefix("skill-"))
        except ValueError:
            continue
        activation = bound_by_id.get(skill_id)
        if activation is None:
            raise PermissionError(
                f"selected Agent attempted to load unbound Skill: {skill_id}"
            )
        observed_calls.append(
            {
                **call,
                "status": str(call.get("status") or "success"),
                "invocation_mode": "model_native",
                "skill_id": skill_id,
                "skill_version": activation.version,
                "skill_content_hash": activation.content_hash,
            }
        )
        if str(call.get("status") or "success") != "success":
            continue
        if skill_id not in used_ids:
            used_ids.append(skill_id)
    activations = [bound_by_id[item] for item in used_ids]
    sources = [evidence_by_id[item] for item in used_ids if item in evidence_by_id]
    return activations, observed_calls, sources


def _find_agent(config: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    for item in config.get("agents") or []:
        if item.get("agent_id") == agent_id and item.get("enabled"):
            return item
    raise ValueError(f"agent is not enabled in RunConfigSnapshot: {agent_id}")


def router_agent_cards(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Project the strict four-field Router view from one immutable release.

    Bindings and implementation detail are intentionally absent. Missing or
    unrecognised structured scope is a release configuration error; it is
    never guessed from a name, description, instruction, or binding.
    """

    cards: List[Dict[str, Any]] = []
    for agent in config.get("agents") or []:
        if not agent.get("enabled") or agent.get("category") in {"router", "orchestration"}:
            continue
        scope = str(agent.get("domain_scope") or "").strip()
        if scope not in {"property", "isolated_general"}:
            raise ValueError(
                f"published Agent has invalid domain_scope: {agent.get('agent_id')}"
            )
        cards.append(
            {
                "agent_id": str(agent.get("agent_id") or ""),
                "name": str(agent.get("name") or agent.get("agent_id") or ""),
                "description": str(agent.get("description") or ""),
                "scope": scope,
            }
        )
    return cards


def vertical_agent_cards(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    skill_by_id = {
        int(item["skill_id"]): item for item in config.get("skills") or []
    }
    server_by_name = {
        str(item.get("name") or ""): item
        for item in config.get("mcp_servers") or []
        if item.get("enabled")
    }
    knowledge_by_id = {
        int(item["knowledge_doc_id"]): item
        for item in config.get("knowledge") or []
    }
    cards = []
    for agent in config.get("agents") or []:
        if not agent.get("enabled") or agent.get("category") in {"router", "orchestration"}:
            continue
        bound_skills = [
            skill_by_id[skill_id]
            for skill_id in agent.get("skill_ids") or []
            if skill_id in skill_by_id and skill_by_id[skill_id].get("enabled")
        ]
        bound_servers = [
            server_by_name[name]
            for name in agent.get("mcp_server_names") or []
            if name in server_by_name
        ]
        bound_knowledge = [
            knowledge_by_id[doc_id]
            for doc_id in agent.get("knowledge_doc_ids") or []
            if doc_id in knowledge_by_id
        ]
        skill_cards = [
            {
                "id": item["skill_id"],
                "name": item.get("name"),
                "description": item.get("description"),
                "positive_triggers": (
                    item.get("metadata") or {}
                ).get("positive_triggers") or [],
                "tool_hints": (item.get("metadata") or {}).get("tool_hints") or [],
            }
            for item in bound_skills
        ]
        server_cards = [
            {
                "name": item.get("name"),
                "description": item.get("description"),
                "tools": [
                    tool.get("name")
                    for tool in item.get("tools") or []
                    if (tool.get("policy") or {}).get("enabled")
                ],
                "natural_language_intents": [
                    str(intent)
                    for tool in item.get("tools") or []
                    for intent in (
                        (tool.get("tool_metadata") or {}).get(
                            "natural_language_intents"
                        )
                        or []
                    )
                    if str(intent).strip()
                ],
                "trigger_keywords": [
                    str(trigger)
                    for tool in item.get("tools") or []
                    for trigger in (
                        (tool.get("tool_metadata") or {}).get(
                            "trigger_keywords"
                        )
                        or []
                    )
                    if str(trigger).strip()
                ],
            }
            for item in bound_servers
        ]
        knowledge_cards = [
            {
                "id": item.get("knowledge_doc_id"),
                "title": item.get("title") or "",
                "category": item.get("category") or "",
            }
            for item in bound_knowledge
        ]
        cards.append(
            {
                "agent_id": agent["agent_id"],
                "name": agent.get("name") or agent["agent_id"],
                "description": agent.get("description") or "",
                "instructions": agent.get("instructions") or "",
                "domain_scope": agent.get("domain_scope") or "property",
                "enabled": True,
                "skills": [
                    {
                        "id": item["skill_id"],
                        "name": item.get("name"),
                        "description": item.get("description"),
                        "trigger_condition": item.get("trigger_condition"),
                        "skill_metadata": item.get("metadata") or {},
                    }
                    for item in bound_skills
                ],
                "mcp_tools": list(agent.get("mcp_server_names") or []),
                "capability_card": {
                    "service_scope": agent.get("description") or "",
                    "domain_scope": agent.get("domain_scope") or "property",
                    "routing_hints": agent.get("instructions") or "",
                    "skills": skill_cards,
                    "mcp_servers": server_cards,
                    "knowledge_docs": knowledge_cards,
                },
            }
        )
    return cards


def build_agent_from_snapshot(
    snapshot: RunConfigSnapshot,
    agent_id: str,
    message: str,
    tools: Optional[List[Any]] = None,
    evidence_prompt: str = "",
    enable_skills: bool = True,
) -> AgentBuild:
    config = snapshot.config
    agent_config = _find_agent(config, agent_id)
    skills_by_id = {
        int(item["skill_id"]): item for item in config.get("skills") or []
    }
    candidates = (
        [
            skills_by_id[skill_id]
            for skill_id in agent_config.get("skill_ids") or []
            if skill_id in skills_by_id and skills_by_id[skill_id].get("enabled")
        ]
        if enable_skills
        else []
    )
    # Binding is the authority boundary, not proof of use. Project every Skill
    # bound to this frozen Agent as a progressive-discovery candidate and let
    # the same Agent decide whether to call get_skill_instructions. No keyword,
    # bigram, Resolver, pre-invocation, or extra Provider request participates.
    skills_root, bound_activations = project_skills(
        snapshot.release_id,
        candidates,
    )
    agno_skills = None
    if skills_root and Skills is not None and LocalSkills is not None:
        agno_skills = Skills(loaders=[LocalSkills(str(skills_root))])
    decisions = [
        {
            "skill_id": item.skill_id,
            "bound": True,
            "used": False,
            "reason": "available_to_frozen_agent",
        }
        for item in bound_activations
    ]
    bound_evidence_sources = [
        {
            "skill_id": int(item["skill_id"]),
            "name": str(item.get("name") or f"Skill {item['skill_id']}"),
            "version": str(item.get("version") or ""),
            "snapshot_id": snapshot.snapshot_id,
            "content_hash": str(item.get("content_hash") or ""),
            "content_snapshot": str(item.get("instructions_fallback") or ""),
        }
        for item in candidates
    ]

    instructions = [
        str(agent_config.get("instructions") or ""),
        "You may use only capabilities bound to this Agent in the pinned RuntimeRelease snapshot.",
        "Bound Skills are candidates, not preloaded instructions. Call get_skill_instructions only for Skills relevant to this turn; zero, some, or all bound Skills may be used.",
        "Never create, update, or delete business data directly. A write request can only be expressed as a pending proposal_request.",
        "Claim write success only when the backend supplies a committed ActionReceipt with a real resource_id.",
    ]
    instructions.extend(
        [
            "Return exactly one JSON object and no prose, markdown fence, schema explanation, or control-plane fields.",
            "The object must contain exactly: answer, answer_status, citations, proposal_request, capability_usage.",
            "answer_status must be answered, insufficient_evidence, or insufficient_capability. Never request another Agent or routing decision.",
            "citations is a list of RAG evidence IDs supplied in this turn. Skill, MCP, Tool, model output, and configuration are never citations.",
            "capability_usage must contain skill_ids, rag_evidence_ids, mcp_calls, and tool_calls. Report only capabilities actually used in this run; use empty lists when none were used.",
            "For Skill use, report the numeric id from a successfully loaded get_skill_instructions package named skill-<id>. tool_calls must list every non-MCP model-native function actually called, including Skill helper functions; mcp_calls lists MCP functions separately.",
            "Only a selected property Agent may request work-order creation. It must use proposal_request with room_id, issue_type, issue_desc, urgency, contact_name, contact_phone, and appointment_time. Do not encode a write request in prose.",
            "If work-order fields are missing, include known values in proposal_request and ask for the missing values in answer. If no work-order state is requested, proposal_request must be null.",
        ]
    )
    if (agent_config.get("domain_scope") or "property") == "isolated_general":
        instructions.extend(
            [
                "你处于非物业隔离域：不得把通用回答表述为物业官方结论。",
                "只能使用本Agent在当前Snapshot中真实绑定的Skill、RAG和只读Tool；不得调用物业ActionGateway。",
                "没有实时Tool时不得确认当前天气、价格、名额等实时事实；医疗、法律、金融或暴力风险只给保守边界和求助建议。",
            ]
        )
    else:
        instructions.append(
            "你处于物业业务域：价格、时效、责任、服务是否存在等受控事实必须来自本轮合法Skill、RAG、成功Tool或Receipt证据。"
        )
    if evidence_prompt:
        instructions.append(evidence_prompt)
    snapshot_default_model = (
        (config.get("model_policy") or {}).get("default") or {}
    ).get("model_id")
    resolved_model_id = (
        agent_config.get("model_id")
        or snapshot_default_model
        or MODEL_ID
    )
    snapshot_model_config = next(
        (
            item
            for item in [
                (config.get("model_policy") or {}).get("default"),
                *((config.get("model_policy") or {}).get("available") or []),
            ]
            if isinstance(item, dict)
            and item.get("model_id") == resolved_model_id
        ),
        {},
    )
    model_params = snapshot_model_config.get("model_params") or {}
    model_overrides: Dict[str, Any] = {}
    if snapshot_model_config.get("base_url"):
        model_overrides["base_url"] = snapshot_model_config["base_url"]
    if "use_thinking" in model_params:
        model_overrides["use_thinking"] = bool(model_params["use_thinking"])
    agent = Agent(
        id=agent_id,
        name=str(agent_config.get("name") or agent_id),
        model=build_model(resolved_model_id, **model_overrides),
        db=agent_db,
        instructions=instructions,
        tools=list(tools or []),
        skills=agno_skills,
        output_schema=AgentTurnResult,
        use_json_mode=True,
        markdown=False,
        # The coordinator injects only successful user-visible chat_messages.
        # Agno stage history may contain Router control JSON and must stay off.
        add_history_to_context=False,
        num_history_runs=0,
    )
    return AgentBuild(
        agent=agent,
        agent_config=agent_config,
        activated_skills=[],
        skill_decisions=decisions,
        skill_tool_calls=[],
        skill_evidence_sources=[],
        bound_skills=bound_activations,
        bound_skill_evidence_sources=bound_evidence_sources,
    )


class RuntimeAgentFactoryInput(dict):
    """Documentation marker for the required factory_input shape."""


def build_runtime_agent(ctx: RequestContext) -> Agent:
    from app.runtime.snapshot_resolver import resolve_snapshot

    raw = ctx.input or {}
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    agent_id = str(raw.get("agent_id") or "customer_service")
    message = str(raw.get("message") or "")
    snapshot = resolve_snapshot(ctx.session_id or f"agentos-{ctx.user_id or 'anonymous'}")
    return build_agent_from_snapshot(snapshot, agent_id, message).agent


runtime_agent_factory = AgentFactory(
    id="runtime-agent",
    db=agent_db,
    factory=build_runtime_agent,
    name="YIAI Published Runtime Agent",
    description="Builds an Agno Agent from one immutable RunConfigSnapshot.",
)
