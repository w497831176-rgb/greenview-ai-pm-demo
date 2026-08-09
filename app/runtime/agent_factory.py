"""Build Agno Agents from a pinned RunConfigSnapshot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agno.agent import Agent, AgentFactory
from agno.factory import RequestContext

from app.runtime.contracts import (
    AgentResponseEnvelope,
    RunConfigSnapshot,
    SkillActivation,
)
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


def _preload_skill_instructions(
    skills: Any,
    activations: List[SkillActivation],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load selected Agno Skills deterministically before the model call.

    Trigger selection belongs to the runtime control plane. Relying on the
    model to optionally call ``get_skill_instructions`` made an otherwise
    valid Skill disappear from real runs. We still use Agno's own Skill access
    tool, but invoke it as a governed pre-invocation and preserve the evidence.
    """

    if skills is None or not activations:
        return [], []
    access_tool = next(
        (
            tool
            for tool in skills.get_tools()
            if getattr(tool, "name", "") == "get_skill_instructions"
        ),
        None,
    )
    if access_tool is None or not getattr(access_tool, "entrypoint", None):
        raise RuntimeError("Agno get_skill_instructions tool is unavailable")

    contexts: List[str] = []
    calls: List[Dict[str, Any]] = []
    for activation in activations:
        skill_name = f"skill-{activation.skill_id}"
        raw = access_tool.entrypoint(skill_name)
        payload = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        if payload.get("error"):
            raise RuntimeError(
                f"failed to load published Skill {activation.skill_id}: "
                f"{payload['error']}"
            )
        contexts.append(
            "\n".join(
                [
                    f"[已加载动态 Skill：{activation.name}]",
                    str(payload.get("instructions") or ""),
                ]
            )
        )
        calls.append(
            {
                "tool_name": "get_skill_instructions",
                "arguments": {"skill_name": skill_name},
                "status": "success",
                "invocation_mode": "policy_preinvoke",
                "skill_id": activation.skill_id,
                "skill_version": activation.version,
                "skill_content_hash": activation.content_hash,
            }
        )
    return contexts, calls


def _skills_exposed_to_model(
    skills: Any,
    preload_calls: List[Dict[str, Any]],
) -> Any:
    """Hide a Skill tool after the same Skill was deterministically preloaded."""
    return None if preload_calls else skills


def _find_agent(config: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    for item in config.get("agents") or []:
        if item.get("agent_id") == agent_id and item.get("enabled"):
            return item
    raise ValueError(f"agent is not enabled in RunConfigSnapshot: {agent_id}")


VALID_AGENT_SCOPES = {"property", "isolated_general"}
BINDING_FIELDS = ("skill_ids", "knowledge_doc_ids", "mcp_server_names")


def _strict_agent_scope(agent: Dict[str, Any]) -> str:
    agent_id = str(agent.get("agent_id") or "").strip() or "<missing>"
    scope = agent.get("domain_scope")
    if not isinstance(scope, str) or scope not in VALID_AGENT_SCOPES:
        raise ValueError(
            f"enabled Agent has missing or invalid structured domain_scope: "
            f"{agent_id}/{scope!r}"
        )
    return scope


def _enabled_vertical_agents(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    agents: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in config.get("agents") or []:
        if not raw.get("enabled") or raw.get("category") in {"router", "orchestration"}:
            continue
        agent_id = str(raw.get("agent_id") or "").strip()
        if not agent_id:
            raise ValueError("enabled vertical Agent is missing agent_id")
        if agent_id in seen_ids:
            raise ValueError(f"duplicate enabled vertical Agent id: {agent_id}")
        _strict_agent_scope(raw)
        seen_ids.add(agent_id)
        agents.append(raw)
    return agents


def _binding_ids(agent: Dict[str, Any], field: str) -> set[Any]:
    values = agent.get(field) or []
    if not isinstance(values, list):
        raise ValueError(
            f"Agent binding must be a list: {agent.get('agent_id')}/{field}"
        )
    normalized: set[Any] = set()
    for value in values:
        try:
            identity = str(value).strip() if field == "mcp_server_names" else int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Agent binding has an invalid identity: "
                f"{agent.get('agent_id')}/{field}/{value!r}"
            ) from exc
        if identity == "":
            raise ValueError(
                f"Agent binding has an empty identity: {agent.get('agent_id')}/{field}"
            )
        normalized.add(identity)
    return normalized


def validate_agent_binding_isolation(
    config: Dict[str, Any],
    agent_id: str,
    expected_scope: Optional[str] = None,
) -> str:
    """Validate one Agent using only immutable scopes and binding relations.

    A capability shared by enabled property and isolated-general Agents has no
    unambiguous domain. It is a configuration error; names, descriptions and
    prompt text are deliberately never consulted as a fallback.
    """

    if expected_scope is not None and expected_scope not in VALID_AGENT_SCOPES:
        raise ValueError(f"invalid expected Agent scope: {expected_scope!r}")
    agents = _enabled_vertical_agents(config)
    selected = next(
        (item for item in agents if str(item.get("agent_id")) == str(agent_id)),
        None,
    )
    if selected is None:
        raise ValueError(f"selected Agent is not an enabled vertical Agent: {agent_id}")
    selected_scope = _strict_agent_scope(selected)
    if expected_scope is not None and selected_scope != expected_scope:
        raise ValueError(
            f"selected Agent scope mismatch: {agent_id}/{selected_scope}/{expected_scope}"
        )

    available: Dict[str, set[Any]] = {
        "skill_ids": {
            int(item["skill_id"])
            for item in config.get("skills") or []
            if item.get("enabled") and item.get("skill_id") is not None
        },
        "knowledge_doc_ids": {
            int(item["knowledge_doc_id"])
            for item in config.get("knowledge") or []
            if item.get("knowledge_doc_id") is not None
        },
        "mcp_server_names": {
            str(item.get("name") or "").strip()
            for item in config.get("mcp_servers") or []
            if item.get("enabled") and str(item.get("name") or "").strip()
        },
    }
    bindings_by_agent = {
        str(item["agent_id"]): {
            field: _binding_ids(item, field) for field in BINDING_FIELDS
        }
        for item in agents
    }
    for field in BINDING_FIELDS:
        # Validate the complete enabled Release graph.  A selected Agent must
        # not be allowed to build merely because an ambiguous cross-domain
        # binding happens to belong to another enabled Agent.
        for owner in agents:
            owner_id = str(owner["agent_id"])
            missing = bindings_by_agent[owner_id][field] - available[field]
            if missing:
                raise ValueError(
                    f"enabled Agent has unavailable published binding: "
                    f"{owner_id}/{field}/{sorted(missing, key=str)}"
                )
        binding_ids = {
            binding_id
            for owner_bindings in bindings_by_agent.values()
            for binding_id in owner_bindings[field]
        }
        for binding_id in binding_ids:
            owner_scopes = {
                _strict_agent_scope(owner)
                for owner in agents
                if binding_id
                in bindings_by_agent[str(owner["agent_id"])][field]
            }
            if len(owner_scopes) > 1:
                raise ValueError(
                    f"cross-domain shared binding is forbidden: "
                    f"{field}/{binding_id}/{sorted(owner_scopes)}"
                )
    return selected_scope


def router_agent_cards(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the exact minimal card surface visible to the production Router."""

    return [
        {
            "agent_id": str(agent["agent_id"]),
            "name": str(agent.get("name") or agent["agent_id"]),
            "description": str(agent.get("description") or ""),
            "scope": _strict_agent_scope(agent),
        }
        for agent in _enabled_vertical_agents(config)
    ]


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
        domain_scope = _strict_agent_scope(agent)
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
                "domain_scope": domain_scope,
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
                    "domain_scope": domain_scope,
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
    domain_scope = validate_agent_binding_isolation(config, agent_id)
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
    # Agent selection already happened in the Router. Every enabled Skill bound
    # to that one frozen Agent is loaded; message keywords have no authority
    # over capability construction.
    selected = list(candidates)
    decisions = [
        {
            "skill_id": int(item["skill_id"]),
            "selected": True,
            "outcome": "published_agent_binding",
            "match_reason": "enabled binding of the frozen selected Agent",
        }
        for item in selected
    ]
    reasons = {
        int(item["skill_id"]): str(
            item.get("match_reason") or item.get("outcome") or "published binding"
        )
        for item in decisions
        if item.get("selected")
    }
    skills_root, activations = project_skills(
        snapshot.release_id,
        selected,
        match_reasons=reasons,
    )
    agno_skills = None
    if skills_root and Skills is not None and LocalSkills is not None:
        agno_skills = Skills(loaders=[LocalSkills(str(skills_root))])
    skill_contexts, skill_tool_calls = _preload_skill_instructions(
        agno_skills,
        activations,
    )
    model_skills = _skills_exposed_to_model(agno_skills, skill_tool_calls)
    envelope_schema = json.dumps(
        AgentResponseEnvelope.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    instructions = [
        str(agent_config.get("instructions") or ""),
        "若运行时已加载业务 Skill，请直接依据已加载的 Skill 原文回答，不要重复调用 Skill 工具。",
        "不得自行创建、更新、删除业务数据；写操作只能描述为待确认 Proposal。",
        "只有后端 ActionReceipt.status=committed 且包含真实 resource_id 时，才能声称操作成功。",
    ]
    instructions.append(
        "你只能使用本次已发布快照装配的能力。"
        if domain_scope == "property"
        else (
            "回答非物业通用问题时可以直接使用基础模型知识；本Agent在当前Snapshot中"
            "真实绑定的Skill、RAG、MCP和Tool只作为可选增强。"
        )
    )
    instructions.extend(
        [
            "Your final response must be exactly one JSON object. Do not emit Markdown fences, a preface, a suffix, or any text outside the object.",
            "The JSON must strictly match AgentResponseEnvelope and must not contain additional fields. citation_ids may contain only identifiers supplied by this run.",
            "A successful read Tool result exposes its exact citation_id. If the answer relies on that result, copy that citation_id into citation_ids; never invent or transform it.",
            f"AgentResponseEnvelope JSON Schema: {envelope_schema}",
        ]
    )
    if domain_scope == "isolated_general":
        instructions.extend(
            [
                "你处于非物业隔离域：不得把通用回答表述为物业官方结论。",
                "若使用增强能力，只能加载或调用本Agent在当前Snapshot中真实绑定的非物业Skill、RAG和只读Tool；不得访问物业能力或ActionGateway。",
                "没有实时Tool时不得确认当前天气、价格、名额等实时事实；医疗、法律、金融或暴力风险只给保守边界和求助建议。",
            ]
        )
    else:
        instructions.append(
            "你处于物业业务域：价格、时效、责任、服务是否存在等受控事实必须来自本轮合法Skill、RAG、成功Tool或Receipt证据。"
        )
    if domain_scope == "isolated_general":
        instructions.append(
            "You are a C-lane Agent: proposal_request and confirmation_request must both be null. Answer normally even when no enhancement is available."
        )
    else:
        instructions.extend(
            [
                "Only when you decide from the complete conversation to request a work order may proposal_request contain a strict work_order.create request; the backend will not infer write intent from prose.",
                "Only when you decide from the complete conversation and a persisted pending Proposal to approve or reject it may confirmation_request be non-null. Otherwise both request fields must be null.",
            ]
        )
    instructions.extend(skill_contexts)
    if evidence_prompt:
        instructions.append(evidence_prompt)
    if domain_scope == "isolated_general":
        # Keep the product-level C contract as the final authority after the
        # selected Agent's own Skill/RAG context.  Those optional enhancements
        # may narrow factual claims, but they must never turn a general Agent
        # back into a property-only assistant or make missing evidence a reply
        # precondition.
        instructions.append(
            "Final C-lane authority: answer the user's non-property request directly "
            "using your general model knowledge and any selected Agent bindings that "
            "are genuinely relevant. Skill, RAG, MCP, and Tool results are optional "
            "enhancements, never prerequisites for answering. Do not refuse merely "
            "because no enhancement or citation is available, and do not claim that "
            "you only handle property matters. This structured domain_scope overrides "
            "any conflicting property-only persona text. proposal_request and "
            "confirmation_request must remain null."
        )
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
        skills=model_skills,
        markdown=False,
        # The coordinator injects only successful user-visible chat_messages.
        # Agno stage history may contain Router control JSON and must stay off.
        add_history_to_context=False,
        num_history_runs=0,
    )
    return AgentBuild(
        agent=agent,
        agent_config=agent_config,
        activated_skills=activations,
        skill_decisions=decisions,
        skill_tool_calls=skill_tool_calls,
        skill_evidence_sources=[
            {
                "skill_id": int(item["skill_id"]),
                "name": str(item.get("name") or f"Skill {item['skill_id']}"),
                "version": str(item.get("version") or ""),
                "snapshot_id": snapshot.snapshot_id,
                "content_hash": str(item.get("content_hash") or ""),
                "content_snapshot": str(item.get("instructions_fallback") or ""),
            }
            for item in selected
        ],
    )


class RuntimeAgentFactoryInput(dict):
    """Documentation marker for the required factory_input shape."""


def build_runtime_agent(ctx: RequestContext) -> Agent:
    from app.runtime.snapshot_resolver import resolve_snapshot

    raw = ctx.input or {}
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    agent_id = str(raw.get("agent_id") or "").strip()
    if not agent_id:
        raise ValueError("factory_input.agent_id is required; no default Agent is allowed")
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
