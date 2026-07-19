"""Build Agno Agents from a pinned RunConfigSnapshot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agno.agent import Agent, AgentFactory
from agno.factory import RequestContext

from app.runtime.contracts import RunConfigSnapshot, SkillActivation
from app.runtime.skill_projector import project_skills
from app.settings import MODEL_ID, agent_db, build_model
from app.skill_runtime import select_skills

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


def _find_agent(config: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    for item in config.get("agents") or []:
        if item.get("agent_id") == agent_id and item.get("enabled"):
            return item
    raise ValueError(f"agent is not enabled in RunConfigSnapshot: {agent_id}")


def vertical_agent_cards(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    skill_by_id = {
        int(item["skill_id"]): item for item in config.get("skills") or []
    }
    server_by_name = {
        str(item.get("name") or ""): item
        for item in config.get("mcp_servers") or []
        if item.get("enabled")
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
            }
            for item in bound_servers
        ]
        cards.append(
            {
                "agent_id": agent["agent_id"],
                "name": agent.get("name") or agent["agent_id"],
                "description": agent.get("description") or "",
                "instructions": agent.get("instructions") or "",
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
                    "routing_hints": agent.get("instructions") or "",
                    "skills": skill_cards,
                    "mcp_servers": server_cards,
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
) -> AgentBuild:
    config = snapshot.config
    agent_config = _find_agent(config, agent_id)
    skills_by_id = {
        int(item["skill_id"]): item for item in config.get("skills") or []
    }
    candidates = [
        skills_by_id[skill_id]
        for skill_id in agent_config.get("skill_ids") or []
        if skill_id in skills_by_id and skills_by_id[skill_id].get("enabled")
    ]
    # Reuse the deterministic runtime selector.  Adapt the compiled field names
    # to its legacy-compatible input contract.
    selector_candidates = [
        {
            "id": item["skill_id"],
            "name": item.get("name"),
            "description": item.get("description"),
            "instructions": item.get("instructions_fallback"),
            "enabled": item.get("enabled"),
            "trigger_condition": item.get("trigger_condition"),
            "skill_metadata": item.get("metadata") or {},
        }
        for item in candidates
    ]
    selected_legacy, decisions = select_skills(selector_candidates, message)
    selected_ids = {int(item["skill_id"]) for item in selected_legacy}
    selected = [item for item in candidates if int(item["skill_id"]) in selected_ids]
    reasons = {
        int(item["skill_id"]): str(item.get("match_reason") or item.get("outcome") or "trigger matched")
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

    instructions = [
        str(agent_config.get("instructions") or ""),
        "你只能使用本次已发布快照装配的能力。",
        "若有可用 Skill，先调用 get_skill_instructions 读取命中 Skill，再回答。",
        "不得自行创建、更新、删除业务数据；写操作只能描述为待确认 Proposal。",
        "只有后端 ActionReceipt.status=committed 且包含真实 resource_id 时，才能声称操作成功。",
    ]
    instructions.extend(skill_contexts)
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
        markdown=True,
        add_history_to_context=True,
        num_history_runs=5,
    )
    return AgentBuild(
        agent=agent,
        agent_config=agent_config,
        activated_skills=activations,
        skill_decisions=decisions,
        skill_tool_calls=skill_tool_calls,
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
