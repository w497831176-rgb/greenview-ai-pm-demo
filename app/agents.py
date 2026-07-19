"""
Agent Management API
====================

REST endpoints for creating and managing Router and Vertical Agents.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from db.property_db import (
    create_agent as db_create_agent,
    delete_agent as db_delete_agent,
    get_agent as db_get_agent,
    get_agent_by_agent_id,
    get_agent_knowledge_bindings,
    get_agent_skills,
    get_agent_tools,
    get_skill as db_get_skill,
    get_skill_by_name,
    list_agents as db_list_agents,
    set_agent_skills,
    set_agent_tools,
    set_agent_knowledge_bindings,
    update_agent as db_update_agent,
)

router = APIRouter(prefix="/api/agents", tags=["agents"])


class AgentCreate(BaseModel):
    agent_id: Optional[str] = None
    name: str
    description: str = ""
    instructions: Optional[str] = ""
    # Alias is optional; when omitted it must not erase ``instructions``.
    system_prompt: Optional[str] = None
    category: Optional[str] = "vertical"  # "router" or "vertical"
    is_router: Optional[bool] = False  # frontend alias for category
    enabled: Optional[bool] = True
    model_id: Optional[str] = None
    skill_ids: Optional[List[int]] = []
    available_skills: Optional[List[str]] = []  # frontend sends skill names
    tool_names: Optional[List[str]] = []
    available_mcp_tools: Optional[List[str]] = []  # frontend alias for tool_names
    # V1.8 vertical Agents always own an explicit RAG scope.  An omitted
    # selection means "no RAG", never the legacy "all published docs" scope.
    knowledge_doc_ids: List[int] = Field(default_factory=list)


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    system_prompt: Optional[str] = None  # frontend alias for instructions
    category: Optional[str] = None
    is_router: Optional[bool] = None
    enabled: Optional[bool] = None
    model_id: Optional[str] = None
    skill_ids: Optional[List[int]] = None
    available_skills: Optional[List[str]] = None
    tool_names: Optional[List[str]] = None
    available_mcp_tools: Optional[List[str]] = None
    knowledge_doc_ids: Optional[List[int]] = None


class AgentToggleRequest(BaseModel):
    enabled: bool


def _resolve_agent(identifier: str) -> Dict[str, Any]:
    """Resolve an agent by numeric row id or string agent_id."""
    if identifier.isdigit():
        agent = db_get_agent(int(identifier))
        if agent:
            return agent
    agent = get_agent_by_agent_id(identifier)
    if agent:
        return agent
    raise HTTPException(status_code=404, detail="agent not found")


def _resolve_skill_ids(skill_names: List[str]) -> List[int]:
    """Resolve a list of skill names to skill ids, ignoring unknown names."""
    ids = []
    for name in skill_names:
        skill = get_skill_by_name(name)
        if skill:
            ids.append(skill["id"])
    return ids


def _serialize_agent(agent: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not agent:
        return None
    agent = dict(agent)
    agent["skill_ids"] = get_agent_skills(agent["agent_id"])
    agent["tools"] = get_agent_tools(agent["agent_id"])
    # Frontend-compatible aliases.
    agent["is_router"] = agent.get("category") in ("router", "orchestration")
    agent["system_prompt"] = agent.get("instructions") or ""
    # The frontend checkbox values are skill names; return structured objects.
    agent["available_skills"] = [
        {"id": int(x), "name": (db_get_skill(int(x)) or {}).get("name") or str(x)}
        for x in agent["skill_ids"]
    ]
    agent["available_mcp_tools"] = [
        t.get("tool_name") for t in agent["tools"] if t.get("tool_name")
    ]
    agent["knowledge_doc_ids"] = get_agent_knowledge_bindings(agent["agent_id"])
    agent["knowledge_scope_mode"] = (
        "legacy_all_published"
        if agent["knowledge_doc_ids"] is None
        else "explicit"
    )
    agent["runtime_registration"] = {
        "router_candidate": bool(agent.get("enabled")) and not agent["is_router"],
        "effective_on": "next_published_release_new_session" if bool(agent.get("enabled")) and not agent["is_router"] else None,
        "skill_binding_count": len(agent["skill_ids"]),
        "mcp_server_binding_count": len(agent["available_mcp_tools"]),
        "note": (
            "配置保存后仍是 Draft；在“V1.8 运行时发布”校验并发布后，"
            "该 Agent 才会进入下一新会话的路由候选池。"
        ),
    }
    if agent.get("is_router"):
        agent["members"] = _get_router_members()
    return agent


def _get_router_members() -> List[Dict[str, Any]]:
    """Return enabled vertical agents as router routing candidates."""
    members = []
    for a in db_list_agents(category="vertical"):
        if not a.get("enabled"):
            continue
        aid = a.get("agent_id")
        skills = [db_get_skill(int(s)).get("name") or str(s) for s in get_agent_skills(aid)]
        tools = [t.get("tool_name") for t in get_agent_tools(aid) if t.get("tool_name")]
        members.append({
            "agent_id": aid,
            "name": a.get("name"),
            "description": a.get("description") or "",
            "enabled": a.get("enabled"),
            "skills": skills,
            "mcp_tools": tools,
        })
    return members


def _is_router(agent: Dict[str, Any]) -> bool:
    return agent.get("category") in ("router", "orchestration")


@router.get("")
async def list_agents(category: Optional[str] = None):
    """List all agents, optionally filtered by category."""
    agents = db_list_agents(category=category)
    return {"agents": [_serialize_agent(a) for a in agents], "count": len(agents)}


@router.get("/{agent_id}")
async def get_agent(agent_id: str):
    """Get a single agent by numeric row id or string agent_id."""
    agent = _resolve_agent(agent_id)
    return {"agent": _serialize_agent(agent)}


@router.post("")
async def create_agent(request: AgentCreate):
    """Create a new agent. Only vertical agents can be created."""
    agent_id = (request.agent_id or request.name).strip()
    if get_agent_by_agent_id(agent_id):
        raise HTTPException(status_code=409, detail="agent_id already exists")

    # Router is singleton and seeded; users cannot create a second router.
    if request.is_router or (request.category in ("router", "orchestration")):
        raise HTTPException(status_code=400, detail="router agent cannot be created")

    instructions = request.system_prompt if request.system_prompt is not None else request.instructions
    category = "vertical"
    agent = db_create_agent(
        agent_id=agent_id,
        name=request.name,
        description=request.description,
        instructions=instructions,
        category=category,
        enabled=request.enabled if request.enabled is not None else True,
        model_id=request.model_id,
    )
    skill_ids = request.skill_ids or []
    if request.available_skills:
        skill_ids = _resolve_skill_ids(request.available_skills)
    tool_names = request.tool_names or []
    if request.available_mcp_tools:
        tool_names = request.available_mcp_tools
    if skill_ids:
        set_agent_skills(agent_id, skill_ids)
    if tool_names:
        tools = [{"tool_name": name} for name in tool_names]
        set_agent_tools(agent_id, tools)
    # Persist an explicit binding row even when the selection is empty.  This
    # prevents a newly created Agent from silently inheriting every legacy
    # knowledge document.
    set_agent_knowledge_bindings(agent_id, request.knowledge_doc_ids)
    return {"agent": _serialize_agent(agent)}


@router.put("/{agent_id}")
async def update_agent(agent_id: str, request: AgentUpdate):
    """Update an agent with partial-update semantics.

    Only fields explicitly present in the request body are changed.
    Omitting skill/tool fields preserves existing bindings; passing an
    empty array explicitly clears them.

    Router is a singleton: its name/description/instructions may be edited,
    but category/is_router/enabled/model_id cannot be changed and it has no
    skill/tool bindings.
    """
    agent = _resolve_agent(agent_id)
    is_router = _is_router(agent)

    # Pydantic V2 / V1 compatible way to know which fields were sent.
    fields_set = getattr(request, "model_fields_set", getattr(request, "__fields_set__", set()))

    # 1. Basic scalar fields: keep original if not sent.
    name = request.name if "name" in fields_set else agent.get("name")
    description = request.description if "description" in fields_set else agent.get("description")

    # 2. Instructions / system_prompt alias: system_prompt wins if sent.
    if "system_prompt" in fields_set:
        instructions = request.system_prompt
    elif "instructions" in fields_set:
        instructions = request.instructions
    else:
        instructions = agent.get("instructions")

    # 3. Router-only restrictions.
    if is_router:
        # Router cannot change category, enabled, model, skills or tools.
        if any(f in fields_set for f in ("category", "is_router", "enabled", "model_id", "skill_ids", "available_skills", "tool_names", "available_mcp_tools", "knowledge_doc_ids")):
            raise HTTPException(status_code=400, detail="router agent can only edit name/description/system_prompt")
        category = agent.get("category")
        enabled = agent.get("enabled")
        model_id = agent.get("model_id")
        updated = db_update_agent(
            agent_row_id=agent["id"],
            name=name,
            description=description,
            instructions=instructions,
            category=category,
            enabled=enabled,
            model_id=model_id,
        )
        return {"agent": _serialize_agent(updated)}

    # 4. Vertical agent fields.
    enabled = request.enabled if "enabled" in fields_set else agent.get("enabled")
    model_id = request.model_id if "model_id" in fields_set else agent.get("model_id")
    category = "vertical"

    updated = db_update_agent(
        agent_row_id=agent["id"],
        name=name,
        description=description,
        instructions=instructions,
        category=category,
        enabled=enabled,
        model_id=model_id,
    )

    # 5. Skill bindings: update only when skill fields are explicitly sent.
    skill_ids = None
    if "available_skills" in fields_set:
        skill_ids = _resolve_skill_ids(request.available_skills or [])
    elif "skill_ids" in fields_set:
        skill_ids = request.skill_ids or []
    if skill_ids is not None:
        set_agent_skills(agent["agent_id"], skill_ids)

    # 6. MCP tool bindings: update only when tool fields are explicitly sent.
    tool_names = None
    if "available_mcp_tools" in fields_set:
        tool_names = request.available_mcp_tools or []
    elif "tool_names" in fields_set:
        tool_names = request.tool_names or []
    if tool_names is not None:
        tools = [{"tool_name": name} for name in tool_names]
        set_agent_tools(agent["agent_id"], tools)
    if "knowledge_doc_ids" in fields_set:
        set_agent_knowledge_bindings(
            agent["agent_id"], request.knowledge_doc_ids or []
        )

    return {"agent": _serialize_agent(updated)}


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent."""
    agent = _resolve_agent(agent_id)
    if _is_router(agent):
        raise HTTPException(status_code=400, detail="router agent cannot be deleted")
    deleted = db_delete_agent(agent["id"])
    return {"ok": deleted}


@router.post("/{agent_id}/toggle")
async def toggle_agent(agent_id: str, request: AgentToggleRequest):
    """Enable or disable an agent."""
    agent = _resolve_agent(agent_id)
    if _is_router(agent):
        raise HTTPException(status_code=400, detail="router agent cannot be disabled")
    updated = db_update_agent(agent["id"], enabled=request.enabled)
    return {"agent": _serialize_agent(updated)}


@router.patch("/{agent_id}")
async def patch_agent(agent_id: str, request: AgentToggleRequest):
    """Alias for toggle via PATCH (used by the frontend)."""
    return await toggle_agent(agent_id, request)
