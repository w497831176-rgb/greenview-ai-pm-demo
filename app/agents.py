"""
Agent Management API
====================

REST endpoints for creating and managing Router and Vertical Agents.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db.property_db import (
    create_agent as db_create_agent,
    delete_agent as db_delete_agent,
    get_agent as db_get_agent,
    get_agent_by_agent_id,
    get_agent_skills,
    get_agent_tools,
    get_skill as db_get_skill,
    get_skill_by_name,
    list_agents as db_list_agents,
    set_agent_skills,
    set_agent_tools,
    update_agent as db_update_agent,
)

router = APIRouter(prefix="/api/agents", tags=["agents"])


class AgentCreate(BaseModel):
    agent_id: Optional[str] = None
    name: str
    description: str = ""
    instructions: Optional[str] = ""
    system_prompt: Optional[str] = ""  # frontend alias for instructions
    category: Optional[str] = "vertical"  # "router" or "vertical"
    is_router: Optional[bool] = False  # frontend alias for category
    enabled: Optional[bool] = True
    model_id: Optional[str] = None
    skill_ids: Optional[List[int]] = []
    available_skills: Optional[List[str]] = []  # frontend sends skill names
    tool_names: Optional[List[str]] = []
    available_mcp_tools: Optional[List[str]] = []  # frontend alias for tool_names


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
    # The frontend checkbox values are skill names; return names for checked state.
    agent["available_skills"] = [
        (db_get_skill(int(x)) or {}).get("name") or str(x)
        for x in agent["skill_ids"]
    ]
    agent["available_mcp_tools"] = [
        t.get("tool_name") for t in agent["tools"] if t.get("tool_name")
    ]
    return agent


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
    """Create a new agent."""
    agent_id = (request.agent_id or request.name).strip()
    if get_agent_by_agent_id(agent_id):
        raise HTTPException(status_code=409, detail="agent_id already exists")

    instructions = request.system_prompt if request.system_prompt is not None else request.instructions
    category = request.category
    if request.is_router is not None:
        category = "router" if request.is_router else "vertical"
    category = "router" if category in ("router", "orchestration") else "vertical"
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
    return {"agent": _serialize_agent(agent)}


@router.put("/{agent_id}")
async def update_agent(agent_id: str, request: AgentUpdate):
    """Update an agent with partial-update semantics.

    Only fields explicitly present in the request body are changed.
    Omitting skill/tool fields preserves existing bindings; passing an
    empty array explicitly clears them.
    """
    agent = _resolve_agent(agent_id)

    # Pydantic V2 / V1 compatible way to know which fields were sent.
    fields_set = getattr(request, "model_fields_set", getattr(request, "__fields_set__", set()))

    # 1. Basic scalar fields: keep original if not sent.
    name = request.name if "name" in fields_set else agent.get("name")
    description = request.description if "description" in fields_set else agent.get("description")
    enabled = request.enabled if "enabled" in fields_set else agent.get("enabled")
    model_id = request.model_id if "model_id" in fields_set else agent.get("model_id")

    # 2. Instructions / system_prompt alias: system_prompt wins if sent.
    if "system_prompt" in fields_set:
        instructions = request.system_prompt
    elif "instructions" in fields_set:
        instructions = request.instructions
    else:
        instructions = agent.get("instructions")

    # 3. Category / is_router alias: is_router wins if sent.
    if "is_router" in fields_set:
        category = "router" if request.is_router else "vertical"
    elif "category" in fields_set:
        category = "router" if request.category in ("router", "orchestration") else "vertical"
    else:
        category = agent.get("category")
        category = "router" if category in ("router", "orchestration") else "vertical"

    updated = db_update_agent(
        agent_row_id=agent["id"],
        name=name,
        description=description,
        instructions=instructions,
        category=category,
        enabled=enabled,
        model_id=model_id,
    )

    # 4. Skill bindings: update only when skill fields are explicitly sent.
    skill_ids = None
    if "available_skills" in fields_set:
        skill_ids = _resolve_skill_ids(request.available_skills or [])
    elif "skill_ids" in fields_set:
        skill_ids = request.skill_ids or []
    if skill_ids is not None:
        set_agent_skills(agent["agent_id"], skill_ids)

    # 5. MCP tool bindings: update only when tool fields are explicitly sent.
    tool_names = None
    if "available_mcp_tools" in fields_set:
        tool_names = request.available_mcp_tools or []
    elif "tool_names" in fields_set:
        tool_names = request.tool_names or []
    if tool_names is not None:
        tools = [{"tool_name": name} for name in tool_names]
        set_agent_tools(agent["agent_id"], tools)

    return {"agent": _serialize_agent(updated)}


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent."""
    agent = _resolve_agent(agent_id)
    deleted = db_delete_agent(agent["id"])
    return {"ok": deleted}


@router.post("/{agent_id}/toggle")
async def toggle_agent(agent_id: str, request: AgentToggleRequest):
    """Enable or disable an agent."""
    agent = _resolve_agent(agent_id)
    updated = db_update_agent(agent["id"], enabled=request.enabled)
    return {"agent": _serialize_agent(updated)}


@router.patch("/{agent_id}")
async def patch_agent(agent_id: str, request: AgentToggleRequest):
    """Alias for toggle via PATCH (used by the frontend)."""
    return await toggle_agent(agent_id, request)
