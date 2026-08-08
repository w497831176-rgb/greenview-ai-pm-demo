"""
Agent Management API
====================

REST endpoints for creating and managing Router and Vertical Agents.
"""

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.runtime.capability_catalog import (
    CapabilityCatalogError,
    assert_trusted_capability,
    set_trusted_agent_bindings,
    set_trusted_capability_enabled,
    trusted_capability_ids,
)

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


def _supply_chain_locked(operation: str) -> None:
    raise HTTPException(
        status_code=410,
        detail={
            "code": "trusted_catalog_supply_chain_locked",
            "operation": operation,
            "message": (
                "新增、删除或修改 Agent 实现已停用；新增能力需经过代码级"
                "受控入库，目录内既有能力仅支持 Draft 启停与绑定。"
            ),
        },
    )


class AgentCreate(BaseModel):
    agent_id: Optional[str] = None
    name: str
    description: str = ""
    instructions: Optional[str] = ""
    # Alias is optional; when omitted it must not erase ``instructions``.
    system_prompt: Optional[str] = None
    category: Optional[str] = "vertical"  # "router" or "vertical"
    domain_scope: Literal["property", "isolated_general"] = "property"
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
    domain_scope: Optional[Literal["property", "isolated_general"]] = None
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
            identifier = str(agent.get("agent_id") or "")
    try:
        return assert_trusted_capability("agent", identifier)["object"]
    except CapabilityCatalogError:
        pass
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
    agent["domain_scope"] = agent.get("domain_scope") or "property"
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
    allowed = {str(item) for item in trusted_capability_ids("agent")}
    for a in db_list_agents(category="vertical"):
        if not a.get("enabled") or str(a.get("agent_id") or "") not in allowed:
            continue
        aid = a.get("agent_id")
        skills = [
            (db_get_skill(int(skill_id)) or {}).get("name") or str(skill_id)
            for skill_id in get_agent_skills(aid)
        ]
        tools = [t.get("tool_name") for t in get_agent_tools(aid) if t.get("tool_name")]
        members.append({
            "agent_id": aid,
            "name": a.get("name"),
            "description": a.get("description") or "",
            "enabled": a.get("enabled"),
            "domain_scope": a.get("domain_scope") or "property",
            "skills": skills,
            "mcp_tools": tools,
        })
    return members


def _is_router(agent: Dict[str, Any]) -> bool:
    return agent.get("category") in ("router", "orchestration")


@router.get("")
async def list_agents(category: Optional[str] = None):
    """List all agents, optionally filtered by category."""
    allowed = {str(item) for item in trusted_capability_ids("agent")}
    agents = [
        agent
        for agent in db_list_agents(category=category)
        if str(agent.get("agent_id") or "") in allowed
    ]
    return {"agents": [_serialize_agent(a) for a in agents], "count": len(agents)}


@router.get("/{agent_id}")
async def get_agent(agent_id: str):
    """Get a single agent by numeric row id or string agent_id."""
    agent = _resolve_agent(agent_id)
    return {"agent": _serialize_agent(agent)}


@router.post("")
async def create_agent(request: AgentCreate):
    """Create a new agent. Only vertical agents can be created."""
    del request
    _supply_chain_locked("agent.create")

    # Unreachable legacy implementation retained only to preserve source
    # history; the production API is fail-closed above.
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
        domain_scope=request.domain_scope,
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
    fields_set = getattr(request, "model_fields_set", getattr(request, "__fields_set__", set()))
    allowed = {
        "skill_ids",
        "knowledge_doc_ids",
        "tool_names",
        "available_mcp_tools",
    }
    if not fields_set or not set(fields_set).issubset(allowed):
        _supply_chain_locked("agent.implementation.update")
    skill_ids = (
        request.skill_ids
        if "skill_ids" in fields_set
        else get_agent_skills(str(agent["agent_id"]))
    )
    knowledge_doc_ids = (
        request.knowledge_doc_ids
        if "knowledge_doc_ids" in fields_set
        else get_agent_knowledge_bindings(str(agent["agent_id"]))
    )
    if knowledge_doc_ids is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "explicit_knowledge_binding_required",
                "message": "旧式隐式全量 RAG 绑定不能进入可信目录发布。",
            },
        )
    mcp_names = (
        request.available_mcp_tools
        if "available_mcp_tools" in fields_set
        else (
            request.tool_names
            if "tool_names" in fields_set
            else [
                str(item.get("tool_name") or "")
                for item in get_agent_tools(str(agent["agent_id"]))
                if str(item.get("tool_name") or "")
            ]
        )
    )
    try:
        set_trusted_agent_bindings(
            str(agent["agent_id"]),
            skill_ids or [],
            knowledge_doc_ids or [],
            mcp_server_names=mcp_names or [],
            system_tool_ids=[],
        )
    except CapabilityCatalogError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "trusted_catalog_binding_rejected", "message": str(exc)},
        ) from exc
    return {"agent": _serialize_agent(_resolve_agent(agent_id))}

    # Unreachable legacy implementation retained for source archaeology.
    is_router = _is_router(agent)

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
        if any(f in fields_set for f in ("category", "domain_scope", "is_router", "enabled", "model_id", "skill_ids", "available_skills", "tool_names", "available_mcp_tools", "knowledge_doc_ids")):
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
            domain_scope="property",
            enabled=enabled,
            model_id=model_id,
        )
        return {"agent": _serialize_agent(updated)}

    # 4. Vertical agent fields.
    enabled = request.enabled if "enabled" in fields_set else agent.get("enabled")
    model_id = request.model_id if "model_id" in fields_set else agent.get("model_id")
    category = "vertical"
    domain_scope = (
        request.domain_scope
        if "domain_scope" in fields_set
        else (agent.get("domain_scope") or "property")
    )

    updated = db_update_agent(
        agent_row_id=agent["id"],
        name=name,
        description=description,
        instructions=instructions,
        category=category,
        domain_scope=domain_scope,
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
    del agent_id
    _supply_chain_locked("agent.delete")


@router.post("/{agent_id}/toggle")
async def toggle_agent(agent_id: str, request: AgentToggleRequest):
    """Enable or disable an agent."""
    agent = _resolve_agent(agent_id)
    try:
        set_trusted_capability_enabled(
            "agent", str(agent["agent_id"]), request.enabled
        )
    except CapabilityCatalogError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "trusted_catalog_toggle_rejected", "message": str(exc)},
        ) from exc
    return {"agent": _serialize_agent(_resolve_agent(agent_id))}


@router.patch("/{agent_id}")
async def patch_agent(agent_id: str, request: AgentToggleRequest):
    """Alias for toggle via PATCH (used by the frontend)."""
    return await toggle_agent(agent_id, request)
