"""
MCP Server Management API
=========================

REST endpoints for MCP server configuration CRUD and tool discovery.
"""

import asyncio
import json
import os
import traceback
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

from app.runtime.tool_planner import (
    DEFAULT_RESULT_CONTRACT,
    effective_tool_metadata,
    validate_tool_metadata,
)
from app.runtime.mcp_importer import (
    McpImportError,
    prepare_git_mcp_package,
    suggest_tool_effect,
)
from db.property_db import (
    create_mcp_server as db_create_mcp_server,
    delete_mcp_server as db_delete_mcp_server,
    delete_mcp_tools_for_server,
    get_mcp_server as db_get_mcp_server,
    get_mcp_tool,
    list_mcp_servers as db_list_mcp_servers,
    list_mcp_tools,
    save_mcp_tool,
    toggle_mcp_server_enabled,
    update_mcp_server as db_update_mcp_server,
    update_mcp_server_import_metadata,
)

router = APIRouter(prefix="/api/mcp-servers", tags=["mcp-servers"])


def _resolve_server(identifier: str) -> Dict[str, Any]:
    """Resolve an MCP server by integer id or string id (e.g. 'weather')."""
    if identifier.isdigit():
        server = db_get_mcp_server(int(identifier))
        if server:
            return server
    # Fall back to matching by string id/name.
    lowered = identifier.lower()
    for server in db_list_mcp_servers():
        if str(server.get("id")) == identifier:
            return server
        server_id = str(server.get("server_id", "")).lower()
        name = str(server.get("name", "")).lower()
        if server_id == lowered or name == lowered or lowered in name:
            return server
    raise HTTPException(status_code=404, detail="mcp server not found")


class McpServerCreate(BaseModel):
    name: str
    command: str = ""
    args: Optional[List[str]] = Field(default_factory=list)
    env: Optional[Dict[str, str]] = Field(default_factory=dict)
    description: str = ""
    enabled: bool = True


class McpServerUpdate(BaseModel):
    name: str
    command: str = ""
    args: Optional[List[str]] = Field(default_factory=list)
    env: Optional[Dict[str, str]] = Field(default_factory=dict)
    description: str = ""
    enabled: bool = True


class McpServerToggle(BaseModel):
    enabled: bool


class McpGitImportRequest(BaseModel):
    git_url: str
    name: str = ""
    description: str = ""
    runtime_type: str = "auto"
    env: Dict[str, str] = Field(default_factory=dict)


class McpToolPolicyUpdate(BaseModel):
    effect: str


class McpToolRuntimePolicyUpdate(BaseModel):
    effect: str
    risk_level: str = "L1"
    result_contract: Dict[str, Any] = Field(
        default_factory=lambda: dict(DEFAULT_RESULT_CONTRACT)
    )
    natural_language_intents: List[str] = Field(default_factory=list)
    trigger_keywords: List[str] = Field(default_factory=list)
    trigger_mode: str = "any"
    execution_mode: str = "model_native"
    argument_bindings: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


def _save_tool_runtime_policy(
    server: Dict[str, Any],
    tool: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    effect = str(payload.get("effect") or "").strip().lower()
    if effect not in {"read", "create", "update", "delete", "unknown"}:
        raise HTTPException(status_code=400, detail="invalid tool effect")
    metadata = dict(tool.get("tool_metadata") or {})
    metadata.update(payload)
    metadata["effect"] = effect
    metadata["effect_source"] = "operator_declared"
    risk_level = str(metadata.get("risk_level") or "").upper()
    if effect in {"create", "update"} and risk_level in {"L0", "L1"}:
        risk_level = "L2"
    if effect in {"delete", "unknown"}:
        risk_level = "L3"
    metadata["risk_level"] = risk_level or (
        "L1" if effect == "read" else "L2"
    )
    metadata.setdefault("result_contract", dict(DEFAULT_RESULT_CONTRACT))
    if effect in {"create", "update"}:
        metadata["execution_mode"] = "proposal"
    elif effect != "read":
        metadata["execution_mode"] = "model_native"
    errors = validate_tool_metadata(metadata, tool.get("input_schema") or {})
    if errors:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_tool_runtime_policy", "errors": errors},
        )
    return save_mcp_tool(
        server_id=int(server["id"]),
        name=str(tool["name"]),
        description=str(tool.get("description") or ""),
        input_schema=tool.get("input_schema") or {},
        tool_metadata=metadata,
    )


@router.get("")
async def list_mcp_servers():
    """List all MCP server configurations."""
    servers = db_list_mcp_servers()
    return {"mcp_servers": servers, "count": len(servers)}


@router.get("/{server_id}")
async def get_mcp_server(server_id: str):
    """Get a single MCP server configuration."""
    server = _resolve_server(server_id)
    return {"mcp_server": server}


@router.post("")
async def create_mcp_server(request: McpServerCreate):
    """Create a new MCP server configuration."""
    server = db_create_mcp_server(
        name=request.name,
        command=request.command,
        args=request.args,
        env=request.env,
        description=request.description,
        enabled=request.enabled,
    )
    return {"mcp_server": server}


@router.post("/import-git")
async def import_mcp_server_from_git(request: McpGitImportRequest):
    """Prepare, connect and discover a public Python/Node MCP repository.

    Import is Draft-only.  A successful connection does not grant any Tool
    permission and does not affect existing sessions; the operator still
    classifies tools, binds an Agent and publishes a RuntimeRelease.
    """

    try:
        prepared = await asyncio.to_thread(
            prepare_git_mcp_package,
            request.git_url,
            requested_name=request.name,
            requested_runtime=request.runtime_type,
        )
    except McpImportError as exc:
        raise HTTPException(status_code=422, detail=exc.as_dict()) from exc

    if any(
        str(item.get("name") or "").casefold() == prepared.name.casefold()
        for item in db_list_mcp_servers()
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "mcp_name_exists",
                "message": "已存在同名 MCP Server，请修改显示名称后重试",
                "name": prepared.name,
            },
        )

    server = db_create_mcp_server(
        name=prepared.name,
        command=prepared.command,
        args=prepared.args,
        env=request.env,
        description=(
            request.description.strip()
            or f"从 Git 导入的 {prepared.runtime_type} MCP Server"
        ),
        enabled=True,
        source_type=prepared.source_type,
        source_url=prepared.source_url,
        runtime_type=prepared.runtime_type,
        install_status="discovering",
        install_detail="代码与依赖已准备，正在连接并发现 Tool",
        package_path=prepared.package_path,
        detected_entrypoint=prepared.detected_entrypoint,
    )
    try:
        discovered = await discover_server_tools(server)
    except Exception as exc:
        update_mcp_server_import_metadata(
            int(server["id"]),
            install_status="failed",
            install_detail=f"连接或工具发现失败：{type(exc).__name__}: {str(exc)[:500]}",
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "mcp_discovery_failed",
                "message": "仓库已准备，但 MCP 连接或工具发现失败",
                "server_id": server["id"],
                "detail": str(exc)[:500],
                "steps": prepared.steps,
            },
        ) from exc
    if not discovered:
        update_mcp_server_import_metadata(
            int(server["id"]),
            install_status="failed",
            install_detail="连接成功，但 Server 没有返回任何 Tool",
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "mcp_no_tools",
                "message": "连接成功，但该 MCP Server 没有返回任何 Tool",
                "server_id": server["id"],
                "steps": prepared.steps,
            },
        )

    server = update_mcp_server_import_metadata(
        int(server["id"]),
        install_status="ready",
        install_detail=f"连接成功，已发现 {len(discovered)} 个 Tool",
    )
    return {
        "mcp_server": server,
        "prepared_package": prepared.as_dict(),
        "discovered": discovered,
        "count": len(discovered),
        "next_steps": [
            "确认每个 Tool 是只读查询还是需要确认的写操作",
            "把 MCP Server 绑定到垂直 Agent",
            "校验并发布 RuntimeRelease",
            "在下一新会话用自然语言验收并查看 Trace",
        ],
    }


@router.put("/{server_id}")
async def update_mcp_server(server_id: str, request: McpServerUpdate):
    """Update an MCP server configuration."""
    server = _resolve_server(server_id)
    updated = db_update_mcp_server(
        server_id=server["id"],
        name=request.name,
        command=request.command,
        args=request.args,
        env=request.env,
        description=request.description,
        enabled=request.enabled,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="not found")
    return {"mcp_server": updated}


@router.delete("/{server_id}")
async def delete_mcp_server(server_id: str):
    """Delete an MCP server configuration."""
    server = _resolve_server(server_id)
    deleted = db_delete_mcp_server(server["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="not found")
    delete_mcp_tools_for_server(server["id"])
    return {"ok": True, "deleted_id": server_id}


@router.post("/{server_id}/toggle")
async def toggle_mcp_server(server_id: str, request: Optional[McpServerToggle] = None):
    """Enable or disable an MCP server. If no body is provided, flip the current state."""
    server = _resolve_server(server_id)
    current_enabled = bool(server.get("enabled"))
    target = request.enabled if request else not current_enabled
    updated = toggle_mcp_server_enabled(server["id"], target)
    return {"mcp_server": updated}


@router.get("/{server_id}/tools")
async def get_mcp_server_tools(server_id: str):
    """List cached tools for an MCP server."""
    server = _resolve_server(server_id)
    tools = list_mcp_tools(server_id=server["id"])
    for tool in tools:
        tool["effective_runtime_metadata"] = effective_tool_metadata(
            str(server.get("name") or ""),
            str(tool.get("name") or ""),
            tool.get("tool_metadata") or {},
        )
    return {"mcp_server": server, "tools": tools, "count": len(tools)}


@router.put("/{server_id}/tools/{tool_id}/policy")
async def update_mcp_tool_policy(
    server_id: str,
    tool_id: int,
    request: McpToolPolicyUpdate,
):
    server = _resolve_server(server_id)
    tool = get_mcp_tool(tool_id)
    if not tool or int(tool.get("server_id") or 0) != int(server["id"]):
        raise HTTPException(status_code=404, detail="mcp tool not found")
    updated = _save_tool_runtime_policy(
        server,
        tool,
        {
            **dict(tool.get("tool_metadata") or {}),
            "effect": request.effect,
        },
    )
    return {
        "mcp_tool": updated,
        "runtime_effective_on": "after_runtime_release_publish",
    }


@router.put("/{server_id}/tools/{tool_id}/runtime-policy")
async def update_mcp_tool_runtime_policy(
    server_id: str,
    tool_id: int,
    request: McpToolRuntimePolicyUpdate,
):
    """Save the generic natural-language ToolPlan contract as Draft."""

    server = _resolve_server(server_id)
    tool = get_mcp_tool(tool_id)
    if not tool or int(tool.get("server_id") or 0) != int(server["id"]):
        raise HTTPException(status_code=404, detail="mcp tool not found")
    updated = _save_tool_runtime_policy(
        server,
        tool,
        request.model_dump(),
    )
    return {
        "mcp_tool": updated,
        "runtime_effective_on": "after_runtime_release_publish_new_session",
        "contract": {
            "planner": "configuration_driven",
            "writes": "proposal_confirmation_receipt",
        },
    }


@router.post("/{server_id}/discover")
async def discover_mcp_server_tools(server_id: str):
    """Discover tools from a running MCP server and cache them."""
    server = _resolve_server(server_id)
    try:
        discovered = await discover_server_tools(server)
        update_mcp_server_import_metadata(
            int(server["id"]),
            install_status="ready" if discovered else "failed",
            install_detail=(
                f"连接成功，已发现 {len(discovered)} 个 Tool"
                if discovered
                else "连接成功，但 Server 没有返回任何 Tool"
            ),
        )
        return {"mcp_server": server, "discovered": discovered, "count": len(discovered)}
    except Exception as e:
        update_mcp_server_import_metadata(
            int(server["id"]),
            install_status="failed",
            install_detail=f"连接或工具发现失败：{type(e).__name__}: {str(e)[:500]}",
        )
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"discover failed: {e}")


async def discover_server_tools(server: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Connect to a single MCP server via stdio and cache its tools."""
    command = server.get("command") or "python"
    args = server.get("args") or []
    env = server.get("env") or {}
    server_id = server["id"]

    # Resolve relative paths inside the container.
    resolved_args = []
    for arg in args:
        if arg.startswith("/app/tools/") and not os.path.exists(arg):
            # Fallback to local tools directory for development/tests.
            local = os.path.join(os.path.dirname(__file__), "..", "tools", os.path.basename(arg))
            if os.path.exists(local):
                arg = os.path.abspath(local)
        resolved_args.append(arg)

    merged_env = {**dict(os.environ), **env}
    params = StdioServerParameters(command=command, args=resolved_args, env=merged_env)

    discovered: List[Dict[str, Any]] = []
    existing_by_name = {
        str(item.get("name") or ""): item
        for item in list_mcp_tools(server_id=server_id)
    }
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                tools = getattr(tools_result, "tools", tools_result)
                for tool in tools:
                    name = getattr(tool, "name", "")
                    description = getattr(tool, "description", "") or ""
                    input_schema = getattr(tool, "inputSchema", {}) or {}
                    prior_metadata = (
                        existing_by_name.get(str(name), {}).get("tool_metadata") or {}
                    )
                    tool_metadata = {
                        **prior_metadata,
                        "source": "discovered",
                        "effect_suggestion": suggest_tool_effect(name, description),
                    }
                    save_mcp_tool(
                        server_id=server_id,
                        name=name,
                        description=description,
                        input_schema=input_schema,
                        tool_metadata=tool_metadata,
                    )
                    discovered.append({
                        "name": name,
                        "description": description,
                        "input_schema": input_schema,
                    })
    except Exception:
        traceback.print_exc()
        raise
    return discovered


async def discover_all_mcp_tools() -> Dict[str, Any]:
    """Discover tools from all enabled MCP servers and cache them.

    Called during application lifespan startup.
    """
    summary: Dict[str, Any] = {"servers": 0, "tools": 0, "errors": []}
    for server in db_list_mcp_servers():
        if not server.get("enabled"):
            continue
        server_id = server["id"]
        try:
            discovered = await discover_server_tools(server)
            summary["servers"] += 1
            summary["tools"] += len(discovered)
        except Exception as e:
            summary["errors"].append({"server_id": server_id, "error": str(e)})
    return summary
