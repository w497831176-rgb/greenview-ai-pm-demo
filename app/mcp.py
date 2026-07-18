"""
MCP Server Management API
=========================

REST endpoints for MCP server configuration CRUD and tool discovery.
"""

import json
import os
import traceback
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

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


class McpToolPolicyUpdate(BaseModel):
    effect: str


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
    effect = request.effect.strip().lower()
    if effect not in {"read", "create", "update", "delete", "unknown"}:
        raise HTTPException(status_code=400, detail="invalid tool effect")
    metadata = dict(tool.get("tool_metadata") or {})
    metadata["effect"] = effect
    metadata["effect_source"] = "operator_declared"
    updated = save_mcp_tool(
        server_id=int(server["id"]),
        name=str(tool["name"]),
        description=str(tool.get("description") or ""),
        input_schema=tool.get("input_schema") or {},
        tool_metadata=metadata,
    )
    return {
        "mcp_tool": updated,
        "runtime_effective_on": "after_runtime_release_publish",
    }


@router.post("/{server_id}/discover")
async def discover_mcp_server_tools(server_id: str):
    """Discover tools from a running MCP server and cache them."""
    server = _resolve_server(server_id)
    try:
        discovered = await discover_server_tools(server)
        return {"mcp_server": server, "discovered": discovered, "count": len(discovered)}
    except Exception as e:
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
                tool_metadata = {**prior_metadata, "source": "discovered"}
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
