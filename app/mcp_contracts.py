"""Read-only MCP contract API used by the platform console.

This endpoint makes the runtime boundary inspectable: server discovery tells us
what a server *can* expose, while this contract tells us what an owner-facing
Agent is actually permitted to use.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter

from app.mcp_policy import (
    MCP_SERVER_CONTRACTS,
    WORK_ORDER_WRITE_BOUNDARY,
    allowed_tools_for_agent,
    tool_outcome_label,
)
from db.property_db import get_agent_tools, list_agents, list_mcp_servers, list_mcp_tools


router = APIRouter(prefix="/api/mcp-contracts", tags=["mcp-contracts"])


def _agent_label(agent: Dict[str, Any]) -> str:
    return str(agent.get("name") or agent.get("agent_id") or "unknown")


@router.get("")
def list_mcp_contracts() -> Dict[str, Any]:
    servers = list_mcp_servers()
    agents = list_agents()
    contract_rows: List[Dict[str, Any]] = []

    for server in servers:
        server_name = str(server.get("name") or "")
        policy = MCP_SERVER_CONTRACTS.get(server_name, {})
        discovered = list_mcp_tools(server_id=server.get("id"))
        discovered_by_name = {str(tool.get("name")): tool for tool in discovered}

        bindings: List[Dict[str, Any]] = []
        for agent in agents:
            agent_id = str(agent.get("agent_id") or "")
            bound_servers = {
                str(item.get("tool_name"))
                for item in (get_agent_tools(agent_id) or [])
                if item.get("tool_name")
            }
            allowlisted = sorted(allowed_tools_for_agent(agent_id, server_name))
            if server_name in bound_servers or allowlisted:
                bindings.append(
                    {
                        "agent_id": agent_id,
                        "agent_name": _agent_label(agent),
                        "bound": server_name in bound_servers,
                        "allowed_tools": allowlisted,
                    }
                )

        tools: List[Dict[str, Any]] = []
        known_tool_names = list(policy.get("tools", {}).keys())
        for tool_name in sorted(set(known_tool_names) | set(discovered_by_name)):
            discovered_tool = discovered_by_name.get(tool_name, {})
            allowed_by = [
                binding["agent_name"]
                for binding in bindings
                if binding["bound"] and tool_name in binding["allowed_tools"]
            ]
            tools.append(
                {
                    "name": tool_name,
                    "description": discovered_tool.get("description", ""),
                    "input_schema": discovered_tool.get("input_schema") or {},
                    "policy": policy.get("tools", {}).get(tool_name, {}),
                    "allowed_by": allowed_by,
                    "runtime_allowed": bool(allowed_by),
                }
            )

        contract_rows.append(
            {
                "server_id": server.get("id"),
                "server_name": server_name,
                "enabled": bool(server.get("enabled")),
                "description": server.get("description", ""),
                "mode": policy.get("mode", "unclassified"),
                "mode_label": policy.get("mode_label", "未定义"),
                "source": policy.get("source", "未声明"),
                "data_scope": policy.get("data_scope", "未声明"),
                "write_boundary": policy.get("write_boundary", "未声明"),
                "bindings": bindings,
                "tools": tools,
            }
        )

    return {
        "servers": contract_rows,
        "tool_outcomes": [
            {"code": code, "label": tool_outcome_label(code)}
            for code in ("success", "empty", "not_found", "invalid_input", "unauthorized", "timeout", "upstream_error")
        ],
        "write_boundary": WORK_ORDER_WRITE_BOUNDARY,
    }
