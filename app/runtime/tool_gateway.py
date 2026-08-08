"""Policy enforcement for tool discovery, exposure and invocation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.runtime.contracts import RuntimePath, ToolEffect, ToolPolicy
from app.runtime.capability_catalog import get_trusted_metadata


class ToolPolicyError(PermissionError):
    pass


class ToolGateway:
    def __init__(self, snapshot_config: Dict[str, Any]):
        self.config = snapshot_config

    def _agent(self, agent_id: str) -> Dict[str, Any]:
        for agent in self.config.get("agents") or []:
            if agent.get("agent_id") == agent_id:
                return agent
        raise ToolPolicyError(f"agent is not in RunConfigSnapshot: {agent_id}")

    def _validated_agent(self, agent_id: str) -> Dict[str, Any]:
        # Import lazily to avoid coupling agent construction to gateway module
        # import order. The validator uses only immutable scope/binding fields.
        from app.runtime.agent_factory import validate_agent_binding_isolation

        try:
            validate_agent_binding_isolation(self.config, agent_id)
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolPolicyError(
                f"agent binding isolation configuration error: {exc}"
            ) from exc
        return self._agent(agent_id)

    def policies_for_agent(
        self,
        agent_id: str,
        runtime_path: RuntimePath,
    ) -> List[ToolPolicy]:
        agent = self._validated_agent(agent_id)
        bound_servers = set(agent.get("mcp_server_names") or [])
        result: List[ToolPolicy] = []
        for server in self.config.get("mcp_servers") or []:
            if server.get("name") not in bound_servers or not server.get("enabled"):
                continue
            server_manifest = get_trusted_metadata(
                "mcp_server", server.get("server_id")
            )
            if not server_manifest:
                raise ToolPolicyError(
                    f"bound MCP server is absent from trusted catalog: "
                    f"{server.get('server_id')}"
                )
            if server_manifest.get("domain_scope") != agent.get("domain_scope"):
                raise ToolPolicyError(
                    f"bound MCP server domain mismatch: {agent_id}/"
                    f"{server.get('server_id')}"
                )
            for tool in server.get("tools") or []:
                policy = ToolPolicy.model_validate(tool.get("policy") or {})
                if not policy.enabled:
                    continue
                tool_manifest = get_trusted_metadata(
                    "mcp_tool", tool.get("tool_id")
                )
                if not tool_manifest:
                    raise ToolPolicyError(
                        f"enabled MCP Tool is absent from trusted catalog: "
                        f"{server.get('server_id')}/{tool.get('tool_id')}"
                    )
                if (
                    int(tool_manifest.get("server_id") or 0)
                    != int(server.get("server_id") or 0)
                    or tool_manifest.get("domain_scope")
                    != agent.get("domain_scope")
                    or tool_manifest.get("effect") != ToolEffect.READ.value
                ):
                    raise ToolPolicyError(
                        f"enabled MCP Tool violates trusted catalog policy: "
                        f"{server.get('server_id')}/{tool.get('tool_id')}"
                    )
                if policy.effect != ToolEffect.READ:
                    raise ToolPolicyError(
                        f"published MCP policy is not read-only: "
                        f"{server.get('server_id')}/{tool.get('tool_id')}/"
                        f"{policy.effect.value}"
                    )
                if runtime_path in policy.allowed_paths:
                    result.append(policy)
        return result

    def include_tools(
        self,
        agent_id: str,
        runtime_path: RuntimePath,
        server_name: str,
    ) -> List[str]:
        return [
            policy.tool_name
            for policy in self.policies_for_agent(agent_id, runtime_path)
            if policy.server_name == server_name
        ]

    def assert_read_invocation(
        self,
        agent_id: str,
        runtime_path: RuntimePath,
        server_name: str,
        tool_name: str,
    ) -> ToolPolicy:
        for policy in self.policies_for_agent(agent_id, runtime_path):
            if policy.server_name == server_name and policy.tool_name == tool_name:
                return policy
        raise ToolPolicyError(
            f"tool not allowed by published snapshot: {agent_id}/{server_name}/{tool_name}/{runtime_path.value}"
        )

    def write_policy(
        self,
        server_name: str,
        tool_name: str,
        agent_id: Optional[str] = None,
    ) -> ToolPolicy:
        del server_name, tool_name, agent_id
        raise ToolPolicyError(
            "MCP is permanently read-only; write policies and confirmed MCP "
            "execution are forbidden"
        )
