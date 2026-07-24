"""Execute only published, read-only MCP calls and preserve four statuses."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.runtime.contracts import RuntimePath, ToolEffect, ToolInvocation, ToolPlan
from app.runtime.tool_gateway import ToolGateway
from app.runtime.tool_planner import plan_tools, validate_arguments

try:
    from agno.tools.mcp import MCPTools
except Exception:  # pragma: no cover
    MCPTools = None  # type: ignore


if MCPTools is not None:
    class GovernedMCPTools(MCPTools):
        """Expose only published read tools and retain model-native evidence."""

        def __init__(self, *args: Any, **kwargs: Any):
            self.server_name = str(kwargs.pop("server_name"))
            self.allowed_function_names: Set[str] = set(
                kwargs.pop("allowed_function_names", [])
            )
            self.result_contracts: Dict[str, Dict[str, Any]] = dict(
                kwargs.pop("result_contracts", {})
            )
            self.recorded_invocations: List[ToolInvocation] = []
            super().__init__(*args, **kwargs)

        async def build_tools(self) -> None:
            await super(GovernedMCPTools, self).build_tools()
            functions = getattr(self, "functions", None) or {}
            functions = {
                name: function
                for name, function in functions.items()
                if name in self.allowed_function_names
            }
            self.functions = functions
            for function_name, function in functions.items():
                original = getattr(function, "entrypoint", None)
                if original is None or getattr(original, "_v18_governed", False):
                    continue
                wrapped = self._wrap_entrypoint(original, function_name)
                wrapped._v18_governed = True  # type: ignore[attr-defined]
                function.entrypoint = wrapped

        def _wrap_entrypoint(self, original: Any, function_name: str):
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                arguments = kwargs if kwargs else (args[0] if args else {})
                if not isinstance(arguments, dict):
                    arguments = {"value": str(arguments)}
                started = time.time()
                try:
                    if asyncio.iscoroutinefunction(original):
                        result = await original(*args, **kwargs)
                    else:
                        result = await asyncio.to_thread(original, *args, **kwargs)
                    business_status, result_summary = _business_status(
                        result,
                        self.result_contracts.get(function_name),
                    )
                    self.recorded_invocations.append(
                        ToolInvocation(
                            server_name=self.server_name,
                            tool_name=function_name,
                            effect=ToolEffect.READ,
                            arguments=arguments,
                            discovery_status="success",
                            transport_status="success",
                            invocation_status="success",
                            business_status=business_status,
                            latency_ms=int((time.time() - started) * 1000),
                            result_summary=result_summary,
                        )
                    )
                    return result
                except Exception as exc:
                    self.recorded_invocations.append(
                        ToolInvocation(
                            server_name=self.server_name,
                            tool_name=function_name,
                            effect=ToolEffect.READ,
                            arguments=arguments,
                            discovery_status="success",
                            transport_status="success",
                            invocation_status="failed",
                            business_status="unknown",
                            latency_ms=int((time.time() - started) * 1000),
                            error_summary=str(exc)[:500],
                        )
                    )
                    raise

            return wrapper
else:
    GovernedMCPTools = None  # type: ignore


def _business_status(
    result: Any,
    result_contract: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    parsed = _structured_result(result)
    if parsed is not None:
        raw_status = str(parsed.get("status") or "unknown")
        success_statuses = {
            str(item).lower()
            for item in (
                (result_contract or {}).get("success_statuses")
                or ["success"]
            )
        }
        return (
            "success" if raw_status.lower() in success_statuses else raw_status,
            json.dumps(parsed, ensure_ascii=False, default=str)[:500],
        )
    text = result if isinstance(result, str) else str(result)
    return "unknown", text[:500]


def _structured_result(value: Any, depth: int = 0) -> Optional[Dict[str, Any]]:
    """Unwrap Agno/MCP result envelopes into the server's business payload."""

    if value is None or depth > 6:
        return None
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump()
        except Exception:
            pass
    if isinstance(value, str):
        try:
            return _structured_result(json.loads(value), depth + 1)
        except Exception:
            return None
    if isinstance(value, (list, tuple)):
        for item in value:
            parsed = _structured_result(item, depth + 1)
            if parsed is not None:
                return parsed
        return None
    if not isinstance(value, dict):
        for attribute in ("structured_content", "metadata", "content", "text"):
            if hasattr(value, attribute):
                parsed = _structured_result(
                    getattr(value, attribute),
                    depth + 1,
                )
                if parsed is not None:
                    return parsed
        return None
    if value.get("status") is not None:
        return value
    for key in (
        "structured_content",
        "result",
        "content",
        "text",
        "metadata",
        "data",
    ):
        if key in value:
            parsed = _structured_result(value[key], depth + 1)
            if parsed is not None:
                return parsed
    return None


async def preinvoke_read_tools(
    snapshot_config: Dict[str, Any],
    agent_id: str,
    message: str,
) -> Tuple[str, List[ToolInvocation]]:
    if MCPTools is None:
        return "", []
    gateway = ToolGateway(snapshot_config)
    invocations: List[ToolInvocation] = []
    context: List[str] = []
    plans = plan_tools(
        snapshot_config,
        agent_id,
        message,
        RuntimePath.CONSULTATION,
        effects=[ToolEffect.READ],
        execution_modes=["auto_preinvoke"],
    )
    plans_by_server: Dict[str, List[ToolPlan]] = {}
    for plan in plans:
        plans_by_server.setdefault(plan.server_name, []).append(plan)
    for server in snapshot_config.get("mcp_servers") or []:
        server_name = str(server.get("name") or "")
        if not server.get("enabled"):
            continue
        planned = plans_by_server.get(server_name) or []
        if not planned:
            continue
        executable_plans: List[ToolPlan] = []
        for plan in planned:
            if not plan.missing_required and not plan.schema_errors:
                executable_plans.append(plan)
                continue
            invocations.append(
                ToolInvocation(
                    plan_id=plan.plan_id,
                    server_name=server_name,
                    tool_name=plan.tool_name,
                    effect=plan.effect,
                    arguments=plan.arguments,
                    planner_source=plan.planner_source,
                    match_reason=plan.match_reason,
                    discovery_status="not_started",
                    transport_status="not_started",
                    invocation_status="not_started",
                    business_status="invalid_input",
                    error_summary=(
                        "ToolPlan arguments failed schema validation: "
                        + "; ".join(
                            plan.schema_errors
                            or [
                                "missing required arguments: "
                                + ", ".join(plan.missing_required)
                            ]
                        )
                    ),
                )
            )
        if not executable_plans:
            continue
        command = server.get("command")
        if not command:
            continue
        full_command = shlex.join([str(command), *[str(arg) for arg in (server.get("args") or [])]])
        toolkit: Optional[Any] = None
        try:
            toolkit = MCPTools(
                command=full_command,
                env={
                    **dict(os.environ),
                    **{
                        key: os.environ[key]
                        for key in (server.get("env_keys") or [])
                        if key in os.environ
                    },
                },
                name=server_name,
                transport="stdio",
                timeout_seconds=15,
            )
            await asyncio.wait_for(toolkit.__aenter__(), timeout=8)
            functions = getattr(toolkit, "functions", None) or {}
            discovery_status = "success"
        except Exception as exc:
            invocations.append(
                ToolInvocation(
                    server_name=server_name,
                    tool_name="discovery",
                    effect=ToolEffect.READ,
                    discovery_status="failed",
                    transport_status="failed",
                    invocation_status="not_started",
                    business_status="unknown",
                    error_summary=str(exc)[:500],
                )
            )
            if toolkit and hasattr(toolkit, "close"):
                try:
                    await asyncio.wait_for(toolkit.close(), timeout=3)
                except Exception:
                    pass
            continue

        for plan in executable_plans:
            tool_name = plan.tool_name
            arguments = plan.arguments
            policy = gateway.assert_read_invocation(
                agent_id,
                RuntimePath.CONSULTATION,
                server_name,
                tool_name,
            )
            function = functions.get(tool_name)
            if function is None or not getattr(function, "entrypoint", None):
                invocations.append(
                    ToolInvocation(
                        plan_id=plan.plan_id,
                        server_name=server_name,
                        tool_name=tool_name,
                        effect=policy.effect,
                        arguments=arguments,
                        planner_source=plan.planner_source,
                        match_reason=plan.match_reason,
                        discovery_status=discovery_status,
                        transport_status="success",
                        invocation_status="failed",
                        business_status="unknown",
                        error_summary="published tool was not exposed by MCP discovery",
                    )
                )
                continue
            started = time.time()
            try:
                result = await asyncio.wait_for(
                    function.entrypoint(**arguments),
                    timeout=8,
                )
                business_status, result_summary = _business_status(
                    result,
                    plan.result_contract,
                )
                invocations.append(
                    ToolInvocation(
                        plan_id=plan.plan_id,
                        server_name=server_name,
                        tool_name=tool_name,
                        effect=policy.effect,
                        arguments=arguments,
                        planner_source=plan.planner_source,
                        match_reason=plan.match_reason,
                        discovery_status=discovery_status,
                        transport_status="success",
                        invocation_status="success",
                        business_status=business_status,
                        latency_ms=int((time.time() - started) * 1000),
                        result_summary=result_summary,
                    )
                )
                context.append(
                    f"[MCP {server_name}/{tool_name}] business_status={business_status}; result={result_summary}"
                )
            except asyncio.TimeoutError:
                invocations.append(
                    ToolInvocation(
                        plan_id=plan.plan_id,
                        server_name=server_name,
                        tool_name=tool_name,
                        effect=policy.effect,
                        arguments=arguments,
                        planner_source=plan.planner_source,
                        match_reason=plan.match_reason,
                        discovery_status=discovery_status,
                        transport_status="timeout",
                        invocation_status="failed",
                        business_status="unknown",
                        latency_ms=int((time.time() - started) * 1000),
                        error_summary="MCP invocation timed out",
                    )
                )
            except Exception as exc:
                invocations.append(
                    ToolInvocation(
                        plan_id=plan.plan_id,
                        server_name=server_name,
                        tool_name=tool_name,
                        effect=policy.effect,
                        arguments=arguments,
                        planner_source=plan.planner_source,
                        match_reason=plan.match_reason,
                        discovery_status=discovery_status,
                        transport_status="success",
                        invocation_status="failed",
                        business_status="unknown",
                        latency_ms=int((time.time() - started) * 1000),
                        error_summary=str(exc)[:500],
                    )
                )
        if toolkit and hasattr(toolkit, "close"):
            try:
                await asyncio.wait_for(toolkit.close(), timeout=3)
            except Exception:
                pass
    if not context:
        return "", invocations
    return (
        "\n\n以下为后端按 ToolPolicy 真实调用的只读 MCP 结果。"
        "不得猜测或把调用成功等同于业务成功。"
        "MCP 结果只由 mcp_calls 展示来源；回答正文中绝不能为 MCP 生成"
        "任何双中括号、引用编号、参考号、脚注或来源占位符，也不能占用 "
        "RAG 引用编号：\n"
        + "\n".join(context),
        invocations,
    )


def build_model_native_read_tools(
    snapshot_config: Dict[str, Any],
    agent_id: str,
    message: str,
    excluded_servers: Optional[Set[str]] = None,
    excluded_tools: Optional[Set[Tuple[str, str]]] = None,
) -> List[Any]:
    """Build only message-matched model-native tools for Agno's tool loop.

    A published binding makes a tool eligible; it does not mean every request
    should start that MCP server.  Matching the immutable Tool metadata before
    Agent construction both preserves the control-plane contract and prevents
    unrelated route-only prompts from being coupled to external tool startup.
    """

    if GovernedMCPTools is None:
        return []
    excluded = set(excluded_servers or set())
    excluded_tool_keys = set(excluded_tools or set())
    planned_tool_keys = {
        (plan.server_name, plan.tool_name)
        for plan in plan_tools(
            snapshot_config,
            agent_id,
            message,
            RuntimePath.CONSULTATION,
            effects=[ToolEffect.READ],
            execution_modes=["model_native"],
        )
    }
    if not planned_tool_keys:
        return []
    gateway = ToolGateway(snapshot_config)
    toolkits: List[Any] = []
    for server in snapshot_config.get("mcp_servers") or []:
        server_name = str(server.get("name") or "")
        if (
            not server.get("enabled")
            or server_name in excluded
        ):
            continue
        policy_allowed = [
            tool_name
            for tool_name in gateway.include_tools(
                agent_id, RuntimePath.CONSULTATION, server_name
            )
            if (server_name, tool_name) not in excluded_tool_keys
        ]
        tool_definitions = {
            str(tool.get("name") or ""): tool
            for tool in server.get("tools") or []
        }
        allowed = [
            tool_name
            for tool_name in policy_allowed
            if (
                (server_name, tool_name) in planned_tool_keys
                and
                (
                    tool_definitions.get(tool_name, {}).get("tool_metadata")
                    or {}
                ).get("execution_mode")
                == "model_native"
            )
        ]
        if not allowed or not server.get("command"):
            continue
        result_contracts = {
            str(tool.get("name") or ""): (
                (tool.get("tool_metadata") or {}).get("result_contract") or {}
            )
            for tool in server.get("tools") or []
            if str(tool.get("name") or "") in allowed
        }
        full_command = shlex.join(
            [
                str(server["command"]),
                *[str(argument) for argument in (server.get("args") or [])],
            ]
        )
        toolkits.append(
            GovernedMCPTools(
                command=full_command,
                env={
                    **dict(os.environ),
                    **{
                        key: os.environ[key]
                        for key in (server.get("env_keys") or [])
                        if key in os.environ
                    },
                },
                name=server_name,
                server_name=server_name,
                allowed_function_names=allowed,
                result_contracts=result_contracts,
                transport="stdio",
                timeout_seconds=15,
            )
        )
    return toolkits


async def invoke_confirmed_write(
    snapshot_config: Dict[str, Any],
    agent_id: str,
    server_name: str,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Invoke one approved create/update MCP tool from its immutable snapshot."""

    if MCPTools is None:
        raise RuntimeError("Agno MCP toolkit is unavailable")
    gateway = ToolGateway(snapshot_config)
    policy = gateway.write_policy(server_name, tool_name, agent_id=agent_id)
    server = next(
        (
            item
            for item in snapshot_config.get("mcp_servers") or []
            if item.get("name") == server_name and item.get("enabled")
        ),
        None,
    )
    if not server or not server.get("command"):
        raise RuntimeError("published MCP server has no executable command")
    tool = next(
        (
            item
            for item in server.get("tools") or []
            if item.get("name") == tool_name
        ),
        None,
    )
    if not tool:
        raise RuntimeError("published MCP tool is absent from snapshot")
    schema_errors = validate_arguments(
        arguments,
        tool.get("input_schema") or {},
    )
    if schema_errors:
        raise ValueError(
            "MCP arguments failed published JSON Schema: "
            + "; ".join(schema_errors)
        )
    full_command = shlex.join(
        [
            str(server["command"]),
            *[str(argument) for argument in (server.get("args") or [])],
        ]
    )
    toolkit: Optional[Any] = None
    try:
        toolkit = MCPTools(
            command=full_command,
            env={
                **dict(os.environ),
                **{
                    key: os.environ[key]
                    for key in (server.get("env_keys") or [])
                    if key in os.environ
                },
            },
            name=server_name,
            transport="stdio",
            timeout_seconds=15,
        )
        await asyncio.wait_for(toolkit.__aenter__(), timeout=8)
        functions = getattr(toolkit, "functions", None) or {}
        function = functions.get(tool_name)
        if function is None or not getattr(function, "entrypoint", None):
            raise RuntimeError("approved MCP function was not exposed by discovery")
        started = time.time()
        result = await asyncio.wait_for(
            function.entrypoint(**arguments),
            timeout=12,
        )
        business_status, result_summary = _business_status(
            result,
            (tool.get("tool_metadata") or {}).get("result_contract") or {},
        )
        parsed = _structured_result(result) or {}
        if business_status != "success":
            raise RuntimeError(
                f"MCP business outcome is not successful: {business_status}"
            )
        resource_id = ""
        if isinstance(parsed, dict):
            for key in (
                "resource_id",
                "id",
                "work_order_id",
                "order_id",
                "ticket_id",
                "booking_id",
            ):
                if parsed.get(key):
                    resource_id = str(parsed[key])
                    break
            nested = parsed.get("data") or parsed.get("result")
            if not resource_id and isinstance(nested, dict):
                for key in (
                    "resource_id",
                    "id",
                    "work_order_id",
                    "order_id",
                    "ticket_id",
                    "booking_id",
                ):
                    if nested.get(key):
                        resource_id = str(nested[key])
                        break
        if not resource_id:
            raise RuntimeError(
                "MCP write returned no durable resource id; committed Receipt is forbidden"
            )
        return {
            "resource_type": f"mcp:{server_name}:{tool_name}",
            "resource_id": resource_id,
            "server_name": server_name,
            "tool_name": tool_name,
            "effect": policy.effect.value,
            "arguments": arguments,
            "business_status": business_status,
            "latency_ms": int((time.time() - started) * 1000),
            "result_summary": result_summary,
            "raw_result": parsed,
        }
    finally:
        if toolkit and hasattr(toolkit, "close"):
            try:
                await asyncio.wait_for(toolkit.close(), timeout=3)
            except Exception:
                pass
