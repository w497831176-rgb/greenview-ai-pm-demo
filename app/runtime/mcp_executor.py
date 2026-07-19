"""Execute only published, read-only MCP calls and preserve four statuses."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.runtime.contracts import RuntimePath, ToolEffect, ToolInvocation
from app.runtime.tool_gateway import ToolGateway

try:
    from agno.tools.mcp import MCPTools
except Exception:  # pragma: no cover
    MCPTools = None  # type: ignore


DEMO_DEFAULT_CITY = os.getenv("YIAI_DEMO_CITY", "杭州")


if MCPTools is not None:
    class GovernedMCPTools(MCPTools):
        """Expose only published read tools and retain model-native evidence."""

        def __init__(self, *args: Any, **kwargs: Any):
            self.server_name = str(kwargs.pop("server_name"))
            self.allowed_function_names: Set[str] = set(
                kwargs.pop("allowed_function_names", [])
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
                    business_status, result_summary = _business_status(result)
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


def _relevant(server_name: str, message: str) -> bool:
    rules = {
        "weather-server": ("天气", "气温", "下雨", "降雨", "湿度", "明天", "户外", "巡检"),
        "workorder-server": ("工单进度", "查询工单", "我的工单", "工单状态", "维修进度", "工单数量"),
        "calendar-server": ("今天", "明天", "日期", "星期", "几点", "预约"),
    }
    if server_name in rules:
        return any(term in message for term in rules[server_name])
    return server_name.lower() in message.lower()


def _plan(server_name: str, message: str) -> List[Tuple[str, Dict[str, Any]]]:
    if server_name == "weather-server":
        cities = ("北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安")
        city = next((item for item in cities if item in message), None)
        if not city and "本小区" in message:
            city = DEMO_DEFAULT_CITY
        if not city:
            return []
        calls = [("get_current_weather", {"city": city})]
        if any(term in message for term in ("建议", "风险", "户外", "巡检", "暴雨", "下雨", "降雨")):
            calls.append(("get_weather_advice", {"city": city}))
        return calls
    if server_name == "workorder-server":
        calls: List[Tuple[str, Dict[str, Any]]] = []
        if any(term in message for term in ("最近工单", "我的工单", "我家工单", "维修记录", "工单进度")):
            calls.append(("get_my_recent_work_orders", {"limit": 5}))
        if any(term in message for term in ("待处理", "待派单", "未处理", "还有多少")):
            calls.append(("count_my_open_work_orders", {}))
        return calls
    if server_name == "calendar-server":
        if any(term in message for term in ("今天", "明天", "日期", "星期", "几点", "预约")):
            return [("get_current_datetime", {})]
    return []


def _business_status(result: Any) -> Tuple[str, str]:
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    try:
        parsed = json.loads(text)
        status = str(parsed.get("status") or "unknown")
    except Exception:
        status = "unknown"
    return status, text[:500]


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
    for server in snapshot_config.get("mcp_servers") or []:
        server_name = str(server.get("name") or "")
        if not server.get("enabled") or not _relevant(server_name, message):
            continue
        allowed = set(
            gateway.include_tools(agent_id, RuntimePath.CONSULTATION, server_name)
        )
        planned = [(name, args) for name, args in _plan(server_name, message) if name in allowed]
        if not planned:
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

        for tool_name, arguments in planned:
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
                        server_name=server_name,
                        tool_name=tool_name,
                        effect=policy.effect,
                        arguments=arguments,
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
                business_status, result_summary = _business_status(result)
                invocations.append(
                    ToolInvocation(
                        server_name=server_name,
                        tool_name=tool_name,
                        effect=policy.effect,
                        arguments=arguments,
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
                        server_name=server_name,
                        tool_name=tool_name,
                        effect=policy.effect,
                        arguments=arguments,
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
                        server_name=server_name,
                        tool_name=tool_name,
                        effect=policy.effect,
                        arguments=arguments,
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
        "不得猜测或把调用成功等同于业务成功：\n" + "\n".join(context),
        invocations,
    )


def build_model_native_read_tools(
    snapshot_config: Dict[str, Any],
    agent_id: str,
    excluded_servers: Optional[Set[str]] = None,
) -> List[Any]:
    """Build published dynamic read MCP toolkits for Agno's native tool loop.

    Formal built-in servers use deterministic pre-invocation. A user-added,
    explicitly bound server is exposed here so a newly published MCP can
    participate in the very next new session without hard-coded server names.
    """

    if GovernedMCPTools is None:
        return []
    excluded = set(excluded_servers or set())
    gateway = ToolGateway(snapshot_config)
    toolkits: List[Any] = []
    for server in snapshot_config.get("mcp_servers") or []:
        server_name = str(server.get("name") or "")
        if (
            not server.get("enabled")
            or server.get("is_builtin")
            or server_name in excluded
        ):
            continue
        allowed = gateway.include_tools(
            agent_id, RuntimePath.CONSULTATION, server_name
        )
        if not allowed or not server.get("command"):
            continue
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
        business_status, result_summary = _business_status(result)
        try:
            parsed = (
                result
                if isinstance(result, dict)
                else json.loads(result if isinstance(result, str) else json.dumps(result, default=str))
            )
        except Exception:
            parsed = {}
        if business_status not in {"success", "unknown"}:
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
