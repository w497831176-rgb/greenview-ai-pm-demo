"""Execute only published, read-only MCP calls and preserve four statuses."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shlex
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from app.runtime.contracts import (
    RuntimePath,
    ToolEffect,
    ToolEvidence,
    ToolEvidenceFact,
    ToolInvocation,
    ToolPlan,
    content_hash,
    stable_id,
)
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
                invocation_id = f"tool_{uuid.uuid4().hex}"
                started = time.time()
                try:
                    if asyncio.iscoroutinefunction(original):
                        result = await original(*args, **kwargs)
                    else:
                        result = await asyncio.to_thread(original, *args, **kwargs)
                    business_status, result_summary, tool_evidence = (
                        _evaluate_read_tool_result(
                            result,
                            self.result_contracts.get(function_name),
                            invocation_id=invocation_id,
                            server_name=self.server_name,
                            tool_name=function_name,
                        )
                    )
                    invocation = ToolInvocation(
                        invocation_id=invocation_id,
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
                        tool_evidence=tool_evidence,
                    )
                    self.recorded_invocations.append(invocation)
                    # The model sees the same safe structured facts that enter
                    # RunEvidenceBundle. Raw payloads, request parameters and
                    # non-success result bodies never become answer evidence.
                    return _tool_evidence_context(invocation)
                except Exception as exc:
                    self.recorded_invocations.append(
                        ToolInvocation(
                            invocation_id=invocation_id,
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


_MCP_ENVELOPE_KEYS = {
    "content",
    "isError",
    "is_error",
    "metadata",
    "structuredContent",
    "structured_content",
    "text",
    "type",
}
_SENSITIVE_FIELD_MARKERS = {
    "access_key",
    "api_key",
    "apikey",
    "auth_header",
    "authorization",
    "cookie",
    "credential",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "session_cookie",
    "token",
}


def _json_safe(value: Any, depth: int = 0) -> Any:
    """Return a deterministic JSON-compatible value without truncating it."""

    if depth > 24:
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump(mode="json")
        except TypeError:
            value = value.model_dump()
        except Exception:
            pass
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(nested, depth + 1)
            for key, nested in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth + 1) for item in value]
    return str(value)


def _canonical_result_payload(
    value: Any,
    depth: int = 0,
    *,
    allow_root_result_wrapper: bool = True,
) -> Any:
    """Unwrap only MCP transport envelopes and retain the complete payload.

    Unlike ``_structured_result`` this function never selects the first matching
    child from a business object.  The complete result is needed for hashing and
    scalar evidence extraction; display truncation happens separately.
    """

    if depth > 12:
        return _json_safe(value)
    for attribute in ("structured_content", "structuredContent"):
        if hasattr(value, attribute):
            structured = getattr(value, attribute)
            if structured is not None:
                return _canonical_result_payload(
                    structured,
                    depth + 1,
                    allow_root_result_wrapper=allow_root_result_wrapper,
                )
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump(mode="json")
        except TypeError:
            value = value.model_dump()
        except Exception:
            pass
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            return _canonical_result_payload(
                json.loads(stripped),
                depth + 1,
                allow_root_result_wrapper=allow_root_result_wrapper,
            )
        except Exception:
            return value
    if isinstance(value, dict):
        for key in ("structured_content", "structuredContent"):
            if key in value and value[key] is not None:
                return _canonical_result_payload(
                    value[key],
                    depth + 1,
                    allow_root_result_wrapper=allow_root_result_wrapper,
                )
        metadata = value.get("metadata")
        if isinstance(metadata, dict):
            for key in ("structured_content", "structuredContent"):
                if key in metadata and metadata[key] is not None:
                    return _canonical_result_payload(
                        metadata[key],
                        depth + 1,
                        allow_root_result_wrapper=allow_root_result_wrapper,
                    )
        # Agno's production MCP adapter wraps a complete business JSON object
        # or array as the string value of a root-level, single-key ``result``
        # object.  Unwrap exactly that transport shape once.  A calculator's
        # business payload ``{"result": 84}``, JSON scalars, multi-key business
        # objects and nested ``result`` fields retain their original semantics.
        if (
            allow_root_result_wrapper
            and set(value) == {"result"}
            and isinstance(value.get("result"), str)
        ):
            try:
                parsed_result = json.loads(value["result"])
            except Exception:
                parsed_result = None
            if isinstance(parsed_result, (dict, list)):
                return _json_safe(parsed_result, depth + 1)
        # Content blocks are a transport envelope only when no business fields
        # are present.  A business object containing a ``content`` field remains
        # intact and is hashed as returned.
        if set(value).issubset(_MCP_ENVELOPE_KEYS) and "content" in value:
            return _canonical_result_payload(
                value.get("content"),
                depth + 1,
                allow_root_result_wrapper=allow_root_result_wrapper,
            )
        return _json_safe(value, depth + 1)
    if isinstance(value, (list, tuple)):
        # A single MCP text content block is an envelope around its text.  Real
        # business arrays retain every item rather than the former first match.
        if len(value) == 1 and isinstance(value[0], dict):
            block = value[0]
            if set(block).issubset(_MCP_ENVELOPE_KEYS) and "text" in block:
                return _canonical_result_payload(
                    block.get("text"),
                    depth + 1,
                    allow_root_result_wrapper=allow_root_result_wrapper,
                )
        return [
            _canonical_result_payload(
                item,
                depth + 1,
                allow_root_result_wrapper=False,
            )
            for item in value
        ]
    return _json_safe(value, depth + 1)


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
    if result is None:
        return "empty", ""
    text = result if isinstance(result, str) else str(result)
    summary = text.strip()[:500]
    if not summary or summary.lower() in {"none", "null"}:
        return "empty", summary
    lowered = summary.lower()
    if any(marker in lowered for marker in ("not found", "未找到", "不存在")):
        return "not_found", summary
    if any(marker in lowered for marker in ("timeout", "timed out", "超时")):
        return "timeout", summary
    if any(marker in lowered for marker in ("unauthorized", "forbidden", "无权限")):
        return "unauthorized", summary
    if any(
        marker in lowered
        for marker in (
            "error",
            "failed",
            "failure",
            "invalid",
            "错误",
            "失败",
            "参数不合法",
        )
    ):
        return "upstream_error", summary
    # A read Tool may validly return a scalar or plain text without a business
    # status field. Reaching here means discovery, transport and invocation all
    # succeeded and the result is non-empty, so it is a real success.
    return "success", summary


def _normalized_scalar(value: Any) -> Tuple[str, str]:
    if isinstance(value, bool):
        return ("true" if value else "false"), "boolean"
    if isinstance(value, int):
        return str(value), "integer"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value), "string"
        if value.is_integer():
            return str(int(value)), "number"
        return format(value, ".15g"), "number"
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    return normalized, "string"


def _safe_field_name(field_name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", field_name.lower()).strip("_")
    return not any(marker in normalized for marker in _SENSITIVE_FIELD_MARKERS)


def _json_path(parent: str, key: Any) -> str:
    if isinstance(key, int):
        return f"{parent}[{key}]"
    name = str(key)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return f"{parent}.{name}"
    return f"{parent}[{json.dumps(name, ensure_ascii=False)}]"


def _unit_hint(
    field_name: str,
    json_path: str,
    result_contract: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    contract = result_contract or {}
    mappings = (
        contract.get("evidence_units")
        or contract.get("field_units")
        or contract.get("unit_by_path")
        or {}
    )
    hint = mappings.get(json_path) or mappings.get(field_name)
    if isinstance(hint, dict):
        return (
            str(hint.get("unit")) if hint.get("unit") is not None else None,
            str(hint.get("semantic_type"))
            if hint.get("semantic_type")
            else None,
        )
    if hint is not None:
        return str(hint), None

    normalized = re.sub(r"[^a-z0-9]+", "_", field_name.lower()).strip("_")
    unit_rules = (
        (("pct", "percent", "percentage"), "%"),
        (("temperature_c", "temp_c", "celsius", "c"), "℃"),
        (("minutes", "minute", "mins"), "分钟"),
        (("seconds", "second", "secs"), "秒"),
        (("hours", "hour", "hrs"), "小时"),
        (("days", "day"), "天"),
        (("yuan", "cny"), "元"),
        (("level", "grade"), "级"),
    )
    for suffixes, unit in unit_rules:
        if any(normalized == suffix or normalized.endswith(f"_{suffix}") for suffix in suffixes):
            return unit, None
    return None, None


def _semantic_type(field_name: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    normalized = re.sub(r"[^a-z0-9]+", "_", field_name.lower()).strip("_")
    if normalized == "result":
        return "calculated_result"
    if normalized == "status" or normalized.endswith("_status"):
        return "business_status"
    if normalized == "id" or normalized.endswith("_id"):
        return "business_identifier"
    if any(token in normalized for token in ("condition", "weather")):
        return "weather_condition"
    if "wind" in normalized:
        return "wind_condition"
    if any(token in normalized for token in ("city", "location", "address")):
        return "location"
    return "business_result"


def _safe_scalar_facts(
    payload: Any,
    *,
    invocation_id: str,
    result_contract: Optional[Dict[str, Any]],
) -> List[ToolEvidenceFact]:
    facts: List[ToolEvidenceFact] = []

    def visit(value: Any, path: str, field_name: str) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                key_name = str(key)
                if not _safe_field_name(key_name):
                    continue
                visit(nested, _json_path(path, key_name), key_name)
            return
        if isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, _json_path(path, index), field_name)
            return
        if value is None or isinstance(value, (dict, list, tuple, set)):
            return
        # Root operation status is represented by ToolEvidence.business_status;
        # nested business statuses remain useful facts (for example work orders).
        if path == "$.status":
            return
        normalized, value_type = _normalized_scalar(value)
        if not normalized:
            return
        unit, explicit_semantic = _unit_hint(
            field_name,
            path,
            result_contract,
        )
        display_value = (
            f"{normalized}{unit}"
            if unit and value_type in {"integer", "number"}
            else normalized
        )
        fact_payload = {
            "invocation_id": invocation_id,
            "json_path": path,
            "normalized_value": normalized,
            "unit": unit,
        }
        facts.append(
            ToolEvidenceFact(
                fact_id=stable_id("tool_fact", fact_payload),
                json_path=path,
                field_name=field_name,
                value_type=value_type,
                value=value,
                normalized_value=normalized,
                unit=unit,
                display_value=display_value,
                semantic_type=_semantic_type(field_name, explicit_semantic),
            )
        )

    visit(payload, "$", "result")
    return facts


def _evaluate_read_tool_result(
    result: Any,
    result_contract: Optional[Dict[str, Any]],
    *,
    invocation_id: str,
    server_name: str,
    tool_name: str,
) -> Tuple[str, str, Optional[ToolEvidence]]:
    """Freeze one successful read result; request arguments are not accepted."""

    business_status, result_summary = _business_status(result, result_contract)
    if business_status != "success":
        return business_status, result_summary, None
    payload = _canonical_result_payload(result)
    payload_hash = content_hash(payload)
    facts = _safe_scalar_facts(
        payload,
        invocation_id=invocation_id,
        result_contract=result_contract,
    )
    evidence_payload = {
        "invocation_id": invocation_id,
        "server_name": server_name,
        "tool_name": tool_name,
        "payload_hash": payload_hash,
    }
    return (
        business_status,
        result_summary,
        ToolEvidence(
            evidence_id=stable_id("tool_ev", evidence_payload),
            invocation_id=invocation_id,
            server_name=server_name,
            tool_name=tool_name,
            payload_hash=payload_hash,
            facts=facts,
        ),
    )


def _tool_evidence_context(invocation: ToolInvocation) -> str:
    """Render model context from safe facts, never from the display summary."""

    header = (
        f"[MCP {invocation.server_name}/{invocation.tool_name}] "
        f"business_status={invocation.business_status}"
    )
    evidence = invocation.tool_evidence
    if evidence is None:
        return header
    lines = [f"{header}; evidence_id={evidence.evidence_id}"]
    lines.extend(
        f"- {fact.json_path} = {fact.display_value}"
        for fact in evidence.facts
    )
    return "\n".join(lines)


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
            invocation_id = f"tool_{uuid.uuid4().hex}"
            started = time.time()
            try:
                result = await asyncio.wait_for(
                    function.entrypoint(**arguments),
                    timeout=8,
                )
                business_status, result_summary, tool_evidence = (
                    _evaluate_read_tool_result(
                        result,
                        plan.result_contract,
                        invocation_id=invocation_id,
                        server_name=server_name,
                        tool_name=tool_name,
                    )
                )
                invocation = ToolInvocation(
                    invocation_id=invocation_id,
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
                    tool_evidence=tool_evidence,
                )
                invocations.append(invocation)
                context.append(_tool_evidence_context(invocation))
            except asyncio.TimeoutError:
                invocations.append(
                    ToolInvocation(
                        invocation_id=invocation_id,
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
                        invocation_id=invocation_id,
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
