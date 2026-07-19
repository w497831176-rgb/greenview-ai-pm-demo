"""Configuration-driven MCP ToolPlan compiler.

There are deliberately no business-domain branches in the planner.  Every
runtime decision comes from the immutable RuntimeRelease:

* Agent -> MCP binding controls which servers are candidates.
* ToolPolicy controls effect, path, confirmation and enablement.
* ``tool_metadata`` controls natural-language activation and argument mapping.

The small built-in compatibility map below is compile-time metadata for the
three V1.7 demo servers.  Operator-declared metadata always wins, and newly
added servers use exactly the same generic planner without code changes.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from jsonschema import Draft202012Validator

from app.runtime.contracts import RuntimePath, ToolEffect, ToolPlan
from app.runtime.tool_gateway import ToolGateway


VALID_EXECUTION_MODES = {"model_native", "auto_preinvoke", "proposal"}
VALID_TRIGGER_MODES = {"any", "all"}
VALID_ARGUMENT_SOURCES = {
    "constant",
    "message",
    "regex",
    "enum",
    "keyword_map",
}
DEFAULT_RESULT_CONTRACT = {
    "success_statuses": ["success"],
    "non_success_statuses": [
        "empty",
        "not_found",
        "invalid_input",
        "unauthorized",
        "timeout",
        "upstream_error",
        "unknown",
    ],
    "claim_rule": "Only an explicit success business status may be described as success.",
}


# Compatibility only: this replaces the former server-name ``if/elif`` runtime
# planner.  It is merged into the published snapshot, never written back over
# platform configuration.
BUILTIN_COMPATIBILITY_METADATA: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("weather-server", "get_current_weather"): {
        "effect": "read",
        "natural_language_intents": ["查询城市天气", "判断天气对上门服务的影响"],
        "trigger_keywords": ["天气", "气温", "下雨", "降雨", "湿度", "本小区"],
        "trigger_mode": "any",
        "execution_mode": "auto_preinvoke",
        "argument_bindings": {
            "city": {
                "source": "keyword_map",
                "mapping": {
                    "北京": "北京",
                    "上海": "上海",
                    "广州": "广州",
                    "深圳": "深圳",
                    "杭州": "杭州",
                    "成都": "成都",
                    "武汉": "武汉",
                    "西安": "西安",
                    "本小区": "杭州",
                },
            }
        },
    },
    ("weather-server", "get_weather_advice"): {
        "effect": "read",
        "natural_language_intents": ["查询天气风险建议", "判断是否适合户外或上门作业"],
        "trigger_keywords": ["建议", "风险", "户外", "巡检", "暴雨", "下雨", "降雨"],
        "trigger_mode": "any",
        "execution_mode": "auto_preinvoke",
        "argument_bindings": {
            "city": {
                "source": "keyword_map",
                "mapping": {
                    "北京": "北京",
                    "上海": "上海",
                    "广州": "广州",
                    "深圳": "深圳",
                    "杭州": "杭州",
                    "成都": "成都",
                    "武汉": "武汉",
                    "西安": "西安",
                    "本小区": "杭州",
                },
            }
        },
    },
    ("workorder-server", "get_my_recent_work_orders"): {
        "effect": "read",
        "natural_language_intents": ["查询我的最近工单或维修记录"],
        "trigger_keywords": ["最近工单", "我的工单", "我家工单", "维修记录", "工单进度"],
        "trigger_mode": "any",
        "execution_mode": "auto_preinvoke",
        "argument_bindings": {"limit": {"source": "constant", "value": 5}},
    },
    ("workorder-server", "count_my_open_work_orders"): {
        "effect": "read",
        "natural_language_intents": ["统计我的未关闭工单"],
        "trigger_keywords": ["待处理", "待派单", "未处理", "还有多少"],
        "trigger_mode": "any",
        "execution_mode": "auto_preinvoke",
        "argument_bindings": {},
    },
    ("workorder-server", "get_my_work_order_by_id"): {
        "effect": "read",
        "natural_language_intents": ["按工单号查询我的工单"],
        "trigger_keywords": ["查询工单", "工单号"],
        "trigger_mode": "any",
        "execution_mode": "model_native",
        "argument_bindings": {},
    },
    ("workorder-server", "count_work_orders"): {
        "effect": "read",
        "natural_language_intents": ["查询脱敏工单数量"],
        "trigger_keywords": ["工单数量", "多少工单"],
        "trigger_mode": "any",
        "execution_mode": "model_native",
        "argument_bindings": {},
    },
    ("calendar-server", "get_current_date"): {
        "effect": "read",
        "natural_language_intents": ["查询当前日期"],
        "trigger_keywords": ["今天", "日期"],
        "trigger_mode": "any",
        "execution_mode": "model_native",
        "argument_bindings": {},
    },
    ("calendar-server", "get_current_datetime"): {
        "effect": "read",
        "natural_language_intents": ["查询当前日期时间"],
        "trigger_keywords": ["今天", "明天", "日期", "星期", "几点", "预约"],
        "trigger_mode": "any",
        "execution_mode": "auto_preinvoke",
        "argument_bindings": {},
    },
    ("calendar-server", "get_weekday"): {
        "effect": "read",
        "natural_language_intents": ["查询指定日期是星期几"],
        "trigger_keywords": ["星期", "周几"],
        "trigger_mode": "any",
        "execution_mode": "model_native",
        "argument_bindings": {},
    },
    ("calendar-server", "add_days"): {
        "effect": "read",
        "natural_language_intents": ["进行日期加减计算"],
        "trigger_keywords": ["几天后", "几天前", "日期计算"],
        "trigger_mode": "any",
        "execution_mode": "model_native",
        "argument_bindings": {},
    },
}


def effective_tool_metadata(
    server_name: str,
    tool_name: str,
    declared: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return effective metadata without mutating the editable DB draft."""

    declared_metadata = dict(declared or {})
    compatibility = BUILTIN_COMPATIBILITY_METADATA.get(
        (str(server_name), str(tool_name)),
        {},
    )
    metadata = {**compatibility, **declared_metadata}
    if declared_metadata.get("effect"):
        metadata.setdefault(
            "effect_source",
            declared_metadata.get("effect_source") or "operator_declared_legacy",
        )
    elif compatibility.get("effect"):
        metadata.setdefault("effect_source", "builtin_compatibility")
    if compatibility:
        metadata.setdefault("runtime_metadata_source", "builtin_compatibility")
        metadata.setdefault("risk_level", "L1")
        metadata.setdefault("result_contract", dict(DEFAULT_RESULT_CONTRACT))
    elif declared_metadata:
        metadata.setdefault("runtime_metadata_source", "operator_declared")
    else:
        metadata.setdefault("runtime_metadata_source", "discovery_default")
    effect = str(metadata.get("effect") or "").lower()
    metadata.setdefault(
        "execution_mode",
        "proposal" if effect in {"create", "update"} else "model_native",
    )
    metadata.setdefault("natural_language_intents", [])
    metadata.setdefault("trigger_keywords", [])
    metadata.setdefault("trigger_mode", "any")
    metadata.setdefault("argument_bindings", {})
    return metadata


def validate_tool_metadata(
    metadata: Dict[str, Any],
    input_schema: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Validate the operator-editable planner contract."""

    errors: List[str] = []
    execution_mode = str(metadata.get("execution_mode") or "model_native")
    if execution_mode not in VALID_EXECUTION_MODES:
        errors.append(
            "execution_mode must be model_native, auto_preinvoke or proposal"
        )
    trigger_mode = str(metadata.get("trigger_mode") or "any")
    if trigger_mode not in VALID_TRIGGER_MODES:
        errors.append("trigger_mode must be any or all")
    risk_level = str(metadata.get("risk_level") or "")
    if metadata.get("effect") and risk_level not in {"L0", "L1", "L2", "L3"}:
        errors.append("risk_level must be L0, L1, L2 or L3")
    result_contract = metadata.get("result_contract")
    if metadata.get("effect") and not isinstance(result_contract, dict):
        errors.append("result_contract must be an object")
    elif metadata.get("effect"):
        success_statuses = result_contract.get("success_statuses") or []
        if not isinstance(success_statuses, list) or not success_statuses:
            errors.append("result_contract.success_statuses must be a non-empty array")
        elif any(
            str(item).lower()
            in {
                "unknown",
                "empty",
                "not_found",
                "invalid_input",
                "unauthorized",
                "timeout",
                "upstream_error",
                "failed",
                "error",
            }
            for item in success_statuses
        ):
            errors.append(
                "result_contract.success_statuses contains a non-success status"
            )
    for key in ("natural_language_intents", "trigger_keywords"):
        value = metadata.get(key, [])
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            errors.append(f"{key} must be a string array")
    bindings = metadata.get("argument_bindings") or {}
    if not isinstance(bindings, dict):
        errors.append("argument_bindings must be an object")
        return errors
    properties = set(((input_schema or {}).get("properties") or {}).keys())
    for argument, rule in bindings.items():
        if properties and argument not in properties:
            errors.append(f"argument binding is absent from input schema: {argument}")
        if not isinstance(rule, dict):
            errors.append(f"argument binding must be an object: {argument}")
            continue
        source = str(rule.get("source") or "")
        if source not in VALID_ARGUMENT_SOURCES:
            errors.append(f"unsupported argument source for {argument}: {source}")
            continue
        if source == "regex":
            pattern = str(rule.get("pattern") or "")
            try:
                re.compile(pattern)
            except re.error as exc:
                errors.append(f"invalid regex for {argument}: {exc}")
        if source == "keyword_map" and not isinstance(rule.get("mapping"), dict):
            errors.append(f"keyword_map requires mapping object: {argument}")
        if source == "enum" and not isinstance(rule.get("values"), list):
            errors.append(f"enum requires values array: {argument}")
    return errors


def _message_json(message: str) -> Dict[str, Any]:
    start = (message or "").find("{")
    if start < 0:
        return {}
    try:
        value, _ = json.JSONDecoder().raw_decode(message[start:])
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _match_tool(
    message: str,
    tool_name: str,
    description: str,
    metadata: Dict[str, Any],
) -> Optional[Tuple[float, str]]:
    normalized = (message or "").casefold()
    keywords = [
        str(item).strip()
        for item in metadata.get("trigger_keywords") or []
        if str(item).strip()
    ]
    matched_keywords = [item for item in keywords if item.casefold() in normalized]
    trigger_mode = str(metadata.get("trigger_mode") or "any")
    intents = [
        str(item).strip()
        for item in metadata.get("natural_language_intents") or []
        if str(item).strip()
    ]
    matched_intents = [item for item in intents if item.casefold() in normalized]
    matched_signals: List[str] = []
    if not keywords:
        generic = {
            "创建",
            "新增",
            "添加",
            "查询",
            "获取",
            "记录",
            "信息",
            "一个",
            "一条",
            "数据",
            "工具",
            "请求",
        }
        signals: set[str] = set()
        for source in [description, *intents]:
            for phrase in re.findall(
                r"[\u4e00-\u9fff]{2,24}|[a-zA-Z][a-zA-Z0-9_-]{3,}",
                str(source or ""),
            ):
                lowered = phrase.casefold()
                if re.fullmatch(r"[\u4e00-\u9fff]+", lowered):
                    for size in range(2, min(8, len(lowered)) + 1):
                        for start in range(len(lowered) - size + 1):
                            term = lowered[start : start + size]
                            if term not in generic:
                                signals.add(term)
                elif lowered not in generic:
                    signals.add(lowered)
        matched_signals = sorted(
            (item for item in signals if item in normalized),
            key=lambda item: (-len(item), item),
        )
    if keywords:
        if trigger_mode == "all" and len(matched_keywords) != len(keywords):
            return None
        if trigger_mode == "any" and not matched_keywords:
            return None
    else:
        normalized_tool_name = str(tool_name).casefold()
        explicit_tool_name = bool(
            normalized_tool_name and normalized_tool_name in normalized
        )
        # A high-specificity phrase from the operator-declared intent or tool
        # description is sufficient.  Generic verbs/nouns are excluded, and
        # equal-score candidates are rejected by ``unique_write_plan``.
        strong_signal = next(
            (
                item
                for item in matched_signals
                if len(item) >= 4
                or (
                    not re.fullmatch(r"[\u4e00-\u9fff]+", item)
                    and len(item) >= 4
                )
            ),
            None,
        )
        if not explicit_tool_name and not matched_intents and not strong_signal:
            return None

    score = float(len(matched_keywords) * 100 + len(matched_intents) * 30)
    if matched_signals:
        score += min(80, len(matched_signals[0]) * 10)
    if str(tool_name).casefold() in normalized:
        score += 25
    reason_parts = []
    if matched_keywords:
        reason_parts.append("触发词=" + "、".join(matched_keywords))
    if matched_intents:
        reason_parts.append("意图=" + "、".join(matched_intents))
    if matched_signals:
        reason_parts.append("描述信号=" + "、".join(matched_signals[:3]))
    if not reason_parts:
        reason_parts.append("用户显式指定工具名")
    return score, "；".join(reason_parts)


def _resolve_argument(
    message: str,
    rule: Dict[str, Any],
) -> Tuple[bool, Any]:
    source = str(rule.get("source") or "")
    if source == "constant":
        return "value" in rule, rule.get("value")
    if source == "message":
        return True, message
    if source == "regex":
        match = re.search(str(rule.get("pattern") or ""), message or "")
        if not match:
            return False, None
        group = rule.get("group", 1 if match.lastindex else 0)
        try:
            return True, match.group(group)
        except (IndexError, KeyError):
            return False, None
    if source == "enum":
        for value in rule.get("values") or []:
            if str(value).casefold() in (message or "").casefold():
                return True, value
        return False, None
    if source == "keyword_map":
        for keyword, value in (rule.get("mapping") or {}).items():
            if str(keyword).casefold() in (message or "").casefold():
                return True, value
        return False, None
    return False, None


def _arguments_for_tool(
    message: str,
    input_schema: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    properties = input_schema.get("properties") or {}
    arguments: Dict[str, Any] = {}
    # Explicit JSON has highest priority and is still schema/policy validated by
    # the backend gateway before any write is executed.
    arguments.update(_message_json(message))
    for name, rule in (metadata.get("argument_bindings") or {}).items():
        if name in arguments or not isinstance(rule, dict):
            continue
        found, value = _resolve_argument(message, rule)
        if found:
            arguments[name] = value
    for name, schema in properties.items():
        if name not in arguments and isinstance(schema, dict) and "default" in schema:
            arguments[name] = schema["default"]
    required = list(input_schema.get("required") or [])
    missing = [name for name in required if name not in arguments]
    return arguments, missing


def validate_arguments(
    arguments: Dict[str, Any],
    input_schema: Optional[Dict[str, Any]],
) -> List[str]:
    """Return stable, user-safe JSON Schema errors for a ToolPlan."""

    schema = input_schema or {}
    if not schema:
        return []
    errors = sorted(
        Draft202012Validator(schema).iter_errors(arguments),
        key=lambda item: list(item.absolute_path),
    )
    return [
        (
            (".".join(str(part) for part in error.absolute_path) + ": ")
            if error.absolute_path
            else ""
        )
        + error.message
        for error in errors
    ]


def plan_tools(
    snapshot_config: Dict[str, Any],
    agent_id: str,
    message: str,
    runtime_path: RuntimePath,
    effects: Optional[Sequence[ToolEffect]] = None,
    execution_modes: Optional[Iterable[str]] = None,
) -> List[ToolPlan]:
    """Compile matching, policy-admitted ToolPlans for one published Agent."""

    gateway = ToolGateway(snapshot_config)
    allowed_effects = set(effects or list(ToolEffect))
    allowed_modes = set(execution_modes or VALID_EXECUTION_MODES)
    agent = next(
        (
            item
            for item in snapshot_config.get("agents") or []
            if item.get("agent_id") == agent_id and item.get("enabled")
        ),
        None,
    )
    if not agent:
        return []
    bound_servers = set(agent.get("mcp_server_names") or [])
    plans: List[ToolPlan] = []
    for server in snapshot_config.get("mcp_servers") or []:
        server_name = str(server.get("name") or "")
        if not server.get("enabled") or server_name not in bound_servers:
            continue
        for tool in server.get("tools") or []:
            tool_name = str(tool.get("name") or "")
            policy_data = tool.get("policy") or {}
            try:
                effect = ToolEffect(str(policy_data.get("effect") or "unknown"))
            except ValueError:
                effect = ToolEffect.UNKNOWN
            if effect not in allowed_effects:
                continue
            try:
                if effect == ToolEffect.READ:
                    gateway.assert_read_invocation(
                        agent_id,
                        runtime_path,
                        server_name,
                        tool_name,
                    )
                elif effect in {ToolEffect.CREATE, ToolEffect.UPDATE}:
                    gateway.write_policy(
                        server_name,
                        tool_name,
                        agent_id=agent_id,
                    )
                else:
                    continue
            except Exception:
                continue
            metadata = effective_tool_metadata(
                server_name,
                tool_name,
                tool.get("tool_metadata") or {},
            )
            execution_mode = str(metadata.get("execution_mode") or "")
            if execution_mode not in allowed_modes:
                continue
            matched = _match_tool(
                message,
                tool_name,
                str(tool.get("description") or ""),
                metadata,
            )
            if not matched:
                continue
            arguments, missing = _arguments_for_tool(
                message,
                tool.get("input_schema") or {},
                metadata,
            )
            schema_errors = validate_arguments(
                arguments,
                tool.get("input_schema") or {},
            )
            plans.append(
                ToolPlan(
                    agent_id=agent_id,
                    server_name=server_name,
                    tool_name=tool_name,
                    effect=effect,
                    execution_mode=execution_mode,
                    arguments=arguments,
                    missing_required=missing,
                    schema_errors=schema_errors,
                    match_score=matched[0],
                    match_reason=matched[1],
                    result_contract=metadata.get("result_contract") or {},
                )
            )
    return sorted(
        plans,
        key=lambda item: (-item.match_score, item.server_name, item.tool_name),
    )


def unique_write_plan(
    snapshot_config: Dict[str, Any],
    message: str,
) -> Optional[ToolPlan]:
    """Return one unambiguous governed write plan across all vertical Agents."""

    candidates: List[ToolPlan] = []
    for agent in snapshot_config.get("agents") or []:
        if (
            not agent.get("enabled")
            or agent.get("category") in {"router", "orchestration"}
        ):
            continue
        candidates.extend(
            plan_tools(
                snapshot_config,
                str(agent.get("agent_id") or ""),
                message,
                RuntimePath.CONTROLLED_ACTION,
                effects=[ToolEffect.CREATE, ToolEffect.UPDATE],
                execution_modes=["proposal"],
            )
        )
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (-item.match_score, item.agent_id, item.server_name, item.tool_name)
    )
    if len(candidates) > 1 and candidates[0].match_score == candidates[1].match_score:
        return None
    return candidates[0]
