"""Runtime policy for the three formal MCP servers.

The policy is intentionally code-owned rather than editable by a prompt.  An
Agent may decide whether a tool is useful, but it cannot expand the server or
tool permissions that the product grants to it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Set


TOOL_OUTCOME_LABELS = {
    "success": "成功",
    "empty": "无匹配数据",
    "not_found": "未找到",
    "invalid_input": "参数不合法",
    "unauthorized": "无权限",
    "timeout": "调用超时",
    "upstream_error": "上游异常",
}


# This is the Host-side least-privilege contract.  It is deliberately stricter
# than the list discovered from a server: discovering a tool does not grant an
# owner-facing Agent permission to call it.
AGENT_SERVER_TOOL_ALLOWLIST: Dict[str, Dict[str, Set[str]]] = {
    "maintenance": {
        "weather-server": {"get_current_weather", "get_weather_advice"},
        "workorder-server": {
            "get_my_recent_work_orders",
            "count_my_open_work_orders",
            "get_my_work_order_by_id",
            "count_work_orders",
        },
    },
    "customer_service": {
        "calendar-server": {
            "get_current_date",
            "get_current_datetime",
            "get_weekday",
            "add_days",
        },
    },
}


MCP_SERVER_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "weather-server": {
        "mode": "readonly",
        "mode_label": "只读查询",
        "source": "演示固定天气样例（非实时互联网天气）",
        "data_scope": "仅返回城市级天气，不包含用户数据。",
        "write_boundary": "不执行报修、派单或任何写操作。",
        "agents": ["maintenance"],
        "tools": {
            "get_current_weather": {
                "purpose": "查询城市天气事实，用于判断天气对维修风险和上门作业的影响。",
                "input_rule": "城市必须在演示样例覆盖范围内。",
                "result_rule": "success 才能作为事实；invalid_input 必须说明不支持的城市。",
            },
            "get_weather_advice": {
                "purpose": "根据已查询的演示天气给出户外作业建议。",
                "input_rule": "只在用户明确要求风险、建议或户外作业判断时调用。",
                "result_rule": "仅是天气建议，不替代维修安全判断。",
            },
        },
    },
    "workorder-server": {
        "mode": "readonly",
        "mode_label": "只读查询",
        "source": "物业演示 SQLite 数据库",
        "data_scope": "业主端仅可读取默认演示房号 3-2-1201 的明细；全小区仅返回脱敏聚合数量。",
        "write_boundary": "不创建、不修改、不派发工单。正式写入由独立会话工作流在用户确认后完成。",
        "agents": ["maintenance"],
        "tools": {
            "get_my_recent_work_orders": {
                "purpose": "读取当前演示业主的最近工单。",
                "input_rule": "没有 room_id 参数，服务端固定按当前演示业主范围查询。",
                "result_rule": "empty/not_found 只能表示没有匹配记录，不能声称工单已处理。",
            },
            "count_my_open_work_orders": {
                "purpose": "统计当前演示业主未关闭工单数。",
                "input_rule": "无输入；服务端固定业主范围。",
                "result_rule": "返回聚合数量，不暴露其他业主信息。",
            },
            "get_my_work_order_by_id": {
                "purpose": "查询当前演示业主的一张工单。",
                "input_rule": "工单号必须属于当前演示业主。",
                "result_rule": "not_found/unauthorized 不得推断其他业主工单情况。",
            },
            "count_work_orders": {
                "purpose": "返回全小区工单的脱敏聚合数量。",
                "input_rule": "只接受可识别的状态筛选；不返回明细。",
                "result_rule": "回答中必须说明这是系统聚合数，不是个人工单明细。",
            },
        },
    },
    "calendar-server": {
        "mode": "readonly",
        "mode_label": "只读计算",
        "source": "服务端北京时间",
        "data_scope": "不读取用户或业务数据。",
        "write_boundary": "不创建预约、不修改日程。",
        "agents": ["customer_service"],
        "tools": {
            "get_current_date": {"purpose": "读取当前北京时间日期。", "input_rule": "无输入。", "result_rule": "success 才可用于日期说明。"},
            "get_current_datetime": {"purpose": "读取当前北京时间日期和时间。", "input_rule": "无输入。", "result_rule": "success 才可用于时间说明。"},
            "get_weekday": {"purpose": "计算指定日期的星期。", "input_rule": "日期格式 YYYY-MM-DD。", "result_rule": "invalid_input 时提示正确格式。"},
            "add_days": {"purpose": "进行日期加减计算。", "input_rule": "日期格式 YYYY-MM-DD，days 为整数。", "result_rule": "只返回计算结果，不创建预约。"},
        },
    },
}


WORK_ORDER_WRITE_BOUNDARY = {
    "name": "work_order_workflow",
    "mode": "application_workflow",
    "mode_label": "应用层受控写入（非 MCP）",
    "summary": "先收集为待确认草稿；只有用户明确说“确认创建”后，服务端才写入正式工单并返回真实工单号。",
    "guardrails": [
        "模型语言不能代替真实写入结果。",
        "草稿阶段不产生正式工单。",
        "缺少必要信息时只追问，不写入。",
        "写入成功必须以服务端真实工单号为凭据。",
    ],
}


def allowed_tools_for_agent(agent_id: Optional[str], server_name: str) -> Set[str]:
    """Return the Host-side allowlist for one Agent/server pair."""
    return set(AGENT_SERVER_TOOL_ALLOWLIST.get(agent_id or "", {}).get(server_name, set()))


def allowed_server_names(agent_id: Optional[str]) -> Set[str]:
    return set(AGENT_SERVER_TOOL_ALLOWLIST.get(agent_id or "", {}).keys())


def tool_contract(server_name: str, tool_name: str) -> Dict[str, Any]:
    return dict(MCP_SERVER_CONTRACTS.get(server_name, {}).get("tools", {}).get(tool_name, {}))


def tool_outcome_label(outcome: Optional[str]) -> str:
    return TOOL_OUTCOME_LABELS.get(str(outcome or ""), str(outcome or "未知"))


def extract_tool_outcome(result: Any) -> str:
    """Normalize the explicit MCP result envelope or legacy failure text.

    New formal tools return JSON with a `status` field.  The legacy fallback
    keeps Trace honest when a server exits, times out, or returns an old-style
    error string.
    """
    text = str(result or "").strip()
    if not text:
        return "empty"

    candidates: List[str] = [text]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("status") in TOOL_OUTCOME_LABELS:
            return str(payload["status"])

    lowered = text.lower()
    if "timeout" in lowered or "超时" in text:
        return "timeout"
    if "error from mcp tool" in lowered or "upstream" in lowered or "异常" in text:
        return "upstream_error"
    if "未找到" in text:
        return "not_found"
    if "参数" in text and ("错误" in text or "不合法" in text):
        return "invalid_input"
    if "无权限" in text or "不允许" in text:
        return "unauthorized"
    return "success"


def outcome_instruction(outcome: str) -> str:
    """One sentence injected into Agent context after a pre-invocation."""
    rules = {
        "success": "可将返回数据作为本轮事实使用，并注明其演示数据来源。",
        "empty": "只能说明没有匹配数据，不能推断业务已完成或不存在风险。",
        "not_found": "只能说明未找到该范围内记录，不能推断其他范围的数据。",
        "invalid_input": "必须说明参数或能力范围不满足，请用户补充或改正输入。",
        "unauthorized": "必须说明当前业主端无权查询该信息，不能绕过范围。",
        "timeout": "必须说明工具超时、结果未确认；不能把超时说成无结果或成功。",
        "upstream_error": "必须说明工具异常、结果未确认；必要时建议人工处理。",
    }
    return rules.get(outcome, "结果状态未知，不能据此编造事实。")


def contract_tools(server_name: str) -> Iterable[str]:
    return MCP_SERVER_CONTRACTS.get(server_name, {}).get("tools", {}).keys()
