"""Calendar MCP Server with explicit machine-readable outcome states."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("calendar-server")


def _result(status: str, data: Any = None, message: str = "") -> str:
    return json.dumps(
        {"status": status, "data": data, "message": message, "meta": {"timezone": "Asia/Shanghai"}},
        ensure_ascii=False,
    )


def _now_cn() -> datetime:
    return datetime.now(timezone(timedelta(hours=8)))


def _parse_date(value: str) -> datetime | None:
    try:
        return datetime.strptime((value or "").strip(), "%Y-%m-%d")
    except ValueError:
        return None


@mcp.tool()
def get_current_date() -> str:
    """获取当前北京时间日期。"""
    return _result("success", {"date": _now_cn().strftime("%Y-%m-%d")})


@mcp.tool()
def get_current_datetime() -> str:
    """获取当前北京时间日期和时间。"""
    return _result("success", {"datetime": _now_cn().strftime("%Y-%m-%d %H:%M:%S")})


@mcp.tool()
def get_weekday(date: str) -> str:
    """根据 YYYY-MM-DD 计算星期几。"""
    parsed = _parse_date(date)
    if parsed is None:
        return _result("invalid_input", None, "日期格式应为 YYYY-MM-DD。")
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return _result("success", {"date": parsed.strftime("%Y-%m-%d"), "weekday": weekdays[parsed.weekday()]})


@mcp.tool()
def add_days(date: str, days: int) -> str:
    """在指定日期上加减整数天数；不创建任何预约。"""
    parsed = _parse_date(date)
    if parsed is None or not isinstance(days, int):
        return _result("invalid_input", None, "date 必须为 YYYY-MM-DD，days 必须为整数。")
    result = parsed + timedelta(days=days)
    return _result(
        "success",
        {"base_date": parsed.strftime("%Y-%m-%d"), "days": days, "result_date": result.strftime("%Y-%m-%d")},
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
