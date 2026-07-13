"""
Calendar MCP Server
===================

Built-in MCP server providing date/calendar tools via stdio transport.
"""

from datetime import datetime, timedelta, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calendar-server")


def _now_cn() -> datetime:
    return datetime.now(timezone(timedelta(hours=8)))


@mcp.tool()
def get_current_date() -> str:
    """获取今天的日期（北京时间）。"""
    return _now_cn().strftime("%Y-%m-%d")


@mcp.tool()
def get_current_datetime() -> str:
    """获取当前的日期和时间（北京时间）。"""
    return _now_cn().strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
def get_weekday(date: str) -> str:
    """根据日期字符串（YYYY-MM-DD）返回星期几。"""
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return weekdays[dt.weekday()]
    except ValueError:
        return "日期格式错误，请使用 YYYY-MM-DD 格式。"


@mcp.tool()
def add_days(date: str, days: int) -> str:
    """在指定日期上增加若干天，返回结果日期（YYYY-MM-DD）。"""
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        result = dt + timedelta(days=days)
        return result.strftime("%Y-%m-%d")
    except ValueError:
        return "日期格式错误，请使用 YYYY-MM-DD 格式。"


if __name__ == "__main__":
    mcp.run(transport="stdio")
