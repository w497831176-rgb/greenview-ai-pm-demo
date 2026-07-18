"""Weather MCP Server for the YIAI interview demo.

The data is deliberately deterministic.  It demonstrates MCP result contracts
without claiming a live weather-provider integration.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("weather-server")

CITY_WEATHER: Dict[str, Dict[str, Any]] = {
    "北京": {"temperature_c": 32, "condition": "晴", "humidity_pct": 45, "wind": "东南风 2 级"},
    "上海": {"temperature_c": 29, "condition": "多云", "humidity_pct": 70, "wind": "东风 3 级"},
    "广州": {"temperature_c": 33, "condition": "雷阵雨", "humidity_pct": 80, "wind": "南风 3 级"},
    "深圳": {"temperature_c": 32, "condition": "阵雨", "humidity_pct": 78, "wind": "东南风 2 级"},
    "杭州": {"temperature_c": 30, "condition": "阴", "humidity_pct": 65, "wind": "东北风 2 级"},
    "成都": {"temperature_c": 27, "condition": "多云", "humidity_pct": 72, "wind": "北风 1 级"},
    "武汉": {"temperature_c": 31, "condition": "晴", "humidity_pct": 55, "wind": "南风 2 级"},
    "西安": {"temperature_c": 30, "condition": "晴", "humidity_pct": 40, "wind": "东风 2 级"},
}


def _result(status: str, data: Any = None, message: str = "") -> str:
    return json.dumps(
        {
            "status": status,
            "data": data,
            "message": message,
            "meta": {"source": "demo_fixture", "live_weather": False},
        },
        ensure_ascii=False,
    )


def _weather(city: str) -> tuple[str, Dict[str, Any] | None]:
    normalized = (city or "").strip()
    if not normalized:
        return "invalid_input", None
    return ("success", CITY_WEATHER.get(normalized)) if normalized in CITY_WEATHER else ("invalid_input", None)


@mcp.tool()
def get_current_weather(city: str) -> str:
    """查询演示样例城市的天气；非实时互联网天气。"""
    status, data = _weather(city)
    if status != "success":
        return _result("invalid_input", None, "暂不支持该城市；请使用演示样例覆盖的城市。")
    return _result("success", {"city": city.strip(), **(data or {})}, "演示固定天气样例，非实时互联网天气。")


@mcp.tool()
def get_weather_advice(city: str) -> str:
    """基于演示天气样例，给出户外维修作业建议。"""
    status, data = _weather(city)
    if status != "success":
        return _result("invalid_input", None, "暂不支持该城市；无法生成天气建议。")
    assert data is not None
    if data["condition"] in ("雷阵雨", "阵雨", "大雨"):
        advice = "有降雨风险，优先做好防水防滑；不建议安排非必要露天作业。"
    elif data["temperature_c"] >= 35:
        advice = "高温天气，户外维修需注意防暑并合理安排时段。"
    else:
        advice = "天气条件适合正常户外维修作业；仍需结合现场安全情况判断。"
    return _result("success", {"city": city.strip(), "advice": advice}, "演示固定天气样例推导的建议。")


if __name__ == "__main__":
    mcp.run(transport="stdio")
