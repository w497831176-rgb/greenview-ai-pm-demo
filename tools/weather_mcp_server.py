"""
Weather MCP Server
==================

Built-in MCP server providing weather query tools via stdio transport.
Uses a deterministic mock implementation so it works offline on NAS.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-server")


CITY_WEATHER = {
    "北京": {"temperature": 32, "condition": "晴", "humidity": 45, "wind": "东南风 2 级"},
    "上海": {"temperature": 29, "condition": "多云", "humidity": 70, "wind": "东风 3 级"},
    "广州": {"temperature": 33, "condition": "雷阵雨", "humidity": 80, "wind": "南风 3 级"},
    "深圳": {"temperature": 32, "condition": "阵雨", "humidity": 78, "wind": "东南风 2 级"},
    "杭州": {"temperature": 30, "condition": "阴", "humidity": 65, "wind": "东北风 2 级"},
    "成都": {"temperature": 27, "condition": "多云", "humidity": 72, "wind": "北风 1 级"},
    "武汉": {"temperature": 31, "condition": "晴", "humidity": 55, "wind": "南风 2 级"},
    "西安": {"temperature": 30, "condition": "晴", "humidity": 40, "wind": "东风 2 级"},
}


@mcp.tool()
def get_current_weather(city: str) -> str:
    """查询指定城市当前天气，返回温度、天气状况、湿度和风力。"""
    city = city.strip()
    data = CITY_WEATHER.get(city)
    if not data:
        return f"暂不支持查询 {city} 的天气，请提供北京、上海、广州、深圳、杭州、成都、武汉、西安等城市。"
    return (
        f"{city}当前天气：{data['condition']}，"
        f"气温 {data['temperature']}℃，"
        f"湿度 {data['humidity']}%，"
        f"风力 {data['wind']}。"
    )


@mcp.tool()
def get_weather_advice(city: str) -> str:
    """根据天气给出简单的户外活动建议。"""
    city = city.strip()
    data = CITY_WEATHER.get(city)
    if not data:
        return f"暂不支持查询 {city} 的天气。"
    if data["condition"] in ("雷阵雨", "阵雨", "大雨"):
        return f"{city}今天有雨，外出请带伞，不建议安排露天维修作业。"
    if data["temperature"] >= 35:
        return f"{city}今天高温，户外作业请注意防暑降温。"
    return f"{city}今天天气{data['condition']}，适合正常户外维修作业。"


if __name__ == "__main__":
    mcp.run(transport="stdio")
