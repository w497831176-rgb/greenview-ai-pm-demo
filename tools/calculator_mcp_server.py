"""
Calculator MCP Server
=====================

Built-in MCP server providing arithmetic tools via stdio transport.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator-server")


@mcp.tool()
def add(a: float, b: float) -> float:
    """计算两个数的和。"""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """计算两个数的差。"""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """计算两个数的乘积。"""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> str:
    """计算两个数的商。"""
    if b == 0:
        return "错误：除数不能为零。"
    return str(a / b)


@mcp.tool()
def calculate_fee(unit_price: float, quantity: float) -> str:
    """根据单价和数量计算总费用。"""
    total = unit_price * quantity
    return f"总费用为 {total:.2f} 元。"


if __name__ == "__main__":
    mcp.run(transport="stdio")
