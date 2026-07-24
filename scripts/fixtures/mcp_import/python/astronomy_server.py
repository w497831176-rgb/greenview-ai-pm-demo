"""Tiny stdio MCP fixture used by the isolated Git-import contract check."""

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("fixture-astronomy-mcp")


@mcp.tool()
def lookup_transit_window(target: str = "GVX-42") -> dict:
    """查询虚构目标星体的观测窗口。"""

    return {
        "status": "success",
        "target": target,
        "window": "21:10-21:57",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
