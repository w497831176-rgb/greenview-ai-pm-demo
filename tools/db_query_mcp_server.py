"""
DB Query MCP Server
===================

Built-in MCP server providing read-only work order queries via stdio transport.
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("db-query-server")

DB_PATH = Path(os.getenv("PROPERTY_DATA_DIR", "/app/data")) / "property_demo.db"


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool()
def count_work_orders(status: Optional[str] = None) -> str:
    """统计工单数量，可按状态筛选。"""
    conn = _get_conn()
    cursor = conn.cursor()
    query = "SELECT COUNT(*) AS cnt FROM work_orders WHERE 1=1"
    params = []
    if status:
        query += " AND status = ?"
        params.append(status)
    cursor.execute(query, params)
    row = cursor.fetchone()
    conn.close()
    return f"工单数量：{row['cnt']}"


@mcp.tool()
def list_recent_work_orders(room_id: Optional[str] = None, limit: int = 5) -> str:
    """查询最近工单，可按房号筛选。"""
    conn = _get_conn()
    cursor = conn.cursor()
    query = "SELECT * FROM work_orders WHERE 1=1"
    params = []
    if room_id:
        query += " AND room_id = ?"
        params.append(room_id)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return "未找到工单。"
    results: List[Dict[str, Any]] = [dict(r) for r in rows]
    return json.dumps(results, ensure_ascii=False, indent=2)


@mcp.tool()
def get_work_order_by_id(work_order_id: str) -> str:
    """根据工单号查询工单详情。"""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM work_orders WHERE id = ?", (work_order_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return f"未找到工单 {work_order_id}。"
    return json.dumps(dict(row), ensure_ascii=False, indent=2)


@mcp.tool()
def list_urgent_work_orders(limit: int = 5) -> str:
    """查询最近的紧急/高优先级工单。"""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM work_orders WHERE urgency IN ('紧急', '高') ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return "暂无紧急或高优先级工单。"
    return json.dumps([dict(r) for r in rows], ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
