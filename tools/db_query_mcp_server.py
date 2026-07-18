"""Read-only property work-order MCP server.

The owner-facing demo does not expose arbitrary room or community work-order
details.  The server enforces the scope; the model cannot widen it by changing
arguments.  Formal work-order creation remains in app.work_order_workflow,
outside MCP, and requires explicit owner confirmation.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("workorder-server")

DB_PATH = Path(os.getenv("PROPERTY_DATA_DIR", "/app/data")) / "property_demo.db"
DEMO_OWNER_ROOM_ID = os.getenv("DEMO_OWNER_ROOM_ID", "3-2-1201")
TERMINAL_STATUSES = ("已完成", "已关闭", "已取消")
ALLOWED_STATUS_FILTERS = {"待派单", "处理中", "待处理", "已完成", "已关闭", "已取消"}


def _result(status: str, data: Any = None, message: str = "", *, scope: str = "") -> str:
    return json.dumps(
        {
            "status": status,
            "data": data,
            "message": message,
            "meta": {"source": "property_demo.sqlite", "scope": scope, "read_only": True},
        },
        ensure_ascii=False,
    )


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _safe_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Return only fields needed by the owner conversation.

    The table has changed over iterations, so this projection is defensive and
    does not assume every optional field exists.
    """
    raw = dict(row)
    allowed = (
        "id",
        "room_id",
        "category",
        "issue_type",
        "description",
        "urgency",
        "status",
        "created_at",
        "updated_at",
        "assigned_to",
        "completion_note",
    )
    return {key: raw.get(key) for key in allowed if key in raw}


def _rows_to_data(rows: Iterable[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [_safe_row(row) for row in rows]


@mcp.tool()
def get_my_recent_work_orders(limit: int = 5) -> str:
    """查询当前演示业主最近工单；服务端固定为 3-2-1201 范围。"""
    if not isinstance(limit, int) or limit < 1 or limit > 10:
        return _result("invalid_input", None, "limit 必须是 1 到 10 的整数。", scope="owner")
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM work_orders WHERE room_id = ? ORDER BY created_at DESC LIMIT ?",
            (DEMO_OWNER_ROOM_ID, limit),
        ).fetchall()
        conn.close()
    except Exception as exc:
        return _result("upstream_error", None, f"读取工单数据失败：{type(exc).__name__}", scope="owner")
    if not rows:
        return _result("empty", [], "当前演示业主没有匹配工单。", scope="owner")
    return _result("success", _rows_to_data(rows), "返回当前演示业主最近工单。", scope="owner")


@mcp.tool()
def count_my_open_work_orders() -> str:
    """统计当前演示业主的未关闭工单数量。"""
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM work_orders WHERE room_id = ? AND COALESCE(status, '') NOT IN (?, ?, ?)",
            (DEMO_OWNER_ROOM_ID, *TERMINAL_STATUSES),
        ).fetchone()
        conn.close()
    except Exception as exc:
        return _result("upstream_error", None, f"统计工单失败：{type(exc).__name__}", scope="owner")
    return _result("success", {"open_count": int(row["cnt"] if row else 0)}, "当前演示业主未关闭工单聚合数。", scope="owner")


@mcp.tool()
def get_my_work_order_by_id(work_order_id: str) -> str:
    """查询当前演示业主的一张工单；不能跨房号读取。"""
    normalized = (work_order_id or "").strip()
    if not normalized:
        return _result("invalid_input", None, "work_order_id 不能为空。", scope="owner")
    try:
        conn = _get_conn()
        row = conn.execute("SELECT * FROM work_orders WHERE id = ?", (normalized,)).fetchone()
        conn.close()
    except Exception as exc:
        return _result("upstream_error", None, f"读取工单失败：{type(exc).__name__}", scope="owner")
    if row is None:
        return _result("not_found", None, "未找到该工单。", scope="owner")
    if str(dict(row).get("room_id") or "") != DEMO_OWNER_ROOM_ID:
        return _result("unauthorized", None, "当前业主端不能查询其他房号工单。", scope="owner")
    return _result("success", _safe_row(row), "返回当前演示业主工单。", scope="owner")


@mcp.tool()
def count_work_orders(status: Optional[str] = None) -> str:
    """返回全小区的脱敏工单聚合数量，不返回工单明细。"""
    normalized = (status or "").strip()
    if normalized and normalized not in ALLOWED_STATUS_FILTERS:
        return _result("invalid_input", None, "status 仅支持演示系统定义的工单状态。", scope="community_aggregate")
    try:
        conn = _get_conn()
        if normalized:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM work_orders WHERE status = ?", (normalized,)).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM work_orders").fetchone()
        conn.close()
    except Exception as exc:
        return _result("upstream_error", None, f"统计工单失败：{type(exc).__name__}", scope="community_aggregate")
    return _result(
        "success",
        {"count": int(row["cnt"] if row else 0), "status_filter": normalized or None},
        "这是全小区脱敏聚合数量，不包含其他业主工单明细。",
        scope="community_aggregate",
    )


# Compatibility aliases keep an older prompt/database discovery record from
# breaking.  They enforce the same server-side owner scope, so a model cannot
# use a legacy parameter to access other rooms.
@mcp.tool()
def list_recent_work_orders(room_id: Optional[str] = None, limit: int = 5) -> str:
    """兼容旧接口：仅可查询当前演示业主的最近工单。"""
    if room_id and room_id.strip() != DEMO_OWNER_ROOM_ID:
        return _result("unauthorized", None, "当前业主端不能按其他房号查询工单。", scope="owner")
    return get_my_recent_work_orders(limit=limit)


@mcp.tool()
def get_work_order_by_id(work_order_id: str) -> str:
    """兼容旧接口：仅可查询当前演示业主的一张工单。"""
    return get_my_work_order_by_id(work_order_id=work_order_id)


@mcp.tool()
def list_urgent_work_orders(limit: int = 5) -> str:
    """业主端不提供全小区紧急工单明细。"""
    return _result("unauthorized", None, "当前业主端没有查看全小区紧急工单明细的权限。", scope="owner")


if __name__ == "__main__":
    mcp.run(transport="stdio")
