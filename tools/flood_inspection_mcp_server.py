"""Dynamic read/write MCP fixture for the V1.8 extension acceptance demo."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("flood-inspection-server")
DB_PATH = Path(os.getenv("PROPERTY_DATA_DIR", "/app/data")) / "property_demo.db"


def _result(status: str, data: Any = None, message: str = "") -> str:
    return json.dumps(
        {
            "status": status,
            "data": data,
            "message": message,
            "meta": {
                "source": "property_demo.sqlite",
                "demo_capability": "flood_inspection",
            },
        },
        ensure_ascii=False,
    )


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS flood_inspection_records (
            id TEXT PRIMARY KEY,
            zone TEXT NOT NULL,
            risk_note TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


@mcp.tool()
def get_flood_inspection_status(zone: str) -> str:
    """查询指定演示区域的最新汛期巡检状态，不修改任何记录。"""

    normalized = (zone or "").strip()
    if not normalized:
        return _result("invalid_input", None, "zone 不能为空。")
    conn = _conn()
    row = conn.execute(
        """
        SELECT id, zone, risk_note, status, created_at
        FROM flood_inspection_records
        WHERE zone = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (normalized,),
    ).fetchone()
    conn.close()
    if row is None:
        return _result(
            "success",
            {"zone": normalized, "latest_record": None, "risk_level": "待巡检"},
            "该演示区域暂无巡检记录。",
        )
    return _result(
        "success",
        {"zone": normalized, "latest_record": dict(row)},
        "返回该演示区域最新巡检记录。",
    )


@mcp.tool()
def create_flood_inspection_record(zone: str, risk_note: str) -> str:
    """创建一条真实持久化的汛期巡检记录；调用前必须取得业主确认。"""

    normalized_zone = (zone or "").strip()
    normalized_note = (risk_note or "").strip()
    if not normalized_zone or not normalized_note:
        return _result("invalid_input", None, "zone 和 risk_note 均不能为空。")
    resource_id = f"FLOOD-{uuid.uuid4().hex[:12].upper()}"
    conn = _conn()
    conn.execute(
        """
        INSERT INTO flood_inspection_records (id, zone, risk_note, status)
        VALUES (?, ?, ?, ?)
        """,
        (resource_id, normalized_zone, normalized_note, "待执行"),
    )
    conn.commit()
    row = conn.execute(
        """
        SELECT id, zone, risk_note, status, created_at
        FROM flood_inspection_records WHERE id = ?
        """,
        (resource_id,),
    ).fetchone()
    conn.close()
    return _result(
        "success",
        {
            "resource_id": resource_id,
            "record": dict(row) if row else {"id": resource_id},
        },
        "汛期巡检记录已持久化。",
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")

