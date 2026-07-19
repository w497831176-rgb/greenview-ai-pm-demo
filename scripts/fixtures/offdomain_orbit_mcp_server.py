"""Off-domain MCP fixture used to prove configuration-driven extensibility.

The runtime has no knowledge of this server or its tools.  Acceptance creates
the MCP configuration, discovers these schemas, declares read/write policies,
binds the server to a newly created vertical Agent and publishes a release.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("offdomain-orbit-server")
STORE_PATH = Path(
    os.getenv(
        "OFFDOMAIN_ORBIT_STORE_PATH",
        str(
            Path(os.getenv("PROPERTY_DATA_DIR", "/tmp"))
            / "offdomain_orbit_reservations.jsonl"
        ),
    )
)


def _result(status: str, data: Any = None, message: str = "") -> str:
    return json.dumps(
        {
            "status": status,
            "data": data,
            "message": message,
            "meta": {"fixture": "offdomain-orbit", "durable_write": True},
        },
        ensure_ascii=False,
    )


@mcp.tool()
def lookup_orbit_window(mission_code: str) -> str:
    """查询与物业无关的轨道观测窗口演示数据。"""

    normalized = (mission_code or "").strip()
    if not normalized:
        return _result("invalid_input", None, "mission_code 不能为空。")
    return _result(
        "success",
        {
            "mission_code": normalized,
            "window": "2032-04-08T09:30:00+08:00",
            "visibility_minutes": 18,
        },
        "已返回轨道观测窗口演示数据。",
    )


@mcp.tool()
def create_orbit_reservation(mission_code: str, owner_name: str) -> str:
    """创建与物业无关的轨道观测预约，并持久化到独立 JSONL 存储。"""

    mission = (mission_code or "").strip()
    owner = (owner_name or "").strip()
    if not mission or not owner:
        return _result(
            "invalid_input",
            None,
            "mission_code 与 owner_name 均不能为空。",
        )
    resource_id = f"ORBIT-{uuid.uuid4().hex[:10].upper()}"
    record = {
        "resource_id": resource_id,
        "mission_code": mission,
        "owner_name": owner,
    }
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STORE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return _result("success", record, "轨道观测预约已持久化。")


if __name__ == "__main__":
    mcp.run(transport="stdio")
