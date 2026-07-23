"""Historical deterministic contract helpers.

These functions exist only so older no-model contract scripts retain their
original assertions. They are not imported by the V1.8 owner-chat runtime and
hold no runtime authority.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

DEFAULT_ROOM_ID = "3-2-1201"


def _unique_rag_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for result in results:
        key = (result.get("doc_id"), result.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def _policy_mcp_args(
    server_name: str,
    tool_name: str,
    message: str,
) -> Optional[Dict[str, Any]]:
    cities = ("北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安")
    if server_name == "weather-server":
        city = next((item for item in cities if item in message), None)
        return {"city": city} if city else None
    if server_name == "workorder-server":
        room_match = re.search(
            r"(\d{1,2})\s*[-#—]\s*(\d{1,2})\s*[-#—]\s*(\d{3,4})",
            message,
        )
        room = (
            f"{room_match.group(1)}-{room_match.group(2)}-{room_match.group(3)}"
            if room_match
            else (
                DEFAULT_ROOM_ID
                if any(word in message for word in ("我的", "我家", "本房号"))
                else None
            )
        )
        if tool_name == "list_recent_work_orders":
            return {"room_id": room, "limit": 5}
        if tool_name == "count_work_orders":
            return (
                {"status": "pending"}
                if any(word in message for word in ("待处理", "待派单", "未处理"))
                else {}
            )
    if server_name == "calendar-server":
        return {}
    return {}
