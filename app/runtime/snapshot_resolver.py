"""Resolve one immutable published configuration snapshot per chat session."""

from __future__ import annotations

import uuid
from typing import Any, Dict

from app.runtime.contracts import RunConfigSnapshot
from db.property_db import (
    get_current_runtime_release,
    get_run_config_snapshot,
    now_cn,
    save_run_config_snapshot,
)


def _to_contract(row: Dict[str, Any]) -> RunConfigSnapshot:
    return RunConfigSnapshot(
        snapshot_id=row["snapshot_id"],
        release_id=row["release_id"],
        snapshot_hash=row["config_hash"],
        session_id=row["session_id"],
        config=row["snapshot"],
        created_at=row["created_at"],
    )


def resolve_snapshot(session_id: str) -> RunConfigSnapshot:
    existing = get_run_config_snapshot(session_id)
    if existing:
        return _to_contract(existing)

    release = get_current_runtime_release()
    if not release or release.get("status") != "published":
        raise RuntimeError("no published RuntimeRelease available")
    saved = save_run_config_snapshot(
        snapshot_id=f"snap_{uuid.uuid4().hex}",
        session_id=session_id,
        release_id=release["release_id"],
        config_hash=release["config_hash"],
        snapshot=release["config"],
    )
    return _to_contract(saved)
