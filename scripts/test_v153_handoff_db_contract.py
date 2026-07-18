"""Isolated SQLite state-machine test for V1.5.3; never touches /app/data."""

import os
import shutil
import tempfile


TEMP_DIR = tempfile.mkdtemp(prefix="yiai-v153-handoff-")
os.environ["PROPERTY_DATA_DIR"] = TEMP_DIR

from db.property_db import (  # noqa: E402
    claim_handoff,
    close_handoff,
    create_chat_session,
    get_handoff_package,
    _get_conn,
    request_handoff,
    resolve_handoff,
    resume_handoff_after_owner_message,
    wait_for_handoff_user,
)


def main() -> None:
    try:
        # Minimal isolated schema: this verifies the state-machine functions
        # without invoking the demo's unrelated seed-price migrations.
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE chat_sessions (
                session_id TEXT PRIMARY KEY, handoff_status TEXT, handoff_reason TEXT,
                handoff_requested_at TEXT, handoff_active_at TEXT, handoff_resolved_at TEXT,
                assigned_to TEXT, title TEXT, last_message_at TEXT, last_message_preview TEXT,
                last_agent TEXT, created_at TEXT, updated_at TEXT, handoff_risk_level TEXT,
                handoff_reason_code TEXT, handoff_queue TEXT, handoff_package_json TEXT,
                handoff_summary TEXT, handoff_outcome TEXT, handoff_waiting_at TEXT,
                handoff_closed_at TEXT, handoff_cancelled_at TEXT, handoff_last_actor TEXT,
                handoff_last_action_at TEXT
            );
            CREATE TABLE handoff_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                action_type TEXT NOT NULL, status_before TEXT, status_after TEXT,
                actor TEXT, action_detail TEXT, created_at TEXT
            );
        """)
        conn.commit()
        conn.close()
        session_id = create_chat_session()["session_id"]
        requested = request_handoff(
            session_id,
            "DEMO_TEST_V153_业主明确请求人工",
            risk_level="L3",
            reason_code="owner_requested",
            queue="property_service",
            handoff_package={"owner_request": {"content": "DEMO_TEST_V153"}},
        )
        assert requested["handoff_status"] == "requested"
        assert requested["handoff_package"]["owner_request"]["content"] == "DEMO_TEST_V153"

        active = claim_handoff(session_id, "测试员工")
        assert active["handoff_status"] == "active" and active["assigned_to"] == "测试员工"
        waiting = wait_for_handoff_user(session_id, "测试员工", "请补充预约时间")
        assert waiting["handoff_status"] == "waiting_user"
        resumed = resume_handoff_after_owner_message(session_id)
        assert resumed["handoff_status"] == "active"
        resolved = resolve_handoff(session_id, "已安排复核", "测试员工")
        assert resolved["handoff_status"] == "resolved" and resolved["handoff_summary"] == "已安排复核"
        closed = close_handoff(session_id, "测试员工")
        assert closed["handoff_status"] == "closed"
        bundle = get_handoff_package(session_id)
        assert len(bundle["actions"]) >= 5
        print("V1.5.3 human-copilot SQLite contract passed")
    finally:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
