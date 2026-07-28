"""Deterministic V1.8.2-S9-B.2 range and server-pagination contract.

Uses only a temporary SQLite database and static frontend source inspection.
No HTTP, model, RuntimeRelease, or business-data call is made.
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import app.observability as observability


FIXED_NOW = datetime(2026, 7, 28, 12, 34, 56)


def assert_ranges() -> None:
    expected = {
        "today": ("今天", "2026-07-28 00:00:00", "2026-07-28 12:34:56"),
        "yesterday": ("昨天", "2026-07-27 00:00:00", "2026-07-27 23:59:59"),
        "last_7_days": ("近7天", "2026-07-22 00:00:00", "2026-07-28 12:34:56"),
        "this_month": ("本月", "2026-07-01 00:00:00", "2026-07-28 12:34:56"),
        "last_month": ("上月", "2026-06-01 00:00:00", "2026-06-30 23:59:59"),
    }
    for key, (label, start, end) in expected.items():
        scope = observability._reporting_scope(key, now=FIXED_NOW)
        assert (scope["label"], scope["start"], scope["end"]) == (
            label,
            start,
            end,
        )
        assert scope["timezone"] == "Asia/Shanghai (UTC+8)"

    custom = observability._reporting_scope(
        "custom", "2026-06-28", "2026-07-03", now=FIXED_NOW
    )
    assert custom["label"] == "2026-06-28至2026-07-03期间"
    assert custom["start"] == "2026-06-28"
    assert custom["end"] == "2026-07-03 23:59:59"
    try:
        observability._reporting_scope(
            "custom", "2026-07-04", "2026-07-03", now=FIXED_NOW
        )
    except ValueError:
        pass
    else:
        raise AssertionError("reversed custom range must be rejected")


def create_fixture(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE chat_traces (
            trace_id TEXT PRIMARY KEY,
            session_id TEXT,
            user_message TEXT,
            intent TEXT,
            agent_name TEXT,
            status TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE model_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT,
            model_id TEXT,
            total_tokens INTEGER,
            estimated_cost_cny REAL,
            usage_source TEXT,
            stage TEXT,
            status TEXT,
            created_at TEXT
        );
        """
    )

    # Enough ordinary rows to require two server pages.
    for index in range(22):
        trace_id = f"chat-{index:02d}"
        created_at = f"2026-07-28 00:00:{index:02d}"
        conn.execute(
            "INSERT INTO chat_traces VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trace_id,
                f"session-{index:02d}",
                f"普通物业问题 {index}",
                "consult",
                "customer_service",
                "complete",
                created_at,
                created_at,
            ),
        )
        conn.execute(
            """INSERT INTO model_calls
               (trace_id, model_id, total_tokens, estimated_cost_cny,
                usage_source, stage, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace_id,
                "deepseek-v4-flash",
                100 + index,
                0.001,
                "provider_actual",
                "vertical_agent",
                "success",
                created_at,
            ),
        )

    special_chat_rows = [
        (
            "duplicate-trace",
            "session-duplicate",
            "同时存在于Trace和模型调用中的问题",
            "consult",
            "customer_service",
            "complete",
            "2026-07-28 12:00:00",
            "2026-07-28 12:00:00",
        ),
        (
            "no-model",
            "session-rule",
            "确定性规则操作",
            "rule_operation",
            None,
            "complete",
            "2026-07-28 11:59:00",
            "2026-07-28 11:59:00",
        ),
        (
            "usage-missing",
            "session-missing",
            "Provider用量缺失",
            "consult",
            "customer_service",
            "failed",
            "2026-07-28 11:58:00",
            "2026-07-28 11:58:00",
        ),
    ]
    conn.executemany("INSERT INTO chat_traces VALUES (?, ?, ?, ?, ?, ?, ?, ?)", special_chat_rows)
    conn.executemany(
        """INSERT INTO model_calls
           (trace_id, model_id, total_tokens, estimated_cost_cny,
            usage_source, stage, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                "duplicate-trace",
                "deepseek-v4-flash",
                5164,
                0.0030102,
                "provider_actual",
                "router",
                "success",
                "2026-07-28 12:00:00",
            ),
            (
                "usage-missing",
                "deepseek-v4-flash",
                99,
                None,
                "unavailable",
                "router",
                "failed",
                "2026-07-28 11:58:00",
            ),
            (
                "darwin-only",
                "deepseek-v4-pro",
                4670,
                0.0118872,
                "provider_actual",
                "darwin",
                "success",
                "2026-07-28 11:57:00",
            ),
        ],
    )
    conn.commit()
    conn.close()


def assert_pagination() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "cost-pagination.db"
        create_fixture(db_path)
        original_get_conn = observability._get_conn

        def fixture_conn() -> sqlite3.Connection:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            return conn

        observability._get_conn = fixture_conn
        try:
            query = {
                "range_key": "custom",
                "start": "2026-07-28",
                "end": "2026-07-28",
                "limit": 20,
            }
            first = observability._list_trace_page(**query, offset=0)
            second = observability._list_trace_page(**query, offset=20)
            assert first["total"] == 26
            assert (first["page"], first["pages"]) == (1, 2)
            assert len(first["traces"]) == 20
            assert not first["has_previous"] and first["has_next"]
            assert (second["page"], second["pages"]) == (2, 2)
            assert len(second["traces"]) == 6
            assert second["has_previous"] and not second["has_next"]
            assert first["start"] == "2026-07-28"
            assert first["end"] == "2026-07-28 23:59:59"

            all_items = first["traces"] + second["traces"]
            trace_ids = [item["trace_id"] for item in all_items]
            assert len(trace_ids) == len(set(trace_ids)) == 26
            assert trace_ids.count("duplicate-trace") == 1
            assert [item["created_at"] for item in all_items] == sorted(
                (item["created_at"] for item in all_items), reverse=True
            )

            by_id = {item["trace_id"]: item for item in all_items}
            assert by_id["duplicate-trace"]["cost_status"] == "provider_actual"
            assert by_id["darwin-only"]["models"] == ["deepseek-v4-pro"]
            assert by_id["no-model"]["cost_status"] == "not_applicable"
            assert by_id["no-model"]["total_tokens"] is None
            assert by_id["usage-missing"]["cost_status"] == "partial_unavailable"
            assert by_id["usage-missing"]["model_call_count"] == 1
            assert by_id["usage-missing"]["provider_actual_priced_calls"] == 0
            assert by_id["usage-missing"]["unavailable_calls"] == 1

            darwin = observability._list_trace_page(
                **query,
                offset=0,
                model_id="deepseek-v4-pro",
                stage="darwin",
            )
            assert darwin["total"] == 1
            assert darwin["traces"][0]["trace_id"] == "darwin-only"
        finally:
            observability._get_conn = original_get_conn


def assert_frontend() -> None:
    source = Path("frontend/index.html").read_text(encoding="utf-8")
    section = source[
        source.index("async function renderCostGovernancePage") : source.index(
            "async function renderCostStrategyPage"
        )
    ]
    rendered = section[section.rindex("function renderPage()") : section.index("function bindEvents()")]
    for token in (
        "['today', '今天']",
        "['yesterday', '昨天']",
        "['last_7_days', '近7天']",
        "['this_month', '本月']",
        "['last_month', '上月']",
        "['custom', '自定义']",
        "state.rangeKey === 'custom'",
        "state.resolvedScope || {}",
        "new URLSearchParams({ start: scope.start, end: scope.end })",
        "traceParams.set('limit', String(state.pagination.limit))",
        "traceParams.set('offset', String((state.pagination.page - 1) * state.pagination.limit))",
        "每页20条",
        "共${Number(pagination.total || 0).toLocaleString()}条",
        "第${Number(pagination.page || 1)}/${Number(pagination.pages || 1)}页",
        "上一页",
        "下一页",
    ):
        assert token in section
    assert section.count("state.pagination.page = 1") >= 3
    assert section.count("loadAll({ refreshSummary: false })") >= 3
    assert ".slice(0, 10)" not in section
    assert "今日调用记录" not in section
    assert all(part in rendered for part in (
        "renderOverview()",
        "renderTraces()",
        "renderGovernancePrinciples()",
    ))


def main() -> None:
    assert_ranges()
    assert_pagination()
    assert_frontend()
    print("PASS: V1.8.2-S9-B.2 deterministic range/pagination contract (3 groups)")


if __name__ == "__main__":
    main()
