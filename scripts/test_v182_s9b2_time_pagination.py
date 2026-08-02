"""Deterministic V1.8.2-S9-B.2 range and server-pagination contract.

Uses only a temporary SQLite database and static frontend source inspection.
No HTTP, model, RuntimeRelease, or business-data call is made.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import app.observability as observability


FIXED_NOW = datetime(2026, 7, 28, 12, 34, 56)


def assert_ranges() -> None:
    expected = {
        "today": ("今天", "2026-07-28 00:00:00", "2026-07-28 12:34:56"),
        "yesterday": ("昨天", "2026-07-27 00:00:00", "2026-07-27 23:59:59.999999"),
        "last_7_days": ("近7天", "2026-07-22 00:00:00", "2026-07-28 12:34:56"),
        "this_month": ("本月", "2026-07-01 00:00:00", "2026-07-28 12:34:56"),
        "last_month": ("上月", "2026-06-01 00:00:00", "2026-06-30 23:59:59.999999"),
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
    assert custom["end"] == "2026-07-03 23:59:59.999999"
    # Lexically reversed strings can still be a valid chronological range.
    mixed = observability._reporting_scope(
        "custom",
        "2026-08-01T00:00:00+08:00",
        "2026-07-31T17:00:00Z",
        now=FIXED_NOW,
    )
    assert mixed["start"].endswith("+08:00")
    try:
        observability._reporting_scope(
            "custom", "2026-07-04", "2026-07-03", now=FIXED_NOW
        )
    except ValueError:
        pass
    else:
        raise AssertionError("reversed custom range must be rejected")

    subsecond_today = observability._reporting_scope(
        "today",
        now=datetime(
            2026,
            8,
            1,
            12,
            0,
            0,
            500000,
            tzinfo=timezone(timedelta(hours=8)),
        ),
    )
    assert subsecond_today["end"] == "2026-08-01 12:00:00.5"


def create_schema(conn: sqlite3.Connection) -> None:
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
            updated_at TEXT,
            run_type TEXT DEFAULT 'chat'
        );
        CREATE TABLE model_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT,
            model_id TEXT,
            total_tokens INTEGER,
            estimated_cost_cny REAL,
            usage_source TEXT,
            usage_normalized TEXT,
            stage TEXT,
            status TEXT,
            created_at TEXT,
            record_kind TEXT DEFAULT 'provider_attempt',
            usage_status TEXT,
            finished_at TEXT
        );
        CREATE TABLE evaluation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT
        );
        """
    )


def create_fixture(path: Path) -> None:
    conn = sqlite3.connect(path)
    create_schema(conn)

    # Enough ordinary rows to require two server pages.
    for index in range(22):
        trace_id = f"chat-{index:02d}"
        created_at = f"2026-07-28 00:00:{index:02d}"
        conn.execute(
            "INSERT INTO chat_traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trace_id,
                f"session-{index:02d}",
                f"普通物业问题 {index}",
                "consult",
                "customer_service",
                "complete",
                created_at,
                created_at,
                "chat",
            ),
        )
        conn.execute(
            """INSERT INTO model_calls
               (trace_id, model_id, total_tokens, estimated_cost_cny,
                usage_source, usage_normalized, stage, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace_id,
                "deepseek-v4-flash",
                100 + index,
                0.001,
                "provider_actual",
                json.dumps(
                    {
                        "record_kind": "provider_attempt",
                        "include_in_provider_aggregate": True,
                        "provider_request_sent": True,
                        "usage_status": "provider_actual",
                    }
                ),
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
            "chat",
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
            "chat",
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
            "chat",
        ),
    ]
    conn.executemany(
        "INSERT INTO chat_traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        special_chat_rows,
    )
    conn.executemany(
        """INSERT INTO model_calls
           (trace_id, model_id, total_tokens, estimated_cost_cny,
            usage_source, usage_normalized, stage, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                "duplicate-trace",
                "deepseek-v4-flash",
                5164,
                0.0030102,
                "provider_actual",
                json.dumps(
                    {
                        "record_kind": "provider_attempt",
                        "include_in_provider_aggregate": True,
                        "provider_request_sent": True,
                        "usage_status": "provider_actual",
                    }
                ),
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
                json.dumps(
                    {
                        "record_kind": "provider_attempt",
                        "include_in_provider_aggregate": True,
                        "provider_request_sent": True,
                        "usage_status": "unavailable_done_without_usage",
                        "usage_unavailable_reason": "stream_done_without_usage_chunk",
                    }
                ),
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
                json.dumps(
                    {
                        "record_kind": "provider_attempt",
                        "include_in_provider_aggregate": True,
                        "provider_request_sent": True,
                        "usage_status": "provider_actual",
                    }
                ),
                "darwin",
                "success",
                "2026-07-28 11:57:00",
            ),
        ],
    )
    conn.commit()
    conn.close()


def insert_provider_attempt(
    conn: sqlite3.Connection,
    trace_id: str,
    stage: str,
    created_at: str,
    total_tokens: int,
    amount: float = 0.001,
) -> None:
    output_tokens = min(10, total_tokens)
    usage = {
        "record_kind": "provider_attempt",
        "include_in_provider_aggregate": True,
        "provider_request_sent": True,
        "usage_status": "provider_actual",
        "token_source": "provider_actual",
        "input_cache_hit_tokens": 0,
        "input_cache_miss_tokens": total_tokens - output_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": min(5, output_tokens),
        "total_tokens": total_tokens,
    }
    conn.execute(
        """INSERT INTO model_calls
           (trace_id, model_id, total_tokens, estimated_cost_cny,
            usage_source, usage_normalized, stage, status, created_at,
            record_kind, usage_status, finished_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trace_id,
            "deepseek-v4-flash",
            total_tokens,
            amount,
            "provider_actual",
            json.dumps(usage),
            stage,
            "success",
            created_at,
            "provider_attempt",
            "provider_actual",
            created_at,
        ),
    )


def assert_mixed_time_cross_day_and_model_only() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "mixed-time.db"
        conn = sqlite3.connect(db_path)
        create_schema(conn)
        conn.executemany(
            "INSERT INTO chat_traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "cross-midnight-chat",
                    "session-cross",
                    "跨午夜对话",
                    "consult",
                    "customer_service",
                    "complete",
                    "2026-07-31 23:59:50",
                    "2026-08-01 23:59:59",
                    "chat",
                ),
                (
                    "cn-offset-chat",
                    "session-offset",
                    "带时区请求",
                    "consult",
                    "customer_service",
                    "complete",
                    "2026-08-01 10:00:00",
                    "2026-08-01 10:00:00",
                    "chat",
                ),
                (
                    "out-of-range-stage",
                    "session-stage",
                    "范围外阶段不得命中",
                    "consult",
                    "customer_service",
                    "complete",
                    "2026-08-01 08:00:00",
                    "2026-08-01 08:00:00",
                    "chat",
                ),
            ],
        )
        # UTC 15:59:59 is Beijing 23:59:59 on July 31: outside Aug 1.
        insert_provider_attempt(
            conn, "cross-midnight-chat", "router", "2026-07-31T15:59:59Z", 90
        )
        # UTC 16:00:00 is Beijing midnight on Aug 1: included.
        insert_provider_attempt(
            conn, "cross-midnight-chat", "router", "2026-07-31T16:00:00Z", 100
        )
        # A naive Beijing row at the final microsecond is included.
        insert_provider_attempt(
            conn,
            "cross-midnight-chat",
            "vertical_agent",
            "2026-08-01 23:59:59.999999",
            200,
        )
        # UTC 16:00:00 on Aug 1 is Beijing Aug 2 midnight: excluded.
        insert_provider_attempt(
            conn, "cross-midnight-chat", "vertical_agent", "2026-08-01T16:00:00Z", 300
        )
        insert_provider_attempt(
            conn, "cn-offset-chat", "agent_selector", "2026-08-01T10:00:00+08:00", 400
        )
        insert_provider_attempt(
            conn,
            "model-only-knowledge",
            "badcase_extract_knowledge",
            "2026-08-01T09:00:00Z",
            500,
        )
        insert_provider_attempt(
            conn,
            "model-only-evaluation",
            "router",
            "2026-08-01T08:30:00Z",
            50,
        )
        conn.execute(
            "INSERT INTO evaluation_runs (trace_id) VALUES (?)",
            ("model-only-evaluation",),
        )
        insert_provider_attempt(
            conn,
            "out-of-range-stage",
            "darwin",
            "2026-08-01T16:00:01Z",
            600,
        )
        conn.commit()
        conn.close()

        original_get_conn = observability._get_conn
        original_get_budget_thresholds = observability.get_budget_thresholds
        original_evaluation_summary = observability.evaluation_summary
        original_get_chat_trace = observability.get_chat_trace

        def fixture_conn() -> sqlite3.Connection:
            fixture = sqlite3.connect(db_path)
            fixture.row_factory = sqlite3.Row
            return fixture

        observability._get_conn = fixture_conn
        observability.get_budget_thresholds = lambda: {}
        observability.evaluation_summary = lambda: {}
        observability.get_chat_trace = lambda _trace_id: None
        try:
            scope = observability._reporting_scope(
                "custom", "2026-08-01", "2026-08-01", now=FIXED_NOW
            )
            calls = observability._fetch_model_calls(scope["start"], scope["end"])
            aggregate = observability._aggregate_model_calls(calls)
            assert aggregate["calls"] == 5
            assert aggregate["total_tokens"] == 1250
            assert aggregate["input_cache_hit_tokens"] == 0
            assert aggregate["input_cache_miss_tokens"] == 1200
            assert aggregate["output_tokens"] == 50
            assert aggregate["reasoning_tokens"] == 25
            assert aggregate["reasoning_known_calls"] == 5
            # Reasoning is retained separately and remains a subset of output;
            # it is not added to total_tokens a second time.
            assert aggregate["total_tokens"] == (
                aggregate["input_cache_hit_tokens"]
                + aggregate["input_cache_miss_tokens"]
                + aggregate["output_tokens"]
            )
            assert aggregate["by_stage"]["router"]["reasoning_tokens"] == 10
            assert aggregate["by_stage"]["vertical_agent"]["reasoning_tokens"] == 5
            assert aggregate["by_stage"]["agent_selector"]["reasoning_tokens"] == 5
            assert aggregate["by_stage"]["badcase_extract_knowledge"]["reasoning_tokens"] == 5

            overview = asyncio.run(
                observability.overview(
                    start="2026-08-01",
                    end="2026-08-01",
                    model_id=None,
                    stage=None,
                    trace_id=None,
                    range_key="custom",
                )
            )
            assert overview["provider_request_count"] == aggregate["calls"]
            assert overview["trace_group_count"] == 5
            assert overview["scope_consistent"] is True
            assert overview["known_usage"] == {
                "input_cache_hit_tokens": 0,
                "input_cache_miss_tokens": 1200,
                "output_tokens": 50,
                "reasoning_tokens": 25,
                "reasoning_known_calls": 5,
                "reasoning_unavailable_calls": 0,
                "reasoning_is_output_subset": True,
            }

            daily_trends = asyncio.run(
                observability.trends(group_by="day", start=None, end=None)
            )
            assert [item["period"] for item in daily_trends["items"]] == [
                "2026-07-31",
                "2026-08-01",
                "2026-08-02",
            ]
            assert [item["provider_request_count"] for item in daily_trends["items"]] == [
                1,
                5,
                2,
            ]
            hourly_trends = asyncio.run(
                observability.trends(group_by="hour", start=None, end=None)
            )
            midnight = {
                item["period"]: item["provider_request_count"]
                for item in hourly_trends["items"]
            }
            assert midnight["2026-08-01 00:00"] == 1

            page = observability._list_trace_page(
                range_key="custom",
                start="2026-08-01",
                end="2026-08-01",
                limit=20,
            )
            assert page["trace_group_count"] == 5
            assert page["provider_request_count"] == 5
            assert page["provider_request_count"] == overview["provider_request_count"]
            assert page["start"] == overview["scope"]["start"]
            assert page["end"] == overview["scope"]["end"]
            assert page["traces"][0]["trace_id"] == "cross-midnight-chat"
            by_id = {item["trace_id"]: item for item in page["traces"]}
            assert by_id["cross-midnight-chat"]["provider_request_count"] == 2
            assert by_id["cross-midnight-chat"]["total_tokens"] == 300
            assert by_id["cross-midnight-chat"]["reasoning_tokens"] == 10
            assert by_id["model-only-knowledge"]["operation_type"] == "badcase_extract_knowledge"
            assert by_id["model-only-knowledge"]["operation_label"] == "Badcase · 提取知识"
            assert by_id["model-only-evaluation"]["operation_type"] == "evaluation"
            assert by_id["model-only-evaluation"]["operation_label"] == "Evaluation评估"

            model_only = observability._list_trace_page(
                range_key="custom",
                start="2026-08-01",
                end="2026-08-01",
                stage="badcase_extract_knowledge",
                limit=20,
            )
            assert model_only["trace_group_count"] == 1
            assert model_only["provider_request_count"] == 1
            assert model_only["traces"][0]["trace_id"] == "model-only-knowledge"

            outside_stage = observability._list_trace_page(
                range_key="custom",
                start="2026-08-01",
                end="2026-08-01",
                stage="darwin",
                limit=20,
            )
            assert outside_stage["trace_group_count"] == 0
            assert outside_stage["provider_request_count"] == 0

            yesterday = observability._reporting_scope(
                "yesterday",
                now=datetime(2026, 8, 2, 9, 0, tzinfo=timezone(timedelta(hours=8))),
            )
            assert yesterday["start"] == "2026-08-01 00:00:00"
            assert yesterday["end"] == "2026-08-01 23:59:59.999999"
        finally:
            observability._get_conn = original_get_conn
            observability.get_budget_thresholds = original_get_budget_thresholds
            observability.evaluation_summary = original_evaluation_summary
            observability.get_chat_trace = original_get_chat_trace


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
            assert first["end"] == "2026-07-28 23:59:59.999999"
            assert first["trace_group_count"] == 26
            assert first["provider_request_count"] == 25
            assert first["scope_consistent"] is True

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
            assert by_id["darwin-only"]["operation_type"] == "badcase_darwin"
            assert by_id["darwin-only"]["operation_label"] == "Badcase · Darwin建议"
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
        "条Trace分组",
        "次Provider请求",
        "三类Token与Reasoning",
        "属于输出子集，不重复计入总量",
        "统计口径异常",
        "state.overview.scope_consistent !== false",
        "Number(state.overview.trace_group_count ?? 0) === state.traceSummary.traceGroupCount",
        "Trace与成本加载失败",
        "接口失败没有被转换成0次、¥0或“不适用”",
        "第${Number(pagination.page || 1)}/${Number(pagination.pages || 1)}页",
        "上一页",
        "下一页",
    ):
        assert token in section
    assert section.count("state.pagination.page = 1") >= 3
    assert "apiGet(`/api/observability/overview?${overviewParams.toString()}`).catch" not in section
    assert "apiGet(`/api/observability/traces?${traceParams.toString()}`).catch" not in section
    assert "apiGet('/api/observability/prices').catch" not in section
    assert "apiGet('/api/observability/budget').catch" not in section
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
    assert_mixed_time_cross_day_and_model_only()
    assert_frontend()
    print("PASS: V1.8.2 Trace timepoint/range/pagination contract (4 groups)")


if __name__ == "__main__":
    main()
