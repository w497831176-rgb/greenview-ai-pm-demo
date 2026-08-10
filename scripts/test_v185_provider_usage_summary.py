"""Focused Provider usage-summary and Trace date-filter contract checks.

Only symbolic rows in temporary SQLite databases are used.  The script does
not call a model, production API, RuntimeRelease, or business workflow.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST_DATA = tempfile.TemporaryDirectory(prefix="yiai-v185-usage-summary-")
os.environ["PROPERTY_DATA_DIR"] = TEST_DATA.name
os.environ["DEEPSEEK_API_KEY"] = ""

try:
    import fastapi  # noqa: F401
    import pydantic  # noqa: F401
except ModuleNotFoundError:
    fastapi_stub = types.ModuleType("fastapi")

    class _Router:
        def __init__(self, *args, **kwargs):
            pass

        def _route(self, *args, **kwargs):
            return lambda function: function

        get = post = put = delete = patch = _route

    class _HTTPException(Exception):
        def __init__(self, status_code: int, detail=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    fastapi_stub.APIRouter = _Router
    fastapi_stub.HTTPException = _HTTPException
    fastapi_stub.Query = lambda default=None, **kwargs: default
    sys.modules["fastapi"] = fastapi_stub

    pydantic_stub = types.ModuleType("pydantic")

    class _BaseModel:
        pass

    pydantic_stub.BaseModel = _BaseModel
    pydantic_stub.Field = lambda default=None, **kwargs: default
    sys.modules["pydantic"] = pydantic_stub

import app.observability as observability  # noqa: E402


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE model_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT,
            stage TEXT,
            model_id TEXT,
            status TEXT,
            usage_source TEXT,
            usage_normalized TEXT,
            record_kind TEXT,
            usage_status TEXT,
            created_at TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            reasoning_tokens INTEGER,
            cached_tokens INTEGER,
            total_tokens INTEGER
        )
        """
    )


def _usage(
    *,
    actual_model: str | None,
    hit: int | None,
    miss: int | None,
    output: int | None,
    total: int | None,
    request_sent: bool = True,
) -> dict:
    payload = {
        "record_kind": "provider_attempt",
        "include_in_provider_aggregate": True,
        "provider_request_sent": request_sent,
        "usage_status": "provider_actual",
        "token_source": "provider_actual",
        "reasoning_tokens": 0 if output is not None else None,
    }
    if actual_model is not None:
        payload["provider_response_model"] = actual_model
    if hit is not None:
        payload["input_cache_hit_tokens"] = hit
    if miss is not None:
        payload["input_cache_miss_tokens"] = miss
    if hit is not None and miss is not None:
        payload["input_tokens"] = hit + miss
    if output is not None:
        payload["output_tokens"] = output
    if total is not None:
        payload["total_tokens"] = total
    return payload


def _insert_call(
    conn: sqlite3.Connection,
    *,
    created_at: str,
    requested_model: str,
    actual_model: str | None,
    hit: int | None,
    miss: int | None,
    output: int | None,
    total: int | None,
    sequence: int,
    request_sent: bool = True,
    record_kind: str = "provider_attempt",
) -> None:
    usage = _usage(
        actual_model=actual_model,
        hit=hit,
        miss=miss,
        output=output,
        total=total,
        request_sent=request_sent,
    )
    conn.execute(
        """
        INSERT INTO model_calls (
            trace_id, stage, model_id, status, usage_source,
            usage_normalized, record_kind, usage_status, created_at,
            input_tokens, output_tokens, reasoning_tokens, cached_tokens,
            total_tokens
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"trace-symbolic-{sequence}",
            "symbolic-stage",
            requested_model,
            "success",
            "provider_actual",
            json.dumps(usage),
            record_kind,
            "provider_actual",
            created_at,
            (hit + miss) if hit is not None and miss is not None else None,
            output,
            0 if output is not None else None,
            hit,
            total,
        ),
    )


def _with_fixture(run) -> None:
    with tempfile.TemporaryDirectory(prefix="yiai-usage-db-") as temp_dir:
        path = Path(temp_dir) / "usage.db"
        conn = sqlite3.connect(path)
        _create_schema(conn)
        run(conn, path)
        conn.close()


def _query(path: Path, start: str, end: str) -> dict:
    original_get_conn = observability._get_conn

    def fixture_conn() -> sqlite3.Connection:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn

    observability._get_conn = fixture_conn
    try:
        return observability._query_provider_usage_summary(start, end)
    finally:
        observability._get_conn = original_get_conn


def test_groups_by_provider_actual_model_and_sums_direct_usage() -> None:
    def run(conn: sqlite3.Connection, path: Path) -> None:
        _insert_call(
            conn,
            created_at="2026-08-09 10:00:00",
            requested_model="deepseek-v4-flash",
            actual_model="deepseek-v4-flash",
            hit=11,
            miss=13,
            output=17,
            total=41,
            sequence=1,
        )
        # Requested Flash but actual Pro: the row must be grouped under Pro.
        _insert_call(
            conn,
            created_at="2026-08-09 11:00:00",
            requested_model="deepseek-v4-flash",
            actual_model="deepseek-v4-pro",
            hit=0,
            miss=19,
            output=23,
            total=42,
            sequence=2,
        )
        conn.commit()
        result = _query(path, "2026-08-09", "2026-08-09 23:59:59.999999")
        flash, pro = result["rows"]
        assert flash["provider_request_count"] == 1
        assert flash["cache_hit_input_tokens"] == 11
        assert flash["cache_miss_input_tokens"] == 13
        assert flash["output_tokens"] == 17
        assert flash["total_tokens"] == 41
        assert pro["provider_request_count"] == 1
        assert pro["cache_miss_input_tokens"] == 19
        assert pro["output_tokens"] == 23
        assert pro["total_tokens"] == 42
        assert result["total"]["provider_request_count"] == 2
        assert result["total"]["cache_hit_input_tokens"] == 11
        assert result["total"]["cache_miss_input_tokens"] == 32
        assert result["total"]["output_tokens"] == 40
        assert result["total"]["total_tokens"] == 83
        assert result["complete"] is True

    _with_fixture(run)


def test_no_calls_are_real_zeroes() -> None:
    def run(conn: sqlite3.Connection, path: Path) -> None:
        conn.commit()
        result = _query(path, "2026-08-09", "2026-08-09 23:59:59.999999")
        assert result["complete"] is True
        for row in [*result["rows"], result["total"]]:
            assert row["provider_request_count"] == 0
            assert row["cache_hit_input_tokens"] == 0
            assert row["cache_miss_input_tokens"] == 0
            assert row["output_tokens"] == 0
            assert row["total_tokens"] == 0

    _with_fixture(run)


def test_nonphysical_rows_are_excluded_and_unknown_actual_model_is_explicit() -> None:
    def run(conn: sqlite3.Connection, path: Path) -> None:
        _insert_call(
            conn,
            created_at="2026-08-09 09:00:00",
            requested_model="symbolic-request-model",
            actual_model=None,
            hit=2,
            miss=3,
            output=4,
            total=9,
            sequence=20,
        )
        _insert_call(
            conn,
            created_at="2026-08-09 09:01:00",
            requested_model="deepseek-v4-flash",
            actual_model="deepseek-v4-flash",
            hit=50,
            miss=60,
            output=70,
            total=180,
            sequence=21,
            request_sent=False,
        )
        _insert_call(
            conn,
            created_at="2026-08-09 09:02:00",
            requested_model="deepseek-v4-pro",
            actual_model="deepseek-v4-pro",
            hit=80,
            miss=90,
            output=100,
            total=270,
            sequence=22,
            record_kind="logical",
        )
        conn.commit()
        result = _query(path, "2026-08-09", "2026-08-09 23:59:59.999999")
        assert [row["provider_request_count"] for row in result["rows"]] == [0, 0]
        assert result["total"]["provider_request_count"] == 1
        assert result["total"]["total_tokens"] is None
        assert result["unclassified_provider_request_count"] == 1
        assert result["complete"] is False

    _with_fixture(run)


def test_incomplete_usage_is_not_rendered_as_zero() -> None:
    def run(conn: sqlite3.Connection, path: Path) -> None:
        _insert_call(
            conn,
            created_at="2026-08-09 12:00:00",
            requested_model="deepseek-v4-flash",
            actual_model="deepseek-v4-flash",
            hit=None,
            miss=7,
            output=5,
            total=12,
            sequence=3,
        )
        conn.commit()
        result = _query(path, "2026-08-09", "2026-08-09 23:59:59.999999")
        flash = result["rows"][0]
        assert flash["provider_request_count"] == 1
        assert flash["cache_hit_input_tokens"] is None
        assert flash["cache_miss_input_tokens"] is None
        assert flash["output_tokens"] is None
        assert flash["total_tokens"] is None
        assert flash["usage_complete"] is False
        assert result["complete"] is False
        assert result["incomplete_provider_request_count"] == 1
        assert result["total"]["total_tokens"] is None

    _with_fixture(run)


def test_beijing_midnight_boundary() -> None:
    def run(conn: sqlite3.Connection, path: Path) -> None:
        _insert_call(
            conn,
            created_at="2026-08-09T15:59:59+00:00",
            requested_model="deepseek-v4-flash",
            actual_model="deepseek-v4-flash",
            hit=1,
            miss=2,
            output=3,
            total=6,
            sequence=4,
        )
        _insert_call(
            conn,
            created_at="2026-08-09T16:00:00+00:00",
            requested_model="deepseek-v4-pro",
            actual_model="deepseek-v4-pro",
            hit=4,
            miss=5,
            output=6,
            total=15,
            sequence=5,
        )
        conn.commit()
        august_ninth = _query(
            path, "2026-08-09", "2026-08-09 23:59:59.999999"
        )
        august_tenth = _query(
            path, "2026-08-10", "2026-08-10 23:59:59.999999"
        )
        assert august_ninth["rows"][0]["provider_request_count"] == 1
        assert august_ninth["rows"][1]["provider_request_count"] == 0
        assert august_tenth["rows"][0]["provider_request_count"] == 0
        assert august_tenth["rows"][1]["provider_request_count"] == 1

    _with_fixture(run)


def test_custom_range_rejects_reverse_dates() -> None:
    try:
        observability._reporting_scope(
            "custom", "2026-08-10", "2026-08-09"
        )
    except ValueError as exc:
        assert "开始日期不能晚于结束日期" in str(exc)
    else:
        raise AssertionError("reverse custom dates must fail")


def test_frontend_uses_one_scope_for_parallel_summary_and_trace_reads() -> None:
    source = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    section = source[
        source.index("async function renderCostGovernancePage") : source.index(
            "async function renderCostStrategyPage"
        )
    ]
    load = section[
        section.index("async function loadAll") : section.index(
            "async function reloadPage"
        )
    ]
    scope_builder = section[
        section.index("function scopeParams") : section.index("function fmtCost")
    ]
    assert "rangeKey: 'yesterday'" in section
    assert "const usageParams = scopeParams()" in load
    assert "new URLSearchParams(usageParams.toString())" in load
    assert "Promise.all" in load
    assert "/api/observability/usage-summary?" in load
    assert "/api/observability/traces?" in load
    assert "day_before_yesterday" in section
    assert "this_week" in section
    assert "beijingWeekStart()" in section
    assert "state.rangeKey === 'today'" in scope_builder
    assert "state.rangeKey === 'this_month'" in scope_builder
    assert scope_builder.count("params.set('range_key', 'custom')") >= 5
    assert "开始日期不能晚于结束日期" in section
    assert "state.pagination.page = 1" in section
    assert "Reasoning属于Output子集，不重复计入Total。" in section
    assert "value === null || row[field] === undefined" not in section
    assert "row[field] === null || row[field] === undefined" in section
    assert "provider_request_count: null" in section
    assert "row.provider_request_count === null" in section


def main() -> None:
    tests = (
        test_groups_by_provider_actual_model_and_sums_direct_usage,
        test_no_calls_are_real_zeroes,
        test_nonphysical_rows_are_excluded_and_unknown_actual_model_is_explicit,
        test_incomplete_usage_is_not_rendered_as_zero,
        test_beijing_midnight_boundary,
        test_custom_range_rejects_reverse_dates,
        test_frontend_uses_one_scope_for_parallel_summary_and_trace_reads,
    )
    try:
        for test in tests:
            test()
            print(f"PASS {test.__name__}")
        print(
            f"Provider usage summary: PASS ({len(tests)} checks; Provider calls: 0)"
        )
    finally:
        TEST_DATA.cleanup()


if __name__ == "__main__":
    main()
