"""Deterministic S10-F Trace truthfulness and background-budget contract.

Uses an isolated temporary SQLite database.  It never opens HTTP and never
constructs or invokes a model.  The deployment test container must additionally
run with ``--network none``.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_TEST_DATA_TEMP = tempfile.TemporaryDirectory(prefix="yiai-s10f-data-")
_TEST_DATA_DIR = Path(_TEST_DATA_TEMP.name).resolve()
_normalized_test_data_dir = str(_TEST_DATA_DIR).replace("\\", "/").lower()
if (
    _normalized_test_data_dir == "/app/data"
    or _normalized_test_data_dir.startswith("/app/data/")
    or _normalized_test_data_dir == "/volume3/docker/agno-demo-os"
    or _normalized_test_data_dir.startswith("/volume3/docker/agno-demo-os/")
):
    raise RuntimeError("unsafe S10-F PROPERTY_DATA_DIR")
# Never inherit a production data directory or Provider key into this test.
os.environ["PROPERTY_DATA_DIR"] = str(_TEST_DATA_DIR)
os.environ["DEEPSEEK_API_KEY"] = ""

try:
    import fastapi  # noqa: F401
    import pydantic  # noqa: F401
    HAS_RUNTIME_DEPENDENCIES = True
except ModuleNotFoundError:
    HAS_RUNTIME_DEPENDENCIES = False

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: object = None):
            super().__init__(str(detail))
            self.status_code = status_code
            self.detail = detail

    class APIRouter:
        def __init__(self, *_args, **_kwargs):
            pass

        def _decorator(self, *_args, **_kwargs):
            return lambda function: function

        get = post = put = delete = _decorator

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.APIRouter = APIRouter
    fastapi_stub.HTTPException = HTTPException
    fastapi_stub.Query = lambda default=None, **_kwargs: default
    pydantic_stub = types.ModuleType("pydantic")
    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = lambda default=None, **_kwargs: default
    sys.modules["fastapi"] = fastapi_stub
    sys.modules["pydantic"] = pydantic_stub

    property_stub = types.ModuleType("db.property_db")
    property_names = (
        "create_model_price delete_model_price evaluation_summary "
        "get_badcase_id_by_trace_id get_budget_thresholds get_chat_trace "
        "get_evaluation_run_by_trace_id get_mcp_call_audits_for_trace "
        "get_model_call get_model_calls_for_trace get_model_price "
        "list_chat_messages list_chat_traces list_trace_events list_model_prices "
        "update_budget_thresholds update_model_price"
    ).split()
    for name in property_names:
        setattr(property_stub, name, lambda *_args, **_kwargs: None)
    property_stub._get_conn = lambda: None
    property_stub.now_cn = lambda: "2026-08-02 12:00:00"
    property_stub.now_cn_dt = lambda: datetime(
        2026, 8, 2, 12, 0, tzinfo=timezone(timedelta(hours=8))
    )
    sys.modules["db.property_db"] = property_stub

if os.getenv("S10F_REQUIRE_FULL_DEPENDENCIES") == "1" and not HAS_RUNTIME_DEPENDENCIES:
    raise SystemExit("FAIL: S10-F full dependencies are required; fallback is forbidden")

import app.observability as observability
from fastapi import HTTPException

if HAS_RUNTIME_DEPENDENCIES:
    from db import property_db

    # This initializes only the freshly forced temporary fixture database.
    # The standalone fixed-history verifier is a separate subprocess and
    # independently proves that its read-only target never calls init_db.
    property_db.init_db()
    import app.badcases as badcases
    import app.evaluations as evaluations
    import app.model_configs as model_configs
else:
    badcases = evaluations = model_configs = None


CHECKS: list[str] = []
DEFERRED_CHECKS: list[str] = []
PRICE = {
    "price_snapshot_id": "s10f-fixture",
    "model_id": "deepseek-v4-flash",
    "input_price_per_1m": 1.0,
    "cached_input_price_per_1m": 0.1,
    "output_price_per_1m": 2.0,
}


def check(name: str, condition: object) -> None:
    if not condition:
        raise AssertionError(name)
    CHECKS.append(name)


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
            session_id TEXT,
            model_id TEXT,
            total_tokens INTEGER,
            estimated_cost_cny REAL,
            usage_source TEXT,
            usage_normalized TEXT,
            stage TEXT,
            status TEXT,
            error_summary TEXT,
            created_at TEXT,
            finished_at TEXT,
            record_kind TEXT,
            usage_status TEXT,
            price_snapshot TEXT,
            cost_source TEXT,
            local_attempt_id TEXT,
            provider_request_id TEXT,
            model_selection_reason TEXT
        );
        CREATE TABLE evaluation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT
        );
        """
    )


def insert_attempt(
    conn: sqlite3.Connection,
    *,
    trace_id: str,
    created_at: object,
    usage_status: str = "provider_actual",
    request_sent: object = True,
    include: bool = True,
    hit: object = 10,
    miss: object = 20,
    output: object = 30,
    reasoning: object = 12,
    total: object = 60,
    priced: bool = True,
    stage: str = "router",
    attempt_key: str | None = None,
    calculated_direct_cost: float | None = 0.001,
    inconsistency_reasons: list[str] | None = None,
) -> None:
    attempt_key = attempt_key or trace_id
    usage = {
        "record_kind": "provider_attempt",
        "local_attempt_id": f"attempt-{attempt_key}",
        "provider_request_id": f"request-{attempt_key}" if request_sent is True else None,
        "include_in_provider_aggregate": include,
        "provider_request_sent": request_sent,
        "usage_status": usage_status,
        "token_source": "provider_actual" if usage_status == "provider_actual" else "unavailable",
        "input_cache_hit_tokens": hit,
        "input_cache_miss_tokens": miss,
        "output_tokens": output,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
        "provider_actual_model": "deepseek-v4-flash",
        "cost_source": "platform_price_snapshot" if priced else None,
        "price_snapshot": PRICE if priced else {},
        "calculated_direct_cost": calculated_direct_cost if priced else None,
    }
    if inconsistency_reasons:
        usage["provider_usage_inconsistency_reasons"] = list(
            inconsistency_reasons
        )
    conn.execute(
        """INSERT INTO model_calls (
               trace_id, session_id, model_id, total_tokens,
               estimated_cost_cny, usage_source, usage_normalized, stage,
               status, created_at, finished_at, record_kind, usage_status,
               price_snapshot, cost_source, local_attempt_id,
               provider_request_id, model_selection_reason
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trace_id,
            f"session-{trace_id}",
            "deepseek-v4-flash",
            total,
            calculated_direct_cost if priced else None,
            "provider_actual" if usage_status == "provider_actual" else "unavailable",
            json.dumps(usage),
            stage,
            "success" if request_sent is True else "pending",
            created_at,
            created_at,
            "provider_attempt",
            usage_status,
            json.dumps(PRICE) if priced else None,
            "platform_price_snapshot" if priced else None,
            f"attempt-{attempt_key}",
            f"request-{attempt_key}" if request_sent is True else None,
            "S10-F deterministic fixture",
        ),
    )


def create_fixture(path: Path) -> None:
    conn = sqlite3.connect(path)
    create_schema(conn)
    insert_attempt(conn, trace_id="healthy", created_at="2026-08-01T08:00:00+08:00")
    insert_attempt(conn, trace_id="null-time", created_at=None)
    insert_attempt(conn, trace_id="invalid-time", created_at="not-a-time")
    insert_attempt(
        conn,
        trace_id="orphaned",
        created_at="2026-08-01T09:00:00+08:00",
        usage_status="orphaned_pending",
        request_sent=None,
        include=False,
        hit=None,
        miss=None,
        output=None,
        reasoning=None,
        total=None,
        priced=False,
        stage="badcase_classify",
    )
    insert_attempt(
        conn,
        trace_id="usage-inconsistent",
        created_at="2026-08-01T10:00:00+08:00",
        hit=10,
        miss=20,
        output=60,
        reasoning=70,
        total=100,
    )
    insert_attempt(
        conn,
        trace_id="price-missing",
        created_at="2026-08-01T11:00:00+08:00",
        priced=False,
    )
    insert_attempt(
        conn,
        trace_id="usage-unavailable",
        created_at="2026-08-01T12:00:00+08:00",
        usage_status="unavailable_done_without_usage",
        hit=None,
        miss=None,
        output=None,
        reasoning=None,
        total=None,
        priced=False,
    )
    conn.commit()
    conn.close()


def _raw_attempt(
    *,
    trace_id: str,
    created_at: object = "2026-08-01T08:00:00+08:00",
    usage_status: str = "provider_actual",
    request_sent: object = True,
    include: bool = True,
    hit: object = 10,
    miss: object = 20,
    output: object = 30,
    reasoning: object = 12,
    total: object = 60,
    priced: bool = True,
    cost: object = 0.001,
    stage: str = "router",
    attempt_key: str | None = None,
    inconsistency_reasons: list[str] | None = None,
) -> dict:
    attempt_key = attempt_key or trace_id
    usage = {
        "record_kind": "provider_attempt",
        "local_attempt_id": f"attempt-{attempt_key}",
        "provider_request_id": (
            f"request-{attempt_key}" if request_sent is True else None
        ),
        "include_in_provider_aggregate": include,
        "provider_request_sent": request_sent,
        "usage_status": usage_status,
        "token_source": (
            "provider_actual" if usage_status == "provider_actual" else "unavailable"
        ),
        "input_cache_hit_tokens": hit,
        "input_cache_miss_tokens": miss,
        "output_tokens": output,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
        "provider_actual_model": "deepseek-v4-flash",
        "cost_source": "platform_price_snapshot" if priced else None,
        "price_snapshot": PRICE if priced else {},
        "calculated_direct_cost": cost if priced else None,
    }
    if inconsistency_reasons:
        usage["provider_usage_inconsistency_reasons"] = list(
            inconsistency_reasons
        )
    return {
        "trace_id": trace_id,
        "session_id": f"session-{trace_id}",
        "model_id": "deepseek-v4-flash",
        "total_tokens": total,
        "estimated_cost_cny": cost if priced else None,
        "usage_source": (
            "provider_actual" if usage_status == "provider_actual" else "unavailable"
        ),
        "usage_normalized": usage,
        "stage": stage,
        "status": "success" if request_sent is True else "pending",
        "created_at": created_at,
        "finished_at": created_at,
        "record_kind": "provider_attempt",
        "usage_status": usage_status,
        "price_snapshot": PRICE if priced else None,
        "cost_source": "platform_price_snapshot" if priced else None,
        "local_attempt_id": f"attempt-{attempt_key}",
        "provider_request_id": (
            f"request-{attempt_key}" if request_sent is True else None
        ),
    }


def assert_fixed_history_mirror_contract() -> None:
    """Protect the accepted 2026-08-01 totals without touching production."""
    rows: list[dict] = []
    stages = [
        *("router" for _ in range(5)),
        *("agent_selector" for _ in range(5)),
        *("vertical_agent" for _ in range(3)),
        *("badcase_extract_knowledge" for _ in range(2)),
    ]
    chat_traces = [f"history-chat-{index}" for index in range(1, 6)]
    for index in range(15):
        trace_id = (
            chat_traces[index % 5]
            if index < 13
            else f"history-badcase-{index - 12}"
        )
        is_last = index == 14
        hit = 274 if is_last else 273
        miss = 837 if is_last else 824
        output = 456 if is_last else 447
        reasoning = 347 if is_last else 333
        total = hit + miss + output
        rows.append(
            _raw_attempt(
                trace_id=trace_id,
                stage=stages[index],
                attempt_key=f"history-{index + 1}",
                hit=hit,
                miss=miss,
                output=output,
                reasoning=reasoning,
                total=total,
                cost=0.00208292 if is_last else 0.0017,
                created_at=f"2026-08-01T{8 + index // 2:02d}:{(index % 2) * 30:02d}:00+08:00",
            )
        )
    summary = observability._aggregate_model_calls(rows)
    check("fixed-history mirror has seven Trace groups", len({row["trace_id"] for row in rows}) == 7)
    check("fixed-history mirror has fifteen Provider requests", summary["calls"] == 15)
    check("fixed-history cache-hit total remains 4096", summary["input_cache_hit_tokens"] == 4096)
    check("fixed-history cache-miss total remains 12373", summary["input_cache_miss_tokens"] == 12373)
    check("fixed-history output total remains 6714", summary["output_tokens"] == 6714)
    check("fixed-history reasoning total remains 5009", summary["reasoning_tokens"] == 5009)
    check("fixed-history Total excludes reasoning double-count", summary["total_tokens"] == 23183)
    check("fixed-history Reasoning remains Output subset", summary["reasoning_is_output_subset"] is True)
    check("fixed-history platform snapshot cost remains exact", summary["provider_actual_cost_cny"] == 0.02588292)
    check("fixed-history background split remains two requests", summary["by_stage"]["badcase_extract_knowledge"]["calls"] == 2)


def assert_provider_actual_core_completeness() -> None:
    with tempfile.TemporaryDirectory(prefix="yiai-s10f-incomplete-") as temp_dir:
        db_path = Path(temp_dir) / "incomplete.db"
        conn = sqlite3.connect(db_path)
        create_schema(conn)
        fixtures = (
            ("missing-hit", {"hit": None}),
            ("missing-miss", {"miss": None}),
            ("missing-output", {"output": None}),
            ("missing-total", {"total": None}),
        )
        for trace_id, override in fixtures:
            insert_attempt(
                conn,
                trace_id=trace_id,
                created_at="2026-08-01T08:00:00+08:00",
                **override,
            )
        conn.commit()
        conn.close()

        original_get_conn = observability._get_conn
        original_thresholds = observability.get_budget_thresholds
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
            calls = observability._fetch_model_calls(
                "2026-08-01", "2026-08-01 23:59:59.999999"
            )
            summary = observability._aggregate_model_calls(calls)
            check("four incomplete actual rows still count as Provider requests", summary["calls"] == 4 and summary["provider_actual_calls"] == 4)
            check("incomplete actual rows never become known Token calls", summary["token_known_calls"] == 0 and summary["token_unavailable_calls"] == 4 and summary["known_usage_calls"] == 0)
            check("incomplete actual rows do not expose complete cost", summary["provider_actual_priced_calls"] == 0 and summary["cost_complete"] is False)
            quality = observability._data_quality_summary(
                observability._fetch_reporting_model_calls(
                    "2026-08-01", "2026-08-01 23:59:59.999999"
                )
            )
            check("all four missing core fields use one explicit anomaly code", quality["provider_actual_usage_incomplete_count"] == 4 and all("provider_actual_usage_incomplete" in item["issue_codes"] for item in quality["anomaly_attempts"]))
            missing_sets = {
                tuple(item["provider_actual_usage_missing_fields"])
                for item in quality["anomaly_attempts"]
            }
            check("Hit Miss Output and Total are distinguished", missing_sets == {("cache_hit_input_tokens",), ("cache_miss_input_tokens",), ("output_tokens",), ("total_tokens",)})
            check("incomplete actual Usage is a hard data-quality error", quality["data_quality_status"] == "data_quality_error")
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
            check("overview counts requests but reports four unknown Token calls", overview["provider_request_count"] == 4 and overview["unknown_token_calls"] == 4)
            check("overview cannot call incomplete actual Usage consistent", overview["statistics_status"] == "data_quality_error" and overview["data_quality_status"] == "data_quality_error")
            check("overview budget is unavailable for incomplete actual Usage", overview["budget_status"] == "unavailable" and overview["platform_price_snapshot_direct_cost_cny"] is None)
            check(
                "incomplete actual Usage suppresses affected period budget percentages",
                overview["this_month"]["budget_status"] == "unavailable"
                and overview["this_month"]["budget_usage_percent"] is None,
            )
        finally:
            observability._get_conn = original_get_conn
            observability.get_budget_thresholds = original_thresholds
            observability.evaluation_summary = original_evaluation_summary
            observability.get_chat_trace = original_get_chat_trace


def assert_mixed_cost_truthfulness() -> None:
    healthy = _raw_attempt(trace_id="mixed-healthy", cost=0.001)
    incomplete = _raw_attempt(
        trace_id="mixed-incomplete",
        hit=None,
        cost=0.002,
    )
    summary = observability._aggregate_model_calls([healthy, incomplete])
    check(
        "mixed complete and incomplete rows retain both Provider requests",
        summary["calls"] == 2 and summary["provider_actual_calls"] == 2,
    )
    check(
        "mixed ledger never exposes known part in the complete direct-cost field",
        summary["platform_price_snapshot_direct_cost_cny"] is None
        and summary["known_partial_provider_actual_cost_cny"] == 0.001
        and summary["cost_complete"] is False,
    )
    trace_summary = observability._cost_summary([healthy, incomplete])
    check(
        "Trace cost summary labels the known amount as partial",
        trace_summary["platform_price_snapshot_direct_cost_cny"] is None
        and trace_summary["known_partial_cost_cny"] == 0.001
        and trace_summary["complete"] is False,
    )
    explanation = observability._trace_cost_explanation(
        [
            observability._enrich_model_call(healthy, "mixed-session"),
            observability._enrich_model_call(incomplete, "mixed-session"),
        ]
    )
    check(
        "Trace explanation also keeps partial cost out of the complete field",
        explanation["cost_scope"][
            "platform_price_snapshot_direct_cost_cny"
        ]
        is None
        and explanation["cost_scope"]["known_partial_cost_cny"] == 0.001
        and explanation["cost_scope"]["cost_complete"] is False,
    )
    contradictory = _raw_attempt(
        trace_id="top-contradictory",
        output=30,
        reasoning=31,
    )
    top_items = observability._top_provider_actual_traces(
        [healthy, contradictory], limit=5
    )
    check(
        "high-cost Trace list excludes internally contradictory Usage",
        [item["trace_id"] for item in top_items] == ["mixed-healthy"],
    )


def assert_reporting_truthfulness(db_path: Path) -> None:
    original_get_conn = observability._get_conn
    original_thresholds = observability.get_budget_thresholds
    original_evaluation_summary = observability.evaluation_summary
    original_get_chat_trace = observability.get_chat_trace

    def fixture_conn() -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    observability._get_conn = fixture_conn
    observability.get_budget_thresholds = lambda: {"per_call_threshold_cny": 0.0001}
    observability.evaluation_summary = lambda: {}
    observability.get_chat_trace = lambda _trace_id: None
    try:
        calls = observability._fetch_model_calls(
            "2026-08-01", "2026-08-01 23:59:59.999999"
        )
        aggregate = observability._aggregate_model_calls(calls)
        check("normal aggregate counts only confirmed in-range Provider requests", aggregate["calls"] == 4)
        check("price-missing request keeps Token but has no priced cost", aggregate["provider_actual_calls"] == 3 and aggregate["provider_actual_priced_calls"] == 2)
        check("cost completeness is false instead of silently complete", aggregate["cost_complete"] is False)

        reporting = observability._fetch_reporting_model_calls(
            "2026-08-01", "2026-08-01 23:59:59.999999"
        )
        quality = observability._data_quality_summary(reporting)
        check("NULL and malformed timestamps remain visible", quality["invalid_timestamp_count"] == 2)
        invalid_samples = [
            sample
            for sample in quality["anomaly_attempts"]
            if sample["created_at_status"] == "invalid"
        ]
        check("invalid timestamps show their real exclusion destination", (
            len(invalid_samples) == 2
            and all(sample["included_in_provider_summary"] is False for sample in invalid_samples)
            and all(
                sample["exclusion_reason"] == "invalid_timestamp_range_unassignable"
                for sample in invalid_samples
            )
        ))
        check("orphaned send is visible but unresolved", quality["provider_send_unconfirmed_count"] == 1 and quality["orphaned_pending_count"] == 1)
        check("Provider usage mismatch is inherited or derived", quality["provider_usage_inconsistent_count"] == 1)
        check("Provider actual Token without price is separate", quality["provider_actual_price_missing_count"] == 1)
        check("sent request without usage is separate", quality["unavailable_usage_count"] == 1)
        check("hard data defects win status priority", quality["data_quality_status"] == "data_quality_error")
        check("Reasoning relationship is data-derived", quality["reasoning_is_output_subset"] is False)

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
        check("overview keeps four confirmed Provider requests", overview["provider_request_count"] == 4)
        check("overview exposes seven Trace groups distinctly", overview["trace_group_count"] == 7)
        check("shared range counts can match while quality still blocks consistent", overview["scope_consistent"] is True and overview["statistics_status"] == "data_quality_error")
        check("overview exposes every anomaly bucket", (
            overview["invalid_timestamp_count"],
            overview["provider_send_unconfirmed_count"],
            overview["provider_usage_inconsistent_count"],
            overview["provider_actual_usage_incomplete_count"],
            overview["provider_actual_price_missing_count"],
            overview["unavailable_usage_count"],
        ) == (2, 1, 1, 0, 1, 1))
        check("per-call denominator uses only priced requests", overview["known_priced_request_count"] == 2)
        check("incomplete cost prevents per-call threshold alert", overview["per_call_cost_complete"] is False and not any(item.get("type") == "per_call" for item in overview["alerts"]))

        page = observability._list_trace_page(
            range_key="custom",
            start="2026-08-01",
            end="2026-08-01",
            limit=20,
        )
        check("Trace list and overview share Provider count", page["provider_request_count"] == overview["provider_request_count"])
        check("Trace status inherits quality rather than metadata filters", page["statistics_status"] == "data_quality_error")
        by_id = {item["trace_id"]: item for item in page["traces"]}
        orphan = by_id["orphaned"]
        check("orphaned model-only Trace is not silent", orphan["provider_attempt_count"] == 1 and orphan["provider_request_count"] == 0)
        check("orphaned attempt is not called no-model or not-applicable", orphan["no_model_calls"] is False and orphan["cost_status"] == "reconciliation_attention")
        check("invalid-time Trace shows exclusion destination", by_id["invalid-time"]["invalid_timestamp_count"] == 1 and by_id["invalid-time"]["provider_request_count"] == 0)
        missing = by_id["price-missing"]
        check("missing snapshot cost remains unavailable", missing["provider_actual_priced_calls"] == 0 and missing["platform_price_snapshot_direct_cost_cny"] is None)
        contradictory = by_id["usage-inconsistent"]
        check(
            "Trace card never labels contradictory Usage cost as complete",
            contradictory["data_quality_status"] == "data_quality_error"
            and contradictory["cost_status"] == "data_quality_error"
            and contradictory["cost_complete"] is False
            and contradictory["platform_price_snapshot_direct_cost_cny"] is None
            and contradictory["known_partial_cost_cny"] == 0.001,
        )

        check("history total excludes invalid timestamps too", overview["history_total"]["calls"] == 4)
        invalid_row = next(
            row for row in reporting if row.get("trace_id") == "invalid-time"
        )
        invalid_detail = observability._enrich_model_call(invalid_row, None)
        check("Trace detail excludes malformed time explicitly", invalid_detail["included_in_provider_summary"] is False and invalid_detail["aggregate_exclusion_reason"] == "invalid_timestamp_range_unassignable")

        distribution = asyncio.run(
            observability.distribution(group_by="model", start=None, end=None)
        )
        check("distribution excludes invalid timestamps from normal calls", sum(item["provider_request_count"] for item in distribution["items"]) == 4)
        check("distribution still reports invalid-time evidence", distribution["data_quality"]["invalid_timestamp_count"] == 2 and distribution["statistics_status"] == "data_quality_error")
        check(
            "distribution does not call contradictory Usage Token-complete",
            all(item["token_complete"] is False for item in distribution["items"]),
        )
        trends = asyncio.run(
            observability.trends(group_by="hour", start=None, end=None)
        )
        check("trends excludes invalid timestamps from normal calls", sum(item["provider_request_count"] for item in trends["items"]) == 4)
        check("trends never creates an unknown-time normal bucket", all(item["period"] != "unknown" for item in trends["items"]))
        check("trends still reports invalid-time evidence", trends["data_quality"]["invalid_timestamp_count"] == 2 and trends["statistics_status"] == "data_quality_error")
        inconsistent_period = next(
            item for item in trends["items"] if item["period"].endswith("10:00")
        )
        check(
            "trend period does not call contradictory Usage Token-complete",
            inconsistent_period["token_complete"] is False
            and inconsistent_period["data_quality_status"] == "data_quality_error",
        )

        check("true mismatch has priority over clean quality", observability._statistics_status(scope_consistent=False, data_quality_status="normal") == "scope_mismatch")

        for endpoint in (observability.distribution, observability.trends):
            try:
                asyncio.run(endpoint(start="bad-time", end="2026-08-01"))
            except HTTPException as exc:
                check("invalid reporting time returns HTTP 400", exc.status_code == 400)
            else:
                raise AssertionError("invalid reporting time must be HTTP 400")
            try:
                asyncio.run(endpoint(start="2026-08-02", end="2026-08-01"))
            except HTTPException as exc:
                check("reversed reporting range returns HTTP 400", exc.status_code == 400)
            else:
                raise AssertionError("reversed reporting range must be HTTP 400")

        conn = fixture_conn()
        insert_attempt(
            conn,
            trace_id="malformed-endpoint",
            created_at="2026-08-01T11:30:00+08:00",
            output="not-an-integer",
            reasoning="not-an-integer",
            total=60,
        )
        conn.commit()
        conn.close()
        malformed_overview = asyncio.run(
            observability.overview(
                start="2026-08-01",
                end="2026-08-01",
                model_id=None,
                stage=None,
                trace_id=None,
                range_key="custom",
            )
        )
        malformed_page = observability._list_trace_page(
            range_key="custom",
            start="2026-08-01",
            end="2026-08-01",
            limit=20,
        )
        malformed_distribution = asyncio.run(
            observability.distribution(group_by="model", start=None, end=None)
        )
        malformed_trends = asyncio.run(
            observability.trends(group_by="hour", start=None, end=None)
        )
        check(
            "overview explains malformed numeric evidence instead of raising 500",
            malformed_overview["provider_request_count"] == 5
            and malformed_overview["provider_actual_usage_incomplete_count"] == 1
            and malformed_overview["statistics_status"] == "data_quality_error",
        )
        malformed_card = next(
            item
            for item in malformed_page["traces"]
            if item["trace_id"] == "malformed-endpoint"
        )
        check(
            "Trace list keeps malformed numeric attempt locatable",
            malformed_card["provider_request_count"] == 1
            and malformed_card["data_quality_status"] == "data_quality_error"
            and malformed_card["cost_complete"] is False,
        )
        check(
            "distribution explains malformed numeric evidence instead of raising 500",
            sum(
                item["provider_request_count"]
                for item in malformed_distribution["items"]
            )
            == 5
            and malformed_distribution["statistics_status"]
            == "data_quality_error",
        )
        check(
            "trends explains malformed numeric evidence instead of raising 500",
            sum(
                item["provider_request_count"]
                for item in malformed_trends["items"]
            )
            == 5
            and malformed_trends["statistics_status"] == "data_quality_error",
        )

        conn = fixture_conn()
        insert_attempt(
            conn,
            trace_id="nonfinite-endpoint",
            created_at="2026-08-01T12:00:00+08:00",
            output=float("nan"),
            reasoning=float("inf"),
            total=60,
        )
        conn.commit()
        conn.close()
        nonfinite_payloads = {
            "overview": asyncio.run(
                observability.overview(
                    start="2026-08-01",
                    end="2026-08-01",
                    model_id=None,
                    stage=None,
                    trace_id=None,
                    range_key="custom",
                )
            ),
            "traces": observability._list_trace_page(
                range_key="custom",
                start="2026-08-01",
                end="2026-08-01",
                limit=20,
            ),
            "distribution": asyncio.run(
                observability.distribution(
                    group_by="model", start=None, end=None
                )
            ),
            "trends": asyncio.run(
                observability.trends(
                    group_by="hour", start=None, end=None
                )
            ),
        }
        for endpoint_name, payload in nonfinite_payloads.items():
            serialized = json.dumps(payload, allow_nan=False)
            check(
                f"{endpoint_name} response is strict-JSON safe with non-finite history",
                "NaN" not in serialized
                or "invalid_non_finite_number:NaN" in serialized,
            )
        if HAS_RUNTIME_DEPENDENCIES:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            test_app = FastAPI()
            test_app.include_router(observability.router)
            endpoint_urls = (
                "/api/observability/overview?range_key=custom&start=2026-08-01&end=2026-08-01",
                "/api/observability/traces?range_key=custom&start=2026-08-01&end=2026-08-01&limit=20",
                "/api/observability/distribution?group_by=model",
                "/api/observability/trends?group_by=hour",
            )
            with TestClient(test_app) as client:
                responses = [client.get(url) for url in endpoint_urls]
            check(
                "all Trace reporting HTTP endpoints serialize non-finite history",
                all(response.status_code == 200 for response in responses),
            )
        else:
            DEFERRED_CHECKS.append(
                "overview/traces/distribution/trends HTTP JSON checks require FastAPI TestClient"
            )
    finally:
        observability._get_conn = original_get_conn
        observability.get_budget_thresholds = original_thresholds
        observability.evaluation_summary = original_evaluation_summary
        observability.get_chat_trace = original_get_chat_trace


def assert_quality_status_contract() -> None:
    healthy = _raw_attempt(trace_id="quality-healthy")
    missing_price = _raw_attempt(trace_id="quality-price", priced=False)
    unavailable = _raw_attempt(
        trace_id="quality-unavailable",
        usage_status="unavailable_done_without_usage",
        hit=None,
        miss=None,
        output=None,
        reasoning=None,
        total=None,
        priced=False,
    )
    orphaned = _raw_attempt(
        trace_id="quality-orphan",
        usage_status="orphaned_pending",
        request_sent=None,
        include=False,
        hit=None,
        miss=None,
        output=None,
        reasoning=None,
        total=None,
        priced=False,
    )
    inconsistent = _raw_attempt(
        trace_id="quality-inconsistent",
        reasoning=31,
        output=30,
    )
    incomplete = _raw_attempt(trace_id="quality-incomplete", hit=None)

    check("healthy ledger quality is normal", observability._data_quality_summary([healthy])["data_quality_status"] == "normal")
    for label, row in (
        ("price only", missing_price),
        ("usage unavailable only", unavailable),
        ("orphaned only", orphaned),
    ):
        quality = observability._data_quality_summary([row])
        check(f"{label} requires reconciliation attention", quality["data_quality_status"] == "reconciliation_attention")
    inconsistent_quality = observability._data_quality_summary([inconsistent])
    check("Usage contradiction is data quality error", inconsistent_quality["data_quality_status"] == "data_quality_error" and inconsistent_quality["reasoning_is_output_subset"] is False)
    incomplete_quality = observability._data_quality_summary([incomplete])
    check("incomplete provider actual is data quality error", incomplete_quality["data_quality_status"] == "data_quality_error" and incomplete_quality["provider_actual_usage_incomplete_count"] == 1)

    partial_price = _raw_attempt(trace_id="quality-partial-price")
    partial_price["price_snapshot"] = {"currency": "CNY"}
    partial_price["usage_normalized"]["price_snapshot"] = {"currency": "CNY"}
    partial_price_quality = observability._data_quality_summary([partial_price])
    check(
        "incomplete three-price snapshot is classified as price missing",
        partial_price_quality["provider_actual_price_missing_count"] == 1
        and partial_price_quality["provider_actual_cost_unavailable_count"] == 0,
    )
    missing_amount = _raw_attempt(trace_id="quality-missing-amount")
    missing_amount["estimated_cost_cny"] = None
    missing_amount["usage_normalized"]["calculated_direct_cost"] = None
    missing_amount_quality = observability._data_quality_summary([missing_amount])
    check(
        "complete prices with missing amount use a separate cost-unavailable bucket",
        missing_amount_quality["provider_actual_price_missing_count"] == 0
        and missing_amount_quality["provider_actual_cost_unavailable_count"] == 1,
    )

    invalid_price = _raw_attempt(trace_id="quality-invalid-price")
    invalid_price["price_snapshot"] = {
        **PRICE,
        "input_price_per_1m": float("nan"),
    }
    invalid_price["usage_normalized"]["price_snapshot"] = dict(
        invalid_price["price_snapshot"]
    )
    invalid_price["usage_normalized"]["calculated_direct_cost"] = float("nan")
    invalid_price["estimated_cost_cny"] = float("nan")
    invalid_price_quality = observability._data_quality_summary([invalid_price])
    invalid_price_summary = observability._aggregate_model_calls([invalid_price])
    check(
        "non-finite price snapshot is unavailable rather than budget-safe",
        invalid_price_quality["provider_actual_price_missing_count"] == 1
        and invalid_price_summary["platform_price_snapshot_direct_cost_cny"] is None
        and invalid_price_summary["cost_complete"] is False,
    )

    invalid_amount = _raw_attempt(trace_id="quality-invalid-amount")
    invalid_amount["usage_normalized"]["calculated_direct_cost"] = -0.01
    invalid_amount["estimated_cost_cny"] = -0.01
    invalid_amount_quality = observability._data_quality_summary([invalid_amount])
    check(
        "negative calculated amount is a separate cost-unavailable anomaly",
        invalid_amount_quality["provider_actual_price_missing_count"] == 0
        and invalid_amount_quality["provider_actual_cost_unavailable_count"] == 1
        and observability._provider_direct_cost(invalid_amount) is None,
    )

    formal = _raw_attempt(
        trace_id="quality-formal-reasons",
        inconsistency_reasons=["formal_fixture_reason"],
    )
    legacy = _raw_attempt(trace_id="quality-legacy-reasons")
    legacy["usage_normalized"][
        "provider_usage_inconsistent_reasons"
    ] = ["legacy_fixture_reason"]
    reason_quality = observability._data_quality_summary([formal, legacy])
    reasons = {
        reason
        for item in reason_quality["anomaly_attempts"]
        for reason in item["provider_usage_inconsistency_reasons"]
    }
    check("formal and legacy inconsistency fields remain readable", reasons == {"formal_fixture_reason", "legacy_fixture_reason"})


def assert_provider_identity_contract() -> None:
    """A sent request is not fully reconciled without unique response identity."""
    missing_id = _raw_attempt(trace_id="identity-missing")
    missing_id["provider_request_id"] = None
    missing_id["usage_normalized"]["provider_request_id"] = None
    missing_id["usage_normalized"]["provider_request_id_obtained"] = False
    missing_id["usage_normalized"][
        "provider_request_identity_status"
    ] = "unavailable_provider_request_id"
    missing_id["usage_normalized"][
        "reconciliation_status"
    ] = "provider_request_id_unavailable"
    missing_quality = observability._data_quality_summary([missing_id])
    missing_summary = observability._aggregate_model_calls([missing_id])
    check(
        "missing Provider request ID keeps the confirmed request counted",
        missing_summary["calls"] == 1,
    )
    check(
        "missing Provider request ID requires reconciliation attention",
        missing_quality["data_quality_status"] == "reconciliation_attention"
        and missing_quality["provider_request_id_unavailable_count"] == 1
        and "provider_request_id_unavailable"
        in missing_quality["anomaly_attempts"][0]["issue_codes"],
    )

    invalid_id = _raw_attempt(trace_id="identity-invalid-type")
    invalid_id["provider_request_id"] = float("nan")
    invalid_id["usage_normalized"]["provider_request_id"] = {
        "not": "a Provider request ID"
    }
    invalid_id_quality = observability._data_quality_summary([invalid_id])
    check(
        "non-string Provider request identity is unavailable evidence",
        invalid_id_quality["provider_request_id_unavailable_count"] == 1
        and invalid_id_quality["data_quality_status"]
        == "reconciliation_attention",
    )
    fallback_valid_id = _raw_attempt(trace_id="identity-valid-second-candidate")
    fallback_valid_id["provider_request_id"] = float("inf")
    check(
        "invalid row identity cannot hide valid normalized identity",
        observability._provider_request_id_evidence(fallback_valid_id)
        == "request-identity-valid-second-candidate",
    )

    duplicate_a = _raw_attempt(
        trace_id="identity-duplicate-a", attempt_key="duplicate"
    )
    duplicate_b = _raw_attempt(
        trace_id="identity-duplicate-b", attempt_key="duplicate"
    )
    duplicate_quality = observability._data_quality_summary(
        [duplicate_a, duplicate_b]
    )
    check(
        "duplicate Provider request IDs are a hard identity error",
        duplicate_quality["data_quality_status"] == "data_quality_error"
        and duplicate_quality["duplicate_provider_request_id_count"] == 2
        and all(
            "duplicate_provider_request_id" in sample["issue_codes"]
            for sample in duplicate_quality["anomaly_attempts"]
        ),
    )

    identity_conflict = _raw_attempt(trace_id="identity-conflict")
    identity_conflict["usage_normalized"]["provider_id_conflict"] = {
        "first": "request-one",
        "later": "request-two",
    }
    conflict_quality = observability._data_quality_summary([identity_conflict])
    check(
        "one SDK attempt with conflicting Provider IDs is a data-quality error",
        conflict_quality["data_quality_status"] == "data_quality_error"
        and conflict_quality["provider_request_identity_conflict_count"] == 1,
    )
    conflict_status = _raw_attempt(trace_id="identity-conflict-status")
    conflict_status["usage_normalized"][
        "provider_request_identity_status"
    ] = "provider_request_id_conflict"
    conflict_status_quality = observability._data_quality_summary(
        [conflict_status]
    )
    check(
        "persisted identity conflict status is inherited by reporting",
        conflict_status_quality["provider_request_identity_conflict_count"]
        == 1
        and conflict_status_quality["data_quality_status"]
        == "data_quality_error",
    )
    implicit_conflict = _raw_attempt(trace_id="identity-implicit-conflict")
    implicit_conflict["provider_request_id"] = "request-top-level"
    implicit_conflict["usage_normalized"][
        "provider_request_id"
    ] = "request-normalized"
    implicit_conflict_quality = observability._data_quality_summary(
        [implicit_conflict]
    )
    implicit_conflict_enriched = observability._enrich_model_call(
        implicit_conflict, "session-identity-implicit-conflict"
    )
    check(
        "two distinct valid Provider IDs derive an identity conflict",
        implicit_conflict_quality[
            "provider_request_identity_conflict_count"
        ]
        == 1
        and implicit_conflict_quality["data_quality_status"]
        == "data_quality_error",
    )
    check(
        "Trace row inherits implicit Provider identity conflict",
        implicit_conflict_enriched["reconciliation"]["reason"]
        == "provider_request_identity_conflict"
        and implicit_conflict_enriched["reconciliation"]["evidence"][
            "provider_request_id_candidate_count"
        ]
        == 2,
    )

    missing_model = _raw_attempt(trace_id="identity-model-missing")
    missing_model["usage_normalized"].pop("provider_actual_model")
    model_quality = observability._data_quality_summary([missing_model])
    model_summary = observability._aggregate_model_calls([missing_model])
    check(
        "requested model never impersonates missing Provider actual model",
        observability._provider_model(missing_model)
        == "actual_model_unverified"
        and "actual_model_unverified" in model_summary["by_model"],
    )
    check(
        "missing actual model remains a locatable reconciliation anomaly",
        model_quality["data_quality_status"] == "reconciliation_attention"
        and model_quality["provider_actual_model_unverified_count"] == 1,
    )
    invalid_model = _raw_attempt(trace_id="identity-model-invalid")
    invalid_model["usage_normalized"]["provider_actual_model"] = float("nan")
    invalid_model["usage_normalized"]["provider_response_model"] = [
        "not-a-model-id"
    ]
    check(
        "non-string actual model cannot impersonate Provider evidence",
        observability._provider_model(invalid_model)
        == "actual_model_unverified"
        and observability._data_quality_summary([invalid_model])[
            "provider_actual_model_unverified_count"
        ]
        == 1,
    )
    model_conflict = _raw_attempt(trace_id="identity-model-conflict")
    model_conflict["usage_normalized"][
        "provider_response_model"
    ] = "deepseek-other-model"
    model_conflict_quality = observability._data_quality_summary(
        [model_conflict]
    )
    model_conflict_enriched = observability._enrich_model_call(
        model_conflict, "session-identity-model-conflict"
    )
    check(
        "conflicting Provider actual models are never called verified",
        observability._provider_model(model_conflict)
        == "actual_model_conflict"
        and model_conflict_quality["provider_actual_model_conflict_count"]
        == 1
        and model_conflict_quality["data_quality_status"]
        == "data_quality_error",
    )
    check(
        "Trace row inherits Provider actual model conflict",
        model_conflict_enriched["reconciliation"]["reason"]
        == "provider_actual_model_conflict"
        and model_conflict_enriched["provider_actual_model"] is None,
    )

    with tempfile.TemporaryDirectory(prefix="yiai-s10f-global-id-") as temp_dir:
        db_path = Path(temp_dir) / "global-id.db"
        conn = sqlite3.connect(db_path)
        create_schema(conn)
        insert_attempt(
            conn,
            trace_id="global-id-aug1",
            attempt_key="cross-day-duplicate",
            created_at="2026-08-01T08:00:00+08:00",
        )
        insert_attempt(
            conn,
            trace_id="global-id-aug2",
            attempt_key="cross-day-duplicate",
            created_at="2026-08-02T08:00:00+08:00",
        )
        conn.commit()
        conn.close()
        original_get_conn = observability._get_conn

        def global_id_conn() -> sqlite3.Connection:
            fixture = sqlite3.connect(db_path)
            fixture.row_factory = sqlite3.Row
            return fixture

        observability._get_conn = global_id_conn
        try:
            scoped = observability._fetch_reporting_model_calls(
                "2026-08-01", "2026-08-01 23:59:59.999999"
            )
            scoped_quality = observability._data_quality_summary(scoped)
            check(
                "cross-day duplicate ID cannot look unique in a single-day report",
                len(scoped) == 1
                and scoped[0]["provider_request_id_global_occurrences"] == 2
                and scoped_quality["duplicate_provider_request_id_count"] == 1
                and scoped_quality["data_quality_status"]
                == "data_quality_error",
            )
        finally:
            observability._get_conn = original_get_conn

    metadata_conflicts = (
        _raw_attempt(trace_id="metadata-include-false", include=False),
        _raw_attempt(trace_id="metadata-retest", stage="retest"),
        _raw_attempt(trace_id="metadata-blocked"),
    )
    metadata_conflicts[2]["status"] = "blocked"
    for row in metadata_conflicts:
        quality = observability._data_quality_summary([row])
        aggregate = observability._aggregate_model_calls([row])
        enriched = observability._enrich_model_call(
            row, f"session-{row['trace_id']}"
        )
        check(
            "confirmed Provider request stays counted despite contradictory metadata",
            aggregate["calls"] == 1,
        )
        check(
            "confirmed Provider metadata conflict is never consistent",
            quality[
                "confirmed_provider_attempt_metadata_conflict_count"
            ]
            == 1
            and quality["data_quality_status"] == "data_quality_error"
            and enriched["reconciliation"]["reason"]
            == "confirmed_provider_attempt_metadata_conflict",
        )

    misclassified = _raw_attempt(
        trace_id="metadata-misclassified", include=False, stage="retest"
    )
    misclassified["record_kind"] = "logical_aggregate"
    misclassified["usage_normalized"]["record_kind"] = "logical_aggregate"
    misclassified_quality = observability._data_quality_summary(
        [misclassified]
    )
    misclassified_enriched = observability._enrich_model_call(
        misclassified, "session-metadata-misclassified"
    )
    check(
        "misclassified sent evidence is not upgraded into Provider totals",
        observability._aggregate_model_calls([misclassified])["calls"] == 0,
    )
    check(
        "misclassified sent evidence remains a hard locatable anomaly",
        misclassified_quality[
            "confirmed_provider_evidence_misclassified_count"
        ]
        == 1
        and misclassified_quality["data_quality_status"]
        == "data_quality_error"
        and misclassified_enriched["reconciliation"]["reason"]
        == "confirmed_provider_evidence_misclassified",
    )
    true_logical = {
        "trace_id": "metadata-true-logical",
        "record_kind": "logical_aggregate",
        "stage": "retest",
        "created_at": "2026-08-01T10:00:00+08:00",
        "usage_normalized": {
            "record_kind": "logical_aggregate",
            "include_in_provider_aggregate": False,
        },
    }
    check(
        "true logical aggregate without sent evidence remains normally excluded",
        observability._data_quality_summary([true_logical])[
            "data_quality_status"
        ]
        == "normal"
        and observability._aggregate_model_calls([true_logical])["calls"]
        == 0,
    )

    originals = {
        "fetch": observability._fetch_model_calls,
        "reporting": observability._fetch_reporting_model_calls,
        "trace_page": observability._list_trace_page,
        "thresholds": observability.get_budget_thresholds,
        "evaluation": observability.evaluation_summary,
    }
    try:
        observability._fetch_model_calls = (
            lambda *_args, **_kwargs: [missing_id]
        )
        observability._fetch_reporting_model_calls = (
            lambda *_args, **_kwargs: [missing_id]
        )
        observability._list_trace_page = lambda **kwargs: {
            "trace_group_count": 1,
            "provider_request_count": 1,
            "anomaly_attempt_count": 1,
            "start": kwargs.get("start"),
            "end": kwargs.get("end"),
        }
        observability.get_budget_thresholds = lambda: {}
        observability.evaluation_summary = lambda: {}
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
        check(
            "Overview cannot call a missing Provider identity consistent",
            overview["provider_request_count"] == 1
            and overview["statistics_status"] == "reconciliation_attention"
            and overview["provider_request_id_unavailable_count"] == 1
            and overview["budget_status"] == "unavailable",
        )
    finally:
        observability._fetch_model_calls = originals["fetch"]
        observability._fetch_reporting_model_calls = originals["reporting"]
        observability._list_trace_page = originals["trace_page"]
        observability.get_budget_thresholds = originals["thresholds"]
        observability.evaluation_summary = originals["evaluation"]


def assert_malformed_numeric_evidence_contract() -> None:
    """Malformed historical numerics stay explainable instead of causing 500s."""
    core_cases = (
        ("hit", "cache_hit_input_tokens_not_integer"),
        ("miss", "cache_miss_input_tokens_not_integer"),
        ("output", "output_tokens_not_integer"),
        ("total", "total_tokens_not_integer"),
    )
    for field, expected_reason in core_cases:
        kwargs = {field: "not-an-integer"}
        row = _raw_attempt(trace_id=f"malformed-{field}", **kwargs)
        quality = observability._data_quality_summary([row])
        aggregate = observability._aggregate_model_calls([row])
        explanation = observability._trace_cost_explanation(
            [observability._enrich_model_call(row, f"session-malformed-{field}")]
        )
        reasons = {
            reason
            for sample in quality["anomaly_attempts"]
            for reason in sample["provider_usage_inconsistency_reasons"]
        }
        check(
            f"malformed {field} remains a counted Provider request",
            aggregate["calls"] == 1,
        )
        check(
            f"malformed {field} is unavailable rather than zero",
            aggregate["token_unavailable_calls"] == 1
            and aggregate["total_tokens"] is None
            and aggregate["cost_complete"] is False,
        )
        check(
            f"malformed {field} becomes an explicit data-quality error",
            quality["data_quality_status"] == "data_quality_error"
            and quality["provider_actual_usage_incomplete_count"] == 1
            and expected_reason in reasons,
        )
        check(
            f"malformed {field} Trace explanation stays available",
            explanation["cost_scope"]["data_quality_status"]
            == "data_quality_error"
            and explanation["recommendation"]["code"]
            == "evidence_insufficient",
        )

    reasoning = _raw_attempt(
        trace_id="malformed-reasoning", reasoning="not-an-integer"
    )
    reasoning_quality = observability._data_quality_summary([reasoning])
    reasoning_aggregate = observability._aggregate_model_calls([reasoning])
    reasoning_explanation = observability._trace_cost_explanation(
        [observability._enrich_model_call(reasoning, "session-malformed-reasoning")]
    )
    check(
        "malformed Reasoning is an explicit Usage inconsistency",
        reasoning_quality["data_quality_status"] == "data_quality_error"
        and reasoning_quality["provider_usage_inconsistent_count"] == 1
        and reasoning_quality["reasoning_is_output_subset"] is None,
    )
    check(
        "malformed Reasoning does not break aggregate or recommendation",
        reasoning_aggregate["calls"] == 1
        and reasoning_aggregate["reasoning_tokens"] is None
        and reasoning_aggregate["cost_complete"] is False
        and reasoning_explanation["recommendation"]["code"]
        == "evidence_insufficient",
    )

    estimated = _raw_attempt(
        trace_id="malformed-estimated-amount",
        usage_status="estimated",
        priced=False,
    )
    estimated["usage_source"] = "estimated"
    estimated["usage_normalized"]["token_source"] = "estimated"
    estimated["estimated_cost_cny"] = "not-a-cost"
    estimated_aggregate = observability._aggregate_model_calls([estimated])
    estimated_explanation = observability._trace_cost_explanation(
        [observability._enrich_model_call(estimated, "session-malformed-estimated")]
    )
    check(
        "malformed historical estimate is unavailable rather than a 500 or zero",
        estimated_aggregate["calls"] == 1
        and estimated_aggregate["estimated_calls"] == 1
        and estimated_aggregate["estimated_amount_unavailable_calls"] == 1
        and estimated_aggregate["estimated_cost_cny"] == 0.0
        and estimated_aggregate["cost_complete"] is False,
    )
    check(
        "malformed historical estimate keeps Trace explanation available",
        estimated_explanation["cost_scope"]["estimated_calls"] == 1
        and estimated_explanation["cost_scope"]["estimated_cost_cny"] == 0.0,
    )

    malformed_reasons = _raw_attempt(
        trace_id="malformed-inconsistency-reasons"
    )
    malformed_reasons["usage_normalized"][
        "provider_usage_inconsistency_reasons"
    ] = float("nan")
    malformed_reason_quality = observability._data_quality_summary(
        [malformed_reasons]
    )
    check(
        "scalar Usage inconsistency reasons become an explicit error, not a 500",
        malformed_reason_quality["data_quality_status"]
        == "data_quality_error"
        and "provider_usage_inconsistency_reason_malformed_type"
        in malformed_reason_quality["anomaly_attempts"][0][
            "provider_usage_inconsistency_reasons"
        ],
    )

    unhashable_stage = _raw_attempt(trace_id="malformed-stage")
    unhashable_stage["stage"] = {"not": "a stage label"}
    stage_summary = observability._aggregate_model_calls([unhashable_stage])
    check(
        "unhashable stage evidence groups under unknown without raising",
        stage_summary["calls"] == 1 and "unknown" in stage_summary["by_stage"],
    )

    malformed_trace = _raw_attempt(trace_id="malformed-trace-contract")
    malformed_trace["usage_normalized"].pop("provider_actual_model")
    malformed_trace["usage_normalized"]["requested_model"] = {
        "not": "a requested model"
    }
    malformed_trace["usage_normalized"]["cost_contract"] = True
    enriched_malformed_trace = observability._enrich_model_call(
        malformed_trace, "session-malformed-trace-contract"
    )
    malformed_trace_explanation = observability._trace_cost_explanation(
        [enriched_malformed_trace]
    )
    check(
        "malformed requested model cannot become an unhashable Trace key",
        malformed_trace_explanation["chain"][0]["display_model"]
        == "actual_model_unverified"
        and malformed_trace_explanation["chain"][0]["requested_model"]
        == "unknown",
    )
    check(
        "malformed cost contract is explicit and never raises Trace detail 500",
        enriched_malformed_trace["cost_formula"].startswith(
            "cost_contract_malformed_type"
        ),
    )


def assert_trace_detail_json_safety() -> None:
    """Historical NaN/Infinity evidence remains visible and HTTP serializable."""
    row = _raw_attempt(trace_id="nonfinite-trace")
    row["usage_normalized"]["provider_usage_raw"] = {
        "nested_nan": float("nan"),
        "nested_positive_infinity": float("inf"),
        "nested_negative_infinity": float("-inf"),
    }
    row["usage_normalized"]["price_snapshot"] = {
        **PRICE,
        "source_probe": float("nan"),
    }
    row["usage_normalized"]["provider_usage_inconsistency_reasons"] = True
    row["usage_normalized"]["cost_contract"] = True
    originals = {
        "trace": observability.get_chat_trace,
        "calls": observability._fetch_reporting_model_calls,
        "mcp": observability.get_mcp_call_audits_for_trace,
        "messages": observability.list_chat_messages,
        "events": observability.list_trace_events,
        "evaluation": observability.get_evaluation_run_by_trace_id,
    }
    observability.get_chat_trace = lambda _trace_id: {
        "trace_id": "nonfinite-trace",
        "session_id": None,
        "status": "success",
        "created_at": "2026-08-01T08:00:00+08:00",
        "updated_at": "2026-08-01T08:00:01+08:00",
        "version_snapshot": {"raw_probe": float("inf")},
    }
    observability._fetch_reporting_model_calls = (
        lambda _start, _end, **_kwargs: [row]
    )
    observability.get_mcp_call_audits_for_trace = lambda _trace_id: []
    observability.list_chat_messages = lambda _session_id: []
    observability.list_trace_events = lambda _trace_id: []
    observability.get_evaluation_run_by_trace_id = lambda _trace_id: None
    try:
        detail = asyncio.run(observability.trace_detail("nonfinite-trace"))
        serialized = json.dumps(detail, allow_nan=False)
        check(
            "Trace detail replaces NaN with an explicit invalid-evidence marker",
            "invalid_non_finite_number:NaN" in serialized,
        )
        check(
            "Trace detail replaces both Infinity signs without coercing to zero",
            "invalid_non_finite_number:Infinity" in serialized
            and "invalid_non_finite_number:-Infinity" in serialized,
        )
        check(
            "Trace detail preserves malformed cost-contract evidence explicitly",
            detail["model_calls"][0]["cost_formula"].startswith(
                "cost_contract_malformed_type"
            ),
        )
        if HAS_RUNTIME_DEPENDENCIES:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            test_app = FastAPI()
            test_app.include_router(observability.router)
            with TestClient(test_app) as client:
                response = client.get(
                    "/api/observability/traces/nonfinite-trace"
                )
            check(
                "Trace detail non-finite historical evidence returns HTTP 200",
                response.status_code == 200
                and response.json()["trace"]["operation_metadata"][
                    "raw_probe"
                ]
                == "invalid_non_finite_number:Infinity",
            )
        else:
            DEFERRED_CHECKS.append(
                "Trace detail HTTP serialization requires FastAPI TestClient"
            )
    finally:
        observability.get_chat_trace = originals["trace"]
        observability._fetch_reporting_model_calls = originals["calls"]
        observability.get_mcp_call_audits_for_trace = originals["mcp"]
        observability.list_chat_messages = originals["messages"]
        observability.list_trace_events = originals["events"]
        observability.get_evaluation_run_by_trace_id = originals["evaluation"]


def assert_budget_quality_contract() -> None:
    original_query = observability._query_period_summary
    original_reporting = observability._fetch_reporting_model_calls
    original_thresholds = observability.get_budget_thresholds
    original_bounds = observability._period_bounds
    observability.get_budget_thresholds = lambda: {
        "daily_threshold_cny": 1.0,
        "monthly_threshold_cny": 10.0,
    }
    observability._period_bounds = lambda: {
        "today": {"start": "2026-08-01", "end": "2026-08-01 23:59:59.999999"},
        "this_month": {"start": "2026-08-01", "end": "2026-08-31 23:59:59.999999"},
    }
    try:
        invalid_price = _raw_attempt(trace_id="budget-invalid-price")
        invalid_price["price_snapshot"] = {
            **PRICE,
            "output_price_per_1m": float("inf"),
        }
        invalid_price["usage_normalized"]["price_snapshot"] = dict(
            invalid_price["price_snapshot"]
        )
        invalid_price["usage_normalized"]["calculated_direct_cost"] = float(
            "inf"
        )
        invalid_price["estimated_cost_cny"] = float("inf")
        scenarios = (
            (
                "healthy",
                _raw_attempt(trace_id="budget-healthy"),
                "available",
                None,
            ),
            (
                "data-quality",
                _raw_attempt(trace_id="budget-incomplete", hit=None),
                "unavailable",
                "budget_data_quality_error",
            ),
            (
                "price-missing",
                _raw_attempt(trace_id="budget-price", priced=False),
                "unavailable",
                "budget_reconciliation_attention",
            ),
            (
                "usage-unavailable",
                _raw_attempt(
                    trace_id="budget-usage",
                    usage_status="unavailable_done_without_usage",
                    hit=None,
                    miss=None,
                    output=None,
                    reasoning=None,
                    total=None,
                    priced=False,
                ),
                "unavailable",
                "budget_reconciliation_attention",
            ),
            (
                "orphaned",
                _raw_attempt(
                    trace_id="budget-orphan",
                    usage_status="orphaned_pending",
                    request_sent=None,
                    include=False,
                    hit=None,
                    miss=None,
                    output=None,
                    reasoning=None,
                    total=None,
                    priced=False,
                ),
                "unavailable",
                "budget_reconciliation_attention",
            ),
            (
                "provider-id-missing",
                (lambda row: (
                    row.__setitem__("provider_request_id", None),
                    row["usage_normalized"].__setitem__(
                        "provider_request_id", None
                    ),
                    row["usage_normalized"].__setitem__(
                        "provider_request_id_obtained", False
                    ),
                    row,
                )[-1])(_raw_attempt(trace_id="budget-provider-id")),
                "unavailable",
                "budget_reconciliation_attention",
            ),
            (
                "actual-model-missing",
                (lambda row: (
                    row["usage_normalized"].pop("provider_actual_model"),
                    row,
                )[-1])(_raw_attempt(trace_id="budget-actual-model")),
                "unavailable",
                "budget_reconciliation_attention",
            ),
            (
                "invalid-price",
                invalid_price,
                "unavailable",
                "budget_reconciliation_attention",
            ),
        )
        for label, row, expected_status, expected_reason in scenarios:
            observability._query_period_summary = (
                lambda _start, _end, selected=row: observability._aggregate_model_calls(
                    [selected]
                )
            )
            observability._fetch_reporting_model_calls = (
                lambda _start, _end, selected=row, **_kwargs: [selected]
            )
            result = observability._check_budget(f"fixture-{label}")
            check(f"budget {label} status is truthful", result["budget_status"] == expected_status)
            if expected_status == "available":
                check("healthy ledger budget exposes known cost", result["today_cost"] is not None and result["month_cost"] is not None and result["alert_level"] == "none")
            else:
                check(f"budget {label} has explicit reason", result["reason_code"] == expected_reason and result["today_cost"] is None and result["month_cost"] is None and result["alert_level"] == "unavailable")
    finally:
        observability._query_period_summary = original_query
        observability._fetch_reporting_model_calls = original_reporting
        observability.get_budget_thresholds = original_thresholds
        observability._period_bounds = original_bounds


def assert_budget_query_failure() -> None:
    original_query = observability._query_period_summary
    original_thresholds = observability.get_budget_thresholds
    observability.get_budget_thresholds = lambda: {}

    def fail_query(_start: str, _end: str) -> dict:
        raise sqlite3.OperationalError("deterministic fixture failure")

    observability._query_period_summary = fail_query
    try:
        result = observability._check_budget("fixture")
        check("budget query failure is unavailable", result["budget_status"] == "unavailable")
        check("budget query failure is never zero or none", result["today_cost"] is None and result["month_cost"] is None and result["alert_level"] == "unavailable")
        check("budget error exposes only a non-sensitive type", result["reason_code"] == "budget_query_failed" and result["error_type"] == "OperationalError")
    finally:
        observability._query_period_summary = original_query
        observability.get_budget_thresholds = original_thresholds


def assert_paid_background_short_circuit(_db_path: Path) -> None:
    """Run the real FastAPI/Agno contract when full dependencies are present."""
    if not HAS_RUNTIME_DEPENDENCIES:
        DEFERRED_CHECKS.append(
            "paid background HTTP checks require the full API dependency image"
        )
        return
    from scripts.s10f_full_dependency_contract import (
        run_full_dependency_contract,
    )

    run_full_dependency_contract(sys.modules[__name__], check)


def main() -> None:
    # Any accidental outbound use is a deterministic test failure even before
    # the deployment-level ``--network none`` boundary is applied.
    original_create_connection = socket.create_connection

    def blocked_connection(*_args, **_kwargs):
        raise AssertionError("network access is forbidden in S10-F deterministic tests")

    with tempfile.TemporaryDirectory(prefix="yiai-s10f-ledger-") as temp_dir:
        db_path = Path(temp_dir) / "s10f.db"
        create_fixture(db_path)
        socket.create_connection = blocked_connection
        try:
            assert_fixed_history_mirror_contract()
            assert_reporting_truthfulness(db_path)
            assert_provider_actual_core_completeness()
            assert_mixed_cost_truthfulness()
            assert_quality_status_contract()
            assert_provider_identity_contract()
            assert_malformed_numeric_evidence_contract()
            assert_trace_detail_json_safety()
            assert_budget_quality_contract()
            assert_budget_query_failure()
            assert_paid_background_short_circuit(db_path)
        finally:
            socket.create_connection = original_create_connection
    if DEFERRED_CHECKS:
        message = (
            "PARTIAL: S10-F deterministic core passed "
            f"({len(CHECKS)} checks); SKIP: "
            + "; ".join(DEFERRED_CHECKS)
        )
        print(message)
        if os.getenv("S10F_REQUIRE_FULL_DEPENDENCIES") == "1":
            raise SystemExit("FAIL: S10-F DEFERRED_CHECKS must be zero")
    else:
        print(
            "PASS: S10-F Trace truthfulness and budget fail-closed "
            f"({len(CHECKS)} checks)"
        )
        print("DEFERRED_CHECKS=0")


if __name__ == "__main__":
    main()
