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
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("PROPERTY_DATA_DIR", tempfile.mkdtemp(prefix="yiai-s10f-data-"))

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

import app.observability as observability
from fastapi import HTTPException

if HAS_RUNTIME_DEPENDENCIES:
    from db import property_db

    property_db.init_db()
    import app.badcases as badcases
    import app.evaluations as evaluations
    import app.model_configs as model_configs
else:
    badcases = evaluations = model_configs = None


CHECKS: list[str] = []
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
) -> None:
    usage = {
        "record_kind": "provider_attempt",
        "local_attempt_id": f"attempt-{trace_id}",
        "provider_request_id": f"request-{trace_id}" if request_sent is True else None,
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
        "calculated_direct_cost": 0.001 if priced else None,
    }
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
            0.001 if priced else None,
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
            f"attempt-{trace_id}",
            f"request-{trace_id}" if request_sent is True else None,
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
            overview["provider_actual_price_missing_count"],
            overview["unavailable_usage_count"],
        ) == (2, 1, 1, 1, 1))
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
    finally:
        observability._get_conn = original_get_conn
        observability.get_budget_thresholds = original_thresholds
        observability.evaluation_summary = original_evaluation_summary
        observability.get_chat_trace = original_get_chat_trace


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


def assert_paid_background_short_circuit(db_path: Path) -> None:
    if not HAS_RUNTIME_DEPENDENCIES:
        CHECKS.append("paid background direct-call checks deferred to real dependency container")
        return
    original_get_conn = observability._get_conn
    original_thresholds = observability.get_budget_thresholds

    def fixture_conn() -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    observability._get_conn = fixture_conn
    observability.get_budget_thresholds = lambda: {}
    counters = {"classify": 0, "darwin": 0, "evaluation": 0, "ab": 0}

    async def classify_model(*_args, **_kwargs):
        counters["classify"] += 1
        raise AssertionError("classification model must not run")

    async def darwin_model(*_args, **_kwargs):
        counters["darwin"] += 1
        raise AssertionError("Darwin model must not run")

    async def evaluation_model(*_args, **_kwargs):
        counters["evaluation"] += 1
        raise AssertionError("Evaluation runtime must not run")

    def ab_model(*_args, **_kwargs):
        counters["ab"] += 1
        raise AssertionError("A/B model must not be built")

    original_classify_load = badcases._load_case
    original_badcase_llm = badcases._llm_generate
    original_db_get_badcase = badcases.db_get_badcase
    original_find_darwin = badcases._find_darwin_skill
    original_start_darwin = badcases.start_darwin_operation
    original_persist_darwin = badcases.persist_darwin_operation
    original_eval_case = evaluations.get_evaluation_case
    original_eval_chat = evaluations._run_real_chat
    original_build_model = model_configs.build_model
    darwin_persisted: list[dict] = []
    fixture_case = {
        "id": 1,
        "status": "pending",
        "title": "S10-F fixture",
        "description": "",
        "feedback_reason": "",
        "original_query": "",
        "ai_response": "",
        "context_json": {},
        "session_id": "fixture-session",
        "category": "other",
    }
    try:
        badcases._load_case = lambda _case_id: dict(fixture_case)
        badcases._llm_generate = classify_model
        try:
            asyncio.run(badcases.classify_badcase(1, badcases.ClassifyRequest(auto=True)))
        except HTTPException as exc:
            check("Badcase classification fails closed with 503", exc.status_code == 503)
        else:
            raise AssertionError("Badcase classification must fail closed")

        darwin_case = {**fixture_case, "status": "classified"}
        badcases.db_get_badcase = lambda _case_id: dict(darwin_case)
        badcases._find_darwin_skill = lambda: None
        badcases.start_darwin_operation = lambda **_kwargs: None
        badcases.persist_darwin_operation = lambda **kwargs: darwin_persisted.append(kwargs)
        badcases._llm_generate = darwin_model
        try:
            asyncio.run(badcases.darwin_fix(1, badcases.DarwinFixRequest()))
        except HTTPException as exc:
            check("Darwin fails closed with 503", exc.status_code == 503)
        else:
            raise AssertionError("Darwin must fail closed")
        check("Darwin closes logical operation without Provider row", bool(darwin_persisted) and darwin_persisted[-1].get("model_call") is None and darwin_persisted[-1].get("operation_status") == "failed")

        evaluations.get_evaluation_case = lambda _case_id: {
            "id": 1,
            "case_key": "s10f-fixture",
            "status": "active",
        }
        evaluations._run_real_chat = evaluation_model
        try:
            asyncio.run(
                evaluations.run_case(1, evaluations.EvaluationRunRequest())
            )
        except HTTPException as exc:
            check("Evaluation fails closed with 503", exc.status_code == 503)
        else:
            raise AssertionError("Evaluation must fail closed")

        model_configs.build_model = ab_model
        try:
            asyncio.run(
                model_configs.ab_test_models(
                    model_configs.AbTestRequest(prompt="S10-F fixture")
                )
            )
        except HTTPException as exc:
            check("A/B test fails closed with 503", exc.status_code == 503)
        else:
            raise AssertionError("A/B test must fail closed")
        check("all four paid model functions remain unentered", counters == {"classify": 0, "darwin": 0, "evaluation": 0, "ab": 0})
    finally:
        observability._get_conn = original_get_conn
        observability.get_budget_thresholds = original_thresholds
        badcases._load_case = original_classify_load
        badcases._llm_generate = original_badcase_llm
        badcases.db_get_badcase = original_db_get_badcase
        badcases._find_darwin_skill = original_find_darwin
        badcases.start_darwin_operation = original_start_darwin
        badcases.persist_darwin_operation = original_persist_darwin
        evaluations.get_evaluation_case = original_eval_case
        evaluations._run_real_chat = original_eval_chat
        model_configs.build_model = original_build_model


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
            assert_reporting_truthfulness(db_path)
            assert_budget_query_failure()
            assert_paid_background_short_circuit(db_path)
        finally:
            socket.create_connection = original_create_connection
    print(f"PASS: S10-F Trace truthfulness and budget fail-closed ({len(CHECKS)} checks)")


if __name__ == "__main__":
    main()
