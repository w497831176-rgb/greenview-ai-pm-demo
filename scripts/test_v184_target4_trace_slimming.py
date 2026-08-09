"""Deterministic Target 4 Trace-summary and lazy-detail contract checks.

The checks use only symbolic rows in a temporary SQLite database plus static
frontend source inspection.  They do not call a model, HTTP endpoint,
RuntimeRelease, RAG, MCP, Tool, or production business data.
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
TEST_DATA = tempfile.TemporaryDirectory(prefix="yiai-v184-target4-trace-")
os.environ["PROPERTY_DATA_DIR"] = TEST_DATA.name
os.environ["DEEPSEEK_API_KEY"] = ""

# Synology's host Python is intentionally minimal.  The deterministic helper
# tests do not exercise HTTP/Pydantic validation, so a tiny import-only fallback
# keeps this script runnable there without installing packages.  Real API
# import checks still run in the application container with the actual modules.
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
    conn.executescript(
        """
        CREATE TABLE chat_traces (
            trace_id TEXT PRIMARY KEY,
            session_id TEXT,
            user_message TEXT,
            intent TEXT,
            agent_name TEXT,
            agent_id TEXT,
            status TEXT,
            created_at TEXT,
            updated_at TEXT,
            run_type TEXT DEFAULT 'chat'
        );
        CREATE TABLE model_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT,
            model_id TEXT,
            requested_model TEXT,
            provider_actual_model TEXT,
            thinking_enabled INTEGER,
            model_selection_reason TEXT,
            total_tokens INTEGER,
            estimated_cost_cny REAL,
            usage_source TEXT,
            usage_normalized TEXT,
            stage TEXT,
            status TEXT,
            created_at TEXT,
            finished_at TEXT,
            latency_ms INTEGER,
            record_kind TEXT DEFAULT 'provider_attempt',
            usage_status TEXT,
            price_snapshot TEXT,
            cost_source TEXT
        );
        CREATE TABLE evaluation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT
        );
        CREATE TABLE trace_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT,
            span_name TEXT,
            status TEXT,
            latency_ms INTEGER,
            input_summary TEXT,
            output_summary TEXT,
            metadata_json TEXT,
            created_at TEXT
        );
        """
    )


def _usage(sequence: int) -> dict:
    return {
        "record_kind": "provider_attempt",
        "include_in_provider_aggregate": True,
        "provider_request_sent": True,
        "provider_request_id": f"provider-symbolic-{sequence}",
        "provider_request_id_obtained": True,
        "done_received": True,
        "sdk_stream_exhausted": True,
        "usage_received": True,
        "persisted": True,
        "usage_status": "provider_actual",
        "token_source": "provider_actual",
        "cost_source": "platform_price_snapshot",
        "requested_model": "deepseek-v4-flash",
        "provider_actual_model": "deepseek-v4-flash",
        "thinking_enabled": False,
        "cache_hit_input_tokens": 10,
        "cache_miss_input_tokens": 20,
        "input_tokens": 30,
        "output_tokens": 5,
        "reasoning_tokens": 2,
        "total_tokens": 35,
        "calculated_direct_cost": 0.00005,
        "price_snapshot": {
            "input_price_per_1m": 1.0,
            "cached_input_price_per_1m": 0.1,
            "output_price_per_1m": 2.0,
            "currency": "CNY",
            "effective_date": "2026-08-01",
        },
    }


def _insert_fixture(path: Path, count: int = 25) -> None:
    conn = sqlite3.connect(path)
    _create_schema(conn)
    for index in range(count):
        trace_id = f"trace-symbolic-{index:02d}"
        created_at = f"2026-08-09 10:00:{index:02d}"
        conn.execute(
            "INSERT INTO chat_traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trace_id,
                f"session-symbolic-{index:02d}",
                f"symbolic-message-{index:02d}",
                "symbolic-intent",
                "Symbolic Agent",
                "symbolic-agent",
                "complete",
                created_at,
                created_at,
                "chat",
            ),
        )
        conn.execute(
            """INSERT INTO model_calls
               (trace_id, model_id, requested_model, provider_actual_model,
                thinking_enabled, model_selection_reason, total_tokens,
                estimated_cost_cny, usage_source, usage_normalized, stage,
                status, created_at, finished_at, latency_ms, record_kind,
                usage_status, price_snapshot, cost_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace_id,
                "deepseek-v4-flash",
                "deepseek-v4-flash",
                "deepseek-v4-flash",
                0,
                "symbolic published policy",
                35,
                0.00005,
                "provider_actual",
                json.dumps(_usage(index)),
                "router",
                "success",
                created_at,
                created_at,
                100 + index,
                "provider_attempt",
                "provider_actual",
                json.dumps(_usage(index)["price_snapshot"]),
                "platform_price_snapshot",
            ),
        )
        conn.execute(
            """INSERT INTO trace_events
               (trace_id, span_name, status, latency_ms, input_summary,
                output_summary, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace_id,
                "router",
                "success",
                100 + index,
                None,
                None,
                json.dumps({"lane": "C_ISOLATED_GENERAL", "candidate_count": 3}),
                created_at,
            ),
        )
        conn.execute(
            """INSERT INTO trace_events
               (trace_id, span_name, status, latency_ms, input_summary,
                output_summary, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace_id,
                "final_response",
                "success",
                200 + index,
                None,
                "symbolic-answer",
                json.dumps({"answer_status": "answered"}),
                created_at,
            ),
        )
    conn.commit()
    conn.close()


def test_server_paginates_before_provider_aggregation() -> None:
    with tempfile.TemporaryDirectory(prefix="yiai-trace-page-") as temp_dir:
        path = Path(temp_dir) / "trace.db"
        _insert_fixture(path)
        original_get_conn = observability._get_conn
        original_fetch = observability._fetch_model_calls_for_trace_ids
        requested_pages: list[tuple[str, ...]] = []

        def fixture_conn() -> sqlite3.Connection:
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            return conn

        def recorded_fetch(trace_ids, *args, **kwargs):
            requested_pages.append(tuple(sorted(trace_ids)))
            return original_fetch(trace_ids, *args, **kwargs)

        observability._get_conn = fixture_conn
        observability._fetch_model_calls_for_trace_ids = recorded_fetch
        try:
            page = observability._list_trace_page(limit=20, offset=0)
            assert page["total"] == 25
            assert len(page["traces"]) == 20
            returned = {item["trace_id"] for item in page["traces"]}
            assert len(requested_pages) == 2
            assert all(len(ids) == 20 for ids in requested_pages)
            assert all(set(ids) == returned for ids in requested_pages)
            assert page["traces"][0]["trace_id"] == "trace-symbolic-24"
            assert page["traces"][-1]["trace_id"] == "trace-symbolic-05"
            allowed = {
                "trace_id",
                "created_at",
                "question_summary",
                "lane",
                "agent_id",
                "agent_name",
                "result",
                "run_type",
                "provider_request_count",
                "total_tokens",
                "total_cost_cny",
                "known_partial_cost_cny",
                "cost_status",
                "total_latency_ms",
            }
            assert all(set(item) == allowed for item in page["traces"])
            forbidden = {
                "user_message",
                "messages",
                "prompt",
                "response",
                "rag_chunks",
                "mcp_calls",
                "tool_results",
                "raw_json",
            }
            assert not any(forbidden.intersection(item) for item in page["traces"])

            ascending = observability._list_trace_page(
                limit=20, offset=0, sort_order="asc"
            )
            assert ascending["traces"][0]["trace_id"] == "trace-symbolic-00"
        finally:
            observability._fetch_model_calls_for_trace_ids = original_fetch
            observability._get_conn = original_get_conn


def test_compact_detail_and_true_cost_control() -> None:
    raw_call = {
        "id": 1,
        "trace_id": "trace-symbolic-detail",
        "stage": "router",
        "model_id": "deepseek-v4-flash",
        "requested_model": "deepseek-v4-flash",
        "provider_actual_model": "deepseek-v4-flash",
        "thinking_enabled": False,
        "model_selection_reason": "symbolic published policy",
        "status": "success",
        "latency_ms": 123,
        "record_kind": "provider_attempt",
        "usage_source": "provider_actual",
        "usage_status": "provider_actual",
        "cost_source": "platform_price_snapshot",
        "estimated_cost_cny": 0.00005,
        "usage_normalized": _usage(1),
    }
    events = [
        {
            "span_name": "router",
            "status": "success",
            "latency_ms": 123,
            "metadata": {
                "lane": "C_ISOLATED_GENERAL",
                "visible_message_count": 4,
                "candidate_count": 3,
            },
        },
        {
            "span_name": "final_response",
            "status": "success",
            "latency_ms": 200,
            "output_summary": "symbolic-answer",
            "metadata": {
                "answer_status": "answered",
                "second_agent_request_count": 0,
                "automatic_retry_count": 0,
                "citation_violations": [],
            },
        },
    ]
    originals = {
        "get_chat_trace": observability.get_chat_trace,
        "fetch": observability._fetch_reporting_model_calls,
        "events": observability.list_trace_events,
        "mcp": observability.get_mcp_call_audits_for_trace,
    }
    observability.get_chat_trace = lambda _trace_id: {
        "trace_id": "trace-symbolic-detail",
        "session_id": "session-symbolic-detail",
        "user_message": "symbolic-message-detail",
        "agent_id": "symbolic-agent",
        "agent_name": "Symbolic Agent",
        "status": "complete",
        "created_at": "2026-08-09 10:01:00",
        "updated_at": "2026-08-09 10:01:01",
        "version_snapshot": "symbolic-snapshot",
    }
    observability._fetch_reporting_model_calls = lambda *args, **kwargs: [raw_call]
    observability.list_trace_events = lambda _trace_id: events
    observability.get_mcp_call_audits_for_trace = lambda _trace_id: []
    try:
        detail = observability._trace_detail_compact("trace-symbolic-detail")
    finally:
        observability.get_chat_trace = originals["get_chat_trace"]
        observability._fetch_reporting_model_calls = originals["fetch"]
        observability.list_trace_events = originals["events"]
        observability.get_mcp_call_audits_for_trace = originals["mcp"]

    assert detail["provider_requests"] == [
        {
            "stage": "router",
            "stage_name": "Router",
            "requested_model": "deepseek-v4-flash",
            "provider_actual_model": "deepseek-v4-flash",
            "thinking_enabled": False,
            "model_selection_reason": "symbolic published policy",
            "cache_hit_input_tokens": 10,
            "cache_miss_input_tokens": 20,
            "input_tokens": 30,
            "output_tokens": 5,
            "reasoning_tokens": 2,
            "total_tokens": 35,
            "reasoning_is_output_subset": True,
            "total_equation_valid": True,
            "price_snapshot_cost_cny": 0.00005,
            "cost_source": "platform_price_snapshot",
            "price_snapshot_effective_date": "2026-08-01",
            "latency_ms": 123,
            "status": "success",
            "usage_status": "provider_actual",
            "provider_request_sequence": None,
        }
    ]
    assert detail["provider_summary"]["provider_request_count"] == 1
    assert detail["provider_summary"]["usage_totals"] == {
        "cache_hit_input_tokens": 10,
        "cache_miss_input_tokens": 20,
        "input_tokens": 30,
        "output_tokens": 5,
        "reasoning_tokens": 2,
        "total_tokens": 35,
        "complete": True,
        "token_relationship_valid": True,
    }
    assert detail["cost_quality_control"]["call_reduction"] == {
        "router_requests": 1,
        "agent_requests": 0,
        "tool_follow_up_requests": 0,
        "selector_requests": 0,
        "resolver_requests": 0,
        "second_agent_requests": 0,
        "automatic_retries": 0,
    }
    assert detail["cost_quality_control"]["context_loading"][
        "router_session_message_count"
    ] == 4
    for forbidden in (
        "messages",
        "model_calls",
        "mcp_calls",
        "trace_events",
        "evidence_ledger",
        "raw_json",
    ):
        assert forbidden not in detail


def test_frontend_loads_summary_and_advanced_diagnostics_lazily() -> None:
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
    rendered = section[
        section.rindex("function renderPage()") : section.index(
            "function bindEvents()"
        )
    ]
    assert load.count("apiGet(") == 1
    assert "/api/observability/traces?" in load
    assert "state.pagination.limit" in load
    assert "sort_order" in load
    assert "/overview" not in load
    assert "/trends" not in load
    assert "/distribution" not in load
    assert "cost-preview" not in load
    assert "首屏只加载最近20条摘要" in rendered

    headings = (
        "本轮结果摘要",
        "真实执行链",
        "每个物理Provider请求",
        "本轮成本与质量控制",
        "高级诊断",
    )
    slim_detail = section[
        section.index("async function showTraceDetailSlim") : section.index(
            "function renderGovernancePrinciples"
        )
    ]
    positions = [slim_detail.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "apiGet(`/api/observability/traces/${traceId}`)" in slim_detail
    assert "apiGet(`/api/observability/traces/${traceId}/advanced`)" in slim_detail
    assert slim_detail.index("advanced.addEventListener('toggle'") < slim_detail.index(
        "apiGet(`/api/observability/traces/${traceId}/advanced`)"
    )
    assert "展开后加载Evidence、Tool结果与Raw JSON" in slim_detail
    assert "Total = Hit + Miss + Output" in section
    assert "Reasoning 是 Output 子集，不重复相加" in section
    assert "const providerSummary = data.provider_summary || {}" in slim_detail
    assert "Trace合计" in slim_detail
    assert "历史会话按实际需要截断或摘要" not in rendered
    assert "COST-01" not in rendered
    assert "Top-K预估" not in rendered
    assert "Provider账本已对平" not in rendered
    assert "④-A Provider账本" in rendered
    assert "case 'cost-strategy': await renderCostGovernancePage(main)" in source


def main() -> None:
    tests = (
        test_server_paginates_before_provider_aggregation,
        test_compact_detail_and_true_cost_control,
        test_frontend_loads_summary_and_advanced_diagnostics_lazily,
    )
    try:
        for test in tests:
            test()
            print(f"PASS {test.__name__}")
        print(
            f"Target4 Trace slimming: PASS ({len(tests)} checks; Provider calls: 0)"
        )
    finally:
        TEST_DATA.cleanup()


if __name__ == "__main__":
    main()
