"""Deterministic S9-C Provider-request accounting contract.

No HTTP, model, RuntimeRelease, or production database call is made.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import db.property_db as property_db
from app.observability import _aggregate_model_calls, _trace_cost_explanation
from app.runtime.cost_ledger import build_cost_entry, cost_entry_usage_payload
from app.runtime.provider_evidence import remember_provider_request


PRICE = {
    "model_id": "deepseek-v4-flash",
    "currency": "CNY",
    "effective_date": "2026-07-25",
    "input_price_per_1m": 1.0,
    "cached_input_price_per_1m": 0.02,
    "output_price_per_1m": 2.0,
    "source_note": "contract fixture",
}


def request(request_id: str | None, hit: int, miss: int, output: int) -> dict:
    return {
        "provider_request_id": request_id,
        "provider_response_model": "deepseek-v4-flash",
        "usage": {
            "input_cache_hit_tokens": hit,
            "input_cache_miss_tokens": miss,
            "output_tokens": output,
            "total_tokens": hit + miss + output,
        },
    }


def model_call(trace_id: str, stage: str, evidence: dict) -> dict:
    cost = build_cost_entry(
        stage=stage,
        provider="deepseek",
        requested_model="deepseek-v4-flash",
        response_model=None,
        provider_response_model=evidence["provider_response_model"],
        thinking_enabled=True,
        model_policy_version="v1.8",
        provider_usage=evidence["usage"],
        price_row=PRICE,
    )
    sequence = evidence["provider_request_sequence"]
    normalized = cost_entry_usage_payload(
        cost,
        provider_request_id=evidence.get("provider_request_id"),
        provider_request_sequence=sequence,
        provider_request_key=evidence["provider_request_key"],
        provider_request_identity_source=evidence[
            "provider_request_identity_source"
        ],
        evidence_source="provider_response",
    )
    return {
        "trace_id": trace_id,
        "stage": stage,
        "model_id": "deepseek-v4-flash",
        "status": "success",
        "input_tokens": cost.input_tokens,
        "output_tokens": cost.output_tokens,
        "reasoning_tokens": cost.reasoning_tokens,
        "cached_tokens": cost.cached_input_tokens,
        "total_tokens": cost.total_tokens,
        "usage_source": cost.usage_source.value,
        "price_snapshot": cost.price_snapshot.model_dump(mode="json"),
        "estimated_cost_cny": cost.amount,
        "usage_normalized": normalized,
    }


def assert_journal_and_aggregation() -> None:
    journal: list[dict] = []
    remember_provider_request(journal, request("router-1", 384, 1845, 191))
    remember_provider_request(journal, request("vertical-1", 0, 2203, 85))
    remember_provider_request(journal, request("vertical-2", 2176, 404, 164))
    # A repeated stream delta for the same request updates rather than appends.
    remember_provider_request(journal, request("vertical-1", 0, 2203, 85))
    assert len(journal) == 3
    assert [item["provider_request_sequence"] for item in journal] == [1, 2, 3]

    calls = [
        model_call("trace", "router", journal[0]),
        model_call("trace", "vertical_agent", journal[1]),
        model_call("trace", "vertical_agent", journal[2]),
    ]
    aggregate = _aggregate_model_calls(calls)
    assert aggregate["calls"] == 3
    assert aggregate["input_cache_hit_tokens"] == 2560
    assert aggregate["input_cache_miss_tokens"] == 4452
    assert aggregate["output_tokens"] == 440
    assert aggregate["total_tokens"] == 7452
    assert aggregate["provider_actual_cost_cny"] == round(
        sum(item["estimated_cost_cny"] for item in calls), 8
    )
    story = _trace_cost_explanation(calls)
    assert story["provider_request_count"] == 3
    assert [item["stage"] for item in story["chain"]] == [
        "router",
        "vertical_agent",
        "vertical_agent",
    ]

    # Missing request ids use stable ordinals; arbitrary future tool loops keep all rounds.
    anonymous: list[dict] = []
    remember_provider_request(anonymous, request(None, 1, 2, 3))
    remember_provider_request(anonymous, request(None, 4, 5, 6))
    assert [item["provider_request_key"] for item in anonymous] == [
        "run_sequence:1",
        "run_sequence:2",
    ]


def assert_idempotent_storage() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "model-calls.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE model_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT, stage TEXT, model_id TEXT, status TEXT,
                latency_ms INTEGER, input_tokens INTEGER, output_tokens INTEGER,
                reasoning_tokens INTEGER, cached_tokens INTEGER, total_tokens INTEGER,
                usage_source TEXT, model_selection_reason TEXT, error_summary TEXT,
                price_snapshot TEXT, estimated_cost_cny REAL,
                context_breakdown TEXT, usage_normalized TEXT, created_at TEXT
            );
            """
        )
        conn.close()
        original_get_conn = property_db._get_conn

        def fixture_conn() -> sqlite3.Connection:
            value = sqlite3.connect(db_path)
            value.row_factory = sqlite3.Row
            return value

        property_db._get_conn = fixture_conn
        try:
            journal: list[dict] = []
            remember_provider_request(journal, request("same-request", 1, 2, 3))
            call = model_call("trace", "vertical_agent", journal[0])
            first = property_db.record_model_call_idempotent(**call)
            second = property_db.record_model_call_idempotent(**call)
            assert first["id"] == second["id"]
            assert second["deduplicated"] is True
            assert len(property_db.get_model_calls_for_trace("trace")) == 1
        finally:
            property_db._get_conn = original_get_conn


def assert_skill_and_frontend_contract() -> None:
    agent_source = Path("app/runtime/agent_factory.py").read_text(encoding="utf-8")
    assert "return None if preload_calls else skills" in agent_source
    assert "skills=model_skills" in agent_source
    assert "不要重复调用 Skill 工具" in agent_source
    source = Path("frontend/index.html").read_text(encoding="utf-8")
    section = source[
        source.index("async function renderCostGovernancePage") : source.index(
            "async function renderCostStrategyPage"
        )
    ]
    assert "Provider请求次数" in section
    assert "发起Provider请求" in section
    assert "actual_model_id === 'deepseek-v4-flash'" in section
    assert "actual_model_id === 'deepseek-v4-pro'" in section
    assert "模型身份待确认" in section
    assert "String(row.model_id || row.model_name || '').includes('flash')" not in section


def main() -> None:
    assert_journal_and_aggregation()
    assert_idempotent_storage()
    assert_skill_and_frontend_contract()
    print("PASS: V1.8.2-S9-C Provider request accounting (3 groups)")


if __name__ == "__main__":
    main()
