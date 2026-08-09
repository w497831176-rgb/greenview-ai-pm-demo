"""Deterministic Target 4 Provider accounting presentation checks.

No model, HTTP, production database, or RuntimeRelease call is made.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST_DATA = tempfile.TemporaryDirectory(prefix="yiai-v184-target4-")
os.environ["PROPERTY_DATA_DIR"] = TEST_DATA.name
os.environ["DEEPSEEK_API_KEY"] = ""

from app.observability import (  # noqa: E402
    _cost_summary,
    _enrich_model_call,
    _provider_direct_cost,
    _trace_efficiency_notes,
    _trace_event_provider_projection,
)


def physical_call(sequence: int = 1, *, total: int = 350) -> dict:
    usage = {
        "record_kind": "provider_attempt",
        "include_in_provider_aggregate": True,
        "provider_request_sent": True,
        "usage_status": "provider_actual",
        "token_source": "provider_actual",
        "cost_source": "platform_price_snapshot",
        "requested_model": "deepseek-v4-flash",
        "provider_actual_model": "deepseek-v4-flash",
        "thinking_enabled": False,
        "provider_request_id": f"provider-{sequence}",
        "provider_request_sequence": sequence,
        "cache_hit_input_tokens": 100,
        "cache_miss_input_tokens": 200,
        "input_tokens": 300,
        "output_tokens": 50,
        "reasoning_tokens": 20,
        "total_tokens": total,
        "calculated_direct_cost": 0.00031,
        "price_snapshot": {
            "input_price_per_1m": 1.0,
            "cached_input_price_per_1m": 0.1,
            "output_price_per_1m": 2.0,
            "currency": "CNY",
            "effective_date": "2026-08-01",
            "source_note": "symbolic frozen snapshot",
        },
    }
    return {
        "id": sequence,
        "trace_id": "trace-target4",
        "stage": "router" if sequence == 1 else "vertical_agent",
        "model_id": "deepseek-v4-flash",
        "thinking_enabled": False,
        "model_selection_reason": "published Flash policy",
        "status": "success",
        "usage_source": "provider_actual",
        "usage_status": "provider_actual",
        "cost_source": "platform_price_snapshot",
        "estimated_cost_cny": 0.00031,
        "usage_normalized": usage,
    }


def test_token_relationship_and_cost() -> None:
    enriched = _enrich_model_call(physical_call(), "session-target4")
    accounting = enriched["provider_token_accounting"]
    assert accounting["cache_hit_input_tokens"] == 100
    assert accounting["cache_miss_input_tokens"] == 200
    assert accounting["input_tokens"] == 300
    assert accounting["output_tokens"] == 50
    assert accounting["reasoning_tokens"] == 20
    assert accounting["total_tokens"] == 350
    assert accounting["total_equation_valid"] is True
    assert accounting["reasoning_is_output_subset"] is True
    assert _provider_direct_cost(enriched) == 0.00031

    inconsistent = _enrich_model_call(physical_call(total=351), "session-target4")
    assert inconsistent["provider_token_accounting"]["total_equation_valid"] is False
    assert inconsistent["provider_usage_inconsistent"] is True


def test_one_row_per_physical_request_and_identity_truth() -> None:
    rows = [
        _enrich_model_call(physical_call(sequence), "session-target4")
        for sequence in (1, 2)
    ]
    assert len(rows) == 2
    assert [row["provider_request_sequence"] for row in rows] == [1, 2]
    assert [row["stage"] for row in rows] == ["router", "vertical_agent"]
    totals = _cost_summary(rows)["usage_totals"]
    assert totals == {
        "cache_hit_input_tokens": 200,
        "cache_miss_input_tokens": 400,
        "input_tokens": 600,
        "output_tokens": 100,
        "reasoning_tokens": 40,
        "total_tokens": 700,
        "complete": True,
        "token_relationship_valid": True,
    }
    identity = rows[0]["provider_identity_evidence"]
    assert identity == {
        "requested_model": "deepseek-v4-flash",
        "requested_model_collected": True,
        "provider_actual_model": "deepseek-v4-flash",
        "provider_actual_model_collected": True,
        "thinking_enabled": False,
        "thinking_collected": True,
        "model_selection_reason": "published Flash policy",
        "model_selection_reason_collected": True,
    }

    missing = physical_call()
    missing["thinking_enabled"] = None
    missing["model_selection_reason"] = None
    missing["usage_normalized"].pop("provider_actual_model")
    missing["usage_normalized"].pop("thinking_enabled")
    missing_identity = _enrich_model_call(missing, "session-target4")[
        "provider_identity_evidence"
    ]
    assert missing_identity["provider_actual_model"] is None
    assert missing_identity["provider_actual_model_collected"] is False
    assert missing_identity["thinking_enabled"] is None
    assert missing_identity["thinking_collected"] is False
    assert missing_identity["model_selection_reason"] is None
    assert missing_identity["model_selection_reason_collected"] is False


def test_non_provider_nodes_and_small_savings_notes() -> None:
    calls = [_enrich_model_call(physical_call(), "session-target4")]
    events = _trace_event_provider_projection(
        [
            {"span_name": "router", "status": "success"},
            {"span_name": "retrieval", "status": "success"},
            {"span_name": "mcp.read_status", "status": "success"},
            {"span_name": "legacy_unknown", "status": "success"},
        ],
        calls,
    )
    assert events[0]["provider_called"] is True
    assert events[1]["provider_called"] is False
    assert events[2]["provider_called"] is False
    assert events[1]["provider_token_note"] == "未调用Provider，无Provider Token。"
    assert events[3]["provider_called"] is None

    notes = _trace_efficiency_notes(calls, {})
    assert len(notes) == 3
    assert notes[0]["evidence_level"] == "measured"
    assert "Provider请求1次" in notes[0]["cost_result"]
    assert notes[1]["evidence_level"] == "not_collected"
    assert notes[2]["evidence_level"] == "expected"
    assert all("¥" not in note["cost_result"] for note in notes)


def test_frontend_uses_backend_evidence_without_token_inference() -> None:
    source = (Path(__file__).resolve().parents[1] / "frontend/index.html").read_text(
        encoding="utf-8"
    )
    assert "call.provider_token_accounting" in source
    assert "call.provider_identity_evidence" in source
    assert "Input=Hit+Miss；Total=Input+Output" in source
    assert "Trace Token合计" in source
    assert "Flash / Pro 选择理由" in source
    assert "未调用Provider，无Provider Token" in source
    assert "trace_efficiency_notes" in source


def main() -> None:
    tests = (
        test_token_relationship_and_cost,
        test_one_row_per_physical_request_and_identity_truth,
        test_non_provider_nodes_and_small_savings_notes,
        test_frontend_uses_backend_evidence_without_token_inference,
    )
    try:
        for test in tests:
            test()
            print(f"PASS {test.__name__}")
        print(f"Target4 cost explainability: PASS ({len(tests)} checks; Provider calls: 0)")
    finally:
        TEST_DATA.cleanup()


if __name__ == "__main__":
    main()
