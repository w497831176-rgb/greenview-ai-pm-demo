"""Idempotently add two missing Provider requests from persisted Agno runs.

This script is deliberately hard-scoped to the two V1.8.2-S9-C traces. It
does not call a model and refuses to write if any persisted request id or Usage
field differs from the audited source evidence.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from app.runtime.cost_ledger import build_cost_entry, cost_entry_usage_payload
from app.settings import agent_db
from db.property_db import (
    get_chat_trace,
    get_model_calls_for_trace,
    record_model_call_idempotent,
)


TARGETS = {
    "3c1aad38350c4600": {
        "session_id": "s8c-skill-acceptance-20260728-01",
        "run_id": "7d8e7f58-20c8-4f6e-affa-e8e9009e2656",
        "request_id": "359a9562-3deb-4f54-8692-b9815dcadc5e",
        "usage": {
            "input_cache_hit_tokens": 0,
            "input_cache_miss_tokens": 2203,
            "output_tokens": 85,
            "total_tokens": 2288,
        },
        "amount": 0.00237300,
    },
    "cedc795859fc458c": {
        "session_id": "web-9c4ff42f9eaf",
        "run_id": "8a7f6c11-b9c9-47ed-8d6f-3e69ca3b0f60",
        "request_id": "c7db2f7d-79f3-4a8a-97b2-223fab90a5c0",
        "usage": {
            "input_cache_hit_tokens": 512,
            "input_cache_miss_tokens": 1698,
            "output_tokens": 82,
            "total_tokens": 2292,
        },
        "amount": 0.00187224,
    },
}

EXPECTED_AFTER = {
    "calls": 6,
    "input_cache_hit_tokens": 6144,
    "input_cache_miss_tokens": 7891,
    "output_tokens": 903,
    "total_tokens": 14938,
    "cost": 0.00981988,
}


def _provider_requests(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = []
    seen = set()
    for message in run.get("messages") or []:
        provider_data = (message or {}).get("provider_data") or {}
        usage = provider_data.get("usage") or {}
        request_id = provider_data.get("id")
        if not request_id or request_id in seen or not usage:
            continue
        seen.add(request_id)
        result.append(
            {
                "provider_request_id": str(request_id),
                "provider_response_model": provider_data.get("response_model"),
                "usage": dict(usage),
            }
        )
    return result


def _normalized(call: Dict[str, Any]) -> Dict[str, Any]:
    value = call.get("usage_normalized") or {}
    return value if isinstance(value, dict) else {}


def main() -> None:
    for trace_id, target in TARGETS.items():
        trace = get_chat_trace(trace_id) or {}
        assert trace.get("session_id") == target["session_id"]
        session = agent_db.get_session(
            session_id=target["session_id"],
            deserialize=False,
        )
        assert isinstance(session, dict)
        run = next(
            item
            for item in session.get("runs") or []
            if item.get("run_id") == target["run_id"]
        )
        raw_requests = _provider_requests(run)
        source = next(
            item
            for item in raw_requests
            if item["provider_request_id"] == target["request_id"]
        )
        assert source["provider_response_model"] == "deepseek-v4-flash"
        for key, expected in target["usage"].items():
            assert source["usage"].get(key) == expected, (trace_id, key)

        existing = get_model_calls_for_trace(trace_id)
        final_call = next(
            item
            for item in existing
            if item.get("stage") == "vertical_agent"
            and _normalized(item).get("provider_request_id")
            != target["request_id"]
        )
        price_snapshot = final_call.get("price_snapshot") or {}
        assert price_snapshot.get("model_id") == "deepseek-v4-flash"
        cost = build_cost_entry(
            stage="vertical_agent",
            provider="deepseek",
            requested_model="deepseek-v4-flash",
            response_model=None,
            provider_response_model=source["provider_response_model"],
            thinking_enabled=_normalized(final_call).get("thinking_enabled"),
            model_policy_version="v1.8",
            provider_usage=source["usage"],
            price_row=price_snapshot,
        )
        assert cost.usage_source.value == "provider_actual"
        assert cost.amount == target["amount"]
        usage_normalized = cost_entry_usage_payload(
            cost,
            provider_request_id=target["request_id"],
            provider_request_sequence=1,
            provider_request_key=f"request_id:{target['request_id']}",
            provider_request_identity_source="provider_request_id",
            evidence_source="agno_persisted_provider_response",
        )
        usage_normalized["historical_backfill"] = "v1.8.2-s9-c"
        created_at = datetime.fromtimestamp(
            int(run["created_at"]),
            tz=ZoneInfo("Asia/Shanghai"),
        ).strftime("%Y-%m-%d %H:%M:%S")
        record_model_call_idempotent(
            trace_id=trace_id,
            stage="vertical_agent",
            model_id="deepseek-v4-flash",
            model_selection_reason=(
                "V1.8.2-S9-C exact backfill from persisted Agno Provider response"
            ),
            input_tokens=cost.input_tokens,
            output_tokens=cost.output_tokens,
            reasoning_tokens=cost.reasoning_tokens,
            cached_tokens=cost.cached_input_tokens,
            total_tokens=cost.total_tokens,
            usage_source=cost.usage_source.value,
            price_snapshot=cost.price_snapshot.model_dump(mode="json"),
            estimated_cost_cny=cost.amount,
            usage_normalized=usage_normalized,
            created_at=created_at,
        )

    totals = {
        "calls": 0,
        "input_cache_hit_tokens": 0,
        "input_cache_miss_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }
    for trace_id in TARGETS:
        calls = get_model_calls_for_trace(trace_id)
        assert len(calls) == 3
        for call in calls:
            usage = _normalized(call)
            assert call.get("usage_source") == "provider_actual"
            assert (
                usage.get("provider_response_model")
                or call.get("model_id")
            ) == "deepseek-v4-flash"
            totals["calls"] += 1
            for key in (
                "input_cache_hit_tokens",
                "input_cache_miss_tokens",
                "output_tokens",
            ):
                totals[key] += int(usage.get(key) or 0)
            totals["total_tokens"] += int(call.get("total_tokens") or 0)
            totals["cost"] += float(call.get("estimated_cost_cny") or 0)
    totals["cost"] = round(totals["cost"], 8)
    assert totals == EXPECTED_AFTER, totals
    print("PASS: exact S9-C backfill is complete and idempotent", totals)


if __name__ == "__main__":
    main()
