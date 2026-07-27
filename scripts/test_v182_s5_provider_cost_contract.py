"""Deterministic V1.8.2-S5 provider usage and cost contract checks.

No model, HTTP, database, RuntimeRelease, or business-data calls are made.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.runtime.contracts import CostEntry, UsageSource
from app.runtime.cost_ledger import build_cost_entry, cost_entry_usage_payload
from app.runtime.provider_evidence import (
    capture_provider_response,
    provider_evidence_from_run,
    raw_provider_usage,
)


def price(model_id: str) -> dict:
    if model_id == "deepseek-v4-pro":
        uncached, cached, output = 3.0, 0.025, 6.0
    else:
        uncached, cached, output = 1.0, 0.02, 2.0
    return {
        "model_id": model_id,
        "currency": "CNY",
        "effective_date": "2026-07-25",
        "input_price_per_1m": uncached,
        "cached_input_price_per_1m": cached,
        "output_price_per_1m": output,
        "source_note": "v19 published price snapshot",
    }


def actual_usage() -> dict:
    return {
        "input_cache_hit_tokens": 2000,
        "input_cache_miss_tokens": 1000,
        "input_tokens": 3000,
        "output_tokens": 500,
        "reasoning_tokens": 120,
        "total_tokens": 3500,
    }


def cost(stage: str, requested: str, actual: str | None, **kwargs):
    return build_cost_entry(
        stage=stage,
        provider="deepseek",
        requested_model=requested,
        response_model=None,
        provider_response_model=actual,
        thinking_enabled=True,
        model_policy_version="v19",
        provider_usage=kwargs.pop("provider_usage", actual_usage()),
        price_row=kwargs.pop("price_row", price(actual or requested)),
        **kwargs,
    )


def main() -> None:
    # 1. Raw three-class Provider Usage is extracted and JSON-persistable.
    raw = SimpleNamespace(
        prompt_cache_hit_tokens=2000,
        prompt_cache_miss_tokens=1000,
        prompt_tokens=3000,
        completion_tokens=500,
        total_tokens=3500,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=120),
        model_extra={},
    )
    assert raw_provider_usage(raw) == actual_usage()
    parsed = SimpleNamespace(provider_data={"id": "request-demo"})
    response = SimpleNamespace(model="deepseek-v4-pro", usage=raw)
    capture_provider_response(parsed, response)
    evidence = provider_evidence_from_run(
        SimpleNamespace(model_provider_data=parsed.provider_data)
    )
    assert evidence["provider_response_model"] == "deepseek-v4-pro"
    assert evidence["usage"]["input_cache_miss_tokens"] == 1000

    router = cost("router", "deepseek-v4-flash", "deepseek-v4-flash")
    payload = cost_entry_usage_payload(router, provider_request_id="request-demo")
    assert json.loads(json.dumps(payload))["input_cache_hit_tokens"] == 2000

    # 2. Flash and Pro use their own published price snapshots.
    pro = cost("darwin", "deepseek-v4-pro", "deepseek-v4-pro")
    assert router.amount == 0.00204
    assert pro.amount == 0.00605
    assert router.price_snapshot.model_id == "deepseek-v4-flash"
    assert pro.price_snapshot.model_id == "deepseek-v4-pro"

    # 3. requested_model and provider_response_model remain distinct.
    mismatch = cost("router", "deepseek-v4-flash", "deepseek-v4-pro")
    assert mismatch.requested_model == "deepseek-v4-flash"
    assert mismatch.provider_response_model == "deepseek-v4-pro"
    assert mismatch.amount == pro.amount

    # 4. Missing Provider response model is never replaced with requested_model.
    missing_model = cost(
        "vertical_agent",
        "deepseek-v4-flash",
        None,
        price_row=price("deepseek-v4-flash"),
    )
    assert missing_model.provider_response_model is None
    assert missing_model.amount is None

    # 5. total_tokens-only records stay unavailable and never get a split/cost.
    total_only = cost(
        "vertical_agent",
        "deepseek-v4-flash",
        "deepseek-v4-flash",
        provider_usage={"total_tokens": 99},
    )
    assert total_only.usage_source == UsageSource.UNAVAILABLE
    assert total_only.input_cache_hit_tokens is None
    assert total_only.input_cache_miss_tokens is None
    assert total_only.amount is None

    # 6. estimated and provider_actual are mutually explicit.
    estimated = build_cost_entry(
        stage="vertical_agent",
        provider="deepseek",
        requested_model="deepseek-v4-flash",
        response_model=None,
        provider_response_model=None,
        thinking_enabled=True,
        model_policy_version="v19",
        provider_usage=None,
        price_row=price("deepseek-v4-flash"),
        local_estimate_tokens=12,
    )
    assert router.usage_source == UsageSource.PROVIDER_ACTUAL
    assert estimated.usage_source == UsageSource.ESTIMATED
    assert estimated.amount is None

    # 7. Historical contracts using legacy source/response_model still parse.
    historical = CostEntry.model_validate(
        {
            "stage": "router",
            "provider": "deepseek",
            "requested_model": "deepseek-v4-flash",
            "response_model": "deepseek-v4-flash",
            "model_policy_version": "v1.8",
            "usage_source": "provider_reported_total_only",
            "total_tokens": 88,
            "availability_note": "historical total-only record",
        }
    )
    assert historical.total_tokens == 88

    # 8. Provider failure cannot be recorded as a successful cost.
    failed = cost(
        "badcase_classify",
        "deepseek-v4-flash",
        "deepseek-v4-flash",
        provider_succeeded=False,
    )
    assert failed.usage_source == UsageSource.UNAVAILABLE
    assert failed.amount is None

    # 9. Router and vertical Agent remain separate ledger stages.
    vertical = cost(
        "vertical_agent", "deepseek-v4-flash", "deepseek-v4-flash"
    )
    assert [router.stage, vertical.stage] == ["router", "vertical_agent"]

    # 10. Darwin Pro never enters the Flash bucket; thinking is independent.
    buckets: dict[str, int] = {}
    for entry in (router, vertical, pro):
        model_id = entry.provider_response_model or "unknown"
        buckets[model_id] = buckets.get(model_id, 0) + 1
    assert buckets == {"deepseek-v4-flash": 2, "deepseek-v4-pro": 1}
    assert pro.thinking_enabled is True
    assert pro.provider_response_model == "deepseek-v4-pro"

    print("PASS: V1.8.2-S5 deterministic provider cost contract (10 checks)")


if __name__ == "__main__":
    main()
