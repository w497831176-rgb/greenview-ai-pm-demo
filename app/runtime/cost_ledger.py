"""Honest per-stage cost accounting for V1.8."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.runtime.contracts import (
    CostEntry,
    PriceSnapshot,
    UsageSource,
    stable_id,
)


COST_FORMULA = (
    "(uncached_input_tokens*input_price_per_1m + "
    "cached_input_tokens*cached_input_price_per_1m + "
    "output_tokens*output_price_per_1m) / 1_000_000"
)


def _integer_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def build_cost_entry(
    stage: str,
    provider: str,
    requested_model: Optional[str],
    response_model: Optional[str],
    model_policy_version: str,
    provider_usage: Optional[Dict[str, Any]],
    price_row: Optional[Dict[str, Any]],
    local_estimate_tokens: Optional[int] = None,
) -> CostEntry:
    usage = provider_usage or {}
    input_tokens = _integer_or_none(usage.get("input_tokens"))
    cached_tokens = _integer_or_none(
        usage.get("cached_input_tokens", usage.get("cached_tokens"))
    )
    output_tokens = _integer_or_none(usage.get("output_tokens"))
    reasoning_tokens = _integer_or_none(usage.get("reasoning_tokens"))
    total_tokens = _integer_or_none(usage.get("total_tokens"))
    complete = all(
        value is not None
        for value in (input_tokens, cached_tokens, output_tokens, total_tokens)
    )

    source: UsageSource
    amount: Optional[float] = None
    formula: Optional[str] = None
    price: Optional[PriceSnapshot] = None
    availability_note: str
    currency: Optional[str] = None

    if complete:
        source = UsageSource.PROVIDER_REPORTED_COMPLETE
        if price_row:
            price_data = {
                "price_snapshot_id": stable_id(
                    "price",
                    {
                        "model_id": price_row.get("model_id"),
                        "effective_date": price_row.get("effective_date"),
                        "source_note": price_row.get("source_note"),
                    },
                ),
                "model_id": str(price_row.get("model_id") or response_model or requested_model or ""),
                "currency": price_row.get("currency"),
                "effective_date": price_row.get("effective_date"),
                "input_price_per_1m": price_row.get("input_price_per_1m"),
                "cached_input_price_per_1m": price_row.get("cached_input_price_per_1m"),
                "output_price_per_1m": price_row.get("output_price_per_1m"),
                "reasoning_price_per_1m": price_row.get("reasoning_price_per_1m"),
                "source_note": price_row.get("source_note"),
            }
            price = PriceSnapshot.model_validate(price_data)
            required_prices = (
                price.input_price_per_1m,
                price.cached_input_price_per_1m,
                price.output_price_per_1m,
            )
            if all(value is not None for value in required_prices):
                uncached = max(0, int(input_tokens or 0) - int(cached_tokens or 0))
                amount = round(
                    (
                        uncached * float(price.input_price_per_1m)
                        + int(cached_tokens or 0) * float(price.cached_input_price_per_1m)
                        + int(output_tokens or 0) * float(price.output_price_per_1m)
                    )
                    / 1_000_000,
                    8,
                )
                formula = COST_FORMULA
                currency = price.currency
                availability_note = "Provider usage 拆分完整，按发布时价格快照计算。"
            else:
                availability_note = "Provider usage 完整，但价格快照字段不完整，金额不可得。"
        else:
            availability_note = "Provider usage 完整，但没有匹配的价格快照，金额不可得。"
    elif total_tokens is not None:
        source = UsageSource.PROVIDER_REPORTED_TOTAL_ONLY
        availability_note = "Provider 仅返回总 Token；不推导输入/输出拆分，也不计算精确金额。"
    elif local_estimate_tokens is not None:
        source = UsageSource.LOCAL_ESTIMATE
        availability_note = "仅有本地 Token 估算；用于容量观察，不作为实际 Usage 或金额。"
    elif provider_usage is None:
        source = UsageSource.UNAVAILABLE
        availability_note = "Provider 未返回 Usage，且没有本地估算；成本不可得。"
    else:
        source = UsageSource.UNAVAILABLE
        availability_note = "Provider Usage 不可解析；成本不可得。"

    return CostEntry(
        stage=stage,
        provider=provider,
        requested_model=requested_model,
        response_model=response_model,
        model_policy_version=model_policy_version,
        usage_source=source,
        input_tokens=input_tokens if complete else None,
        cached_input_tokens=cached_tokens if complete else None,
        output_tokens=output_tokens if complete else None,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        local_estimate_tokens=local_estimate_tokens,
        price_snapshot=price,
        formula=formula,
        amount=amount,
        currency=currency,
        availability_note=availability_note,
    )
