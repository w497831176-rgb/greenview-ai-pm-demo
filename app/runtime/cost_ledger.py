"""Honest per-stage cost accounting for V1.8.2-S5."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.runtime.contracts import CostEntry, PriceSnapshot, UsageSource, stable_id


COST_FORMULA = (
    "(input_cache_miss_tokens*input_price_per_1m + "
    "input_cache_hit_tokens*cached_input_price_per_1m + "
    "output_tokens*output_price_per_1m) / 1_000_000"
)


def _integer_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _first_integer(usage: Dict[str, Any], *names: str) -> Optional[int]:
    for name in names:
        value = _integer_or_none(usage.get(name))
        if value is not None:
            return value
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
    provider_response_model: Optional[str] = None,
    thinking_enabled: Optional[bool] = None,
    provider_succeeded: bool = True,
) -> CostEntry:
    """Build one immutable cost entry without manufacturing missing evidence."""
    usage = provider_usage or {}
    actual_model = provider_response_model or response_model
    input_tokens = _first_integer(usage, "input_tokens", "prompt_tokens")
    cache_hit = _first_integer(
        usage,
        "input_cache_hit_tokens",
        "prompt_cache_hit_tokens",
        "cached_input_tokens",
    )
    cache_miss = _first_integer(
        usage,
        "input_cache_miss_tokens",
        "prompt_cache_miss_tokens",
        "uncached_input_tokens",
    )
    output_tokens = _first_integer(usage, "output_tokens", "completion_tokens")
    reasoning_tokens = _integer_or_none(usage.get("reasoning_tokens"))
    total_tokens = _integer_or_none(usage.get("total_tokens"))
    split_complete = all(
        value is not None for value in (cache_hit, cache_miss, output_tokens)
    )

    price: Optional[PriceSnapshot] = None
    if price_row:
        price = PriceSnapshot.model_validate(
            {
                "price_snapshot_id": stable_id(
                    "price",
                    {
                        "model_id": price_row.get("model_id"),
                        "effective_date": price_row.get("effective_date"),
                        "source_note": price_row.get("source_note"),
                    },
                ),
                "model_id": str(
                    price_row.get("model_id") or actual_model or requested_model or ""
                ),
                "currency": price_row.get("currency"),
                "effective_date": price_row.get("effective_date"),
                "input_price_per_1m": price_row.get("input_price_per_1m"),
                "cached_input_price_per_1m": price_row.get(
                    "cached_input_price_per_1m"
                ),
                "output_price_per_1m": price_row.get("output_price_per_1m"),
                "reasoning_price_per_1m": price_row.get(
                    "reasoning_price_per_1m"
                ),
                "source_note": price_row.get("source_note"),
            }
        )

    source: UsageSource
    amount: Optional[float] = None
    formula: Optional[str] = None
    currency: Optional[str] = None

    if not provider_succeeded:
        source = UsageSource.UNAVAILABLE
        availability_note = "Provider 调用失败；保留失败状态，不记录成功成本。"
    elif split_complete:
        source = UsageSource.PROVIDER_ACTUAL
        if not actual_model:
            availability_note = (
                "Provider 返回了三类 Usage，但实际响应模型未返回/未采集；金额不可得。"
            )
        elif not price or price.model_id != actual_model:
            availability_note = (
                "Provider 返回了三类 Usage，但没有匹配实际响应模型的价格快照；金额不可得。"
            )
        elif all(
            value is not None
            for value in (
                price.input_price_per_1m,
                price.cached_input_price_per_1m,
                price.output_price_per_1m,
            )
        ):
            amount = round(
                (
                    int(cache_miss or 0) * float(price.input_price_per_1m)
                    + int(cache_hit or 0) * float(price.cached_input_price_per_1m)
                    + int(output_tokens or 0) * float(price.output_price_per_1m)
                )
                / 1_000_000,
                8,
            )
            formula = COST_FORMULA
            currency = price.currency
            availability_note = "Provider 三类 Usage 原样保存，并按实际响应模型价格快照计算。"
        else:
            availability_note = "三类 Usage 完整，但价格快照字段不完整；金额不可得。"
    elif total_tokens is not None:
        source = UsageSource.UNAVAILABLE
        availability_note = (
            "Provider 仅有总 Token；不反推缓存命中、缓存未命中或输出，金额不可得。"
        )
    elif local_estimate_tokens is not None:
        source = UsageSource.ESTIMATED
        availability_note = (
            "仅有本地 Token 估算；用于容量观察，不冒充 Provider Usage，也不计算金额。"
        )
    else:
        source = UsageSource.UNAVAILABLE
        availability_note = "Provider Usage 不可得，且没有可诚实展示的估算。"

    return CostEntry(
        stage=stage,
        provider=provider,
        requested_model=requested_model,
        response_model=actual_model,
        provider_response_model=actual_model,
        thinking_enabled=thinking_enabled,
        model_policy_version=model_policy_version,
        usage_source=source,
        input_tokens=input_tokens,
        cached_input_tokens=cache_hit,
        input_cache_hit_tokens=cache_hit,
        input_cache_miss_tokens=cache_miss,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        local_estimate_tokens=local_estimate_tokens,
        price_snapshot=price,
        formula=formula,
        amount=amount,
        currency=currency,
        availability_note=availability_note,
    )


def cost_entry_usage_payload(
    cost: CostEntry,
    provider_request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Serialize new evidence into the existing model_calls JSON column."""
    actual = cost.usage_source == UsageSource.PROVIDER_ACTUAL
    return {
        "requested_model": cost.requested_model,
        "provider_response_model": cost.provider_response_model,
        "provider_request_id": provider_request_id,
        "thinking_enabled": cost.thinking_enabled,
        "input_cache_hit_tokens": cost.input_cache_hit_tokens if actual else None,
        "input_cache_miss_tokens": cost.input_cache_miss_tokens if actual else None,
        "output_tokens": cost.output_tokens if actual else None,
        # Backward-compatible aliases consumed by the existing Trace UI.
        "cached_input_tokens": cost.input_cache_hit_tokens if actual else None,
        "uncached_input_tokens": cost.input_cache_miss_tokens if actual else None,
        "reasoning_tokens": cost.reasoning_tokens,
        "total_tokens": cost.total_tokens,
        "usage_source": cost.usage_source.value,
        "usage_split_unavailable": not actual,
        "cost_contract": cost.model_dump(mode="json"),
    }
