"""Lightweight usage/cost helpers for model calls.

Ensures cost formulas are centralized and consistent across chat, observability,
and badcase modules.
"""

from typing import Any, Dict, Optional, Tuple


def normalize_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Return a uniform usage dict with explicit unknown handling.

    Fields:
      - uncached_input_tokens
      - cached_input_tokens
      - output_tokens
      - total_tokens
      - reasoning_tokens (optional, not used in cost formula)
      - usage_split_unavailable (bool)
    """
    total = usage.get("total_tokens")
    inp = usage.get("input_tokens")
    out = usage.get("output_tokens")
    cached = usage.get("cached_tokens")

    # Provider gave a split we can trust.
    split_available = (
        inp is not None
        and out is not None
        and cached is not None
        and total is not None
    )

    if split_available:
        uncached = max(0, int(inp) - int(cached))
        return {
            "uncached_input_tokens": uncached,
            "cached_input_tokens": int(cached),
            "output_tokens": int(out),
            "total_tokens": int(total),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "usage_split_unavailable": False,
        }

    # Fallback: total only or nothing at all.
    return {
        "uncached_input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "total_tokens": int(total) if total is not None else None,
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "usage_split_unavailable": True,
    }


def compute_cost_cny(
    price_snapshot: Optional[Dict[str, Any]], usage: Dict[str, Any]
) -> Tuple[Optional[float], str]:
    """Compute CNY cost using official DeepSeek pricing.

    Formula:
        (uncached_input * input_price_per_1m
         + cached_input * cached_input_price_per_1m
         + output * output_price_per_1m) / 1_000_000

    Returns (cost, status) where status is one of:
      - "computed"
      - "usage_split_unavailable"
      - "price_unavailable"
    """
    if not price_snapshot:
        return None, "price_unavailable"

    normalized = normalize_usage(usage)
    if normalized["usage_split_unavailable"]:
        return None, "usage_split_unavailable"

    uncached = normalized["uncached_input_tokens"]
    cached = normalized["cached_input_tokens"]
    output = normalized["output_tokens"]

    if uncached is None or cached is None or output is None:
        return None, "usage_split_unavailable"

    cost = 0.0
    if price_snapshot.get("input_price_per_1m") is not None:
        cost += uncached * (price_snapshot["input_price_per_1m"] / 1_000_000)
    if price_snapshot.get("cached_input_price_per_1m") is not None:
        cost += cached * (price_snapshot["cached_input_price_per_1m"] / 1_000_000)
    if price_snapshot.get("output_price_per_1m") is not None:
        cost += output * (price_snapshot["output_price_per_1m"] / 1_000_000)

    return round(cost, 8), "computed"


def build_price_snapshot(price_row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build a serializable price snapshot from a model_prices row."""
    if not price_row:
        return None
    return {
        "model_id": price_row.get("model_id"),
        "currency": price_row.get("currency"),
        "effective_date": price_row.get("effective_date"),
        "input_price_per_1m": price_row.get("input_price_per_1m"),
        "cached_input_price_per_1m": price_row.get("cached_input_price_per_1m"),
        "output_price_per_1m": price_row.get("output_price_per_1m"),
        "reasoning_price_per_1m": price_row.get("reasoning_price_per_1m"),
        "source_note": price_row.get("source_note"),
    }
