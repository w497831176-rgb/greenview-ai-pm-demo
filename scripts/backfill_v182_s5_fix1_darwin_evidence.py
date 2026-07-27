"""Precisely backfill the one approved S5 Darwin Trace without a model call."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from app.runtime.darwin_evidence import persist_darwin_operation
from app.runtime.cost_ledger import COST_FORMULA
from db.property_db import (
    get_badcase,
    get_evidence_ledger,
    get_model_call,
    get_skill_prompt_draft,
    list_badcase_actions,
)


TRACE_ID = "7a9ef83684304e95"
MODEL_CALL_ID = 944
BADCASE_ID = 663
DRAFT_ID = 17


def _json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value:
        parsed = json.loads(value)
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _started_at(model_call: Dict[str, Any]) -> str:
    completed = _parse_time(model_call.get("created_at"))
    latency_ms = model_call.get("latency_ms")
    if completed is None or latency_ms is None:
        return model_call.get("created_at") or "unavailable"
    return (completed - timedelta(milliseconds=int(latency_ms))).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _darwin_action() -> Optional[Dict[str, Any]]:
    matches = []
    for action in list_badcase_actions(BADCASE_ID):
        if action.get("action_type") != "darwin-fix":
            continue
        detail = _json_object(action.get("action_detail"))
        if detail.get("darwin_trace_id") == TRACE_ID:
            return action
        matches.append(action)
    return matches[-1] if len(matches) == 1 else None


def main() -> None:
    badcase = get_badcase(BADCASE_ID)
    model_call = get_model_call(MODEL_CALL_ID)
    draft = get_skill_prompt_draft(DRAFT_ID)
    if not badcase or not model_call or not draft:
        raise RuntimeError("approved backfill source record is missing")
    if badcase.get("darwin_trace_id") != TRACE_ID or badcase.get("status") != "fixing":
        raise RuntimeError("Badcase #663 no longer matches the approved fixing evidence")
    if (
        model_call.get("trace_id") != TRACE_ID
        or model_call.get("stage") != "darwin"
        or model_call.get("status") != "success"
    ):
        raise RuntimeError("model_call #944 no longer matches the approved Darwin evidence")
    if int(draft.get("badcase_id") or -1) != BADCASE_ID or draft.get("status") != "draft":
        raise RuntimeError("draft #17 no longer matches the approved draft evidence")
    if draft.get("published_at") or draft.get("published_by"):
        raise RuntimeError("draft #17 has been published; refusing historical backfill")

    usage = _json_object(model_call.get("usage_normalized"))
    exact_contract = {
        "requested_model": "deepseek-v4-pro",
        "provider_response_model": "deepseek-v4-pro",
        "usage_source": "provider_actual",
        "input_cache_hit_tokens": 2688,
        "input_cache_miss_tokens": 24,
        "output_tokens": 1958,
        "amount": 0.0118872,
    }
    actual_contract = {
        "requested_model": usage.get("requested_model") or model_call.get("model_id"),
        "provider_response_model": usage.get("provider_response_model"),
        "usage_source": model_call.get("usage_source"),
        "input_cache_hit_tokens": usage.get("input_cache_hit_tokens"),
        "input_cache_miss_tokens": usage.get("input_cache_miss_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "amount": model_call.get("estimated_cost_cny"),
    }
    if actual_contract != exact_contract:
        raise RuntimeError("model_call #944 cost evidence differs from the approved facts")
    cost_contract = usage.get("cost_contract")
    if not isinstance(cost_contract, dict):
        raise RuntimeError("model_call #944 has no stored CostEntry")
    price = cost_contract.get("price_snapshot") or {}
    if (
        not usage.get("provider_request_id")
        or cost_contract.get("formula") != COST_FORMULA
        or cost_contract.get("amount") != 0.0118872
        or price.get("model_id") != "deepseek-v4-pro"
        or price.get("input_price_per_1m") != 3.0
        or price.get("cached_input_price_per_1m") != 0.025
        or price.get("output_price_per_1m") != 6.0
    ):
        raise RuntimeError("model_call #944 request/cost snapshot is incomplete or changed")

    action = _darwin_action()
    completed_at = (
        (action or {}).get("created_at")
        or draft.get("updated_at")
        or model_call.get("created_at")
        or "unavailable"
    )
    status_before = (action or {}).get("status_before") or "classified"
    status_after = (action or {}).get("status_after") or "fixing"
    persist_darwin_operation(
        trace_id=TRACE_ID,
        badcase_id=BADCASE_ID,
        model_call=model_call,
        operation_status="complete",
        started_at=_started_at(model_call),
        completed_at=completed_at,
        drafts=[{"type": "skill_prompt", "draft": draft}],
        status_before=status_before,
        status_after=status_after,
    )
    ledger = get_evidence_ledger(TRACE_ID)
    if not ledger or ledger.get("status") != "complete":
        raise RuntimeError("backfill did not persist a complete Evidence Ledger")
    print(
        json.dumps(
            {
                "trace_id": TRACE_ID,
                "model_call_id": MODEL_CALL_ID,
                "badcase_id": BADCASE_ID,
                "draft_id": DRAFT_ID,
                "trace_status": "complete",
                "ledger_status": ledger.get("status"),
                "provider_calls": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
