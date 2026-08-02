"""
Observability & Cost Governance API
===================================

Endpoints for trace visibility, model-call auditing, MCP audit,
model pricing table, and budget thresholds.
"""
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from db.property_db import (
    create_model_price,
    delete_model_price,
    evaluation_summary,
    get_badcase_id_by_trace_id,
    get_budget_thresholds,
    get_chat_trace,
    get_evaluation_run_by_trace_id,
    get_mcp_call_audits_for_trace,
    get_model_call,
    get_model_calls_for_trace,
    get_model_price,
    list_chat_messages,
    list_chat_traces,
    list_trace_events,
    list_model_prices,
    update_budget_thresholds,
    update_model_price,
    _get_conn,
    now_cn,
    now_cn_dt,
)

router = APIRouter(prefix="/api/observability", tags=["observability"])


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------


class PriceCreate(BaseModel):
    model_id: str
    effective_date: str
    input_price_per_1m: Optional[float] = None
    cached_input_price_per_1m: Optional[float] = None
    output_price_per_1m: Optional[float] = None
    reasoning_price_per_1m: Optional[float] = None
    source_note: Optional[str] = None
    enabled: bool = True


class PriceUpdate(BaseModel):
    model_id: Optional[str] = None
    effective_date: Optional[str] = None
    input_price_per_1m: Optional[float] = None
    cached_input_price_per_1m: Optional[float] = None
    output_price_per_1m: Optional[float] = None
    reasoning_price_per_1m: Optional[float] = None
    source_note: Optional[str] = None
    enabled: Optional[bool] = None


class BudgetUpdate(BaseModel):
    per_call_threshold_cny: Optional[float] = None
    daily_threshold_cny: Optional[float] = None
    monthly_threshold_cny: Optional[float] = None


# -----------------------------------------------------------------------------
# Cost/budget helpers
# -----------------------------------------------------------------------------


REPORTING_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


def _parse_reporting_timepoint(value: Any) -> datetime:
    """Parse one stored/reporting timestamp into an aware UTC timepoint.

    Historical YIAI rows use naive ``YYYY-MM-DD HH:MM:SS`` Beijing time,
    while Provider attempts use offset-aware ISO timestamps.  Converting both
    to a real timepoint is required; lexical TEXT comparison is incorrect.
    """
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            raise ValueError("时间不能为空")
        if len(text) == 10:
            try:
                parsed = datetime.strptime(text, "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError(f"无效时间: {text}") from exc
        else:
            normalized = f"{text[:-1]}+00:00" if text.endswith(("Z", "z")) else text
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError as exc:
                raise ValueError(f"无效时间: {text}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=REPORTING_TIMEZONE)
    return parsed.astimezone(timezone.utc)


def _reporting_epoch(value: Any) -> Optional[float]:
    """SQLite-safe epoch converter; malformed stored values do not match."""
    try:
        return _parse_reporting_timepoint(value).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _required_reporting_epoch(value: Any, label: str) -> float:
    epoch = _reporting_epoch(value)
    if epoch is None:
        raise ValueError(f"{label}无效")
    return epoch


def _time_range_predicates(
    column: str,
    start: Optional[str],
    end: Optional[str],
) -> Tuple[List[str], List[Any]]:
    conditions: List[str] = []
    params: List[Any] = []
    if start:
        conditions.append(f"yiai_time_epoch({column}) >= ?")
        params.append(_required_reporting_epoch(start, "开始时间"))
    if end:
        conditions.append(f"yiai_time_epoch({column}) <= ?")
        params.append(_required_reporting_epoch(end, "结束时间"))
    return conditions, params


def _register_time_sql(conn: Any) -> None:
    conn.create_function("yiai_time_epoch", 1, _reporting_epoch, deterministic=True)


def _model_display_name(model_id: Optional[str]) -> str:
    return {
        "deepseek-v4-flash": "Flash",
        "deepseek-v4-pro": "Pro",
    }.get(model_id or "") or (model_id or "unknown")


def _format_reporting_now(value: datetime) -> str:
    """Keep the real current instant without inventing an end-of-second gap."""
    return value.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")


def _reporting_period(value: Any, group_by: str) -> str:
    """Bucket a stored timestamp by its real Asia/Shanghai timepoint."""
    try:
        local_time = _parse_reporting_timepoint(value).astimezone(
            REPORTING_TIMEZONE
        )
    except (TypeError, ValueError, OverflowError):
        return "unknown"
    if group_by == "hour":
        return local_time.strftime("%Y-%m-%d %H:00")
    return local_time.strftime("%Y-%m-%d")


def _period_bounds() -> Dict[str, Dict[str, Any]]:
    """Return canonical CN-time period bounds used by the overview."""
    dt = now_cn_dt()
    today_start = dt.strftime("%Y-%m-%d 00:00:00")
    today_end = _format_reporting_now(dt)
    week_start = (dt - timedelta(days=6)).strftime("%Y-%m-%d 00:00:00")
    month_start = dt.replace(day=1).strftime("%Y-%m-%d 00:00:00")
    return {
        "today": {"start": today_start, "end": today_end, "days": 1},
        "last_7_days": {"start": week_start, "end": today_end, "days": 7},
        "this_month": {"start": month_start, "end": today_end, "days": dt.day},
    }


def _overview_scope(
    start: Optional[str],
    end: Optional[str],
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Return an explicit Asia/Shanghai reporting range.

    With no filters the page means *today so far*, never all history.
    """
    current = now or now_cn_dt()
    if not start and not end:
        return {
            "label": "今日",
            "start": current.strftime("%Y-%m-%d 00:00:00"),
            "end": _format_reporting_now(current),
            "timezone": "Asia/Shanghai (UTC+8)",
        }
    return {
        "label": "自定义范围",
        "start": start,
        "end": _normalize_end(end) if end else None,
        "timezone": "Asia/Shanghai (UTC+8)",
    }


def _reporting_scope(
    range_key: Optional[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Resolve one canonical Asia/Shanghai reporting range."""
    current = now or now_cn_dt()
    key = range_key or ("custom" if start or end else "today")
    current_end = _format_reporting_now(current)

    if key == "today":
        scope = {
            "label": "今天",
            "start": current.strftime("%Y-%m-%d 00:00:00"),
            "end": current_end,
        }
    elif key == "yesterday":
        day = current - timedelta(days=1)
        scope = {
            "label": "昨天",
            "start": day.strftime("%Y-%m-%d 00:00:00"),
            "end": day.strftime("%Y-%m-%d 23:59:59.999999"),
        }
    elif key == "last_7_days":
        scope = {
            "label": "近7天",
            "start": (current - timedelta(days=6)).strftime("%Y-%m-%d 00:00:00"),
            "end": current_end,
        }
    elif key == "this_month":
        scope = {
            "label": "本月",
            "start": current.replace(day=1).strftime("%Y-%m-%d 00:00:00"),
            "end": current_end,
        }
    elif key == "last_month":
        this_month = current.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        previous_end = this_month - timedelta(seconds=1)
        previous_start = previous_end.replace(day=1, hour=0, minute=0, second=0)
        scope = {
            "label": "上月",
            "start": previous_start.strftime("%Y-%m-%d 00:00:00"),
            "end": previous_end.strftime("%Y-%m-%d 23:59:59.999999"),
        }
    elif key == "custom":
        if not start or not end:
            raise ValueError("自定义日期需要同时提供开始日期和结束日期")
        normalized_end = _normalize_end(end)
        if _required_reporting_epoch(start, "开始时间") > _required_reporting_epoch(
            normalized_end, "结束时间"
        ):
            raise ValueError("开始日期不能晚于结束日期")
        scope = {
            "label": f"{str(start)[:10]}至{str(normalized_end)[:10]}期间",
            "start": start,
            "end": normalized_end,
        }
    else:
        raise ValueError(f"不支持的时间范围: {key}")

    return {
        **scope,
        "range_key": key,
        "timezone": "Asia/Shanghai (UTC+8)",
    }


def _usage_payload(call: Dict[str, Any]) -> Dict[str, Any]:
    value = call.get("usage_normalized") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            value = {}
    return value if isinstance(value, dict) else {}


def _first_present(mapping: Dict[str, Any], *keys: str) -> Any:
    """Return the first explicitly present value, preserving False and zero."""
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (1, "1", "true", "True", "yes", "on"):
        return True
    if value in (0, "0", "false", "False", "no", "off"):
        return False
    return None


def _provider_aggregate_decision(call: Dict[str, Any]) -> Dict[str, Any]:
    """Return the single source of truth for Provider aggregate membership.

    The per-request accounting contract starts with records that explicitly opt
    in as ``provider_attempt``.  Historical rows remain visible in trace detail,
    but are never silently upgraded into the new reconciliation ledger.
    """
    usage = _usage_payload(call)
    stage = str(call.get("stage") or usage.get("stage") or "").strip().lower()
    status = str(call.get("status") or usage.get("status") or "").strip().lower()
    record_kind = str(
        call.get("record_kind")
        or usage.get("record_kind")
        or usage.get("record_type")
        or ""
    ).strip().lower()
    usage_status = str(
        call.get("usage_status")
        or usage.get("usage_status")
        or ""
    ).strip().lower()
    include_value = _first_present(
        usage,
        "include_in_provider_aggregate",
        # Read-only compatibility for early development rows; new writers use
        # the singular ``include_`` spelling above.
        "included_in_provider_aggregate",
    )
    include_flag = _optional_bool(include_value)
    request_sent = _optional_bool(
        _first_present(usage, "provider_request_sent", "request_sent")
    )

    reason = "included_provider_attempt"
    included = True
    if stage == "retest":
        included, reason = False, "logical_retest_aggregate"
    elif status == "blocked":
        included, reason = False, "blocked_before_provider"
    elif record_kind in {
        "logical",
        "logical_aggregate",
        "business_aggregate",
        "not_applicable",
        "legacy",
    }:
        included, reason = False, f"record_kind_{record_kind}"
    elif usage_status == "not_applicable":
        included, reason = False, "usage_not_applicable"
    elif not record_kind:
        included, reason = False, "legacy_record_not_upgraded"
    elif record_kind != "provider_attempt":
        included, reason = False, "record_kind_not_provider_attempt"
    elif request_sent is not True:
        included, reason = False, "provider_request_sent_not_confirmed"
    elif include_flag is not True:
        included, reason = False, "provider_aggregate_flag_not_true"

    return {
        "included": included,
        "reason": reason,
        "record_kind": record_kind or "legacy",
        "include_in_provider_aggregate": include_flag,
        "request_sent": request_sent,
        "legacy": not bool(record_kind),
    }


def _is_provider_aggregate_record(call: Dict[str, Any]) -> bool:
    return bool(_provider_aggregate_decision(call)["included"])


def _split_model_records(
    calls: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    provider_attempts: List[Dict[str, Any]] = []
    logical_or_legacy: List[Dict[str, Any]] = []
    for call in calls:
        target = provider_attempts if _is_provider_aggregate_record(call) else logical_or_legacy
        target.append(call)
    return provider_attempts, logical_or_legacy


def _sqlite_provider_aggregate(
    record_kind: Any,
    stage: Any,
    status: Any,
    usage_status: Any,
    usage_normalized: Any,
) -> int:
    call = {
        "record_kind": record_kind,
        "stage": stage,
        "status": status,
        "usage_status": usage_status,
        "usage_normalized": usage_normalized,
    }
    return 1 if _is_provider_aggregate_record(call) else 0


def _sqlite_provider_attempt(record_kind: Any, usage_normalized: Any) -> int:
    call = {
        "record_kind": record_kind,
        "usage_normalized": usage_normalized,
    }
    return (
        1
        if _provider_aggregate_decision(call)["record_kind"] == "provider_attempt"
        else 0
    )


def _register_reporting_sql(conn: Any) -> None:
    _register_time_sql(conn)
    conn.create_function(
        "yiai_provider_aggregate",
        5,
        _sqlite_provider_aggregate,
        deterministic=True,
    )
    conn.create_function(
        "yiai_provider_attempt",
        2,
        _sqlite_provider_attempt,
        deterministic=True,
    )


def _provider_sql_predicate(alias: str) -> str:
    return (
        f"yiai_provider_aggregate({alias}.record_kind, {alias}.stage, "
        f"{alias}.status, {alias}.usage_status, {alias}.usage_normalized) = 1"
    )


def _provider_attempt_sql_predicate(alias: str) -> str:
    return (
        f"yiai_provider_attempt({alias}.record_kind, "
        f"{alias}.usage_normalized) = 1"
    )


def _compact_identifier(value: Any, *, fallback: str = "unassigned") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    if len(text) <= 20:
        return text
    return f"{text[:10]}...{text[-6:]}"


def _optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _provider_usage_inconsistency(call: Dict[str, Any]) -> Tuple[bool, List[str]]:
    usage = _usage_payload(call)
    reasons = usage.get("provider_usage_inconsistent_reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    reasons = [str(item) for item in reasons if str(item or "").strip()]
    derived: List[str] = []
    hit = _optional_int(
        _token_value(call, "cache_hit_input_tokens", "input_cache_hit_tokens")
    )
    miss = _optional_int(
        _token_value(call, "cache_miss_input_tokens", "input_cache_miss_tokens")
    )
    output = _optional_int(_token_value(call, "output_tokens"))
    reasoning = _optional_int(_token_value(call, "reasoning_tokens"))
    total = _optional_int(_token_value(call, "total_tokens"))
    provider_input = _optional_int(
        _token_value(call, "input_tokens", "prompt_tokens")
    )
    known_values = [value for value in (hit, miss, output, reasoning, total) if value is not None]
    if any(value < 0 for value in known_values):
        derived.append("negative_provider_usage_value")
    if None not in (hit, miss, output, total) and total != hit + miss + output:
        derived.append("provider_total_not_equal_three_class_sum")
    if None not in (provider_input, hit, miss) and provider_input != hit + miss:
        derived.append("provider_input_not_equal_cache_split")
    if None not in (reasoning, output) and reasoning > output:
        derived.append("reasoning_exceeds_output")
    reasons = list(dict.fromkeys([*reasons, *derived]))
    inconsistent = bool(
        usage.get("provider_usage_inconsistent")
        or str(usage.get("reconciliation_status") or "").strip().lower()
        == "provider_usage_inconsistent"
        or reasons
    )
    return inconsistent, reasons


def _data_quality_summary(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Explain reporting exclusions without upgrading them into Provider calls."""
    counts = {
        "invalid_timestamp_count": 0,
        "provider_send_unconfirmed_count": 0,
        "orphaned_pending_count": 0,
        "provider_usage_inconsistent_count": 0,
        "provider_actual_price_missing_count": 0,
        "unavailable_usage_count": 0,
        "unresolved_reconciliation_count": 0,
    }
    samples: List[Dict[str, Any]] = []
    reasoning_comparable_calls = 0
    reasoning_unknown_calls = 0
    reasoning_violation_calls = 0
    anomaly_attempt_count = 0

    for raw in calls:
        call = dict(raw)
        decision = _provider_aggregate_decision(call)
        if decision["record_kind"] != "provider_attempt":
            continue
        usage = _usage_payload(call)
        usage_status = str(
            call.get("usage_status") or usage.get("usage_status") or ""
        ).strip().lower()
        issues: List[str] = []

        if _reporting_epoch(call.get("created_at")) is None:
            counts["invalid_timestamp_count"] += 1
            issues.append("invalid_timestamp")

        if decision.get("request_sent") is not True:
            counts["provider_send_unconfirmed_count"] += 1
            issues.append("provider_send_unconfirmed")
        if usage_status in {"orphaned_pending", "cause_unconfirmed", "pending"}:
            counts["orphaned_pending_count"] += 1
            issues.append("orphaned_pending")

        inconsistent, inconsistency_reasons = _provider_usage_inconsistency(call)
        if inconsistent:
            counts["provider_usage_inconsistent_count"] += 1
            issues.append("provider_usage_inconsistent")

        token_source = _token_source(call)
        core_tokens = (
            _token_value(call, "cache_hit_input_tokens", "input_cache_hit_tokens"),
            _token_value(call, "cache_miss_input_tokens", "input_cache_miss_tokens"),
            _token_value(call, "output_tokens"),
        )
        if (
            token_source == "provider_actual"
            and None not in core_tokens
            and _provider_direct_cost(call) is None
        ):
            counts["provider_actual_price_missing_count"] += 1
            issues.append("provider_actual_price_missing")

        if decision.get("request_sent") is True and usage_status != "provider_actual":
            counts["unavailable_usage_count"] += 1
            issues.append("provider_usage_unavailable")

        if decision.get("included") and token_source == "provider_actual":
            output = _token_value(call, "output_tokens")
            reasoning = _token_value(call, "reasoning_tokens")
            if output is None or reasoning is None:
                reasoning_unknown_calls += 1
            else:
                reasoning_comparable_calls += 1
                if int(reasoning) < 0 or int(reasoning) > int(output):
                    reasoning_violation_calls += 1

        unresolved_codes = {
            "provider_send_unconfirmed",
            "orphaned_pending",
            "provider_usage_inconsistent",
            "provider_actual_price_missing",
            "provider_usage_unavailable",
        }
        if any(code in unresolved_codes for code in issues):
            counts["unresolved_reconciliation_count"] += 1

        if issues:
            anomaly_attempt_count += 1
        if issues and len(samples) < 20:
            timestamp_assignable = "invalid_timestamp" not in issues
            included_in_provider_summary = bool(decision.get("included")) and timestamp_assignable
            if not timestamp_assignable:
                exclusion_reason = "invalid_timestamp_range_unassignable"
            elif included_in_provider_summary:
                exclusion_reason = None
            else:
                exclusion_reason = decision.get("reason")
            samples.append(
                {
                    "trace_id_summary": _compact_identifier(
                        call.get("trace_id"), fallback="trace-unassigned"
                    ),
                    "local_attempt_id_summary": _compact_identifier(
                        usage.get("local_attempt_id")
                        or call.get("local_attempt_id"),
                        fallback=f"row-{call.get('id', 'unknown')}",
                    ),
                    "stage": call.get("stage") or usage.get("stage") or "unknown",
                    "created_at_status": (
                        "invalid" if "invalid_timestamp" in issues else "valid"
                    ),
                    "usage_status": usage_status or "cause_unconfirmed",
                    "provider_request_sent": decision.get("request_sent"),
                    "included_in_provider_summary": included_in_provider_summary,
                    "exclusion_reason": exclusion_reason,
                    "issue_codes": issues,
                    "provider_usage_inconsistent_reasons": inconsistency_reasons,
                }
            )

    if reasoning_violation_calls:
        reasoning_is_output_subset: Optional[bool] = False
    elif reasoning_comparable_calls and not reasoning_unknown_calls:
        reasoning_is_output_subset = True
    else:
        reasoning_is_output_subset = None

    if counts["invalid_timestamp_count"] or counts[
        "provider_usage_inconsistent_count"
    ]:
        status = "data_quality_error"
    elif counts["unresolved_reconciliation_count"]:
        status = "reconciliation_attention"
    else:
        status = "normal"

    return {
        "data_quality_status": status,
        **counts,
        "anomaly_attempt_count": anomaly_attempt_count,
        "anomaly_samples_truncated": anomaly_attempt_count > len(samples),
        "anomaly_attempts": samples,
        "reasoning_comparable_calls": reasoning_comparable_calls,
        "reasoning_unknown_calls": reasoning_unknown_calls,
        "reasoning_violation_calls": reasoning_violation_calls,
        "reasoning_is_output_subset": reasoning_is_output_subset,
    }


def _token_value(call: Dict[str, Any], *keys: str) -> Any:
    usage = _usage_payload(call)
    value = _first_present(usage, *keys)
    if value is not None:
        return value
    return _first_present(call, *keys)


def _token_source(call: Dict[str, Any]) -> str:
    usage = _usage_payload(call)
    status = str(
        call.get("usage_status") or usage.get("usage_status") or ""
    ).strip().lower()
    source = str(
        usage.get("token_source")
        or call.get("usage_source")
        or usage.get("usage_source")
        or ""
    ).strip().lower()
    if status == "provider_actual" or source == "provider_actual":
        return "provider_actual"
    if source == "estimated" or status == "estimated":
        return "estimated"
    return "unavailable"


def _provider_model(call: Dict[str, Any]) -> str:
    usage = _usage_payload(call)
    return (
        usage.get("provider_actual_model")
        or usage.get("provider_response_model")
        or call.get("provider_response_model")
        or usage.get("requested_model")
        or call.get("model_id")
        or "unknown"
    )


def _cost_bucket(call: Dict[str, Any]) -> str:
    """Classify token evidence; cost may still be unavailable for actual usage."""
    return _token_source(call)


def _price_snapshot_payload(call: Dict[str, Any]) -> Dict[str, Any]:
    usage = _usage_payload(call)
    value = usage.get("price_snapshot") or call.get("price_snapshot") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            value = {}
    return value if isinstance(value, dict) else {}


def _provider_direct_cost(call: Dict[str, Any]) -> Optional[float]:
    """Return only cost backed by a frozen platform price snapshot."""
    usage = _usage_payload(call)
    cost_source = str(
        usage.get("cost_source") or call.get("cost_source") or ""
    ).strip().lower()
    if cost_source != "platform_price_snapshot" or not _price_snapshot_payload(call):
        return None
    amount = _first_present(usage, "calculated_direct_cost")
    if amount is None:
        amount = call.get("estimated_cost_cny")
    try:
        return None if amount is None else float(amount)
    except (TypeError, ValueError, OverflowError):
        return None


def _empty_cost_group() -> Dict[str, Any]:
    return {
        "calls": 0,
        "total_tokens": 0,
        "token_known_calls": 0,
        "token_unavailable_calls": 0,
        "provider_actual_calls": 0,
        "provider_actual_priced_calls": 0,
        "provider_actual_cost_cny": 0.0,
        "estimated_calls": 0,
        "estimated_cost_cny": 0.0,
        "estimated_amount_unavailable_calls": 0,
        "unavailable_calls": 0,
        "known_usage_calls": 0,
        "input_cache_hit_tokens": 0,
        "input_cache_miss_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "reasoning_known_calls": 0,
        "reasoning_unavailable_calls": 0,
        "reasoning_comparable_calls": 0,
        "reasoning_violation_calls": 0,
    }


def _add_call_to_group(group: Dict[str, Any], call: Dict[str, Any]) -> None:
    group["calls"] += 1
    bucket = _cost_bucket(call)
    amount = (
        _provider_direct_cost(call)
        if bucket == "provider_actual"
        else call.get("estimated_cost_cny")
    )
    if bucket == "provider_actual":
        group["provider_actual_calls"] += 1
        total_tokens = _token_value(call, "total_tokens")
        if total_tokens is None:
            group["token_unavailable_calls"] += 1
        else:
            group["token_known_calls"] += 1
            group["total_tokens"] += int(total_tokens)
        if amount is None:
            group["unavailable_calls"] += 1
        else:
            group["provider_actual_priced_calls"] += 1
            group["provider_actual_cost_cny"] += float(amount)
    elif bucket == "estimated":
        group["estimated_calls"] += 1
        group["token_unavailable_calls"] += 1
        if amount is None:
            group["estimated_amount_unavailable_calls"] += 1
        else:
            group["estimated_cost_cny"] += float(amount)
    else:
        group["token_unavailable_calls"] += 1
        group["unavailable_calls"] += 1

    hit = _token_value(call, "cache_hit_input_tokens", "input_cache_hit_tokens")
    miss = _token_value(call, "cache_miss_input_tokens", "input_cache_miss_tokens")
    output = _token_value(call, "output_tokens")
    if bucket == "provider_actual" and None not in (hit, miss, output):
        group["known_usage_calls"] += 1
        group["input_cache_hit_tokens"] += int(hit)
        group["input_cache_miss_tokens"] += int(miss)
        group["output_tokens"] += int(output)
        reasoning = _token_value(call, "reasoning_tokens")
        if reasoning is None:
            group["reasoning_unavailable_calls"] += 1
        else:
            group["reasoning_known_calls"] += 1
            group["reasoning_tokens"] += int(reasoning)
            group["reasoning_comparable_calls"] += 1
            if int(reasoning) < 0 or int(reasoning) > int(output):
                group["reasoning_violation_calls"] += 1


def _finalize_cost_group(group: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(group)
    if result["token_known_calls"] == 0:
        result["total_tokens"] = None
    if result["reasoning_known_calls"] == 0:
        result["reasoning_tokens"] = None
    if result["reasoning_violation_calls"]:
        result["reasoning_is_output_subset"] = False
    elif (
        result["reasoning_comparable_calls"]
        and not result["reasoning_unavailable_calls"]
    ):
        result["reasoning_is_output_subset"] = True
    else:
        result["reasoning_is_output_subset"] = None
    result["provider_actual_cost_cny"] = round(
        float(result["provider_actual_cost_cny"]), 8
    )
    result["estimated_cost_cny"] = round(float(result["estimated_cost_cny"]), 8)
    result["platform_price_snapshot_direct_cost_cny"] = (
        result["provider_actual_cost_cny"]
        if (
            not result["provider_actual_calls"]
            or result["provider_actual_priced_calls"]
        )
        else None
    )
    result["cost_source"] = "platform_price_snapshot"
    result["cost_complete"] = result["unavailable_calls"] == 0
    result["token_complete"] = result["token_unavailable_calls"] == 0
    return result


def _aggregate_model_calls(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = _empty_cost_group()
    by_model: Dict[str, Dict[str, Any]] = {}
    by_stage: Dict[str, Dict[str, Any]] = {}
    failed_calls = 0
    provider_calls, excluded_calls = _split_model_records(calls)
    for raw in provider_calls:
        call = dict(raw)
        _add_call_to_group(total, call)
        model_id = _provider_model(call)
        _add_call_to_group(by_model.setdefault(model_id, _empty_cost_group()), call)
        stage = call.get("stage") or "unknown"
        stage = "ab_test" if stage in {"ab_test_a", "ab_test_b"} else stage
        _add_call_to_group(by_stage.setdefault(stage, _empty_cost_group()), call)
        if call.get("status") != "success":
            failed_calls += 1
    result = _finalize_cost_group(total)
    result["failed_calls"] = failed_calls
    result["excluded_record_count"] = len(excluded_calls)
    result["by_model"] = {
        key: _finalize_cost_group(value) for key, value in by_model.items()
    }
    result["by_stage"] = {
        key: _finalize_cost_group(value) for key, value in by_stage.items()
    }
    return result


def _fetch_model_calls(
    start: Optional[str],
    end: Optional[str],
    *,
    model_id: Optional[str] = None,
    stage: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conditions, params = _time_range_predicates("created_at", start, end)
    if model_id:
        conditions.append("model_id = ?")
        params.append(model_id)
    if stage:
        conditions.append("stage = ?")
        params.append(stage)
    if trace_id:
        conditions.append("trace_id = ?")
        params.append(trace_id)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    conn = _get_conn()
    _register_reporting_sql(conn)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM model_calls {where}", params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def _fetch_reporting_model_calls(
    start: Optional[str],
    end: Optional[str],
    *,
    model_id: Optional[str] = None,
    stage: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return scoped records plus Provider attempts whose time is unassignable.

    Invalid or NULL timestamps cannot honestly be assigned to a reporting day.
    They remain globally visible to every scoped reconciliation query and are
    never included in the normal Provider request aggregate.
    """
    range_conditions, params = _time_range_predicates("created_at", start, end)
    conditions: List[str] = []
    if range_conditions:
        valid_range = " AND ".join(range_conditions)
        conditions.append(
            f"(({valid_range}) OR "
            f"({_provider_attempt_sql_predicate('model_calls')} "
            "AND yiai_time_epoch(created_at) IS NULL))"
        )
    if model_id:
        conditions.append("model_id = ?")
        params.append(model_id)
    if stage:
        conditions.append("stage = ?")
        params.append(stage)
    if trace_id:
        conditions.append("trace_id = ?")
        params.append(trace_id)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    conn = _get_conn()
    _register_reporting_sql(conn)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM model_calls {where}", params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def _statistics_status(
    *,
    scope_consistent: bool,
    data_quality_status: str,
) -> str:
    if not scope_consistent:
        return "scope_mismatch"
    if data_quality_status == "data_quality_error":
        return "data_quality_error"
    if data_quality_status == "reconciliation_attention":
        return "reconciliation_attention"
    return "consistent"


def _top_provider_actual_traces(
    calls: List[Dict[str, Any]], limit: int = 5
) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    provider_calls, _ = _split_model_records(calls)
    for call in provider_calls:
        trace_id = call.get("trace_id")
        if not trace_id:
            continue
        item = grouped.setdefault(
            trace_id,
            {
                "trace_id": trace_id,
                "all_calls": 0,
                "provider_actual_calls": 0,
                "total_tokens": 0,
                "token_unavailable_calls": 0,
                "provider_actual_cost_cny": 0.0,
                "models": set(),
                "stages": set(),
            },
        )
        item["all_calls"] += 1
        direct_cost = _provider_direct_cost(call)
        if _cost_bucket(call) != "provider_actual" or direct_cost is None:
            continue
        item["provider_actual_calls"] += 1
        total_tokens = _token_value(call, "total_tokens")
        if total_tokens is None:
            item["token_unavailable_calls"] += 1
        else:
            item["total_tokens"] += int(total_tokens)
        item["provider_actual_cost_cny"] += direct_cost
        item["models"].add(_provider_model(call))
        item["stages"].add(call.get("stage") or "unknown")

    eligible = [
        item
        for item in grouped.values()
        if item["all_calls"] == item["provider_actual_calls"]
        and item["provider_actual_cost_cny"] > 0
    ]
    eligible.sort(key=lambda item: item["provider_actual_cost_cny"], reverse=True)
    results = []
    for item in eligible[:limit]:
        results.append(
            {
                **item,
                "provider_actual_cost_cny": round(
                    item["provider_actual_cost_cny"], 8
                ),
                "platform_price_snapshot_direct_cost_cny": round(
                    item["provider_actual_cost_cny"], 8
                ),
                "total_tokens": (
                    item["total_tokens"]
                    if item["provider_actual_calls"] > item["token_unavailable_calls"]
                    else None
                ),
                "models": sorted(item["models"]),
                "stages": sorted(item["stages"]),
            }
        )
    return results


def _query_period_summary(start: str, end: str) -> Dict[str, Any]:
    return _aggregate_model_calls(_fetch_model_calls(start, end))


def _check_budget(strategy: Optional[str] = None) -> Dict[str, Any]:
    """Fail closed for optional paid background work when cost is unknowable."""
    thresholds: Dict[str, Any] = {}
    try:
        thresholds = get_budget_thresholds()
        daily_threshold = thresholds.get("daily_threshold_cny")
        monthly_threshold = thresholds.get("monthly_threshold_cny")
        bounds = _period_bounds()
        today = _query_period_summary(
            bounds["today"]["start"], bounds["today"]["end"]
        )
        month = _query_period_summary(
            bounds["this_month"]["start"], bounds["this_month"]["end"]
        )
        today_quality = _data_quality_summary(
            _fetch_reporting_model_calls(
                bounds["today"]["start"], bounds["today"]["end"]
            )
        )
        month_quality = _data_quality_summary(
            _fetch_reporting_model_calls(
                bounds["this_month"]["start"], bounds["this_month"]["end"]
            )
        )
        if (
            today.get("cost_complete") is not True
            or month.get("cost_complete") is not True
            or today_quality["data_quality_status"] != "normal"
            or month_quality["data_quality_status"] != "normal"
        ):
            reason_code = (
                "budget_data_quality_error"
                if "data_quality_error"
                in {
                    today_quality["data_quality_status"],
                    month_quality["data_quality_status"],
                }
                else "budget_reconciliation_attention"
                if "reconciliation_attention"
                in {
                    today_quality["data_quality_status"],
                    month_quality["data_quality_status"],
                }
                else "budget_cost_incomplete"
            )
            return {
                "budget_status": "unavailable",
                "daily_usage_percent": None,
                "monthly_usage_percent": None,
                "alert_level": "unavailable",
                "reason": "预算账本当前无法核实，本次额外付费后台操作未执行",
                "reason_code": reason_code,
                "error_type": None,
                "trigger_dimension": None,
                "today_cost": None,
                "month_cost": None,
                "cost_source": "platform_price_snapshot",
                "daily_threshold_cny": daily_threshold,
                "monthly_threshold_cny": monthly_threshold,
                "per_call_threshold_cny": thresholds.get("per_call_threshold_cny"),
                "strategy": strategy,
                "data_quality_status": (
                    "data_quality_error"
                    if reason_code == "budget_data_quality_error"
                    else "reconciliation_attention"
                ),
            }
        today_cost = float(today["provider_actual_cost_cny"]) + float(
            today["estimated_cost_cny"]
        )
        month_cost = float(month["provider_actual_cost_cny"]) + float(
            month["estimated_cost_cny"]
        )
    except Exception as exc:
        return {
            "budget_status": "unavailable",
            "daily_usage_percent": None,
            "monthly_usage_percent": None,
            "alert_level": "unavailable",
            "reason": "预算账本当前无法核实，本次额外付费后台操作未执行",
            "reason_code": "budget_query_failed",
            "error_type": type(exc).__name__,
            "trigger_dimension": None,
            "today_cost": None,
            "month_cost": None,
            "cost_source": "platform_price_snapshot",
            "daily_threshold_cny": thresholds.get("daily_threshold_cny"),
            "monthly_threshold_cny": thresholds.get("monthly_threshold_cny"),
            "per_call_threshold_cny": thresholds.get("per_call_threshold_cny"),
            "strategy": strategy,
            "data_quality_status": "unavailable",
        }

    daily_usage_percent = None
    monthly_usage_percent = None
    alert_level = "none"
    reason = None
    trigger_dimension = None

    if daily_threshold and daily_threshold > 0:
        daily_usage_percent = round((today_cost / daily_threshold) * 100, 4)
        if daily_usage_percent >= 100:
            alert_level = "blocked"
            reason = "今日预估成本已达到或超过日预算上限"
            trigger_dimension = "daily"
        elif daily_usage_percent >= 80 and alert_level == "none":
            alert_level = "warning"
            reason = "今日预估成本接近日预算上限（>=80%）"
            trigger_dimension = "daily"

    if monthly_threshold and monthly_threshold > 0:
        monthly_usage_percent = round((month_cost / monthly_threshold) * 100, 4)
        if monthly_usage_percent >= 100:
            alert_level = "blocked"
            reason = "本月预估成本已达到或超过月预算上限"
            trigger_dimension = "monthly"
        elif monthly_usage_percent >= 80 and alert_level == "none":
            alert_level = "warning"
            reason = "本月预估成本接近月预算上限（>=80%）"
            trigger_dimension = "monthly"

    return {
        "budget_status": "available",
        "daily_usage_percent": daily_usage_percent,
        "monthly_usage_percent": monthly_usage_percent,
        "alert_level": alert_level,
        "reason": reason,
        "reason_code": None,
        "error_type": None,
        "trigger_dimension": trigger_dimension,
        "today_cost": round(today_cost, 8),
        "month_cost": round(month_cost, 8),
        "cost_source": "platform_price_snapshot",
        "daily_threshold_cny": daily_threshold,
        "monthly_threshold_cny": monthly_threshold,
        "per_call_threshold_cny": thresholds.get("per_call_threshold_cny"),
        "strategy": strategy,
        "data_quality_status": "normal",
    }


def _background_budget_gate(strategy: str) -> Dict[str, Any]:
    """One fail-closed gate for optional paid background capabilities."""
    budget = _check_budget(strategy)
    if budget.get("budget_status") != "available":
        return {
            **budget,
            "allowed": False,
            "http_status": 503,
            "detail": {
                "code": "budget_status_unavailable",
                "message": "预算账本当前无法核实，本次额外付费后台操作未执行",
                "budget_status": "unavailable",
                "reason_code": budget.get("reason_code") or "cause_unconfirmed",
                "error_type": budget.get("error_type"),
            },
        }
    if budget.get("alert_level") == "blocked":
        return {
            **budget,
            "allowed": False,
            "http_status": 403,
            "detail": budget.get("reason") or "预算已达上限，本次额外付费后台操作未执行",
        }
    return {**budget, "allowed": True, "http_status": None, "detail": None}


# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------


@router.get("/overview")
async def overview(
    start: Optional[str] = Query(None, description="Start date/time ISO"),
    end: Optional[str] = Query(None, description="End date/time ISO"),
    model_id: Optional[str] = Query(None),
    stage: Optional[str] = Query(None),
    trace_id: Optional[str] = Query(None),
    range_key: Optional[str] = Query(
        None,
        pattern="^(today|yesterday|last_7_days|this_month|last_month|custom)$",
    ),
):
    """Return honest cost buckets for an explicit Asia/Shanghai range."""
    try:
        scope = _reporting_scope(range_key, start, end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    selected_calls = _fetch_model_calls(
        scope.get("start"),
        scope.get("end"),
        model_id=model_id,
        stage=stage,
        trace_id=trace_id,
    )
    reporting_calls = _fetch_reporting_model_calls(
        scope.get("start"),
        scope.get("end"),
        model_id=model_id,
        stage=stage,
        trace_id=trace_id,
    )
    data = _aggregate_model_calls(selected_calls)
    data_quality = _data_quality_summary(reporting_calls)
    trace_scope = _list_trace_page(
        trace_id=trace_id,
        model_id=model_id,
        stage=stage,
        start=scope.get("start"),
        end=scope.get("end"),
        limit=1,
        offset=0,
    )
    trace_group_count = int(trace_scope["trace_group_count"])
    scope_consistent = (
        int(trace_scope["provider_request_count"]) == int(data["calls"])
        and str(trace_scope.get("start") or "") == str(scope.get("start") or "")
        and str(trace_scope.get("end") or "") == str(scope.get("end") or "")
        and int(trace_scope.get("anomaly_attempt_count") or 0)
        == int(data_quality["anomaly_attempt_count"])
    )
    scope["filters"] = {
        "model_id": model_id,
        "stage": stage,
        "trace_id": trace_id,
    }
    history_total = _aggregate_model_calls(_fetch_model_calls(None, None))
    thresholds = get_budget_thresholds()
    known_cost = data["provider_actual_cost_cny"] + data["estimated_cost_cny"]
    priced_calls = data["provider_actual_priced_calls"] + (
        data["estimated_calls"] - data["estimated_amount_unavailable_calls"]
    )
    per_call_cost = known_cost / priced_calls if priced_calls else None
    per_call_cost_complete = bool(data.get("cost_complete"))

    periods = {}
    daily_threshold = thresholds.get("daily_threshold_cny")
    monthly_threshold = thresholds.get("monthly_threshold_cny")
    for name, bounds in _period_bounds().items():
        summary = _query_period_summary(bounds["start"], bounds["end"])
        period_known_cost = (
            summary["provider_actual_cost_cny"] + summary["estimated_cost_cny"]
        )
        usage_percent = None
        if name == "this_month" and monthly_threshold and monthly_threshold > 0:
            usage_percent = round((period_known_cost / monthly_threshold) * 100, 4)
        elif daily_threshold and daily_threshold > 0:
            denominator = daily_threshold * bounds["days"]
            if denominator:
                usage_percent = round((period_known_cost / denominator) * 100, 4)
        summary["budget_usage_percent"] = usage_percent
        periods[name] = summary

    daily_cost = (
        periods["today"]["provider_actual_cost_cny"]
        + periods["today"]["estimated_cost_cny"]
    )
    month_cost = (
        periods["this_month"]["provider_actual_cost_cny"]
        + periods["this_month"]["estimated_cost_cny"]
    )

    alerts = []
    if thresholds.get("daily_threshold_cny") and daily_cost > thresholds["daily_threshold_cny"]:
        alerts.append({
            "type": "daily",
            "threshold": thresholds["daily_threshold_cny"],
            "actual": round(daily_cost, 6),
        })
    if thresholds.get("monthly_threshold_cny") and month_cost > thresholds["monthly_threshold_cny"]:
        alerts.append({
            "type": "monthly",
            "threshold": thresholds["monthly_threshold_cny"],
            "actual": round(month_cost, 6),
        })
    if (
        per_call_cost_complete
        and per_call_cost is not None
        and thresholds.get("per_call_threshold_cny")
        and per_call_cost > thresholds["per_call_threshold_cny"]
    ):
        alerts.append({
            "type": "per_call",
            "threshold": thresholds["per_call_threshold_cny"],
            "actual": round(per_call_cost, 6),
        })

    top_traces = _top_provider_actual_traces(selected_calls, limit=5)
    for item in top_traces:
        trace = get_chat_trace(item["trace_id"]) or {}
        scene = trace.get("user_message") or trace.get("intent") or "后台模型分析"
        item["scene"] = str(scene)[:120]
        item["created_at"] = trace.get("created_at")

    by_model = {}
    for model_id, item in data["by_model"].items():
        by_model[model_id] = {**item, "model_name": _model_display_name(model_id)}

    return {
        "scope": scope,
        "calls": data["calls"],
        "provider_request_count": data["calls"],
        "trace_group_count": trace_group_count,
        "scope_consistent": scope_consistent,
        "statistics_status": _statistics_status(
            scope_consistent=scope_consistent,
            data_quality_status=data_quality["data_quality_status"],
        ),
        "data_quality_status": data_quality["data_quality_status"],
        "data_quality": data_quality,
        "invalid_timestamp_count": data_quality["invalid_timestamp_count"],
        "provider_send_unconfirmed_count": data_quality[
            "provider_send_unconfirmed_count"
        ],
        "orphaned_pending_count": data_quality["orphaned_pending_count"],
        "provider_usage_inconsistent_count": data_quality[
            "provider_usage_inconsistent_count"
        ],
        "provider_actual_price_missing_count": data_quality[
            "provider_actual_price_missing_count"
        ],
        "unavailable_usage_count": data_quality["unavailable_usage_count"],
        "unresolved_reconciliation_count": data_quality[
            "unresolved_reconciliation_count"
        ],
        "count_semantics": {
            "trace_group_count": "unique_trace_groups",
            "provider_request_count": "included_outbound_provider_attempts",
        },
        "excluded_record_count": data["excluded_record_count"],
        "total_tokens": data["total_tokens"],
        "known_token_calls": data["token_known_calls"],
        "unknown_token_calls": data["token_unavailable_calls"],
        "provider_actual_calls": data["provider_actual_calls"],
        "provider_actual_priced_calls": data["provider_actual_priced_calls"],
        "provider_actual_cost_cny": data["provider_actual_cost_cny"],
        "platform_price_snapshot_direct_cost_cny": data[
            "platform_price_snapshot_direct_cost_cny"
        ],
        "known_priced_request_count": priced_calls,
        "per_call_cost_cny": round(per_call_cost, 8) if per_call_cost is not None else None,
        "per_call_cost_complete": per_call_cost_complete,
        "estimated_calls": data["estimated_calls"],
        "estimated_cost_cny": data["estimated_cost_cny"],
        "estimated_amount_unavailable_calls": data[
            "estimated_amount_unavailable_calls"
        ],
        "unavailable_calls": data["unavailable_calls"],
        "known_usage_calls": data["known_usage_calls"],
        "known_usage": {
            "input_cache_hit_tokens": data["input_cache_hit_tokens"],
            "input_cache_miss_tokens": data["input_cache_miss_tokens"],
            "output_tokens": data["output_tokens"],
            "reasoning_tokens": data["reasoning_tokens"],
            "reasoning_known_calls": data["reasoning_known_calls"],
            "reasoning_unavailable_calls": data["reasoning_unavailable_calls"],
            "reasoning_is_output_subset": data_quality[
                "reasoning_is_output_subset"
            ],
            "reasoning_comparable_calls": data_quality[
                "reasoning_comparable_calls"
            ],
            "reasoning_unknown_calls": data_quality["reasoning_unknown_calls"],
            "reasoning_violation_calls": data_quality[
                "reasoning_violation_calls"
            ],
        },
        "failed_calls": data["failed_calls"],
        "alerts": alerts,
        "currency": "CNY",
        "cost_source": "platform_price_snapshot",
        "cost_note": (
            "Provider actual 仅表示该请求的实际Token证据；人民币金额由平台冻结"
            "价格快照换算，不是DeepSeek最终账单。历史估算单列且不进入Provider actual"
            "对账；金额不可得不按0元处理。"
        ),
        "today": periods["today"],
        "last_7_days": periods["last_7_days"],
        "this_month": periods["this_month"],
        "history_total": history_total,
        "by_model": by_model,
        "by_stage": data["by_stage"],
        "top_provider_actual_traces": top_traces,
        "price_missing": data["unavailable_calls"] > 0,
        "evaluation_quality": evaluation_summary(),
    }


# -----------------------------------------------------------------------------
# Traces
# -----------------------------------------------------------------------------


def _normalize_end(end: Optional[str]) -> Optional[str]:
    """Expand a day-level end boundary through the final microsecond.

    The frontend historically sent ``23:59:59``.  Treating that as the end of
    a reporting day must not drop rows that contain fractional seconds.
    Offset-aware timestamps keep their offset and are compared as timepoints.
    """
    if not end:
        return end
    text = str(end).strip()
    if len(text) == 10:
        try:
            datetime.strptime(text, "%Y-%m-%d")
            return f"{text} 23:59:59.999999"
        except ValueError:
            return text
    try:
        parsed = datetime.fromisoformat(
            f"{text[:-1]}+00:00" if text.endswith(("Z", "z")) else text
        )
        if (
            parsed.hour == 23
            and parsed.minute == 59
            and parsed.second == 59
            and parsed.microsecond == 0
        ):
            parsed = parsed.replace(microsecond=999999)
            return parsed.isoformat(sep="T" if "T" in text else " ")
        return text
    except ValueError:
        return text


def _validated_reporting_range(
    start: Optional[str], end: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    normalized_end = _normalize_end(end)
    start_epoch = (
        _required_reporting_epoch(start, "开始时间") if start else None
    )
    end_epoch = (
        _required_reporting_epoch(normalized_end, "结束时间")
        if normalized_end
        else None
    )
    if start_epoch is not None and end_epoch is not None and start_epoch > end_epoch:
        raise ValueError("开始时间不能晚于结束时间")
    return start, normalized_end


_RUN_TYPE_LABELS = {
    "chat": ("chat", "业主对话"),
    "evaluation": ("evaluation", "Evaluation评估"),
    "badcase_darwin": ("badcase_darwin", "Badcase · Darwin建议"),
}

_BACKGROUND_STAGE_LABELS = {
    "darwin": ("badcase_darwin", "Badcase · Darwin建议"),
    "badcase_classify": ("badcase_classify", "Badcase · AI分类"),
    "badcase_extract_knowledge": (
        "badcase_extract_knowledge",
        "Badcase · 提取知识",
    ),
    "badcase_switch_model_retry": (
        "badcase_switch_model_retry",
        "Badcase · 切换模型重试",
    ),
    "badcase_check_tools": ("badcase_check_tools", "Badcase · 检查工具"),
    "ab_test_a": ("model_ab_test", "模型A/B测试"),
    "ab_test_b": ("model_ab_test", "模型A/B测试"),
}


def _trace_operation_metadata(
    trace: Dict[str, Any], provider_calls: List[Dict[str, Any]]
) -> Dict[str, Any]:
    run_type = str(trace.get("run_type") or "").strip().lower()
    if run_type and run_type not in {"model_only", "background"}:
        operation_type, operation_label = _RUN_TYPE_LABELS.get(
            run_type, (run_type, run_type)
        )
    else:
        stages = [str(call.get("stage") or "").strip().lower() for call in provider_calls]
        selected = next(
            (_BACKGROUND_STAGE_LABELS[stage] for stage in stages if stage in _BACKGROUND_STAGE_LABELS),
            ("background_model_call", "后台模型任务"),
        )
        operation_type, operation_label = selected
    return {
        "operation_type": operation_type,
        "operation_label": operation_label,
        "is_background_task": operation_type != "chat",
    }


def _list_trace_page(
    *,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    intent: Optional[str] = None,
    agent: Optional[str] = None,
    model_id: Optional[str] = None,
    stage: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    range_key: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    """Return one range-correct page of chat and model-only Trace groups.

    Overview and cards are both driven by the same range-limited Provider
    attempt set.  A Trace crossing midnight stays one group, while its card
    only aggregates attempts that occurred inside the selected range.
    """
    if range_key:
        scope = _reporting_scope(range_key, start, end)
        effective_start = scope["start"]
        effective_end = scope["end"]
    else:
        effective_start, effective_end = _validated_reporting_range(start, end)
        scope = {
            "range_key": "custom" if start or end else None,
            "start": effective_start,
            "end": effective_end,
            "timezone": "Asia/Shanghai (UTC+8)",
        }

    valid_model_range_conditions, model_range_params = _time_range_predicates(
        "m.created_at", effective_start, effective_end
    )
    valid_model_range = " AND ".join(valid_model_range_conditions) or "1=1"
    model_range_conditions = [
        _provider_attempt_sql_predicate("m"),
        f"(({valid_model_range}) OR yiai_time_epoch(m.created_at) IS NULL)",
    ]
    if model_id:
        model_range_conditions.append("m.model_id = ?")
        model_range_params.append(model_id)
    if stage:
        model_range_conditions.append("m.stage = ?")
        model_range_params.append(stage)
    model_range_where = " AND ".join(model_range_conditions)

    chat_range_conditions, chat_range_params = _time_range_predicates(
        "t.created_at", effective_start, effective_end
    )
    chat_range_where = " AND ".join(chat_range_conditions) or "1=1"
    chat_candidate_where = (
        "mg.trace_id IS NOT NULL"
        if model_id or stage
        else f"mg.trace_id IS NOT NULL OR ({chat_range_where})"
    )

    outer_conditions = ["1=1"]
    outer_params: List[Any] = []
    if trace_id:
        outer_conditions.append("a.trace_id = ?")
        outer_params.append(trace_id)
    if session_id:
        outer_conditions.append("a.session_id = ?")
        outer_params.append(session_id)
    if intent:
        outer_conditions.append("a.intent = ?")
        outer_params.append(intent)
    if agent:
        outer_conditions.append("a.agent_name = ?")
        outer_params.append(agent)
    where_sql = " AND ".join(outer_conditions)

    all_traces_sql = f"""
        WITH range_model_calls AS (
            SELECT m.*, yiai_time_epoch(m.created_at) AS created_epoch
            FROM model_calls m
            WHERE {model_range_where}
        ), model_groups AS (
            SELECT
                trace_id,
                SUM(CASE
                    WHEN created_epoch IS NOT NULL
                     AND {_provider_sql_predicate('range_model_calls')}
                    THEN 1 ELSE 0 END
                ) AS provider_request_count,
                COUNT(*) AS provider_attempt_count,
                MAX(created_epoch) AS provider_last_epoch
            FROM range_model_calls
            WHERE trace_id IS NOT NULL
            GROUP BY trace_id
        ), model_only_ranked AS (
            SELECT
                m.*,
                ROW_NUMBER() OVER (
                    PARTITION BY m.trace_id
                    ORDER BY m.created_epoch DESC, m.id DESC
                ) AS range_rank
            FROM range_model_calls m
            WHERE m.trace_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM chat_traces existing
                  WHERE existing.trace_id = m.trace_id
              )
        ), chat_candidates AS (
            SELECT
                t.trace_id, t.session_id, t.user_message, t.intent,
                t.agent_name, t.status, t.created_at, t.updated_at,
                COALESCE(t.run_type, 'chat') AS run_type,
                COALESCE(mg.provider_last_epoch, yiai_time_epoch(t.created_at)) AS sort_epoch,
                COALESCE(mg.provider_request_count, 0) AS provider_request_count,
                COALESCE(mg.provider_attempt_count, 0) AS provider_attempt_count
            FROM chat_traces t
            LEFT JOIN model_groups mg ON mg.trace_id = t.trace_id
            WHERE {chat_candidate_where}
        ), model_only AS (
            SELECT
                m.trace_id,
                NULL AS session_id,
                NULL AS user_message,
                NULL AS intent,
                NULL AS agent_name,
                m.status,
                m.created_at,
                COALESCE(m.finished_at, m.created_at) AS updated_at,
                CASE
                    WHEN EXISTS (
                        SELECT 1 FROM evaluation_runs evaluation
                        WHERE evaluation.trace_id = m.trace_id
                    ) THEN 'evaluation'
                    ELSE 'model_only'
                END AS run_type,
                m.created_epoch AS sort_epoch,
                mg.provider_request_count,
                mg.provider_attempt_count
            FROM model_only_ranked m
            JOIN model_groups mg ON mg.trace_id = m.trace_id
            WHERE m.range_rank = 1
        ), all_traces AS (
            SELECT * FROM chat_candidates
            UNION ALL
            SELECT * FROM model_only
        )
    """
    cte_params = model_range_params + ([] if model_id or stage else chat_range_params)
    query_params = cte_params + outer_params

    conn = _get_conn()
    _register_reporting_sql(conn)
    cursor = conn.cursor()
    cursor.execute(
        f"""
        {all_traces_sql}
        SELECT
            COUNT(*) AS total,
            COALESCE(SUM(a.provider_request_count), 0) AS provider_request_count
        FROM all_traces a
        WHERE {where_sql}
        """,
        query_params,
    )
    totals = cursor.fetchone()
    total = int(totals["total"] or 0)
    scoped_provider_request_count = int(totals["provider_request_count"] or 0)
    cursor.execute(
        f"""
        {all_traces_sql}
        SELECT a.trace_id FROM all_traces a
        WHERE {where_sql}
        """,
        query_params,
    )
    matching_trace_ids = {
        str(row["trace_id"]) for row in cursor.fetchall() if row["trace_id"]
    }
    cursor.execute(
        f"""
        {all_traces_sql}
        SELECT * FROM all_traces a
        WHERE {where_sql}
        ORDER BY a.sort_epoch DESC, a.trace_id DESC
        LIMIT ? OFFSET ?
        """,
        query_params + [limit, offset],
    )
    trace_rows = cursor.fetchall()
    conn.close()

    page_trace_ids = [row["trace_id"] for row in trace_rows]
    calls_by_trace: Dict[str, List[Dict[str, Any]]] = {}
    reporting_calls_by_trace: Dict[str, List[Dict[str, Any]]] = {}
    if page_trace_ids:
        page_ids = set(page_trace_ids)
        range_calls = _fetch_model_calls(
            effective_start,
            effective_end,
            model_id=model_id,
            stage=stage,
        )
        for raw_call in range_calls:
            item_trace_id = raw_call.get("trace_id")
            if item_trace_id in page_ids:
                calls_by_trace.setdefault(item_trace_id, []).append(raw_call)
        reporting_range_calls = _fetch_reporting_model_calls(
            effective_start,
            effective_end,
            model_id=model_id,
            stage=stage,
        )
        for raw_call in reporting_range_calls:
            item_trace_id = raw_call.get("trace_id")
            if item_trace_id in page_ids:
                reporting_calls_by_trace.setdefault(item_trace_id, []).append(raw_call)

    agg_rows: Dict[str, Dict[str, Any]] = {}
    for item_trace_id in page_trace_ids:
        item_calls = calls_by_trace.get(item_trace_id, [])
        reporting_item_calls = reporting_calls_by_trace.get(item_trace_id, [])
        provider_calls, _ = _split_model_records(item_calls)
        provider_attempt_records = [
            call
            for call in reporting_item_calls
            if _provider_aggregate_decision(call)["record_kind"] == "provider_attempt"
        ]
        logical_calls = [
            call
            for call in item_calls
            if _provider_aggregate_decision(call)["record_kind"] != "provider_attempt"
        ]
        unconfirmed_provider_attempts = [
            call
            for call in provider_attempt_records
            if (
                not _is_provider_aggregate_record(call)
                or _reporting_epoch(call.get("created_at")) is None
            )
        ]
        summary = _aggregate_model_calls(item_calls)
        item_data_quality = _data_quality_summary(reporting_item_calls)
        provider_actual_priced_calls = int(
            summary.get("provider_actual_priced_calls") or 0
        )
        estimated_priced_calls = int(summary.get("estimated_calls") or 0) - int(
            summary.get("estimated_amount_unavailable_calls") or 0
        )
        known_cost = None
        if provider_actual_priced_calls or estimated_priced_calls:
            known_cost = round(
                float(summary["provider_actual_cost_cny"])
                + float(summary["estimated_cost_cny"]),
                8,
            )
        agg_rows[item_trace_id] = {
            **summary,
            "provider_calls": provider_calls,
            "provider_attempt_records": provider_attempt_records,
            "model_ids": sorted(
                {_provider_model(call) for call in provider_attempt_records}
            ),
            "call_count": len(provider_calls),
            "provider_attempt_count": len(provider_attempt_records),
            "logical_record_count": len(logical_calls),
            "unconfirmed_provider_attempt_count": len(unconfirmed_provider_attempts),
            "data_quality": item_data_quality,
            "estimated_cost_cny": known_cost,
            "local_estimated_cost_cny": (
                summary["estimated_cost_cny"] if estimated_priced_calls else None
            ),
            "provider_actual_cost_cny": (
                summary["provider_actual_cost_cny"]
                if provider_actual_priced_calls
                else None
            ),
            "estimated_priced_calls": estimated_priced_calls,
            "unknown_cost_calls": int(summary.get("unavailable_calls") or 0),
        }

    results = []
    for row in trace_rows:
        trace = dict(row)
        agg = agg_rows.get(trace["trace_id"], {})
        model_ids = list(agg.get("model_ids") or [])
        call_count = int(agg.get("call_count") or 0)
        provider_attempt_count = int(agg.get("provider_attempt_count") or 0)
        item_data_quality = agg.get("data_quality") or _data_quality_summary([])
        provider_actual_calls = int(agg.get("provider_actual_calls") or 0)
        provider_actual_priced_calls = int(
            agg.get("provider_actual_priced_calls") or 0
        )
        estimated_calls = int(agg.get("estimated_calls") or 0)
        estimated_priced_calls = int(agg.get("estimated_priced_calls") or 0)
        unknown_cost_calls = int(agg.get("unknown_cost_calls") or 0)

        if not model_ids:
            model_summary = "尚无模型调用记录"
        elif len(model_ids) == 1:
            model_summary = _model_display_name(model_ids[0])
        else:
            model_summary = " + ".join(_model_display_name(item) for item in model_ids)

        trace.update({
            "models": model_ids,
            "model_summary": model_summary,
            "total_tokens": agg.get("total_tokens") if call_count else None,
            "input_cache_hit_tokens": (
                agg.get("input_cache_hit_tokens") if call_count else None
            ),
            "input_cache_miss_tokens": (
                agg.get("input_cache_miss_tokens") if call_count else None
            ),
            "output_tokens": agg.get("output_tokens") if call_count else None,
            "reasoning_tokens": (
                agg.get("reasoning_tokens") if call_count else None
            ),
            "reasoning_known_calls": int(
                agg.get("reasoning_known_calls") or 0
            ),
            "reasoning_unavailable_calls": int(
                agg.get("reasoning_unavailable_calls") or 0
            ),
            "estimated_cost_cny": agg.get("estimated_cost_cny"),
            "provider_actual_cost_cny": agg.get("provider_actual_cost_cny"),
            "platform_price_snapshot_direct_cost_cny": agg.get(
                "provider_actual_cost_cny"
            ),
            "local_estimated_cost_cny": agg.get("local_estimated_cost_cny"),
            "provider_actual_calls": provider_actual_calls,
            "provider_actual_priced_calls": provider_actual_priced_calls,
            "estimated_calls": estimated_calls,
            "estimated_priced_calls": estimated_priced_calls,
            "unavailable_calls": unknown_cost_calls,
            "model_call_count": call_count,
            "provider_request_count": call_count,
            "provider_attempt_count": provider_attempt_count,
            "unconfirmed_provider_attempt_count": int(
                agg.get("unconfirmed_provider_attempt_count") or 0
            ),
            "data_quality_status": item_data_quality["data_quality_status"],
            "anomaly_attempt_count": item_data_quality["anomaly_attempt_count"],
            "anomaly_attempts": item_data_quality["anomaly_attempts"],
            "unresolved_reconciliation_count": item_data_quality[
                "unresolved_reconciliation_count"
            ],
            "invalid_timestamp_count": item_data_quality[
                "invalid_timestamp_count"
            ],
            "reasoning_is_output_subset": item_data_quality[
                "reasoning_is_output_subset"
            ],
            "logical_model_record_count": int(
                agg.get("logical_record_count") or 0
            ),
            "cost_status": (
                "not_applicable" if provider_attempt_count == 0
                else item_data_quality["data_quality_status"]
                if call_count == 0
                else "partial_unavailable" if unknown_cost_calls
                else "provider_actual" if provider_actual_calls and not estimated_calls
                else "estimated" if estimated_calls and not provider_actual_calls
                else "mixed"
            ),
            "price_missing": unknown_cost_calls > 0,
            "no_model_calls": provider_attempt_count == 0,
            **_trace_operation_metadata(
                trace, list(agg.get("provider_attempt_records") or [])
            ),
        })
        results.append(trace)

    pages = max(1, (total + limit - 1) // limit)
    page = (offset // limit) + 1
    scoped_normal_calls = _fetch_model_calls(
        effective_start,
        effective_end,
        model_id=model_id,
        stage=stage,
        trace_id=trace_id,
    )
    scoped_reporting_calls = _fetch_reporting_model_calls(
        effective_start,
        effective_end,
        model_id=model_id,
        stage=stage,
        trace_id=trace_id,
    )
    if any([session_id, intent, agent]):
        scoped_normal_calls = [
            call
            for call in scoped_normal_calls
            if str(call.get("trace_id") or "") in matching_trace_ids
        ]
        scoped_reporting_calls = [
            call
            for call in scoped_reporting_calls
            if str(call.get("trace_id") or "") in matching_trace_ids
        ]
    scoped_expected_provider_count = int(
        _aggregate_model_calls(scoped_normal_calls)["calls"]
    )
    trace_data_quality = _data_quality_summary(scoped_reporting_calls)
    scope_consistent = (
        scoped_provider_request_count == scoped_expected_provider_count
    )
    return {
        "traces": results,
        "total": total,
        "trace_group_count": total,
        "provider_request_count": scoped_provider_request_count,
        "scope_consistent": scope_consistent,
        "statistics_status": _statistics_status(
            scope_consistent=scope_consistent,
            data_quality_status=trace_data_quality["data_quality_status"],
        ),
        "data_quality_status": trace_data_quality["data_quality_status"],
        "data_quality": trace_data_quality,
        "anomaly_attempt_count": trace_data_quality["anomaly_attempt_count"],
        "invalid_timestamp_count": trace_data_quality["invalid_timestamp_count"],
        "unresolved_reconciliation_count": trace_data_quality[
            "unresolved_reconciliation_count"
        ],
        "unassigned_timestamp_attempts": [
            item
            for item in trace_data_quality["anomaly_attempts"]
            if "invalid_timestamp" in item.get("issue_codes", [])
        ],
        "count_semantics": {
            "trace_group_count": "unique_trace_groups",
            "provider_request_count": "included_outbound_provider_attempts",
            "trace_card_provider_requests": "range_limited_attempts_only",
        },
        "limit": limit,
        "offset": offset,
        "page": page,
        "pages": pages,
        "has_previous": offset > 0,
        "has_next": offset + len(results) < total,
        "start": effective_start,
        "end": effective_end,
        "range_key": scope.get("range_key"),
        "timezone": scope.get("timezone"),
        "filters": {
            "trace_id": trace_id,
            "session_id": session_id,
            "intent": intent,
            "agent": agent,
            "model_id": model_id,
            "stage": stage,
        },
    }


@router.get("/traces")
async def traces(
    session_id: Optional[str] = Query(None),
    trace_id: Optional[str] = Query(None),
    intent: Optional[str] = Query(None),
    agent: Optional[str] = Query(None),
    model_id: Optional[str] = Query(None),
    stage: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    range_key: Optional[str] = Query(
        None,
        pattern="^(today|yesterday|last_7_days|this_month|last_month|custom)$",
    ),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List one server-paginated page of chat, model-only, and no-model traces."""
    try:
        return _list_trace_page(
            session_id=session_id,
            trace_id=trace_id,
            intent=intent,
            agent=agent,
            model_id=model_id,
            stage=stage,
            start=start,
            end=end,
            range_key=range_key,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _build_cost_formula(call: Dict[str, Any]) -> str:
    """Build a price-snapshot formula without calling it a Provider bill."""
    normalized = _usage_payload(call)
    decision = _provider_aggregate_decision(call)
    if not decision["included"]:
        return f"{decision['reason']}：不计入Provider请求、Token或成本汇总"
    contract = normalized.get("cost_contract") or {}
    if contract.get("formula"):
        return contract["formula"]
    if contract.get("availability_note"):
        return contract["availability_note"]

    usage_status = str(
        call.get("usage_status") or normalized.get("usage_status") or ""
    ).strip()
    unavailable_reason = (
        normalized.get("usage_unavailable_reason")
        or (usage_status if usage_status and usage_status != "provider_actual" else None)
    )
    if _token_source(call) != "provider_actual":
        return (
            f"{unavailable_reason or 'cause_unconfirmed'}：Provider实际Usage不可得；"
            "平台不显示0，也不估算成Provider actual"
        )

    snapshot = call.get("price_snapshot") or {}
    if not snapshot:
        return "price_snapshot_unavailable：未形成平台价格快照直接成本；不按0元处理"

    terms = []
    input_p = snapshot.get("input_price_per_1m")
    cached_p = snapshot.get("cached_input_price_per_1m")
    output_p = snapshot.get("output_price_per_1m")

    if normalized.get("usage_split_unavailable"):
        return "usage_split_unavailable：三类Provider Usage不完整，平台直接成本不可得"

    uncached = _first_present(
        normalized,
        "cache_miss_input_tokens",
        "input_cache_miss_tokens",
        "uncached_input_tokens",
    )
    cached = _first_present(
        normalized,
        "cache_hit_input_tokens",
        "input_cache_hit_tokens",
        "cached_input_tokens",
    )
    output = _first_present(normalized, "output_tokens")

    if uncached is not None and input_p is not None:
        terms.append(f"uncached_input_tokens({uncached}) * {input_p} / 1_000_000")
    if cached is not None and cached_p is not None:
        terms.append(f"cached_input_tokens({cached}) * {cached_p} / 1_000_000")
    if output is not None and output_p is not None:
        terms.append(f"output_tokens({output}) * {output_p} / 1_000_000")

    if not terms:
        return "price_snapshot_incomplete：平台价格快照直接成本不可得；不按0元处理"
    return " + ".join(terms)


def _enrich_model_call(call: Dict[str, Any], session_id: Optional[str]) -> Dict[str, Any]:
    """Expose one record with an evidence-backed reconciliation decision."""
    enriched = dict(call)
    model_id = enriched.get("model_id")
    enriched["model_name"] = _model_display_name(model_id)
    enriched["session_id"] = session_id
    usage_norm = _usage_payload(enriched)
    enriched["usage_normalized"] = usage_norm
    decision = _provider_aggregate_decision(enriched)

    enriched["requested_model"] = (
        usage_norm.get("requested_model") or model_id
    )
    provider_actual_model = usage_norm.get("provider_actual_model") or usage_norm.get(
        "provider_response_model"
    )
    enriched["provider_actual_model"] = provider_actual_model
    enriched["provider_response_model"] = provider_actual_model
    enriched["thinking_enabled"] = _first_present(
        usage_norm, "thinking", "thinking_enabled"
    )
    enriched["stream"] = _optional_bool(
        _first_present(usage_norm, "stream", "streaming")
    )
    enriched["local_attempt_id"] = enriched.get("local_attempt_id") or usage_norm.get(
        "local_attempt_id"
    )
    enriched["provider_request_id"] = enriched.get(
        "provider_request_id"
    ) or usage_norm.get("provider_request_id")
    attempt_sequence = _first_present(
        usage_norm, "attempt_sequence", "provider_request_sequence"
    )
    enriched["attempt_sequence"] = attempt_sequence
    enriched["provider_request_sequence"] = attempt_sequence
    enriched["record_kind"] = enriched.get("record_kind") or decision["record_kind"]
    enriched["usage_status"] = enriched.get("usage_status") or usage_norm.get(
        "usage_status"
    ) or _token_source(enriched)
    enriched["usage_unavailable_reason"] = usage_norm.get(
        "usage_unavailable_reason"
    )
    enriched["token_source"] = usage_norm.get("token_source") or _token_source(
        enriched
    )
    enriched["cost_source"] = enriched.get("cost_source") or usage_norm.get(
        "cost_source"
    )
    enriched["include_in_provider_aggregate"] = decision[
        "include_in_provider_aggregate"
    ]
    enriched["included_in_provider_summary"] = decision["included"]
    enriched["aggregate_exclusion_reason"] = None if decision[
        "included"
    ] else decision["reason"]

    hit = _token_value(
        enriched, "cache_hit_input_tokens", "input_cache_hit_tokens"
    )
    miss = _token_value(
        enriched, "cache_miss_input_tokens", "input_cache_miss_tokens"
    )
    output = _token_value(enriched, "output_tokens")
    reasoning = _token_value(enriched, "reasoning_tokens")
    total = _token_value(enriched, "total_tokens")
    enriched["provider_usage"] = {
        "cache_hit_input_tokens": hit,
        "cache_miss_input_tokens": miss,
        "input_cache_hit_tokens": hit,
        "input_cache_miss_tokens": miss,
        "output_tokens": output,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
    }
    enriched["reasoning_tokens"] = reasoning
    enriched["total_tokens"] = total
    enriched["provider_usage_raw"] = usage_norm.get("provider_usage_raw")
    enriched["provider_usage_inconsistent"] = bool(
        usage_norm.get("provider_usage_inconsistent")
    )

    request_sent = decision["request_sent"]
    done_received = _optional_bool(
        _first_present(usage_norm, "done_received", "received_done")
    )
    stream_completed = _optional_bool(
        _first_present(
            usage_norm,
            "sdk_stream_exhausted",
            "stream_completed",
            "sdk_exhausted",
        )
    )
    usage_received = _optional_bool(
        _first_present(usage_norm, "received_usage", "usage_received")
    )
    if usage_received is None and enriched["usage_status"] == "provider_actual":
        usage_received = True
    persistence_value = _first_present(
        usage_norm, "persistence_succeeded", "persisted", "persistence_status"
    )
    persistence_succeeded = _optional_bool(persistence_value)
    if persistence_succeeded is None and isinstance(persistence_value, str):
        normalized_persistence = persistence_value.strip().lower()
        if normalized_persistence in {"success", "succeeded", "persisted", "complete"}:
            persistence_succeeded = True
        elif normalized_persistence in {"failed", "failure", "error"}:
            persistence_succeeded = False
    retry_of = usage_norm.get("retry_of_local_attempt_id")
    retry_detected = _optional_bool(usage_norm.get("retry_detected"))
    explicit_retry = _optional_bool(usage_norm.get("explicit_retry"))
    if retry_detected is not None:
        retry = retry_detected
    elif explicit_retry is not None:
        retry = explicit_retry
    elif retry_of:
        retry = True
    elif str(attempt_sequence or "").isdigit():
        retry = int(attempt_sequence) > 1
    else:
        retry = None
    reconciliation_reason = decision["reason"]
    if decision["included"]:
        reconciliation_reason = (
            "provider_actual"
            if enriched["usage_status"] == "provider_actual"
            else enriched["usage_unavailable_reason"]
            or enriched["usage_status"]
            or "cause_unconfirmed"
        )
    non_sensitive_error = usage_norm.get("non_sensitive_error_evidence")
    if non_sensitive_error is None:
        non_sensitive_error = usage_norm.get("error_evidence")
    if isinstance(non_sensitive_error, str):
        non_sensitive_error = non_sensitive_error[:1000]
    reconciliation_evidence = {
        "http_status": usage_norm.get("http_status"),
        "sdk_dispatch_started": usage_norm.get("sdk_dispatch_started"),
        "provider_response_seen": usage_norm.get("provider_response_seen"),
        "stream_completed": stream_completed,
        "done_received": done_received,
        "completion_evidence": usage_norm.get("completion_evidence"),
        "non_sensitive_error": non_sensitive_error,
        "provider_usage_inconsistent": enriched["provider_usage_inconsistent"],
    }
    enriched["reconciliation"] = {
        "trace_id": enriched.get("trace_id"),
        "stage": enriched.get("stage"),
        "local_attempt_id": enriched.get("local_attempt_id"),
        "attempt_sequence": attempt_sequence,
        "provider_request_sent": request_sent,
        "provider_request_id": enriched.get("provider_request_id"),
        "done_received": done_received,
        "sdk_stream_exhausted": stream_completed,
        "usage_received": usage_received,
        "persisted": persistence_succeeded,
        "persistence_status": persistence_value,
        "record_persisted": bool(enriched.get("id")),
        "retry": retry,
        "retry_of_local_attempt_id": retry_of,
        "included_in_provider_summary": decision["included"],
        "reason": reconciliation_reason,
        "evidence": reconciliation_evidence,
    }
    enriched["started_at"] = usage_norm.get("started_at") or enriched.get(
        "created_at"
    )
    enriched["finished_at"] = enriched.get("finished_at") or usage_norm.get(
        "finished_at"
    )
    calculated_cost = _first_present(
        usage_norm, "calculated_direct_cost", "calculated_direct_cost_cny"
    )
    if calculated_cost is None and enriched.get("cost_source") == "platform_price_snapshot":
        calculated_cost = enriched.get("estimated_cost_cny")
    enriched["calculated_direct_cost"] = calculated_cost

    snapshot = enriched.get("price_snapshot") or {}
    for key in (
        "input_price_per_1m",
        "cached_input_price_per_1m",
        "output_price_per_1m",
        "currency",
        "effective_date",
        "source_note",
    ):
        enriched[key] = snapshot.get(key)

    stage = enriched.get("stage") or ""
    if stage in {
        "darwin",
        "badcase_classify",
        "badcase_extract_knowledge",
        "badcase_switch_model_retry",
        "badcase_check_tools",
        "retest",
    }:
        enriched["badcase_id"] = get_badcase_id_by_trace_id(enriched.get("trace_id"))
    else:
        enriched["badcase_id"] = None

    enriched["cost_formula"] = _build_cost_formula(enriched)
    return enriched


def _cost_summary(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    provider_calls, excluded_calls = _split_model_records(calls)
    by_model: Dict[str, Dict[str, Any]] = {}
    known_cost_cny = 0.0
    unknown_cost_calls = 0
    known_cost_calls = 0
    known_token_calls = 0
    unknown_token_calls = 0
    total_tokens = 0
    for call in provider_calls:
        model_id = (
            call.get("provider_actual_model")
            or call.get("provider_response_model")
            or call.get("requested_model")
            or call.get("model_id")
            or "unknown"
        )
        item = by_model.setdefault(
            model_id,
            {
                "model_id": model_id,
                "calls": 0,
                "tokens": 0,
                "known_token_calls": 0,
                "unknown_token_calls": 0,
                "cost_cny": 0.0,
                "known_cost_calls": 0,
                "unknown_cost_calls": 0,
            },
        )
        item["calls"] += 1
        token_value = _token_value(call, "total_tokens")
        if _token_source(call) == "provider_actual" and token_value is not None:
            known_token_calls += 1
            item["known_token_calls"] += 1
            total_tokens += int(token_value)
            item["tokens"] += int(token_value)
        else:
            unknown_token_calls += 1
            item["unknown_token_calls"] += 1
        amount = call.get("calculated_direct_cost")
        if amount is None and call.get("cost_source") == "platform_price_snapshot":
            amount = call.get("estimated_cost_cny")
        if amount is None:
            unknown_cost_calls += 1
            item["unknown_cost_calls"] += 1
        else:
            known_cost_calls += 1
            item["known_cost_calls"] += 1
            known_cost_cny += float(amount)
            item["cost_cny"] += float(amount)
    for item in by_model.values():
        if item["known_token_calls"] == 0:
            item["tokens"] = None
        if item["known_cost_calls"] == 0:
            item["cost_cny"] = None
    return {
        "calls": len(provider_calls),
        "provider_request_count": len(provider_calls),
        "excluded_record_count": len(excluded_calls),
        "total_tokens": total_tokens if known_token_calls else None,
        "known_token_calls": known_token_calls,
        "unknown_token_calls": unknown_token_calls,
        "known_cost_cny": round(known_cost_cny, 8) if known_cost_calls else None,
        "platform_price_snapshot_direct_cost_cny": (
            round(known_cost_cny, 8) if known_cost_calls else None
        ),
        "cost_source": "platform_price_snapshot",
        "known_cost_calls": known_cost_calls,
        "unknown_cost_calls": unknown_cost_calls,
        "complete": unknown_cost_calls == 0 and unknown_token_calls == 0,
        "by_model": list(by_model.values()),
    }


def _stage_display_name(stage: Optional[str]) -> str:
    return {
        "router": "Router",
        "vertical_agent": "垂直Agent",
        "badcase_classify": "Badcase分类",
        "darwin": "Darwin/AI专家",
        "retest": "Badcase复测（逻辑聚合）",
    }.get(stage or "") or (stage or "模型调用")


def _trace_cost_explanation(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a request-level story from included Provider attempts only."""
    provider_calls, _ = _split_model_records(calls)
    chain = []
    model_counts: Dict[str, int] = {}
    provider_actual_cost = 0.0
    estimated_cost = 0.0
    provider_actual_calls = 0
    estimated_calls = 0
    unavailable_calls = 0
    total_tokens = 0
    known_token_calls = 0
    unknown_token_calls = 0
    known_cost_calls = 0

    for raw in provider_calls:
        call = dict(raw)
        usage = _usage_payload(call)
        bucket = _cost_bucket(call)
        response_model = (
            call.get("provider_actual_model")
            or usage.get("provider_actual_model")
            or usage.get("provider_response_model")
            or call.get("provider_response_model")
        )
        requested_model = usage.get("requested_model") or call.get(
            "requested_model"
        ) or call.get("model_id")
        display_model = response_model or requested_model or "unknown"
        model_counts[display_model] = model_counts.get(display_model, 0) + 1
        amount = call.get("calculated_direct_cost")
        if amount is None and call.get("cost_source") == "platform_price_snapshot":
            amount = call.get("estimated_cost_cny")
        if bucket == "provider_actual":
            provider_actual_calls += 1
            if amount is not None:
                known_cost_calls += 1
                provider_actual_cost += float(amount)
        elif bucket == "estimated":
            estimated_calls += 1
            if amount is not None:
                estimated_cost += float(amount)
        else:
            unavailable_calls += 1
        attempt_total = _token_value(call, "total_tokens")
        if bucket == "provider_actual" and attempt_total is not None:
            known_token_calls += 1
            total_tokens += int(attempt_total)
        else:
            unknown_token_calls += 1
        hit = _token_value(
            call, "cache_hit_input_tokens", "input_cache_hit_tokens"
        )
        miss = _token_value(
            call, "cache_miss_input_tokens", "input_cache_miss_tokens"
        )
        output = _token_value(call, "output_tokens")
        chain.append(
            {
                "stage": call.get("stage"),
                "stage_name": _stage_display_name(call.get("stage")),
                "requested_model": requested_model,
                "provider_response_model": response_model,
                "display_model": _model_display_name(display_model),
                "thinking_enabled": usage.get(
                    "thinking_enabled", call.get("thinking_enabled")
                ),
                "usage_source": _token_source(call),
                "usage_status": call.get("usage_status")
                or usage.get("usage_status")
                or _token_source(call),
                "usage_unavailable_reason": call.get("usage_unavailable_reason")
                or usage.get("usage_unavailable_reason"),
                "input_cache_hit_tokens": hit,
                "input_cache_miss_tokens": miss,
                "output_tokens": output,
                "reasoning_tokens": _token_value(call, "reasoning_tokens"),
                "total_tokens": attempt_total,
                "amount_cny": amount,
                "calculated_direct_cost": amount,
                "cost_source": call.get("cost_source")
                or usage.get("cost_source"),
                "price_snapshot": call.get("price_snapshot") or {},
                "formula": call.get("cost_formula"),
                "bucket": bucket,
                "local_attempt_id": call.get("local_attempt_id")
                or usage.get("local_attempt_id"),
                "provider_request_id": call.get("provider_request_id")
                or usage.get("provider_request_id"),
                "provider_request_sequence": call.get("attempt_sequence")
                or usage.get("attempt_sequence")
                or usage.get("provider_request_sequence"),
                "reconciliation": call.get("reconciliation") or {},
            }
        )

    if not chain:
        summary = "本轮未调用模型，因此模型Token与费用不适用。"
    else:
        model_parts = [
            f"{count}次{_model_display_name(model_id)}"
            for model_id, count in sorted(model_counts.items())
        ]
        call_text = "、".join(model_parts)
        parts = [f"本轮产生{len(chain)}次Provider请求（{call_text}）"]
        if known_token_calls:
            parts.append(f"已取得Provider实际Usage合计{total_tokens:,} Token")
        if unknown_token_calls:
            parts.append(f"{unknown_token_calls}次实际Token不可得")
        if known_cost_calls:
            parts.append(
                f"平台价格快照直接成本¥{provider_actual_cost:.8f}"
                "（非DeepSeek最终账单）"
            )
        if provider_actual_calls - known_cost_calls:
            parts.append(f"{provider_actual_calls - known_cost_calls}次直接成本不可得")
        if estimated_calls:
            parts.append(
                f"历史估算¥{estimated_cost:.8f}（排除Provider actual对账）"
                if estimated_cost
                else f"{estimated_calls}次历史估算未形成可计金额"
            )
        if unavailable_calls:
            parts.append(f"{unavailable_calls}次Provider实际Usage不可得")
        summary = "；".join(parts) + "。"

    recommendation = _single_trace_recommendation(chain)
    return {
        "summary": summary,
        "model_call_count": len(chain),
        "provider_request_count": len(chain),
        "total_tokens": total_tokens if known_token_calls else None,
        "known_token_calls": known_token_calls,
        "unknown_token_calls": unknown_token_calls,
        "chain": chain,
        "cost_scope": {
            "status": "not_applicable" if not chain else "model_calls_present",
            "provider_actual_calls": provider_actual_calls,
            "provider_actual_cost_cny": (
                round(provider_actual_cost, 8) if known_cost_calls else None
            ),
            "platform_price_snapshot_direct_cost_cny": (
                round(provider_actual_cost, 8) if known_cost_calls else None
            ),
            "cost_source": "platform_price_snapshot",
            "known_cost_calls": known_cost_calls,
            "estimated_calls": estimated_calls,
            "estimated_cost_cny": round(estimated_cost, 8),
            "unavailable_calls": unavailable_calls,
        },
        "recommendation": recommendation,
    }


def _single_trace_recommendation(chain: List[Dict[str, Any]]) -> Dict[str, Any]:
    same_run_retest = (
        "使用同一问题、同一配置快照和同一模型复测；同时比较答案质量、引用、"
        "三类Provider Usage、金额和时延，不自动发起模型请求。"
    )
    if not chain:
        return {
            "code": "normal_no_model",
            "title": "当前流程已避免模型成本",
            "why": "本轮是确定性规则流程，真实model_call为0。",
            "action": "保持规则流程，不为展示而增加模型调用。",
            "expected_direction": "继续保持模型Token与费用不适用。",
            "retest_method": "重复同一规则操作并确认model_call仍为0、业务回执仍正确。",
        }

    non_darwin_pro = next(
        (
            item
            for item in chain
            if item.get("stage") != "darwin"
            and (
                item.get("provider_response_model") == "deepseek-v4-pro"
                or item.get("requested_model") == "deepseek-v4-pro"
            )
        ),
        None,
    )
    if non_darwin_pro:
        return {
            "code": "non_darwin_pro",
            "title": "检查模型分流",
            "why": f"{non_darwin_pro['stage_name']}使用了Pro；普通业务原则上应使用Flash。",
            "action": "核对该阶段的发布模型策略，确认是否确有专家分析必要。",
            "expected_direction": "在质量不下降的前提下，减少Pro进入普通业务链路。",
            "retest_method": same_run_retest,
        }

    insufficient = next(
        (
            item
            for item in chain
            if item.get("bucket") == "unavailable"
            or item.get("provider_response_model") is None
            or None
            in (
                item.get("input_cache_hit_tokens"),
                item.get("input_cache_miss_tokens"),
                item.get("output_tokens"),
            )
        ),
        None,
    )
    if insufficient:
        return {
            "code": "evidence_insufficient",
            "title": "先补齐成本观测证据",
            "why": f"{insufficient['stage_name']}缺少完整Provider模型、三类Usage或价格证据。",
            "action": "检查Provider响应采集与价格快照，不从总Token反推拆分。",
            "expected_direction": "先恢复可对账证据，再讨论成本优化；当前不宣称节省金额。",
            "retest_method": same_run_retest,
        }

    miss_candidates = []
    for item in chain:
        hit = int(item.get("input_cache_hit_tokens") or 0)
        miss = int(item.get("input_cache_miss_tokens") or 0)
        input_total = hit + miss
        ratio = miss / input_total if input_total else 0.0
        if miss >= 1000 and ratio >= 0.6:
            miss_candidates.append((miss * ratio, ratio, item))
    if miss_candidates:
        _, ratio, item = max(miss_candidates, key=lambda value: value[0])
        return {
            "code": "cache_miss_high",
            "title": "优先降低缓存未命中输入",
            "why": (
                f"{item['stage_name']}缓存未命中输入{item['input_cache_miss_tokens']:,} Token，"
                f"占该阶段输入{ratio:.0%}。"
            ),
            "action": "缩短重复上下文、稳定公共Prompt，并提高可缓存内容比例。",
            "expected_direction": "减少未命中输入Token及对应直接成本，不预设具体节省比例。",
            "retest_method": same_run_retest,
        }

    output_candidates = []
    for item in chain:
        output = int(item.get("output_tokens") or 0)
        total = int(item.get("total_tokens") or 0)
        ratio = output / total if total else 0.0
        if output >= 1000 or (output >= 500 and ratio >= 0.35):
            output_candidates.append((output, ratio, item))
    if output_candidates:
        output, ratio, item = max(output_candidates, key=lambda value: value[0])
        return {
            "code": "output_high",
            "title": "收紧输出结构与长度",
            "why": f"{item['stage_name']}输出{output:,} Token，占该阶段总Token约{ratio:.0%}。",
            "action": "限制重复说明，固定必要输出结构，并保留根因与建议的完整性。",
            "expected_direction": "减少输出Token和时延，同时守住答案质量。",
            "retest_method": same_run_retest,
        }

    return {
        "code": "normal_observe",
        "title": "当前成本结构正常，建议继续观察",
        "why": "未发现普通链路误用Pro、明显缓存未命中或异常长输出。",
        "action": "不为降低Token而破坏答案、引用或安全门。",
        "expected_direction": "保持当前质量与成本平衡。",
        "retest_method": same_run_retest,
    }


@router.get("/traces/{trace_id}")
async def trace_detail(trace_id: str):
    """Return a single trace with model calls, MCP audits, and messages.

    Each model call includes token-level explainability, price snapshot,
    cost formula, model display name, and badcase linkage where applicable.
    """
    trace = get_chat_trace(trace_id)
    if not trace:
        conn = _get_conn()
        _register_reporting_sql(conn)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT * FROM model_calls
               WHERE trace_id = ?
               ORDER BY yiai_time_epoch(created_at) DESC, id DESC
               LIMIT 1""",
            (trace_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            trace = {
                "trace_id": trace_id,
                "session_id": None,
                "user_message": None,
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["created_at"],
                "intent": None,
                "agent_name": None,
                "model_id": row["model_id"],
                "model_selection_reason": row["model_selection_reason"],
            }
        else:
            raise HTTPException(status_code=404, detail="Trace not found")

    operation_metadata = trace.get("version_snapshot")
    if isinstance(operation_metadata, str) and operation_metadata:
        import json

        try:
            operation_metadata = json.loads(operation_metadata)
        except (TypeError, ValueError, json.JSONDecodeError):
            operation_metadata = None
    trace["operation_metadata"] = (
        operation_metadata if isinstance(operation_metadata, dict) else None
    )

    raw_calls = get_model_calls_for_trace(trace_id)
    mcp_calls = get_mcp_call_audits_for_trace(trace_id)
    session_id = trace.get("session_id")
    messages = list_chat_messages(session_id or "")
    trace_messages = [m for m in messages if m.get("trace_id") == trace_id]

    # Trace detail must retain every physical Provider attempt, including SDK
    # entries whose outbound status remains unconfirmed. Aggregate membership
    # is decided separately and never turns an unknown into a Provider count.
    provider_raw_calls = [
        call
        for call in raw_calls
        if _provider_aggregate_decision(call)["record_kind"] == "provider_attempt"
    ]
    logical_raw_calls = [
        call
        for call in raw_calls
        if _provider_aggregate_decision(call)["record_kind"] != "provider_attempt"
    ]
    model_calls = [_enrich_model_call(c, session_id) for c in provider_raw_calls]
    logical_model_records = [
        _enrich_model_call(c, session_id) for c in logical_raw_calls
    ]
    session_raw_calls: List[Dict[str, Any]] = []
    if session_id:
        for session_trace in list_chat_traces(session_id=session_id, limit=100):
            session_raw_calls.extend(
                get_model_calls_for_trace(session_trace["trace_id"])
            )
    else:
        session_raw_calls = list(raw_calls)
    session_provider_raw_calls = [
        call
        for call in session_raw_calls
        if _provider_aggregate_decision(call)["record_kind"] == "provider_attempt"
    ]
    session_model_calls = [
        _enrich_model_call(c, session_id) for c in session_provider_raw_calls
    ]
    trace_events = list_trace_events(trace_id)
    evaluation_run = get_evaluation_run_by_trace_id(trace_id)

    # Summarize context composition from the vertical model call if available.
    # Router calls have no usage and no context_breakdown.
    context_breakdown = {}
    vertical_call = next((c for c in model_calls if c.get("stage") == "vertical_agent"), None)
    if vertical_call and vertical_call.get("context_breakdown"):
        context_breakdown = vertical_call["context_breakdown"]
    elif vertical_call:
        context_breakdown = {
            "system_prompt_tokens": None,
            "history_tokens": None,
            "skill_tokens": None,
            "rag_tokens": None,
            "tool_result_tokens": None,
            "user_message_tokens": None,
            "note": "本地上下文估算，不等于 Provider 原始账单",
        }

    reconciliation_rows = [call.get("reconciliation") or {} for call in model_calls]
    request_ids = {
        row.get("provider_request_id")
        for row in reconciliation_rows
        if row.get("provider_request_id")
    }
    non_actual_rows = [
        call
        for call in model_calls
        if str(call.get("usage_status") or "") != "provider_actual"
    ]
    reconciliation_summary = {
        "provider_attempt_count": len(model_calls),
        "provider_request_sent_count": sum(
            1 for row in reconciliation_rows if row.get("provider_request_sent") is True
        ),
        "provider_request_id_count": sum(
            1 for row in reconciliation_rows if row.get("provider_request_id")
        ),
        "unique_provider_request_id_count": len(request_ids),
        "done_received_count": sum(
            1 for row in reconciliation_rows if row.get("done_received") is True
        ),
        "sdk_stream_exhausted_count": sum(
            1 for row in reconciliation_rows if row.get("sdk_stream_exhausted") is True
        ),
        "usage_received_count": sum(
            1 for row in reconciliation_rows if row.get("usage_received") is True
        ),
        "persisted_count": sum(
            1
            for row in reconciliation_rows
            if row.get("persisted") is True
        ),
        "persistence_failure_count": sum(
            1 for row in reconciliation_rows if row.get("persisted") is False
        ),
        "record_row_count": sum(
            1 for row in reconciliation_rows if row.get("record_persisted") is True
        ),
        "retry_count": sum(
            1 for row in reconciliation_rows if row.get("retry") is True
        ),
        "included_in_provider_summary_count": sum(
            1
            for row in reconciliation_rows
            if row.get("included_in_provider_summary") is True
        ),
        "non_actual_count": len(non_actual_rows),
        "non_actual_reasons": [
            {
                "local_attempt_id": call.get("local_attempt_id"),
                "stage": call.get("stage"),
                "usage_status": call.get("usage_status"),
                "reason": call.get("usage_unavailable_reason")
                or (call.get("reconciliation") or {}).get("reason")
                or "cause_unconfirmed",
                "evidence": (call.get("reconciliation") or {}).get("evidence") or {},
            }
            for call in non_actual_rows
        ],
    }

    return {
        "trace": trace,
        "model_calls": model_calls,
        "provider_model_calls": model_calls,
        "logical_model_records": logical_model_records,
        "provider_reconciliation": reconciliation_rows,
        "reconciliation_summary": reconciliation_summary,
        "mcp_calls": mcp_calls,
        "trace_events": trace_events,
        "evaluation_run": evaluation_run,
        "messages": trace_messages,
        "context_breakdown": context_breakdown,
        "trace_cost_summary": _cost_summary(model_calls),
        "session_cost_summary": _cost_summary(session_model_calls),
        "trace_cost_explanation": _trace_cost_explanation(model_calls),
    }


# -----------------------------------------------------------------------------
# Distribution & Trends
# -----------------------------------------------------------------------------


@router.get("/distribution")
async def distribution(
    group_by: str = Query("model", pattern="^(model|agent|intent|session|trace|stage)$"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Return token/cost distribution grouped by model/agent/session/trace/stage.

    Each group item includes a list of trace IDs so the aggregate is traceable.
    """
    try:
        effective_start, effective_end = _validated_reporting_range(start, end)
        raw_calls = _fetch_model_calls(effective_start, effective_end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    provider_calls, excluded_calls = _split_model_records(raw_calls)
    trace_metadata: Dict[str, Dict[str, Any]] = {}
    trace_ids = sorted({str(call.get("trace_id")) for call in provider_calls if call.get("trace_id")})
    if trace_ids and group_by in {"agent", "intent", "session"}:
        conn = _get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in trace_ids)
        cursor.execute(
            f"SELECT trace_id, agent_name, intent, session_id FROM chat_traces WHERE trace_id IN ({placeholders})",
            trace_ids,
        )
        trace_metadata = {row["trace_id"]: dict(row) for row in cursor.fetchall()}
        conn.close()

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for call in provider_calls:
        trace_id = str(call.get("trace_id") or "")
        if group_by == "model":
            key = _provider_model(call)
        elif group_by == "stage":
            key = str(call.get("stage") or "unknown")
        elif group_by == "trace":
            key = trace_id or "unknown"
        else:
            metadata = trace_metadata.get(trace_id) or {}
            metadata_key = {
                "agent": "agent_name",
                "intent": "intent",
                "session": "session_id",
            }[group_by]
            key = str(metadata.get(metadata_key) or "unknown")
        grouped.setdefault(key, []).append(call)

    items = []
    for key, group_calls in grouped.items():
        summary = _aggregate_model_calls(group_calls)
        actual_priced = int(summary.get("provider_actual_priced_calls") or 0)
        estimated_priced = int(summary.get("estimated_calls") or 0) - int(
            summary.get("estimated_amount_unavailable_calls") or 0
        )
        known_cost = None
        if actual_priced or estimated_priced:
            known_cost = round(
                float(summary["provider_actual_cost_cny"])
                + float(summary["estimated_cost_cny"]),
                8,
            )
        items.append(
            {
                group_by: key,
                "calls": summary["calls"],
                "provider_request_count": summary["calls"],
                "tokens": summary["total_tokens"],
                "known_token_calls": summary["token_known_calls"],
                "unknown_token_calls": summary["token_unavailable_calls"],
                "cost": known_cost,
                "platform_price_snapshot_direct_cost_cny": (
                    summary["provider_actual_cost_cny"] if actual_priced else None
                ),
                "estimated_cost_cny": (
                    summary["estimated_cost_cny"] if estimated_priced else None
                ),
                "unknown_cost_calls": summary["unavailable_calls"],
                "cost_source": "platform_price_snapshot",
                "trace_ids": sorted(
                    {str(call.get("trace_id")) for call in group_calls if call.get("trace_id")}
                ),
            }
        )
    return {
        "group_by": group_by,
        "items": items,
        "excluded_record_count": len(excluded_calls),
    }


@router.get("/trends")
async def trends(
    group_by: str = Query("hour", pattern="^(hour|day)$"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Return included Provider attempts over time; logical rows never enter."""
    try:
        effective_start, effective_end = _validated_reporting_range(start, end)
        raw_calls = _fetch_model_calls(effective_start, effective_end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    provider_calls, excluded_calls = _split_model_records(raw_calls)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for call in provider_calls:
        period = _reporting_period(call.get("created_at"), group_by)
        grouped.setdefault(period, []).append(call)

    items = []
    for period in sorted(grouped):
        summary = _aggregate_model_calls(grouped[period])
        actual_priced = int(summary.get("provider_actual_priced_calls") or 0)
        estimated_priced = int(summary.get("estimated_calls") or 0) - int(
            summary.get("estimated_amount_unavailable_calls") or 0
        )
        known_cost = None
        if actual_priced or estimated_priced:
            known_cost = round(
                float(summary["provider_actual_cost_cny"])
                + float(summary["estimated_cost_cny"]),
                8,
            )
        items.append(
            {
                "period": period,
                "calls": summary["calls"],
                "provider_request_count": summary["calls"],
                "tokens": summary["total_tokens"],
                "known_token_calls": summary["token_known_calls"],
                "unknown_token_calls": summary["token_unavailable_calls"],
                "cost": known_cost,
                "platform_price_snapshot_direct_cost_cny": (
                    summary["provider_actual_cost_cny"] if actual_priced else None
                ),
                "estimated_cost_cny": (
                    summary["estimated_cost_cny"] if estimated_priced else None
                ),
                "unknown_cost_calls": summary["unavailable_calls"],
                "cost_source": "platform_price_snapshot",
            }
        )
    return {
        "group_by": group_by,
        "items": items,
        "excluded_record_count": len(excluded_calls),
    }


# -----------------------------------------------------------------------------
# Model prices
# -----------------------------------------------------------------------------


@router.get("/prices")
async def prices(enabled_only: bool = False):
    """Return the model price table."""
    return {"prices": list_model_prices(enabled_only=enabled_only)}


@router.get("/prices/{price_id}")
async def get_price(price_id: int):
    """Return a single model price entry."""
    price = get_model_price(price_id)
    if not price:
        raise HTTPException(status_code=404, detail="Price not found")
    return {"price": price}


@router.post("/prices")
async def create_price(request: PriceCreate):
    """Create a new model price entry."""
    price = create_model_price(
        model_id=request.model_id,
        effective_date=request.effective_date,
        input_price_per_1m=request.input_price_per_1m,
        cached_input_price_per_1m=request.cached_input_price_per_1m,
        output_price_per_1m=request.output_price_per_1m,
        reasoning_price_per_1m=request.reasoning_price_per_1m,
        source_note=request.source_note,
        enabled=request.enabled,
    )
    return {"price": price}


@router.put("/prices/{price_id}")
async def update_price(price_id: int, request: PriceUpdate):
    """Update a model price entry."""
    updates = request.dict(exclude_unset=True)
    price = update_model_price(price_id, **updates)
    if not price:
        raise HTTPException(status_code=404, detail="Price not found")
    return {"price": price}


@router.delete("/prices/{price_id}")
async def delete_price(price_id: int):
    """Delete a model price entry."""
    ok = delete_model_price(price_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Price not found")
    return {"status": "ok"}


# -----------------------------------------------------------------------------
# Budget thresholds
# -----------------------------------------------------------------------------


@router.get("/budget")
async def budget():
    """Return current budget thresholds."""
    return {"budget": get_budget_thresholds()}


@router.put("/budget")
async def update_budget(request: BudgetUpdate):
    """Update budget thresholds."""
    budget = update_budget_thresholds(
        per_call_threshold_cny=request.per_call_threshold_cny,
        daily_threshold_cny=request.daily_threshold_cny,
        monthly_threshold_cny=request.monthly_threshold_cny,
    )
    return {"budget": budget}


@router.get("/cost-strategies")
async def cost_strategies():
    """Return the supported cost optimization strategies with navigation links."""
    return {
        "strategies": [
            {
                "id": "COST-01",
                "title": "Flash 默认，Pro 仅限 Darwin 与 A/B",
                "description": (
                    "业主-facing 对话始终使用 deepseek-v4-flash，控制常规流量成本；"
                    "deepseek-v4-pro 仅用于 Darwin 深度运营分析、A/B 测试等后台评估场景，"
                    "避免高单价模型进入普通问答路径。实际节省只在同题质量门槛通过且"
                    "Provider Usage 可比较时成立。"
                ),
                "mechanism": "published model policy",
                "evidence_required": [
                    "same question set",
                    "quality baseline",
                    "per-stage Provider Usage",
                    "published price snapshot",
                ],
                "claim_policy": "Usage 不完整时只展示模型选择与单价差，不宣称精确节省。",
                "status": "runtime_enforced_measurement_required",
                "links": [
                    {"label": "模型配置", "href": "/platform/models"},
                    {"label": "A/B 测试", "href": "/platform/models/ab"},
                ],
            },
            {
                "id": "COST-02",
                "title": "RAG Top-K 与重排序控制上下文规模",
                "description": (
                    "通过 retrieval_settings.top_k 限制召回片段数量，关闭不必要的重排序，"
                    "可以减少候选上下文；但只有答案质量不下降且 Provider Token 实测降低，"
                    "候选配置才可以发布，不能把片段数下降直接写成成本下降。"
                ),
                "mechanism": "immutable retrieval policy in RuntimeRelease",
                "evidence_required": [
                    "baseline and candidate releases",
                    "same question set",
                    "citation and answer quality gate",
                    "per-stage Provider Usage",
                ],
                "claim_policy": "检索预估不等于模型账单；质量下降或 Token 未降时拒绝候选。",
                "status": "experiment_required",
                "links": [
                    {"label": "检索设置", "href": "/platform/knowledge"},
                ],
            },
            {
                "id": "COST-03",
                "title": "Skill 仅在命中且绑定时注入",
                "description": (
                    "只有被 Agent 显式绑定且触发条件命中的 Skill 才会注入到系统提示中；"
                    "未触发或未绑定的 Skill 不占用上下文，避免无意义 token 开销。"
                ),
                "mechanism": "bound-and-triggered progressive Skill loading",
                "evidence_required": [
                    "snapshot binding",
                    "Skill activation Trace",
                    "context composition or Provider Usage comparison",
                ],
                "claim_policy": "可证明未注入，但没有实测 Token 对照时不写固定节省量。",
                "status": "runtime_enforced_measurement_required",
                "links": [
                    {"label": "Agent 绑定", "href": "/platform/agents"},
                    {"label": "Skill 管理", "href": "/platform/skills"},
                ],
            },
            {
                "id": "COST-04",
                "title": "MCP 按需调用、失败审计",
                "description": (
                    "仅当用户问题命中 MCP 工具绑定的能力域时才初始化对应 Server；"
                    "每次调用进入 mcp_call_audits 并自动捕获失败 badcase，便于识别"
                    "无效/高频失败工具。摘要收益必须用同一工具响应的原始长度、摘要长度"
                    "和后续模型 Usage 验证。"
                ),
                "mechanism": "bound on-demand invocation plus audited summary",
                "evidence_required": [
                    "MCP invocation audit",
                    "raw and summarized result sizes",
                    "downstream Provider Usage",
                    "answer quality gate",
                ],
                "claim_policy": "没有同一响应的实测对照时，不展示虚构的摘要 Token 节省。",
                "status": "runtime_enforced_measurement_required",
                "links": [
                    {"label": "MCP 审计", "href": "/platform/observability"},
                    {"label": "Badcase", "href": "/platform/badcases"},
                ],
            },
        ]
    }
