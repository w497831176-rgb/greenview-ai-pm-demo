"""
Observability & Cost Governance API
===================================

Endpoints for trace visibility, model-call auditing, MCP audit,
model pricing table, and budget thresholds.
"""
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

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


def _model_display_name(model_id: Optional[str]) -> str:
    return {
        "deepseek-v4-flash": "Flash",
        "deepseek-v4-pro": "Pro",
    }.get(model_id or "") or (model_id or "unknown")


def _period_bounds() -> Dict[str, Dict[str, Any]]:
    """Return canonical CN-time period bounds used by the overview."""
    dt = now_cn_dt()
    today_start = dt.strftime("%Y-%m-%d 00:00:00")
    today_end = dt.strftime("%Y-%m-%d %H:%M:%S")
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
            "end": current.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": "Asia/Shanghai (UTC+8)",
        }
    return {
        "label": "自定义范围",
        "start": start,
        "end": _normalize_end(end) if end else None,
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


def _provider_model(call: Dict[str, Any]) -> str:
    usage = _usage_payload(call)
    return (
        usage.get("provider_response_model")
        or call.get("provider_response_model")
        or usage.get("requested_model")
        or call.get("model_id")
        or "unknown"
    )


def _cost_bucket(call: Dict[str, Any]) -> str:
    """Classify one real model_call without upgrading a legacy source."""
    source = call.get("usage_source") or _usage_payload(call).get("usage_source")
    amount = call.get("estimated_cost_cny")
    if source == "provider_actual" and amount is not None:
        return "provider_actual"
    if source == "estimated":
        return "estimated"
    return "unavailable"


def _empty_cost_group() -> Dict[str, Any]:
    return {
        "calls": 0,
        "total_tokens": 0,
        "provider_actual_calls": 0,
        "provider_actual_cost_cny": 0.0,
        "estimated_calls": 0,
        "estimated_cost_cny": 0.0,
        "estimated_amount_unavailable_calls": 0,
        "unavailable_calls": 0,
        "known_usage_calls": 0,
        "input_cache_hit_tokens": 0,
        "input_cache_miss_tokens": 0,
        "output_tokens": 0,
    }


def _add_call_to_group(group: Dict[str, Any], call: Dict[str, Any]) -> None:
    group["calls"] += 1
    group["total_tokens"] += int(call.get("total_tokens") or 0)
    bucket = _cost_bucket(call)
    amount = call.get("estimated_cost_cny")
    if bucket == "provider_actual":
        group["provider_actual_calls"] += 1
        group["provider_actual_cost_cny"] += float(amount)
    elif bucket == "estimated":
        group["estimated_calls"] += 1
        if amount is None:
            group["estimated_amount_unavailable_calls"] += 1
        else:
            group["estimated_cost_cny"] += float(amount)
    else:
        group["unavailable_calls"] += 1

    usage = _usage_payload(call)
    hit = usage.get("input_cache_hit_tokens")
    miss = usage.get("input_cache_miss_tokens")
    output = usage.get("output_tokens")
    if bucket == "provider_actual" and None not in (hit, miss, output):
        group["known_usage_calls"] += 1
        group["input_cache_hit_tokens"] += int(hit)
        group["input_cache_miss_tokens"] += int(miss)
        group["output_tokens"] += int(output)


def _finalize_cost_group(group: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(group)
    result["provider_actual_cost_cny"] = round(
        float(result["provider_actual_cost_cny"]), 8
    )
    result["estimated_cost_cny"] = round(float(result["estimated_cost_cny"]), 8)
    result["cost_complete"] = result["unavailable_calls"] == 0
    return result


def _aggregate_model_calls(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = _empty_cost_group()
    by_model: Dict[str, Dict[str, Any]] = {}
    by_stage: Dict[str, Dict[str, Any]] = {}
    failed_calls = 0
    for raw in calls:
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
    result["by_model"] = {
        key: _finalize_cost_group(value) for key, value in by_model.items()
    }
    result["by_stage"] = {
        key: _finalize_cost_group(value) for key, value in by_stage.items()
    }
    return result


def _fetch_model_calls(start: Optional[str], end: Optional[str]) -> List[Dict[str, Any]]:
    conditions = []
    params: List[Any] = []
    if start:
        conditions.append("created_at >= ?")
        params.append(start)
    if end:
        conditions.append("created_at <= ?")
        params.append(end)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM model_calls {where}", params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def _top_provider_actual_traces(
    calls: List[Dict[str, Any]], limit: int = 5
) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for call in calls:
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
                "provider_actual_cost_cny": 0.0,
                "models": set(),
                "stages": set(),
            },
        )
        item["all_calls"] += 1
        if _cost_bucket(call) != "provider_actual":
            continue
        item["provider_actual_calls"] += 1
        item["total_tokens"] += int(call.get("total_tokens") or 0)
        item["provider_actual_cost_cny"] += float(
            call.get("estimated_cost_cny") or 0.0
        )
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
                "models": sorted(item["models"]),
                "stages": sorted(item["stages"]),
            }
        )
    return results


def _query_period_summary(start: str, end: str) -> Dict[str, Any]:
    return _aggregate_model_calls(_fetch_model_calls(start, end))


def _check_budget(strategy: Optional[str] = None) -> Dict[str, Any]:
    """Return daily and monthly budget usage and the highest alert level.

    - blocked: any configured threshold has reached or exceeded 100%.
    - warning: any configured threshold has reached or exceeded 80%.
    - none: no threshold is configured or all usages are below 80%.
    """
    thresholds = get_budget_thresholds()
    daily_threshold = thresholds.get("daily_threshold_cny")
    monthly_threshold = thresholds.get("monthly_threshold_cny")

    bounds = _period_bounds()
    today_cost = 0.0
    month_cost = 0.0
    try:
        conn = _get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COALESCE(SUM(estimated_cost_cny), 0) as cost FROM model_calls WHERE created_at >= ? AND created_at <= ?",
            (bounds["today"]["start"], bounds["today"]["end"]),
        )
        today_cost = float(cursor.fetchone()["cost"])
        cursor.execute(
            "SELECT COALESCE(SUM(estimated_cost_cny), 0) as cost FROM model_calls WHERE created_at >= ? AND created_at <= ?",
            (bounds["this_month"]["start"], bounds["this_month"]["end"]),
        )
        month_cost = float(cursor.fetchone()["cost"])
        conn.close()
    except Exception:
        today_cost = 0.0
        month_cost = 0.0

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
        "daily_usage_percent": daily_usage_percent,
        "monthly_usage_percent": monthly_usage_percent,
        "alert_level": alert_level,
        "reason": reason,
        "trigger_dimension": trigger_dimension,
        "today_cost": round(today_cost, 8),
        "month_cost": round(month_cost, 8),
        "daily_threshold_cny": daily_threshold,
        "monthly_threshold_cny": monthly_threshold,
        "per_call_threshold_cny": thresholds.get("per_call_threshold_cny"),
        "strategy": strategy,
    }


# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------


@router.get("/overview")
async def overview(
    start: Optional[str] = Query(None, description="Start date/time ISO"),
    end: Optional[str] = Query(None, description="End date/time ISO"),
):
    """Return honest cost buckets for an explicit Asia/Shanghai range."""
    scope = _overview_scope(start, end)
    selected_calls = _fetch_model_calls(scope.get("start"), scope.get("end"))
    data = _aggregate_model_calls(selected_calls)
    history_total = _aggregate_model_calls(_fetch_model_calls(None, None))
    thresholds = get_budget_thresholds()
    known_cost = data["provider_actual_cost_cny"] + data["estimated_cost_cny"]
    priced_calls = data["provider_actual_calls"] + (
        data["estimated_calls"] - data["estimated_amount_unavailable_calls"]
    )
    per_call_cost = known_cost / priced_calls if priced_calls else 0.0

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
    if thresholds.get("per_call_threshold_cny") and per_call_cost > thresholds["per_call_threshold_cny"]:
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
        "total_tokens": data["total_tokens"],
        "provider_actual_calls": data["provider_actual_calls"],
        "provider_actual_cost_cny": data["provider_actual_cost_cny"],
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
        },
        "failed_calls": data["failed_calls"],
        "alerts": alerts,
        "currency": "CNY",
        "cost_note": "Provider真实成本仅汇总 provider_actual；本地估算单列且不是供应商账单；金额不可计算的模型调用不按0元处理。",
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
    """Expand a bare YYYY-MM-DD end date to the last second of that day."""
    if not end:
        return end
    # If already has time component, leave as-is.
    if len(end) > 10 or " " in end or "T" in end:
        return end
    try:
        from datetime import datetime
        datetime.strptime(end, "%Y-%m-%d")
        return f"{end} 23:59:59"
    except ValueError:
        return end


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
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List chat traces aggregated with their model calls.

    Each returned trace includes:
    - models: list of models actually invoked for this trace
    - total_tokens: sum of model_calls.total_tokens
    - estimated_cost_cny: sum of estimated_cost_cny (null if any call lacks a price)
    - price_missing: true when at least one model call had no configured price
    - no_model_calls: true when the trace has no model call records
    """
    effective_end = _normalize_end(end)

    # Build model-call aggregation with its own filters.
    m_conditions = ["1=1"]
    m_params: List[Any] = []
    if model_id:
        m_conditions.append("model_id = ?")
        m_params.append(model_id)
    if stage:
        m_conditions.append("stage = ?")
        m_params.append(stage)
    if start:
        m_conditions.append("created_at >= ?")
        m_params.append(start)
    if effective_end:
        m_conditions.append("created_at <= ?")
        m_params.append(effective_end)
    m_where = " AND ".join(m_conditions)

    # Build chat-trace filters.
    t_conditions = ["1=1"]
    t_params: List[Any] = []
    if trace_id:
        t_conditions.append("trace_id = ?")
        t_params.append(trace_id)
    if session_id:
        t_conditions.append("session_id = ?")
        t_params.append(session_id)
    if intent:
        t_conditions.append("intent = ?")
        t_params.append(intent)
    if agent:
        t_conditions.append("agent_name = ?")
        t_params.append(agent)
    if start:
        t_conditions.append("created_at >= ?")
        t_params.append(start)
    if effective_end:
        t_conditions.append("created_at <= ?")
        t_params.append(effective_end)
    t_where = " AND ".join(t_conditions)

    conn = _get_conn()
    cursor = conn.cursor()

    # Aggregate model calls per trace.
    cursor.execute(
        f"""
        SELECT
            trace_id,
            GROUP_CONCAT(DISTINCT model_id) as model_ids,
            COALESCE(SUM(total_tokens), 0) as total_tokens,
            SUM(estimated_cost_cny) as estimated_cost_cny,
            COALESCE(SUM(CASE WHEN usage_source = 'provider_actual' THEN estimated_cost_cny ELSE 0 END), 0) as provider_actual_cost_cny,
            COALESCE(SUM(CASE WHEN usage_source = 'estimated' THEN estimated_cost_cny ELSE 0 END), 0) as local_estimated_cost_cny,
            SUM(CASE WHEN usage_source = 'provider_actual' THEN 1 ELSE 0 END) as provider_actual_calls,
            SUM(CASE WHEN usage_source = 'estimated' THEN 1 ELSE 0 END) as estimated_calls,
            SUM(CASE WHEN COALESCE(usage_source, 'unavailable') NOT IN ('provider_actual', 'estimated') OR (usage_source = 'provider_actual' AND estimated_cost_cny IS NULL) THEN 1 ELSE 0 END) as unknown_cost_calls,
            COUNT(*) as call_count
        FROM model_calls
        WHERE {m_where}
        GROUP BY trace_id
        """,
        m_params,
    )
    agg_rows = {r["trace_id"]: dict(r) for r in cursor.fetchall()}

    # Main query: chat traces joined with aggregated model-call metrics.
    cursor.execute(
        f"""
        SELECT
            t.trace_id,
            t.session_id,
            t.user_message,
            t.intent,
            t.agent_name,
            t.status,
            t.created_at,
            t.updated_at
        FROM chat_traces t
        WHERE {t_where}
        ORDER BY t.created_at DESC
        LIMIT ? OFFSET ?
        """,
        t_params + [limit, offset],
    )
    trace_rows = cursor.fetchall()

    # Also include traces that only exist in model_calls when no chat-trace
    # filters other than start/end/model/stage are requested.
    if not any([trace_id, session_id, intent, agent]):
        cursor.execute(
            f"""
            SELECT
                m.trace_id,
                NULL as session_id,
                NULL as user_message,
                NULL as intent,
                NULL as agent_name,
                MAX(m.status) as status,
                MAX(m.created_at) as created_at,
                MAX(m.created_at) as updated_at
            FROM model_calls m
            WHERE {m_where}
              AND m.trace_id NOT IN (SELECT trace_id FROM chat_traces)
            GROUP BY m.trace_id
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            m_params + [limit, offset],
        )
        trace_rows.extend(cursor.fetchall())

    conn.close()

    results = []
    for row in trace_rows:
        trace = dict(row)
        agg = agg_rows.get(trace["trace_id"], {})
        model_ids_str = agg.get("model_ids") or ""
        model_ids = [m for m in model_ids_str.split(",") if m]
        call_count = agg.get("call_count") or 0
        total_tokens = agg.get("total_tokens") or 0
        estimated_cost_cny = agg.get("estimated_cost_cny")
        provider_actual_cost_cny = agg.get("provider_actual_cost_cny") or 0.0
        local_estimated_cost_cny = agg.get("local_estimated_cost_cny") or 0.0
        provider_actual_calls = agg.get("provider_actual_calls") or 0
        estimated_calls = agg.get("estimated_calls") or 0
        unknown_cost_calls = agg.get("unknown_cost_calls") or 0
        price_missing = unknown_cost_calls > 0

        # Build a concise model summary.
        if not model_ids:
            model_summary = "尚无模型调用记录"
        elif len(model_ids) == 1:
            display = _model_display_name(model_ids[0])
            model_summary = display
        else:
            display = _model_display_name(model_ids[0])
            model_summary = f"{display}（router + vertical）"

        trace["models"] = model_ids
        trace["model_summary"] = model_summary
        trace["total_tokens"] = total_tokens if call_count else None
        trace["estimated_cost_cny"] = estimated_cost_cny
        trace["provider_actual_cost_cny"] = round(provider_actual_cost_cny, 8)
        trace["local_estimated_cost_cny"] = round(local_estimated_cost_cny, 8)
        trace["provider_actual_calls"] = provider_actual_calls
        trace["estimated_calls"] = estimated_calls
        trace["unavailable_calls"] = unknown_cost_calls
        trace["cost_status"] = (
            "not_applicable"
            if call_count == 0
            else "partial_unavailable"
            if unknown_cost_calls
            else "provider_actual"
            if provider_actual_calls and not estimated_calls
            else "estimated"
            if estimated_calls and not provider_actual_calls
            else "mixed"
        )
        trace["price_missing"] = price_missing
        trace["no_model_calls"] = call_count == 0
        results.append(trace)

    return {"traces": results, "start": start, "end": effective_end}


def _build_cost_formula(call: Dict[str, Any]) -> str:
    """Build a human-readable cost formula from the recorded price snapshot."""
    normalized = call.get("usage_normalized") or {}
    contract = normalized.get("cost_contract") or {}
    if contract.get("formula"):
        return contract["formula"]
    if contract.get("availability_note"):
        return contract["availability_note"]

    snapshot = call.get("price_snapshot") or {}
    if not snapshot:
        return "单价已配置，但 Provider 未返回本次 usage，无法估算本次成本"

    usage_source = call.get("usage_source")
    if usage_source == "unavailable":
        return "单价已配置，但 Provider 未返回本次 usage，无法估算本次成本"

    terms = []
    input_p = snapshot.get("input_price_per_1m")
    cached_p = snapshot.get("cached_input_price_per_1m")
    output_p = snapshot.get("output_price_per_1m")

    # V1.4.3: normalized usage fields are preferred over legacy raw fields.
    if normalized.get("usage_split_unavailable"):
        return "usage_split_unavailable：Provider 未拆分缓存/未缓存输入，成本 --"

    uncached = normalized.get("uncached_input_tokens")
    cached = normalized.get("cached_input_tokens")
    output = normalized.get("output_tokens")

    if uncached is not None and input_p is not None:
        terms.append(f"uncached_input_tokens({uncached}) * {input_p} / 1_000_000")
    if cached is not None and cached_p is not None:
        terms.append(f"cached_input_tokens({cached}) * {cached_p} / 1_000_000")
    if output is not None and output_p is not None:
        terms.append(f"output_tokens({output}) * {output_p} / 1_000_000")

    if not terms:
        return "价格快照中无有效单价，无法估算成本"
    return " + ".join(terms)


def _enrich_model_call(call: Dict[str, Any], session_id: Optional[str]) -> Dict[str, Any]:
    """Add display name, session linkage, badcase linkage, and cost formula."""
    import json

    enriched = dict(call)
    model_id = enriched.get("model_id")
    enriched["model_name"] = _model_display_name(model_id)
    enriched["session_id"] = session_id

    # Parse normalized usage JSON if stored as string.
    usage_norm = enriched.get("usage_normalized")
    if isinstance(usage_norm, str):
        try:
            usage_norm = json.loads(usage_norm)
        except Exception:
            usage_norm = None
    enriched["usage_normalized"] = usage_norm or {}
    enriched["requested_model"] = (
        enriched["usage_normalized"].get("requested_model") or model_id
    )
    # Never fall back to requested_model: null means the Provider/SDK did not
    # return or retain the actual response model for this historical call.
    enriched["provider_response_model"] = enriched["usage_normalized"].get(
        "provider_response_model"
    )
    enriched["thinking_enabled"] = enriched["usage_normalized"].get(
        "thinking_enabled"
    )
    enriched["provider_request_id"] = enriched["usage_normalized"].get(
        "provider_request_id"
    )
    enriched["provider_usage"] = {
        "input_cache_hit_tokens": enriched["usage_normalized"].get(
            "input_cache_hit_tokens"
        ),
        "input_cache_miss_tokens": enriched["usage_normalized"].get(
            "input_cache_miss_tokens"
        ),
        "output_tokens": enriched["usage_normalized"].get("output_tokens"),
    }

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
    if stage in ("darwin", "badcase_classify", "retest"):
        enriched["badcase_id"] = get_badcase_id_by_trace_id(enriched.get("trace_id"))
    else:
        enriched["badcase_id"] = None

    enriched["cost_formula"] = _build_cost_formula(enriched)
    return enriched


def _cost_summary(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, Dict[str, Any]] = {}
    known_cost_cny = 0.0
    unknown_cost_calls = 0
    for call in calls:
        model_id = (
            call.get("provider_response_model")
            or call.get("requested_model")
            or call.get("model_id")
            or "unknown"
        )
        item = by_model.setdefault(
            model_id,
            {"model_id": model_id, "calls": 0, "tokens": 0, "cost_cny": 0.0, "unknown_cost_calls": 0},
        )
        item["calls"] += 1
        item["tokens"] += int(call.get("total_tokens") or 0)
        amount = call.get("estimated_cost_cny")
        if amount is None:
            unknown_cost_calls += 1
            item["unknown_cost_calls"] += 1
        else:
            known_cost_cny += float(amount)
            item["cost_cny"] += float(amount)
    return {
        "calls": len(calls),
        "known_cost_cny": round(known_cost_cny, 8),
        "unknown_cost_calls": unknown_cost_calls,
        "complete": unknown_cost_calls == 0,
        "by_model": list(by_model.values()),
    }


def _stage_display_name(stage: Optional[str]) -> str:
    return {
        "router": "Router",
        "vertical_agent": "垂直Agent",
        "badcase_classify": "Badcase分类",
        "darwin": "Darwin/AI专家",
        "retest": "Badcase复测",
    }.get(stage or "") or (stage or "模型调用")


def _trace_cost_explanation(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build one interview-friendly cost story from persisted model calls."""
    chain = []
    model_counts: Dict[str, int] = {}
    provider_actual_cost = 0.0
    estimated_cost = 0.0
    provider_actual_calls = 0
    estimated_calls = 0
    unavailable_calls = 0
    total_tokens = 0

    for raw in calls:
        call = dict(raw)
        usage = _usage_payload(call)
        bucket = _cost_bucket(call)
        response_model = usage.get("provider_response_model") or call.get(
            "provider_response_model"
        )
        requested_model = usage.get("requested_model") or call.get(
            "requested_model"
        ) or call.get("model_id")
        display_model = response_model or requested_model or "unknown"
        model_counts[display_model] = model_counts.get(display_model, 0) + 1
        amount = call.get("estimated_cost_cny")
        if bucket == "provider_actual":
            provider_actual_calls += 1
            provider_actual_cost += float(amount)
        elif bucket == "estimated":
            estimated_calls += 1
            if amount is not None:
                estimated_cost += float(amount)
        else:
            unavailable_calls += 1
        total_tokens += int(call.get("total_tokens") or 0)
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
                "usage_source": call.get("usage_source")
                or usage.get("usage_source")
                or "unavailable",
                "input_cache_hit_tokens": usage.get("input_cache_hit_tokens"),
                "input_cache_miss_tokens": usage.get("input_cache_miss_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": call.get("total_tokens"),
                "amount_cny": amount if bucket != "unavailable" else None,
                "price_snapshot": call.get("price_snapshot") or {},
                "formula": call.get("cost_formula"),
                "bucket": bucket,
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
        if provider_actual_calls == len(chain):
            summary = (
                f"本轮调用{call_text}，共{total_tokens:,} Token，"
                f"Provider真实成本¥{provider_actual_cost:.8f}。"
            )
        else:
            parts = [f"本轮调用{call_text}，共{total_tokens:,} Token"]
            if provider_actual_calls:
                parts.append(f"Provider真实成本¥{provider_actual_cost:.8f}")
            if estimated_calls:
                parts.append(
                    f"本地估算成本¥{estimated_cost:.8f}（非供应商账单）"
                    if estimated_cost
                    else f"{estimated_calls}次本地估算未形成可计金额"
                )
            if unavailable_calls:
                parts.append(f"{unavailable_calls}次金额不可计算")
            summary = "；".join(parts) + "。"

    recommendation = _single_trace_recommendation(chain)
    return {
        "summary": summary,
        "model_call_count": len(chain),
        "total_tokens": total_tokens if chain else None,
        "chain": chain,
        "cost_scope": {
            "status": "not_applicable" if not chain else "model_calls_present",
            "provider_actual_calls": provider_actual_calls,
            "provider_actual_cost_cny": round(provider_actual_cost, 8),
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
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM model_calls WHERE trace_id = ? ORDER BY created_at DESC LIMIT 1",
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

    model_calls = [_enrich_model_call(c, session_id) for c in raw_calls]
    session_model_calls: List[Dict[str, Any]] = []
    if session_id:
        for session_trace in list_chat_traces(session_id=session_id, limit=100):
            session_model_calls.extend(
                _enrich_model_call(c, session_id)
                for c in get_model_calls_for_trace(session_trace["trace_id"])
            )
    else:
        session_model_calls = list(model_calls)
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

    return {
        "trace": trace,
        "model_calls": model_calls,
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
    group_by: str = Query("model", regex="^(model|agent|intent|session|trace|stage)$"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Return token/cost distribution grouped by model/agent/session/trace/stage.

    Each group item includes a list of trace IDs so the aggregate is traceable.
    """
    column_map = {
        "model": ("model_id", "model_calls"),
        "agent": ("agent_name", "chat_traces"),
        "intent": ("intent", "chat_traces"),
        "session": ("session_id", "chat_traces"),
        "trace": ("trace_id", "model_calls"),
        "stage": ("stage", "model_calls"),
    }
    column, table = column_map.get(group_by, ("model_id", "model_calls"))

    conn = _get_conn()
    cursor = conn.cursor()
    date_filter = ""
    params = []
    if start and end:
        date_filter = "WHERE created_at >= ? AND created_at <= ?"
        params = [start, end]
    elif start:
        date_filter = "WHERE created_at >= ?"
        params = [start]
    elif end:
        date_filter = "WHERE created_at <= ?"
        params = [end]

    if table == "model_calls":
        cursor.execute(
            f"""
            SELECT {column},
                   COUNT(*) as calls,
                   SUM(total_tokens) as tokens,
                   SUM(estimated_cost_cny) as cost,
                   GROUP_CONCAT(DISTINCT trace_id) as trace_ids
            FROM model_calls
            {date_filter}
            GROUP BY {column}
            """,
            params,
        )
    else:
        cursor.execute(
            f"""
            SELECT t.{column},
                   COUNT(m.id) as calls,
                   SUM(m.total_tokens) as tokens,
                   SUM(m.estimated_cost_cny) as cost,
                   GROUP_CONCAT(DISTINCT m.trace_id) as trace_ids
            FROM chat_traces t
            JOIN model_calls m ON t.trace_id = m.trace_id
            {date_filter.replace('WHERE', 'WHERE t.') if date_filter else ''}
            GROUP BY t.{column}
            """,
            params,
        )
    rows = cursor.fetchall()
    conn.close()

    items = []
    for r in rows:
        item = dict(r)
        trace_ids = item.get("trace_ids")
        item["trace_ids"] = trace_ids.split(",") if trace_ids else []
        items.append(item)
    return {"group_by": group_by, "items": items}


@router.get("/trends")
async def trends(
    group_by: str = Query("hour", regex="^(hour|day)$"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Return calls/tokens/cost over time."""
    fmt = "%Y-%m-%d %H:00" if group_by == "hour" else "%Y-%m-%d"
    conn = _get_conn()
    cursor = conn.cursor()
    date_filter = ""
    params = []
    if start and end:
        date_filter = "WHERE created_at >= ? AND created_at <= ?"
        params = [start, end]
    elif start:
        date_filter = "WHERE created_at >= ?"
        params = [start]
    elif end:
        date_filter = "WHERE created_at <= ?"
        params = [end]

    cursor.execute(
        f"""
        SELECT strftime('{fmt}', created_at) as period,
               COUNT(*) as calls,
               SUM(total_tokens) as tokens,
               SUM(estimated_cost_cny) as cost
        FROM model_calls
        {date_filter}
        GROUP BY period
        ORDER BY period ASC
        """,
        params,
    )
    rows = cursor.fetchall()
    conn.close()
    return {"group_by": group_by, "items": [dict(r) for r in rows]}


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
