"""
Observability & Cost Governance API
===================================

Endpoints for trace visibility, model-call auditing, MCP audit,
model pricing table, and budget thresholds.
"""
from datetime import timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from db.property_db import (
    create_model_price,
    delete_model_price,
    get_badcase_id_by_trace_id,
    get_budget_thresholds,
    get_chat_trace,
    get_mcp_call_audits_for_trace,
    get_model_call,
    get_model_calls_for_trace,
    get_model_price,
    list_chat_messages,
    list_chat_traces,
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
    today_end = dt.strftime("%Y-%m-%d 23:59:59")
    week_start = (dt - timedelta(days=6)).strftime("%Y-%m-%d 00:00:00")
    month_start = dt.replace(day=1).strftime("%Y-%m-%d 00:00:00")
    return {
        "today": {"start": today_start, "end": today_end, "days": 1},
        "last_7_days": {"start": week_start, "end": today_end, "days": 7},
        "this_month": {"start": month_start, "end": today_end, "days": dt.day},
    }


def _query_period_summary(start: str, end: str) -> Dict[str, Any]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            COALESCE(SUM(total_tokens), 0) as total_tokens,
            COALESCE(SUM(estimated_cost_cny), 0) as estimated_cost_cny,
            COALESCE(SUM(CASE WHEN usage_source = 'provider_reported' THEN total_tokens ELSE 0 END), 0) as provider_reported_tokens,
            COALESCE(SUM(CASE WHEN usage_source != 'provider_reported' THEN total_tokens ELSE 0 END), 0) as local_estimated_tokens,
            SUM(CASE WHEN estimated_cost_cny IS NULL THEN 1 ELSE 0 END) as unknown_cost_calls
        FROM model_calls
        WHERE created_at >= ? AND created_at <= ?
        """,
        (start, end),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {
            "total_tokens": 0,
            "estimated_cost_cny": 0.0,
            "provider_reported_tokens": 0,
            "local_estimated_tokens": 0,
            "unknown_cost_calls": 0,
        }
    return {
        "total_tokens": row["total_tokens"] or 0,
        "estimated_cost_cny": row["estimated_cost_cny"] or 0.0,
        "provider_reported_tokens": row["provider_reported_tokens"] or 0,
        "local_estimated_tokens": row["local_estimated_tokens"] or 0,
        "unknown_cost_calls": row["unknown_cost_calls"] or 0,
    }


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
    """Return aggregate call/token/cost metrics for the selected time range.

    Also returns pre-computed today / last-7-days / this-month summaries,
    Flash-vs-Pro breakdown, and stage-level breakdown.
    """
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
        SELECT
            COUNT(*) as calls,
            COALESCE(SUM(total_tokens), 0) as total_tokens,
            COALESCE(SUM(estimated_cost_cny), 0) as total_cost,
            COALESCE(AVG(total_tokens), 0) as avg_tokens,
            COALESCE(AVG(estimated_cost_cny), 0) as avg_cost,
            SUM(CASE WHEN estimated_cost_cny IS NULL THEN 1 ELSE 0 END) as unknown_cost_calls,
            SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) as failed_calls
        FROM model_calls
        {date_filter}
        """,
        params,
    )
    row = cursor.fetchone()

    cursor.execute(
        f"""
        SELECT model_id,
               COUNT(*) as calls,
               COALESCE(SUM(total_tokens), 0) as total_tokens,
               COALESCE(SUM(estimated_cost_cny), 0) as total_cost
        FROM model_calls
        {date_filter}
        GROUP BY model_id
        """,
        params,
    )
    model_rows = cursor.fetchall()

    cursor.execute(
        f"""
        SELECT stage,
               COUNT(*) as calls,
               COALESCE(SUM(total_tokens), 0) as total_tokens,
               COALESCE(SUM(estimated_cost_cny), 0) as total_cost
        FROM model_calls
        {date_filter}
        GROUP BY stage
        """,
        params,
    )
    stage_rows = cursor.fetchall()
    conn.close()

    data = dict(row) if row else {}
    thresholds = get_budget_thresholds()
    per_call_cost = data.get("avg_cost") or 0.0

    # Period summaries
    periods = {}
    daily_threshold = thresholds.get("daily_threshold_cny")
    monthly_threshold = thresholds.get("monthly_threshold_cny")
    for name, bounds in _period_bounds().items():
        summary = _query_period_summary(bounds["start"], bounds["end"])
        usage_percent = None
        if name == "this_month" and monthly_threshold and monthly_threshold > 0:
            usage_percent = round((summary["estimated_cost_cny"] / monthly_threshold) * 100, 4)
        elif daily_threshold and daily_threshold > 0:
            denominator = daily_threshold * bounds["days"]
            if denominator:
                usage_percent = round((summary["estimated_cost_cny"] / denominator) * 100, 4)
        summary["budget_usage_percent"] = usage_percent
        periods[name] = summary

    daily_cost = periods["today"]["estimated_cost_cny"]
    month_cost = periods["this_month"]["estimated_cost_cny"]

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

    # Flash vs Pro breakdown
    by_model: Dict[str, Dict[str, Any]] = {}
    price_missing = False
    for r in model_rows:
        model_id = r["model_id"] or "unknown"
        by_model[model_id] = {
            "model_name": _model_display_name(model_id),
            "calls": r["calls"] or 0,
            "total_tokens": r["total_tokens"] or 0,
            "estimated_cost_cny": r["total_cost"] if r["total_cost"] is not None else None,
            "price_missing": r["total_cost"] is None,
        }
        if r["total_cost"] is None:
            price_missing = True

    # Stage breakdown, collapsing ab_test_a / ab_test_b into ab_test
    by_stage: Dict[str, Dict[str, Any]] = {
        "router": {"calls": 0, "total_tokens": 0, "estimated_cost_cny": 0.0},
        "vertical_agent": {"calls": 0, "total_tokens": 0, "estimated_cost_cny": 0.0},
        "darwin": {"calls": 0, "total_tokens": 0, "estimated_cost_cny": 0.0},
        "ab_test": {"calls": 0, "total_tokens": 0, "estimated_cost_cny": 0.0},
        "badcase_classify": {"calls": 0, "total_tokens": 0, "estimated_cost_cny": 0.0},
    }
    for r in stage_rows:
        stage = r["stage"] or "unknown"
        target = "ab_test" if stage in ("ab_test_a", "ab_test_b") else stage
        if target not in by_stage:
            continue
        entry = by_stage[target]
        entry["calls"] += r["calls"] or 0
        entry["total_tokens"] += r["total_tokens"] or 0
        cost = r["total_cost"] if r["total_cost"] is not None else 0.0
        if entry["estimated_cost_cny"] is not None:
            entry["estimated_cost_cny"] += cost
        if r["total_cost"] is None:
            entry["estimated_cost_cny"] = None
            price_missing = True

    return {
        "calls": data.get("calls") or 0,
        "total_tokens": data.get("total_tokens") or 0,
        "total_cost": data.get("total_cost") if data.get("total_cost") is not None else None,
        "avg_tokens": data.get("avg_tokens") or 0,
        "avg_cost": data.get("avg_cost") if data.get("avg_cost") is not None else None,
        "unknown_cost_calls": data.get("unknown_cost_calls") or 0,
        "failed_calls": data.get("failed_calls") or 0,
        "alerts": alerts,
        "currency": "CNY",
        "cost_note": "按配置价格估算成本，非供应商实际结算金额",
        "today": periods["today"],
        "last_7_days": periods["last_7_days"],
        "this_month": periods["this_month"],
        "by_model": by_model,
        "by_stage": by_stage,
        "price_missing": price_missing,
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
            SUM(CASE WHEN estimated_cost_cny IS NULL THEN 1 ELSE 0 END) as unknown_cost_calls,
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
        unknown_cost_calls = agg.get("unknown_cost_calls") or 0
        price_missing = unknown_cost_calls > 0 or (call_count == 0)

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
        trace["price_missing"] = price_missing
        trace["no_model_calls"] = call_count == 0
        results.append(trace)

    return {"traces": results, "start": start, "end": effective_end}


def _build_cost_formula(call: Dict[str, Any]) -> str:
    """Build a human-readable cost formula from the recorded price snapshot."""
    snapshot = call.get("price_snapshot") or {}
    if not snapshot:
        return "价格未配置，无法估算成本"

    terms = []
    input_p = snapshot.get("input_price_per_1m")
    cached_p = snapshot.get("cached_input_price_per_1m")
    output_p = snapshot.get("output_price_per_1m")
    reasoning_p = snapshot.get("reasoning_price_per_1m")

    if input_p is not None:
        terms.append(f"(input_tokens - cached_tokens) * {input_p} / 1_000_000")
    if cached_p is not None:
        terms.append(f"cached_tokens * {cached_p} / 1_000_000")
    if output_p is not None:
        terms.append(f"output_tokens * {output_p} / 1_000_000")
    if reasoning_p is not None:
        terms.append(f"reasoning_tokens * {reasoning_p} / 1_000_000")

    if not terms:
        return "价格快照中无有效单价，无法估算成本"
    return " + ".join(terms)


def _enrich_model_call(call: Dict[str, Any], session_id: Optional[str]) -> Dict[str, Any]:
    """Add display name, session linkage, badcase linkage, and cost formula."""
    enriched = dict(call)
    model_id = enriched.get("model_id")
    enriched["model_name"] = _model_display_name(model_id)
    enriched["session_id"] = session_id

    stage = enriched.get("stage") or ""
    if stage in ("darwin", "badcase_classify", "retest"):
        enriched["badcase_id"] = get_badcase_id_by_trace_id(enriched.get("trace_id"))
    else:
        enriched["badcase_id"] = None

    enriched["cost_formula"] = _build_cost_formula(enriched)
    return enriched


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

    raw_calls = get_model_calls_for_trace(trace_id)
    mcp_calls = get_mcp_call_audits_for_trace(trace_id)
    session_id = trace.get("session_id")
    messages = list_chat_messages(session_id or "")
    trace_messages = [m for m in messages if m.get("trace_id") == trace_id]

    model_calls = [_enrich_model_call(c, session_id) for c in raw_calls]

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
        "messages": trace_messages,
        "context_breakdown": context_breakdown,
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
                    "避免高单价模型进入普通问答路径。"
                ),
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
                    "减少输入到模型的上下文 token 量，从而降低单次调用估算成本。"
                ),
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
                    "无效/高频失败工具，避免重复调用浪费 token。"
                ),
                "links": [
                    {"label": "MCP 审计", "href": "/platform/observability"},
                    {"label": "Badcase", "href": "/platform/badcases"},
                ],
            },
        ]
    }
