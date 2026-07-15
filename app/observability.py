"""
Observability & Cost Governance API
===================================

Endpoints for trace visibility, model-call auditing, MCP audit,
model pricing table, and budget thresholds.
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from db.property_db import (
    create_model_price,
    delete_model_price,
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


# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------


@router.get("/overview")
async def overview(
    start: Optional[str] = Query(None, description="Start date/time ISO"),
    end: Optional[str] = Query(None, description="End date/time ISO"),
):
    """Return aggregate call/token/cost metrics for the selected time range."""
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
    conn.close()

    data = dict(row) if row else {}
    thresholds = get_budget_thresholds()
    daily_cost = data.get("total_cost") or 0.0
    per_call_cost = data.get("avg_cost") or 0.0

    alerts = []
    if thresholds.get("daily_threshold_cny") and daily_cost > thresholds["daily_threshold_cny"]:
        alerts.append({
            "type": "daily",
            "threshold": thresholds["daily_threshold_cny"],
            "actual": round(daily_cost, 6),
        })
    if thresholds.get("per_call_threshold_cny") and per_call_cost > thresholds["per_call_threshold_cny"]:
        alerts.append({
            "type": "per_call_avg",
            "threshold": thresholds["per_call_threshold_cny"],
            "actual": round(per_call_cost, 6),
        })

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
    }


# -----------------------------------------------------------------------------
# Traces
# -----------------------------------------------------------------------------


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
    """List chat traces with optional filters."""
    conn = _get_conn()
    cursor = conn.cursor()
    conditions = ["1=1"]
    params = []
    if trace_id:
        conditions.append("trace_id = ?")
        params.append(trace_id)
    if session_id:
        conditions.append("session_id = ?")
        params.append(session_id)
    if intent:
        conditions.append("intent = ?")
        params.append(intent)
    if agent:
        conditions.append("agent_name = ?")
        params.append(agent)
    if start:
        conditions.append("created_at >= ?")
        params.append(start)
    if end:
        conditions.append("created_at <= ?")
        params.append(end)

    where = " AND ".join(conditions)
    cursor.execute(
        f"""
        SELECT * FROM chat_traces WHERE {where}
        UNION
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
        WHERE m.trace_id NOT IN (SELECT trace_id FROM chat_traces)
        GROUP BY m.trace_id
        ORDER BY created_at DESC LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    )
    rows = cursor.fetchall()
    conn.close()
    return {"traces": [dict(r) for r in rows]}


@router.get("/traces/{trace_id}")
async def trace_detail(trace_id: str):
    """Return a single trace with model calls, MCP audits, and messages."""
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

    model_calls = get_model_calls_for_trace(trace_id)
    mcp_calls = get_mcp_call_audits_for_trace(trace_id)
    messages = list_chat_messages(trace.get("session_id") or "")
    trace_messages = [m for m in messages if m.get("trace_id") == trace_id]

    # Summarize context composition from the assistant message for this trace.
    assistant_msg = next((m for m in trace_messages if m.get("role") == "assistant"), None)
    context_breakdown = {}
    if assistant_msg:
        token_detail = assistant_msg.get("token_detail") or {}
        context_breakdown = {
            "system_prompt_tokens": 0,
            "history_tokens": 0,
            "skill_tokens": 0,
            "rag_tokens": 0,
            "tool_result_tokens": 0,
            "user_message_tokens": 0,
            "note": "本地估算构成，非 Provider 原始账单",
        }
        # We do not have exact per-category token counts; the UI will show the
        # estimate with an honest note.

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
    group_by: str = Query("model", regex="^(model|agent|intent|stage)$"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Return token/cost distribution grouped by model/agent/intent/stage."""
    column = {
        "model": "model_id",
        "agent": "agent_name",
        "intent": "intent",
        "stage": "stage",
    }.get(group_by, "model_id")

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

    if group_by == "stage":
        cursor.execute(
            f"""
            SELECT {column}, COUNT(*) as calls, SUM(total_tokens) as tokens,
                   SUM(estimated_cost_cny) as cost
            FROM model_calls
            {date_filter}
            GROUP BY {column}
            """,
            params,
        )
    else:
        cursor.execute(
            f"""
            SELECT t.{column}, COUNT(m.id) as calls, SUM(m.total_tokens) as tokens,
                   SUM(m.estimated_cost_cny) as cost
            FROM chat_traces t
            JOIN model_calls m ON t.trace_id = m.trace_id
            {date_filter.replace('WHERE', 'WHERE t.') if date_filter else ''}
            GROUP BY t.{column}
            """,
            params,
        )
    rows = cursor.fetchall()
    conn.close()
    return {"group_by": group_by, "items": [dict(r) for r in rows]}


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
    )
    return {"budget": budget}
