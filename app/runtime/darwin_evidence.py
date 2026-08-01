"""Persist one honest top-level Trace and Evidence Ledger for Darwin runs."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.runtime.contracts import RunEvidenceLedger
from db.property_db import (
    _get_conn,
    create_chat_trace,
    get_chat_trace,
    get_evidence_ledger,
    save_evidence_ledger,
    update_chat_trace,
)


OPERATION_TYPE = "badcase_darwin"


def _json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _operation_session_id(trace_id: str, badcase_id: int) -> str:
    # The schema requires a correlation id.  This namespace is explicitly an
    # operations scope, not an owner-chat session.
    return f"badcase-darwin:{int(badcase_id)}:{trace_id}"


def _draft_links(
    drafts: Optional[Iterable[Dict[str, Any]]],
    badcase_id: int,
) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    for wrapper in drafts or []:
        if not isinstance(wrapper, dict) or wrapper.get("error"):
            continue
        draft = wrapper.get("draft")
        if not isinstance(draft, dict) or draft.get("id") is None:
            continue
        if int(draft.get("badcase_id") or -1) != int(badcase_id):
            raise ValueError("Darwin draft does not belong to the evidence badcase")
        links.append(
            {
                "draft_type": wrapper.get("type"),
                "draft_id": int(draft["id"]),
                "badcase_id": int(badcase_id),
                "status": draft.get("status"),
                "created_at": draft.get("created_at"),
                "updated_at": draft.get("updated_at"),
                "published_at": draft.get("published_at"),
            }
        )
    return links


def _side_effect_counts(trace_id: str, session_id: str) -> Dict[str, int]:
    """Count only persisted, trace-linked side effects; never infer them."""
    conn = _get_conn()
    cursor = conn.cursor()
    queries: Tuple[Tuple[str, Tuple[Any, ...], str], ...] = (
        (
            "SELECT COUNT(*) AS count FROM action_proposals WHERE trace_id = ?",
            (trace_id,),
            "action_proposals",
        ),
        (
            """
            SELECT COUNT(*) AS count
            FROM action_receipts r
            JOIN action_proposals p ON p.proposal_id = r.proposal_id
            WHERE p.trace_id = ?
            """,
            (trace_id,),
            "action_receipts",
        ),
        (
            "SELECT COUNT(*) AS count FROM mcp_call_audits WHERE trace_id = ?",
            (trace_id,),
            "mcp_calls",
        ),
        (
            """
            SELECT COUNT(*) AS count FROM trace_events
            WHERE trace_id = ? AND (
                lower(span_name) LIKE '%action_gateway%'
                OR lower(span_name) = 'action.execute'
            )
            """,
            (trace_id,),
            "action_gateway_events",
        ),
        (
            "SELECT COUNT(*) AS count FROM handoff_actions WHERE session_id = ?",
            (session_id,),
            "handoff_actions",
        ),
        (
            "SELECT COUNT(*) AS count FROM work_orders WHERE session_id = ?",
            (session_id,),
            "work_orders",
        ),
    )
    counts: Dict[str, int] = {}
    try:
        for query, params, key in queries:
            cursor.execute(query, params)
            row = cursor.fetchone()
            counts[key] = int(row["count"] if row else 0)
    finally:
        conn.close()
    return counts


def _model_and_cost_evidence(
    model_call: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    usage = _json_object(model_call.get("usage_normalized"))
    requested_model = usage.get("requested_model") or model_call.get("model_id")
    provider_response_model = usage.get("provider_response_model")
    provider_usage = {
        "input_cache_hit_tokens": usage.get("input_cache_hit_tokens"),
        "input_cache_miss_tokens": usage.get("input_cache_miss_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    model_evidence = {
        "model_call_id": model_call.get("id"),
        "record_kind": model_call.get("record_kind"),
        "local_attempt_id": model_call.get("local_attempt_id"),
        "trace_id": model_call.get("trace_id"),
        "stage": model_call.get("stage"),
        "status": model_call.get("status"),
        "requested_model": requested_model,
        # Deliberately no fallback to requested_model.
        "provider_response_model": provider_response_model,
        "provider_request_id": usage.get("provider_request_id"),
        "thinking_enabled": usage.get("thinking_enabled"),
        "usage_source": model_call.get("usage_source"),
        "usage_status": model_call.get("usage_status") or usage.get("usage_status"),
        "provider_usage": provider_usage,
        "latency_ms": model_call.get("latency_ms"),
        "total_tokens": model_call.get("total_tokens"),
        "price_snapshot": _json_object(model_call.get("price_snapshot")),
        "amount": model_call.get("estimated_cost_cny"),
        "cost_source": model_call.get("cost_source") or usage.get("cost_source"),
        "cost_disclaimer": "platform_price_snapshot_not_provider_final_bill",
        "created_at": model_call.get("created_at"),
    }

    stored_contract = usage.get("cost_contract")
    if isinstance(stored_contract, dict):
        cost_evidence = dict(stored_contract)
        cost_evidence["model_call_id"] = model_call.get("id")
        cost_evidence["trace_id"] = model_call.get("trace_id")
        cost_evidence["cost_source"] = model_call.get("cost_source") or usage.get(
            "cost_source"
        )
        cost_evidence["cost_disclaimer"] = (
            "platform_price_snapshot_not_provider_final_bill"
        )
    else:
        # Historical rows without a stored CostEntry keep their real fields;
        # missing fields remain unavailable instead of being reconstructed.
        cost_evidence = {
            "model_call_id": model_call.get("id"),
            "trace_id": model_call.get("trace_id"),
            "stage": model_call.get("stage"),
            "requested_model": requested_model,
            "provider_response_model": provider_response_model,
            "thinking_enabled": usage.get("thinking_enabled"),
            "usage_source": model_call.get("usage_source") or "unavailable",
            **provider_usage,
            "price_snapshot": _json_object(model_call.get("price_snapshot")),
            "formula": None,
            "amount": model_call.get("estimated_cost_cny"),
            "cost_source": model_call.get("cost_source") or usage.get("cost_source"),
            "cost_disclaimer": "platform_price_snapshot_not_provider_final_bill",
            "availability_note": (
                "Stored CostEntry is unavailable; missing fields were not reconstructed."
            ),
        }
    return model_evidence, cost_evidence


def _operation_metadata(
    *,
    trace_id: str,
    badcase_id: int,
    operation_status: str,
    started_at: str,
    completed_at: Optional[str],
    draft_links: List[Dict[str, Any]],
    status_before: Optional[str],
    status_after: Optional[str],
    side_effect_counts: Dict[str, int],
    error_summary: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "v1.8.2-s5-fix1",
        "operation_type": OPERATION_TYPE,
        "trace_id": trace_id,
        "badcase_id": int(badcase_id),
        "session_kind": "badcase_operation",
        "status": operation_status,
        "started_at": started_at,
        "completed_at": completed_at,
        "model_stage": "darwin",
        "draft_results": draft_links,
        "status_transition": {
            "from": status_before,
            "to": status_after,
        },
        "side_effect_counts": side_effect_counts,
        "error_summary": error_summary,
    }


def start_darwin_operation(
    *,
    trace_id: str,
    badcase_id: int,
    started_at: str,
) -> Dict[str, Any]:
    """Create the formal operation Trace before any Provider call."""
    session_id = _operation_session_id(trace_id, badcase_id)
    existing = get_chat_trace(trace_id)
    if existing and existing.get("run_type") not in {None, OPERATION_TYPE}:
        raise ValueError("trace_id already belongs to another operation type")
    if not existing:
        create_chat_trace(
            trace_id=trace_id,
            session_id=session_id,
            user_message="",
            run_type=OPERATION_TYPE,
            version_snapshot=json.dumps(
                {
                    "operation_type": OPERATION_TYPE,
                    "badcase_id": int(badcase_id),
                    "session_kind": "badcase_operation",
                    "started_at": started_at,
                },
                ensure_ascii=False,
            ),
        )
        update_chat_trace(
            trace_id,
            intent=OPERATION_TYPE,
            run_type=OPERATION_TYPE,
            status="in_progress",
            created_at=started_at,
            updated_at=started_at,
        )

    if not get_evidence_ledger(trace_id):
        metadata = _operation_metadata(
            trace_id=trace_id,
            badcase_id=badcase_id,
            operation_status="in_progress",
            started_at=started_at,
            completed_at=None,
            draft_links=[],
            status_before=None,
            status_after=None,
            side_effect_counts=_side_effect_counts(trace_id, session_id),
        )
        contract = RunEvidenceLedger(
            trace_id=trace_id,
            session_id=session_id,
            config_snapshot=metadata,
            badcase_links=[
                {
                    "badcase_id": int(badcase_id),
                    "operation_type": OPERATION_TYPE,
                    "status": "in_progress",
                    "drafts": [],
                }
            ],
        )
        save_evidence_ledger(
            trace_id=trace_id,
            session_id=session_id,
            ledger=contract.model_dump(mode="json"),
            runtime_path=OPERATION_TYPE,
            status="running",
        )
    return get_chat_trace(trace_id) or {}


def persist_darwin_operation(
    *,
    trace_id: str,
    badcase_id: int,
    model_call: Optional[Dict[str, Any]],
    operation_status: str,
    started_at: str,
    completed_at: str,
    drafts: Optional[Iterable[Dict[str, Any]]] = None,
    status_before: Optional[str] = None,
    status_after: Optional[str] = None,
    error_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Idempotently persist final Darwin evidence for one trace."""
    if operation_status not in {"complete", "failed"}:
        raise ValueError("Darwin operation status must be complete or failed")
    if model_call is None:
        if operation_status == "complete":
            raise ValueError("a complete Darwin operation requires a Provider attempt")
        child_status = None
    else:
        if model_call.get("trace_id") != trace_id or model_call.get("stage") != "darwin":
            raise ValueError("model_call does not belong to the Darwin trace")
        child_status = model_call.get("status")
        if operation_status == "complete" and child_status != "success":
            raise ValueError("a complete Darwin operation requires a successful model_call")
        if operation_status == "failed" and child_status == "success" and not error_summary:
            raise ValueError("failed Darwin operation requires an explicit failure reason")

    session_id = _operation_session_id(trace_id, badcase_id)
    if not get_chat_trace(trace_id):
        start_darwin_operation(
            trace_id=trace_id,
            badcase_id=badcase_id,
            started_at=started_at,
        )

    links = _draft_links(drafts, badcase_id)
    counts = _side_effect_counts(trace_id, session_id)
    if model_call is not None:
        model_evidence, cost_evidence = _model_and_cost_evidence(model_call)
        model_evidence_rows = [model_evidence]
        cost_evidence_rows = [cost_evidence]
        model_call_ids = [model_call.get("id")]
    else:
        # Budget/policy blocking is a logical operation outcome, not a fake
        # Provider request with zero tokens.
        model_evidence_rows = []
        cost_evidence_rows = []
        model_call_ids = []
    metadata = _operation_metadata(
        trace_id=trace_id,
        badcase_id=badcase_id,
        operation_status=operation_status,
        started_at=started_at,
        completed_at=completed_at,
        draft_links=links,
        status_before=status_before,
        status_after=status_after,
        side_effect_counts=counts,
        error_summary=error_summary,
    )
    contract = RunEvidenceLedger(
        trace_id=trace_id,
        session_id=session_id,
        config_snapshot=metadata,
        model_calls=model_evidence_rows,
        cost_entries=cost_evidence_rows,
        badcase_links=[
            {
                "badcase_id": int(badcase_id),
                "operation_type": OPERATION_TYPE,
                "status": operation_status,
                "model_call_ids": model_call_ids,
                "drafts": links,
                "status_transition": {
                    "from": status_before,
                    "to": status_after,
                },
                "side_effect_counts": counts,
            }
        ],
    )
    ledger = save_evidence_ledger(
        trace_id=trace_id,
        session_id=session_id,
        ledger=contract.model_dump(mode="json"),
        runtime_path=OPERATION_TYPE,
        status=operation_status,
    )
    update_chat_trace(
        trace_id,
        intent=OPERATION_TYPE,
        status=operation_status,
        run_type=OPERATION_TYPE,
        version_snapshot=json.dumps(metadata, ensure_ascii=False),
        created_at=started_at,
        updated_at=completed_at,
    )
    return ledger
