"""Request-scoped DeepSeek evidence accounting.

This module is deliberately small: one ``ContextVar`` carries business
identity into the model gateway, and every outbound model invocation owns one
durable ``model_calls`` row.  Business stages and retries may aggregate or
reference these rows, but they never create Provider attempts themselves.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4


USAGE_FIELDS = (
    "input_cache_hit_tokens",
    "input_cache_miss_tokens",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


class ProviderAccountingError(RuntimeError):
    """Raised when a paid request cannot be accounted for truthfully."""


class ProviderAccountingPersistenceError(ProviderAccountingError):
    """Raised after a response when its final evidence could not be persisted."""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _non_negative_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def merge_non_null(target: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge Provider evidence without erasing known values.

    DeepSeek emits ``usage=null`` on ordinary stream chunks.  A later partial
    object must likewise never replace an earlier complete usage object.
    """

    for key, value in (incoming or {}).items():
        if value is None:
            continue
        if isinstance(value, dict):
            current = target.get(key)
            if not isinstance(current, dict):
                current = {}
                target[key] = current
            merge_non_null(current, value)
        else:
            target[key] = value
    return target


def _safe_status_code(exc: BaseException) -> Optional[int]:
    for candidate in (exc, getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
        if candidate is None:
            continue
        value = getattr(candidate, "status_code", None)
        if value is None:
            response = getattr(candidate, "response", None)
            value = getattr(response, "status_code", None)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            continue
    return None


def safe_error_evidence(exc: BaseException, *, phase: str) -> Dict[str, Any]:
    """Return only non-sensitive, structural error facts.

    Exception messages, request bodies, headers and prompts are intentionally
    excluded because SDK errors can embed credentials or business content.
    """

    cause = getattr(exc, "__cause__", None)
    return {
        "phase": phase,
        "exception_type": type(exc).__name__,
        "cause_type": type(cause).__name__ if cause is not None else None,
        "http_status": _safe_status_code(exc),
    }


@dataclass
class ProviderAccountingScope:
    trace_id: str
    stage: str
    session_id: Optional[str] = None
    model_selection_reason: Optional[str] = None
    price_snapshot: Optional[Dict[str, Any]] = None
    model_policy_version: Optional[str] = None
    explicit_retry: bool = False
    retry_of_local_attempt_id: Optional[str] = None
    client_cancel_confirmed: bool = False
    client_cancel_evidence_code: Optional[str] = None
    generated_identity: bool = False
    attempts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProviderAttempt:
    local_attempt_id: str
    scope: ProviderAccountingScope
    requested_model: str
    thinking_enabled: Optional[bool]
    stream: bool
    started_at: str
    attempt_sequence: Optional[int] = None
    model_call_id: Optional[int] = None
    dispatched: bool = False
    response_seen: bool = False
    sdk_stream_exhausted: bool = False
    provider_request_id: Optional[str] = None
    provider_actual_model: Optional[str] = None
    usage: Dict[str, Optional[int]] = field(default_factory=dict)
    chunk_count: int = 0
    provider_id_conflict: Optional[Dict[str, str]] = None

    def merge_evidence(self, evidence: Dict[str, Any]) -> None:
        request_id = evidence.get("provider_request_id")
        if request_id:
            request_id = str(request_id)
            if self.provider_request_id and self.provider_request_id != request_id:
                self.provider_id_conflict = {
                    "first": self.provider_request_id,
                    "later": request_id,
                }
            elif not self.provider_request_id:
                self.provider_request_id = request_id
        actual_model = evidence.get("provider_response_model")
        if actual_model:
            self.provider_actual_model = str(actual_model)
        usage = evidence.get("usage")
        if isinstance(usage, dict):
            for key in USAGE_FIELDS:
                value = _non_negative_int(usage.get(key))
                if value is not None:
                    self.usage[key] = value
        self.response_seen = self.response_seen or bool(
            request_id or actual_model or any(value is not None for value in self.usage.values())
        )
        self.chunk_count += 1


_ACTIVE_SCOPE: ContextVar[Optional[ProviderAccountingScope]] = ContextVar(
    "provider_accounting_scope",
    default=None,
)
_ACTIVE_ATTEMPT: ContextVar[Optional[ProviderAttempt]] = ContextVar(
    "provider_accounting_attempt",
    default=None,
)


@contextmanager
def provider_accounting_scope(
    *,
    trace_id: str,
    stage: str,
    session_id: Optional[str] = None,
    model_selection_reason: Optional[str] = None,
    price_snapshot: Optional[Dict[str, Any]] = None,
    model_policy_version: Optional[str] = None,
    explicit_retry: bool = False,
    retry_of_local_attempt_id: Optional[str] = None,
    client_cancel_confirmed: bool = False,
    client_cancel_evidence_code: Optional[str] = None,
) -> Iterator[ProviderAccountingScope]:
    """Bind business metadata to all Provider attempts in this execution path."""

    scope = ProviderAccountingScope(
        trace_id=str(trace_id),
        session_id=str(session_id) if session_id else None,
        stage=str(stage),
        model_selection_reason=model_selection_reason,
        price_snapshot=dict(price_snapshot) if price_snapshot else None,
        model_policy_version=model_policy_version,
        explicit_retry=bool(explicit_retry),
        retry_of_local_attempt_id=retry_of_local_attempt_id,
        client_cancel_confirmed=bool(client_cancel_confirmed),
        client_cancel_evidence_code=(
            str(client_cancel_evidence_code)
            if client_cancel_confirmed and client_cancel_evidence_code
            else None
        ),
    )
    token = _ACTIVE_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _ACTIVE_SCOPE.reset(token)


def _run_identity(run_response: Any) -> Dict[str, Optional[str]]:
    return {
        "trace_id": str(getattr(run_response, "run_id", "") or "") or None,
        "session_id": str(getattr(run_response, "session_id", "") or "") or None,
    }


def _scope_for_attempt(local_attempt_id: str, run_response: Any = None) -> ProviderAccountingScope:
    scope = _ACTIVE_SCOPE.get()
    if scope is not None:
        return scope
    identity = _run_identity(run_response)
    return ProviderAccountingScope(
        trace_id=identity["trace_id"] or f"unscoped:{local_attempt_id}",
        session_id=identity["session_id"],
        stage="agentos_direct_agent",
        model_selection_reason="central model gateway fallback identity",
        generated_identity=True,
    )


def _resolved_price_snapshot(scope: ProviderAccountingScope, model_id: str) -> Optional[Dict[str, Any]]:
    if scope.price_snapshot:
        return dict(scope.price_snapshot)
    try:
        from db.property_db import get_enabled_price_for_model

        row = get_enabled_price_for_model(model_id)
        return dict(row) if row else None
    except Exception:
        return None


def begin_provider_attempt(
    *,
    requested_model: str,
    thinking_enabled: Optional[bool],
    stream: bool,
    run_response: Any = None,
) -> tuple[ProviderAttempt, Any]:
    """Persist a pending row before the paid SDK method may be entered."""

    local_attempt_id = f"pa_{uuid4().hex}"
    scope = _scope_for_attempt(local_attempt_id, run_response)
    attempt = ProviderAttempt(
        local_attempt_id=local_attempt_id,
        scope=scope,
        requested_model=str(requested_model),
        thinking_enabled=thinking_enabled,
        stream=bool(stream),
        started_at=_now_iso(),
    )
    price_snapshot = _resolved_price_snapshot(scope, attempt.requested_model)
    attempt.scope.price_snapshot = price_snapshot
    try:
        from db.property_db import create_provider_attempt

        row = create_provider_attempt(
            local_attempt_id=local_attempt_id,
            trace_id=scope.trace_id,
            session_id=scope.session_id,
            stage=scope.stage,
            model_id=attempt.requested_model,
            model_selection_reason=scope.model_selection_reason,
            thinking_enabled=thinking_enabled,
            stream=bool(stream),
            price_snapshot=price_snapshot,
            model_policy_version=scope.model_policy_version,
            explicit_retry=scope.explicit_retry,
            retry_of_local_attempt_id=scope.retry_of_local_attempt_id,
            started_at=attempt.started_at,
        )
        attempt.attempt_sequence = _non_negative_int(
            (row.get("usage_normalized") or {}).get("attempt_sequence")
        )
        attempt.model_call_id = _non_negative_int(row.get("id"))
    except Exception as exc:
        raise ProviderAccountingError(
            "Provider attempt initialization failed; paid request was blocked"
        ) from exc
    token = _ACTIVE_ATTEMPT.set(attempt)
    return attempt, token


def mark_provider_attempt_dispatched(attempt: ProviderAttempt) -> None:
    """Durably mark that control is about to enter the SDK.

    This is deliberately *not* proof that an HTTP request left the process.
    Provider dispatch is confirmed only from response/request-id evidence or a
    real Provider HTTP status during finalization.
    """

    try:
        from db.property_db import mark_provider_attempt_dispatched as persist_dispatch

        persist_dispatch(attempt.local_attempt_id)
        attempt.dispatched = True
    except Exception as exc:
        raise ProviderAccountingError(
            "Provider attempt dispatch persistence failed; paid request was blocked"
        ) from exc


def capture_active_provider_evidence(evidence: Dict[str, Any]) -> None:
    attempt = _ACTIVE_ATTEMPT.get()
    if attempt is not None:
        attempt.merge_evidence(evidence)


def _price_number(snapshot: Optional[Dict[str, Any]], key: str) -> Optional[Decimal]:
    if not snapshot or snapshot.get(key) is None:
        return None
    try:
        return Decimal(str(snapshot[key]))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _cost(attempt: ProviderAttempt) -> tuple[Optional[float], Optional[str]]:
    hit = attempt.usage.get("input_cache_hit_tokens")
    miss = attempt.usage.get("input_cache_miss_tokens")
    output = attempt.usage.get("output_tokens")
    hit_price = _price_number(attempt.scope.price_snapshot, "cached_input_price_per_1m")
    miss_price = _price_number(attempt.scope.price_snapshot, "input_price_per_1m")
    output_price = _price_number(attempt.scope.price_snapshot, "output_price_per_1m")
    if any(value is None for value in (hit, miss, output, hit_price, miss_price, output_price)):
        return None, None
    if any(int(value) < 0 for value in (hit, miss, output)):
        return None, None
    amount = (
        Decimal(int(hit)) * hit_price
        + Decimal(int(miss)) * miss_price
        + Decimal(int(output)) * output_price
    ) / Decimal(1_000_000)
    return float(amount.quantize(Decimal("0.00000001"))), "platform_price_snapshot"


def _usage_facts(attempt: ProviderAttempt) -> Dict[str, Any]:
    hit = attempt.usage.get("input_cache_hit_tokens")
    miss = attempt.usage.get("input_cache_miss_tokens")
    input_tokens = attempt.usage.get("input_tokens")
    output = attempt.usage.get("output_tokens")
    reasoning = attempt.usage.get("reasoning_tokens")
    total = attempt.usage.get("total_tokens")
    core_complete = all(value is not None for value in (hit, miss, output, total))
    split_total = (int(hit) + int(miss) + int(output)) if core_complete else None
    split_input = (int(hit) + int(miss)) if hit is not None and miss is not None else None
    inconsistent_reasons: List[str] = []
    for key, value in attempt.usage.items():
        if value is not None and int(value) < 0:
            inconsistent_reasons.append(f"negative_provider_value:{key}")
    if split_total is not None and int(total) != split_total:
        inconsistent_reasons.append("total_tokens_not_equal_cache_hit_plus_cache_miss_plus_output")
    if input_tokens is not None and split_input is not None and int(input_tokens) != split_input:
        inconsistent_reasons.append("input_tokens_not_equal_cache_hit_plus_cache_miss")
    if reasoning is not None and output is not None and int(reasoning) > int(output):
        inconsistent_reasons.append("reasoning_tokens_exceed_output_tokens")
    return {
        "core_complete": core_complete,
        "split_total": split_total,
        "split_input": split_input,
        "provider_usage_inconsistent": bool(inconsistent_reasons),
        "provider_usage_inconsistency_reasons": inconsistent_reasons,
    }


def _confirmed_provider_request_sent(
    attempt: ProviderAttempt,
    exception: Optional[BaseException],
) -> Optional[bool]:
    """Return True only when this layer has affirmative outbound evidence."""

    if attempt.provider_request_id or attempt.response_seen:
        return True
    status_code = _safe_status_code(exception) if exception is not None else None
    if status_code is not None and 400 <= status_code <= 599:
        return True
    return None


def _classify_failure(attempt: ProviderAttempt, exc: BaseException, *, phase: str) -> tuple[str, str]:
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        if attempt.scope.client_cancel_confirmed:
            return (
                "unavailable_client_cancelled",
                attempt.scope.client_cancel_evidence_code
                or "client_disconnect_confirmed",
            )
        return "cause_unconfirmed", "cancellation_observed_origin_unconfirmed"
    status_code = _safe_status_code(exc)
    if status_code is not None and 400 <= status_code <= 599:
        return "unavailable_provider_error", "provider_http_error"
    names = " ".join(
        filter(
            None,
            (
                type(exc).__name__,
                type(getattr(exc, "__cause__", None)).__name__
                if getattr(exc, "__cause__", None) is not None
                else None,
            ),
        )
    ).lower()
    if attempt.stream and attempt.response_seen and any(
        marker in names for marker in ("timeout", "connection", "stream", "protocol")
    ):
        return "unavailable_stream_interrupted", "captured_stream_interruption"
    return "cause_unconfirmed", "provider_attempt_failed_cause_unconfirmed"


def _capture_usage_protocol_badcase(
    attempt: ProviderAttempt,
    unavailable_reason: Optional[str],
) -> Dict[str, Any]:
    """Create one human-review Badcase for a completed response without Usage."""

    evidence = {
        "local_attempt_id": attempt.local_attempt_id,
        "provider_request_id": attempt.provider_request_id,
        "trace_id": attempt.scope.trace_id,
        "stage": attempt.scope.stage,
        "stream": attempt.stream,
        "sdk_stream_exhausted": attempt.sdk_stream_exhausted,
        "provider_response_seen": attempt.response_seen,
        "reason_code": unavailable_reason,
        "root_cause": "cause_unconfirmed",
    }
    try:
        from db.property_db import create_badcase

        case = create_badcase(
            title="Provider响应已完成但Usage证据缺失",
            description=(
                "SDK响应已结束，但平台没有取得完整Provider Usage；原因未确认，"
                "需要人工检查协议、SDK解析与原始响应证据。"
            ),
            category="other",
            status="pending",
            evidence=json.dumps(evidence, ensure_ascii=False),
            session_id=attempt.scope.session_id,
            source="runtime_contract",
            feedback_reason=unavailable_reason,
            context_json=json.dumps(evidence, ensure_ascii=False),
            trace_id=attempt.scope.trace_id,
            priority="high",
            symptom="Provider响应结束但未取得完整Usage",
            expected_behavior="每次完整消费的Provider响应都持久化完整Usage",
            actual_behavior="Usage缺失或不完整；Token与成本保持不可得而非0",
            root_cause_domain="unknown",
        )
        return {"status": "created", "badcase_id": case.get("id") if case else None}
    except Exception as exc:
        return {
            "status": "failed",
            "error_evidence": safe_error_evidence(exc, phase="usage_anomaly_badcase_capture"),
        }


def finalize_provider_attempt(
    attempt: ProviderAttempt,
    *,
    normal_completion: bool,
    exception: Optional[BaseException] = None,
    phase: str = "provider_response",
) -> Dict[str, Any]:
    finished_at = _now_iso()
    usage_facts = _usage_facts(attempt)
    usage_received = any(value is not None for value in attempt.usage.values())
    provider_request_sent = _confirmed_provider_request_sent(attempt, exception)
    http_status = _safe_status_code(exception) if exception is not None else None
    provider_request_sent_evidence = (
        "provider_request_id"
        if attempt.provider_request_id
        else "provider_response_seen"
        if attempt.response_seen
        else "provider_http_status"
        if http_status is not None and 400 <= http_status <= 599
        else None
    )
    unavailable_reason: Optional[str] = None
    error_evidence: Optional[Dict[str, Any]] = None
    if exception is not None:
        usage_status, unavailable_reason = _classify_failure(attempt, exception, phase=phase)
        error_evidence = safe_error_evidence(exception, phase=phase)
        status = "failed"
    elif normal_completion and usage_facts["core_complete"]:
        usage_status = "provider_actual"
        status = "success"
    else:
        usage_status = "unavailable_done_without_usage"
        unavailable_reason = (
            "provider_usage_incomplete"
            if usage_received
            else "sdk_completed_without_provider_usage"
        )
        status = "failed"
    if attempt.provider_id_conflict:
        status = "failed"
        usage_status = "cause_unconfirmed"
        unavailable_reason = "multiple_provider_request_ids_within_one_sdk_attempt"

    usage_anomaly_badcase = None
    if usage_status == "unavailable_done_without_usage":
        usage_anomaly_badcase = _capture_usage_protocol_badcase(
            attempt,
            unavailable_reason,
        )

    amount, cost_source = _cost(attempt)
    if usage_status != "provider_actual":
        amount = None
        cost_source = None
    latency_ms: Optional[int] = None
    try:
        start = datetime.fromisoformat(attempt.started_at)
        finish = datetime.fromisoformat(finished_at)
        latency_ms = max(0, int((finish - start).total_seconds() * 1000))
    except Exception:
        pass

    normalized: Dict[str, Any] = {
        "record_kind": "provider_attempt",
        "include_in_provider_aggregate": provider_request_sent is True,
        "local_attempt_id": attempt.local_attempt_id,
        "trace_id": attempt.scope.trace_id,
        "session_id": attempt.scope.session_id,
        "stage": attempt.scope.stage,
        "attempt_sequence": attempt.attempt_sequence,
        "provider_request_sequence": attempt.attempt_sequence,
        "provider_request_key": (
            f"request_id:{attempt.provider_request_id}"
            if attempt.provider_request_id
            else f"local_attempt:{attempt.local_attempt_id}"
        ),
        "provider_request_identity_source": (
            "provider_request_id" if attempt.provider_request_id else "local_attempt_id"
        ),
        "requested_model": attempt.requested_model,
        "provider_response_model": attempt.provider_actual_model,
        "provider_request_id": attempt.provider_request_id,
        "provider_request_id_obtained": bool(attempt.provider_request_id),
        "provider_request_identity_status": (
            "provider_request_id" if attempt.provider_request_id else "unavailable_provider_request_id"
        ),
        "thinking_enabled": attempt.thinking_enabled,
        "stream": attempt.stream,
        "started_at": attempt.started_at,
        "finished_at": finished_at,
        "sdk_dispatch_started": bool(attempt.dispatched),
        "provider_request_sent": provider_request_sent,
        "provider_request_sent_evidence": provider_request_sent_evidence,
        "http_status": http_status,
        "provider_response_seen": bool(attempt.response_seen),
        # Agno/OpenAI SDK consumes the wire-level [DONE] sentinel internally.
        # Do not claim that this layer observed it; iterator exhaustion is the
        # separate, truthful completion evidence available here.
        "received_done": None,
        "done_observation_status": "not_exposed_by_sdk" if attempt.stream else "not_applicable",
        "sdk_stream_exhausted": bool(attempt.sdk_stream_exhausted),
        "completion_evidence": (
            "sdk_stream_iterator_exhausted"
            if attempt.stream and attempt.sdk_stream_exhausted
            else "non_stream_response_returned"
            if normal_completion and not attempt.stream
            else None
        ),
        "received_usage": usage_received,
        "persistence_status": "persisted",
        "explicit_retry": attempt.scope.explicit_retry,
        "retry_detected": bool(
            attempt.scope.explicit_retry or attempt.scope.retry_of_local_attempt_id
        ),
        "retry_of_local_attempt_id": attempt.scope.retry_of_local_attempt_id,
        "client_cancel_confirmed": attempt.scope.client_cancel_confirmed,
        "client_cancel_evidence_code": attempt.scope.client_cancel_evidence_code,
        "usage_status": usage_status,
        "usage_unavailable_reason": unavailable_reason,
        "input_cache_hit_tokens": attempt.usage.get("input_cache_hit_tokens"),
        "input_cache_miss_tokens": attempt.usage.get("input_cache_miss_tokens"),
        "input_tokens": attempt.usage.get("input_tokens"),
        "output_tokens": attempt.usage.get("output_tokens"),
        "reasoning_tokens": attempt.usage.get("reasoning_tokens"),
        "total_tokens": attempt.usage.get("total_tokens"),
        "token_source": "provider_actual" if usage_status == "provider_actual" else "unavailable",
        "provider_usage_inconsistent": usage_facts["provider_usage_inconsistent"],
        "provider_usage_inconsistency_reasons": usage_facts[
            "provider_usage_inconsistency_reasons"
        ],
        "calculated_split_total_tokens": usage_facts["split_total"],
        "calculated_split_input_tokens": usage_facts["split_input"],
        "price_snapshot": attempt.scope.price_snapshot,
        "calculated_direct_cost": amount,
        "cost_source": cost_source,
        "cost_disclaimer": "platform_price_snapshot_not_provider_final_bill",
        "error_evidence": error_evidence,
        "provider_id_conflict": attempt.provider_id_conflict,
        "generated_business_identity": attempt.scope.generated_identity,
        "usage_anomaly_badcase": usage_anomaly_badcase,
        "reconciliation_status": (
            "matched_provider_response"
            if usage_status == "provider_actual"
            and attempt.provider_request_id
            and not usage_facts["provider_usage_inconsistent"]
            else "provider_usage_inconsistent"
            if usage_facts["provider_usage_inconsistent"]
            else "provider_request_id_unavailable"
            if usage_status == "provider_actual" and not attempt.provider_request_id
            else usage_status
        ),
        "status": status,
        "usage": dict(attempt.usage),
    }

    try:
        from db.property_db import finalize_provider_attempt as persist_final

        row = persist_final(
            local_attempt_id=attempt.local_attempt_id,
            provider_request_id=attempt.provider_request_id,
            provider_actual_model=attempt.provider_actual_model,
            status=status,
            latency_ms=latency_ms,
            usage_status=usage_status,
            usage_source=("provider_actual" if usage_status == "provider_actual" else "unavailable"),
            input_tokens=attempt.usage.get("input_tokens"),
            output_tokens=attempt.usage.get("output_tokens"),
            reasoning_tokens=attempt.usage.get("reasoning_tokens"),
            cached_tokens=attempt.usage.get("input_cache_hit_tokens"),
            total_tokens=attempt.usage.get("total_tokens"),
            price_snapshot=attempt.scope.price_snapshot,
            calculated_direct_cost=amount,
            cost_source=cost_source,
            error_summary=unavailable_reason,
            usage_normalized=normalized,
            finished_at=finished_at,
        )
        normalized["model_call_id"] = row.get("id") if isinstance(row, dict) else None
    except Exception as exc:
        normalized["usage_status"] = "unavailable_persistence_failure"
        normalized["usage_unavailable_reason"] = "provider_evidence_finalize_persistence_failed"
        normalized["persistence_status"] = "failed"
        normalized["token_source"] = "unavailable"
        normalized["calculated_direct_cost"] = None
        normalized["cost_source"] = None
        normalized["error_evidence"] = safe_error_evidence(exc, phase="persistence_finalize")
        try:
            from db.property_db import mark_provider_attempt_persistence_failure

            mark_provider_attempt_persistence_failure(
                attempt.local_attempt_id,
                usage_normalized=normalized,
                finished_at=finished_at,
            )
        except Exception:
            pass
        attempt.scope.attempts.append(normalized)
        raise ProviderAccountingPersistenceError(
            "Provider response was obtained but final evidence persistence failed"
        ) from exc

    attempt.scope.attempts.append(normalized)
    return normalized


def reset_active_provider_attempt(token: Any) -> None:
    _ACTIVE_ATTEMPT.reset(token)


def current_provider_attempt() -> Optional[ProviderAttempt]:
    return _ACTIVE_ATTEMPT.get()
