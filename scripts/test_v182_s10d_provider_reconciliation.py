"""Deterministic S10-D DeepSeek per-request reconciliation contract.

This script uses an isolated temporary SQLite database and simulated Provider
chunks only.  It never imports the production model instance, opens HTTP, or
calls a paid model.
"""

from __future__ import annotations

import ast
import asyncio
import atexit
import json
import os
import socket
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# PROPERTY_DATA_DIR must be fixed before importing db.property_db.
_TEMP_DATA = tempfile.TemporaryDirectory(prefix="yiai-s10d-provider-")
atexit.register(_TEMP_DATA.cleanup)
os.environ["PROPERTY_DATA_DIR"] = _TEMP_DATA.name
os.environ["DEEPSEEK_API_KEY"] = ""

from app.handoff_policy import evaluate_handoff_policy
from app.runtime.provider_accounting import (
    ProviderAccountingError,
    ProviderAccountingPersistenceError,
    begin_provider_attempt,
    capture_active_provider_evidence,
    finalize_provider_attempt,
    mark_provider_attempt_dispatched,
    provider_accounting_scope,
    reset_active_provider_attempt,
    safe_error_evidence,
)
from app.runtime.provider_evidence import (
    capture_provider_response,
    provider_evidence_from_run,
)
from db import property_db


PRICE_SNAPSHOT = {
    "price_snapshot_id": "fixture-s10d",
    "model_id": "deepseek-v4-flash",
    "input_price_per_1m": 1.0,
    "cached_input_price_per_1m": 0.02,
    "output_price_per_1m": 2.0,
}
MODEL_ID = "deepseek-v4-flash"
CHECKS: list[str] = []


def check(name: str, condition: Any) -> None:
    if not condition:
        raise AssertionError(name)
    CHECKS.append(name)


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _function_node(source: str, name: str) -> ast.AST:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def _call_name(call: ast.Call) -> str:
    target = call.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call) and _call_name(item) == name
    ]


def _keyword_constant(call: ast.Call, name: str) -> Any:
    for item in call.keywords:
        if item.arg == name and isinstance(item.value, ast.Constant):
            return item.value.value
    return None


def _usage(
    *,
    hit: Optional[int],
    miss: Optional[int],
    output: Optional[int],
    reasoning: Optional[int],
    total: Optional[int],
    input_tokens: Optional[int] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_cache_hit_tokens=hit,
        prompt_cache_miss_tokens=miss,
        prompt_tokens=input_tokens,
        completion_tokens=output,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        total_tokens=total,
    )


def _capture_raw(
    *,
    request_id: Optional[str],
    model: Optional[str] = MODEL_ID,
    usage: Any = None,
) -> Dict[str, Any]:
    parsed = SimpleNamespace(provider_data={})
    raw = SimpleNamespace(id=request_id, model=model, usage=usage)
    capture_provider_response(parsed, raw)
    evidence = provider_evidence_from_run(parsed)
    capture_active_provider_evidence(evidence)
    return evidence


def _complete_attempt(
    *,
    trace_id: str,
    stage: str,
    request_id: str,
    hit: int = 10,
    miss: int = 20,
    output: int = 30,
    reasoning: int = 12,
    explicit_retry: bool = False,
    retry_of_local_attempt_id: Optional[str] = None,
    stream: bool = True,
) -> tuple[Any, Dict[str, Any], Any]:
    with provider_accounting_scope(
        trace_id=trace_id,
        session_id=f"session:{trace_id}",
        stage=stage,
        model_selection_reason="S10-D deterministic fixture",
        price_snapshot=PRICE_SNAPSHOT,
        model_policy_version="fixture",
        explicit_retry=explicit_retry,
        retry_of_local_attempt_id=retry_of_local_attempt_id,
    ) as scope:
        attempt, token = begin_provider_attempt(
            requested_model=MODEL_ID,
            thinking_enabled=True,
            stream=stream,
        )
        try:
            mark_provider_attempt_dispatched(attempt)
            _capture_raw(request_id=request_id, usage=None)
            _capture_raw(
                request_id=request_id,
                usage=_usage(
                    hit=hit,
                    miss=miss,
                    output=output,
                    reasoning=reasoning,
                    total=hit + miss + output,
                    input_tokens=hit + miss,
                ),
            )
            attempt.sdk_stream_exhausted = bool(stream)
            normalized = finalize_provider_attempt(
                attempt,
                normal_completion=True,
            )
        finally:
            reset_active_provider_attempt(token)
    return attempt, normalized, scope


def test_content_and_usage_same_id_one_row() -> None:
    trace_id = "s10d-content-usage"
    _, normalized, _ = _complete_attempt(
        trace_id=trace_id,
        stage="vertical_agent",
        request_id="req-s10d-content-usage",
        hit=120,
        miss=30,
        output=20,
        reasoning=7,
    )
    rows = property_db.get_provider_attempts_for_trace(trace_id)
    check("content plus usage is one Provider row", len(rows) == 1)
    check("Provider request id is preserved", normalized["provider_request_id"] == "req-s10d-content-usage")
    check("three Provider token classes match raw response", (
        normalized["input_cache_hit_tokens"],
        normalized["input_cache_miss_tokens"],
        normalized["output_tokens"],
    ) == (120, 30, 20))
    check("Provider total matches exact three-class sum", normalized["total_tokens"] == 170)
    check("normal completed attempt is Provider actual", normalized["usage_status"] == "provider_actual")
    check("direct cost is explicitly platform price snapshot", normalized["cost_source"] == "platform_price_snapshot")


def test_parsed_without_id_raw_with_id() -> None:
    trace_id = "s10d-raw-id"
    request_id = "req-s10d-raw-only-id"
    with provider_accounting_scope(
        trace_id=trace_id,
        stage="router",
        price_snapshot=PRICE_SNAPSHOT,
    ):
        attempt, token = begin_provider_attempt(
            requested_model=MODEL_ID,
            thinking_enabled=True,
            stream=False,
        )
        try:
            mark_provider_attempt_dispatched(attempt)
            parsed = SimpleNamespace(provider_data={})
            raw = SimpleNamespace(
                id=request_id,
                model=MODEL_ID,
                usage=_usage(
                    hit=1,
                    miss=2,
                    output=3,
                    reasoning=1,
                    total=6,
                    input_tokens=3,
                ),
            )
            capture_provider_response(parsed, raw)
            check("raw response id fills parsed provider data", parsed.provider_data.get("id") == request_id)
            capture_active_provider_evidence(provider_evidence_from_run(parsed))
            normalized = finalize_provider_attempt(attempt, normal_completion=True)
        finally:
            reset_active_provider_attempt(token)
    check("raw-only id reaches durable attempt", normalized["provider_request_id"] == request_id)


def test_none_never_overwrites_complete_usage() -> None:
    trace_id = "s10d-non-null-merge"
    with provider_accounting_scope(
        trace_id=trace_id,
        stage="vertical_agent",
        price_snapshot=PRICE_SNAPSHOT,
    ):
        attempt, token = begin_provider_attempt(
            requested_model=MODEL_ID,
            thinking_enabled=True,
            stream=True,
        )
        try:
            mark_provider_attempt_dispatched(attempt)
            _capture_raw(
                request_id="req-s10d-non-null",
                usage=_usage(
                    hit=9,
                    miss=11,
                    output=13,
                    reasoning=5,
                    total=33,
                    input_tokens=20,
                ),
            )
            _capture_raw(
                request_id="req-s10d-non-null",
                model=None,
                usage=_usage(
                    hit=None,
                    miss=None,
                    output=None,
                    reasoning=None,
                    total=None,
                    input_tokens=None,
                ),
            )
            attempt.sdk_stream_exhausted = True
            normalized = finalize_provider_attempt(attempt, normal_completion=True)
        finally:
            reset_active_provider_attempt(token)
    check("later None values do not erase Provider usage", normalized["usage"] == {
        "input_cache_hit_tokens": 9,
        "input_cache_miss_tokens": 11,
        "input_tokens": 20,
        "output_tokens": 13,
        "reasoning_tokens": 5,
        "total_tokens": 33,
    })


def test_stream_exhausted_without_usage() -> None:
    trace_id = "s10d-done-no-usage"
    with provider_accounting_scope(
        trace_id=trace_id,
        stage="vertical_agent",
        price_snapshot=PRICE_SNAPSHOT,
    ):
        attempt, token = begin_provider_attempt(
            requested_model=MODEL_ID,
            thinking_enabled=True,
            stream=True,
        )
        try:
            mark_provider_attempt_dispatched(attempt)
            _capture_raw(request_id="req-s10d-done-no-usage", usage=None)
            attempt.sdk_stream_exhausted = True
            normalized = finalize_provider_attempt(attempt, normal_completion=True)
        finally:
            reset_active_provider_attempt(token)
    row = property_db.get_provider_attempts_for_trace(trace_id)[0]
    check("exhausted stream without usage has exact status", normalized["usage_status"] == "unavailable_done_without_usage")
    check("exhausted stream records completion evidence", normalized["completion_evidence"] == "sdk_stream_iterator_exhausted")
    check("missing usage tokens remain unavailable not zero", all(
        row.get(key) is None
        for key in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "total_tokens", "estimated_cost_cny")
    ))
    check("done-without-usage anomaly is sent to Badcase review", (normalized.get("usage_anomaly_badcase") or {}).get("status") == "created")


def test_stream_interruption_after_chunk() -> None:
    trace_id = "s10d-stream-interrupted"
    with provider_accounting_scope(
        trace_id=trace_id,
        stage="vertical_agent",
        price_snapshot=PRICE_SNAPSHOT,
    ):
        attempt, token = begin_provider_attempt(
            requested_model=MODEL_ID,
            thinking_enabled=True,
            stream=True,
        )
        try:
            mark_provider_attempt_dispatched(attempt)
            _capture_raw(request_id="req-s10d-interrupted", usage=None)
            normalized = finalize_provider_attempt(
                attempt,
                normal_completion=False,
                exception=ConnectionResetError("fixture stream reset"),
                phase="stream_provider_call",
            )
        finally:
            reset_active_provider_attempt(token)
    check("observed connection interruption is not guessed", normalized["usage_status"] == "unavailable_stream_interrupted")
    check("stream interruption reason is evidence-backed", normalized["usage_unavailable_reason"] == "captured_stream_interruption")
    check("interrupted request does not display zero cost", normalized["calculated_direct_cost"] is None and normalized["cost_source"] is None)


def test_concurrent_contextvar_isolation() -> None:
    async def worker(trace_id: str, request_id: str, hit: int) -> None:
        with provider_accounting_scope(
            trace_id=trace_id,
            stage="router",
            price_snapshot=PRICE_SNAPSHOT,
        ):
            attempt, token = begin_provider_attempt(
                requested_model=MODEL_ID,
                thinking_enabled=True,
                stream=True,
            )
            try:
                mark_provider_attempt_dispatched(attempt)
                await asyncio.sleep(0)
                _capture_raw(
                    request_id=request_id,
                    usage=_usage(
                        hit=hit,
                        miss=2,
                        output=3,
                        reasoning=1,
                        total=hit + 5,
                        input_tokens=hit + 2,
                    ),
                )
                await asyncio.sleep(0)
                attempt.sdk_stream_exhausted = True
                finalize_provider_attempt(attempt, normal_completion=True)
            finally:
                reset_active_provider_attempt(token)

    async def run_workers() -> None:
        await asyncio.gather(
            worker("s10d-concurrent-a", "req-s10d-concurrent-a", 10),
            worker("s10d-concurrent-b", "req-s10d-concurrent-b", 20),
        )

    asyncio.run(run_workers())
    row_a = property_db.get_provider_attempts_for_trace("s10d-concurrent-a")[0]
    row_b = property_db.get_provider_attempts_for_trace("s10d-concurrent-b")[0]
    evidence_a = row_a["usage_normalized"]
    evidence_b = row_b["usage_normalized"]
    check("concurrent trace A keeps its request", evidence_a["provider_request_id"] == "req-s10d-concurrent-a")
    check("concurrent trace B keeps its request", evidence_b["provider_request_id"] == "req-s10d-concurrent-b")
    check("concurrent usage never crosses traces", (
        evidence_a["input_cache_hit_tokens"],
        evidence_b["input_cache_hit_tokens"],
    ) == (10, 20))


def test_explicit_retry_two_attempts() -> None:
    trace_id = "s10d-explicit-retry"
    first, _, _ = _complete_attempt(
        trace_id=trace_id,
        stage="badcase_switch_model_retry",
        request_id="req-s10d-retry-1",
        stream=False,
    )
    _complete_attempt(
        trace_id=trace_id,
        stage="badcase_switch_model_retry",
        request_id="req-s10d-retry-2",
        explicit_retry=True,
        retry_of_local_attempt_id=first.local_attempt_id,
        stream=False,
    )
    rows = property_db.get_provider_attempts_for_trace(trace_id)
    normalized = [row["usage_normalized"] for row in rows]
    check("explicit retry creates two Provider attempts", len(rows) == 2)
    check("retry attempts have stable sequences", [item["attempt_sequence"] for item in normalized] == [1, 2])
    check("retry attempts have unique Provider ids", len({item["provider_request_id"] for item in normalized}) == 2)
    check("second attempt references first local attempt", normalized[1]["retry_of_local_attempt_id"] == first.local_attempt_id)


def test_tool_loop_two_attempts() -> None:
    trace_id = "s10d-tool-loop"
    with provider_accounting_scope(
        trace_id=trace_id,
        stage="vertical_agent_tool_loop",
        price_snapshot=PRICE_SNAPSHOT,
    ) as scope:
        for index in (1, 2):
            attempt, token = begin_provider_attempt(
                requested_model=MODEL_ID,
                thinking_enabled=True,
                stream=False,
            )
            try:
                mark_provider_attempt_dispatched(attempt)
                _capture_raw(
                    request_id=f"req-s10d-tool-loop-{index}",
                    usage=_usage(
                        hit=index,
                        miss=2,
                        output=3,
                        reasoning=1,
                        total=index + 5,
                        input_tokens=index + 2,
                    ),
                )
                finalize_provider_attempt(attempt, normal_completion=True)
            finally:
                reset_active_provider_attempt(token)
    rows = property_db.get_provider_attempts_for_trace(trace_id)
    check("tool loop persists every Provider request", len(rows) == 2 and len(scope.attempts) == 2)
    check("tool loop requests are not merged by business stage", [
        row["usage_normalized"]["provider_request_id"] for row in rows
    ] == ["req-s10d-tool-loop-1", "req-s10d-tool-loop-2"])


def test_badcase_operations_use_central_scope() -> None:
    source = _source("app/badcases.py")
    gateway = _function_node(source, "_llm_generate")
    check("Badcase gateway owns Provider accounting scope", bool(_calls(gateway, "provider_accounting_scope")))
    check("Badcase gateway has no manual model-call writer", not _calls(gateway, "record_model_call"))
    cases = (
        ("extract_knowledge", "badcase_extract_knowledge", False),
        ("switch_model_retry", "badcase_switch_model_retry", True),
        ("check_tools_badcase", "badcase_check_tools", False),
    )
    for function_name, expected_stage, expected_retry in cases:
        node = _function_node(source, function_name)
        calls = _calls(node, "_llm_generate")
        check(f"{function_name} reaches central LLM gateway", len(calls) == 1)
        check(f"{function_name} declares stable accounting stage", _keyword_constant(calls[0], "stage") == expected_stage)
        if expected_retry:
            check(f"{function_name} declares explicit retry", _keyword_constant(calls[0], "explicit_retry") is True)
        check(f"{function_name} has no scattered manual accounting", not _calls(node, "record_model_call"))


def test_retest_logical_aggregate_excluded() -> None:
    badcase_source = _source("app/badcases.py")
    node = _function_node(badcase_source, "retest_badcase")
    segment = ast.get_source_segment(badcase_source, node) or ""
    check("Badcase retest is labelled logical aggregate", '"record_kind": "logical_aggregate"' in segment)
    check("Badcase retest opts out of Provider aggregate", '"include_in_provider_aggregate": False' in segment)
    check("Badcase retest does not create aggregate model call", not _calls(node, "record_model_call"))

    provider_record = {
        "record_kind": "provider_attempt",
        "stage": "router",
        "status": "success",
        "usage_normalized": {
            "record_kind": "provider_attempt",
            "include_in_provider_aggregate": True,
            "provider_request_sent": True,
            "usage_status": "provider_actual",
        },
    }
    logical_record = {
        "record_kind": "logical_aggregate",
        "stage": "retest",
        "usage_normalized": {
            "record_kind": "logical_aggregate",
            "include_in_provider_aggregate": False,
        },
    }
    check("property DB distinguishes Provider from logical records", (
        property_db.is_provider_attempt_record(provider_record)
        and not property_db.is_provider_attempt_record(logical_record)
    ))
    observability_source = _source("app/observability.py")
    aggregate_node = _function_node(observability_source, "_provider_aggregate_decision")
    aggregate_segment = ast.get_source_segment(observability_source, aggregate_node) or ""
    check("cost aggregation explicitly excludes retest logical records", (
        "logical_aggregate" in aggregate_segment
        and "logical_retest_aggregate" in aggregate_segment
        and 'stage == "retest"' in aggregate_segment
    ))


def test_explicit_handoff_zero_provider_requests() -> None:
    decision = evaluate_handoff_policy(
        "owner requests a human handoff",
        explicit_reason="owner explicitly requested staff takeover",
    )
    source = ast.get_source_segment(
        _source("app/handoff_policy.py"),
        _function_node(_source("app/handoff_policy.py"), "evaluate_handoff_policy"),
    ) or ""
    check("explicit handoff deterministically short-circuits", decision["should_request_handoff"] is True and decision["reason_code"] == "owner_requested")
    check("handoff policy contains no model gateway", all(term not in source for term in ("build_model", "DeepSeek", "MODEL.invoke", "_llm_generate")))
    check("explicit handoff fixture produced zero Provider rows", property_db.get_provider_attempts_for_trace("s10d-explicit-handoff") == [])


def test_finalize_persistence_failure_is_visible() -> None:
    trace_id = "s10d-persistence-failure"
    with provider_accounting_scope(
        trace_id=trace_id,
        stage="router",
        price_snapshot=PRICE_SNAPSHOT,
    ) as scope:
        attempt, token = begin_provider_attempt(
            requested_model=MODEL_ID,
            thinking_enabled=True,
            stream=False,
        )
        original_finalize = property_db.finalize_provider_attempt
        try:
            mark_provider_attempt_dispatched(attempt)
            _capture_raw(
                request_id="req-s10d-persistence-failure",
                usage=_usage(
                    hit=1,
                    miss=2,
                    output=3,
                    reasoning=1,
                    total=6,
                    input_tokens=3,
                ),
            )

            def fail_finalize(*args: Any, **kwargs: Any) -> None:
                raise sqlite3.OperationalError("fixture final persistence failure")

            property_db.finalize_provider_attempt = fail_finalize
            try:
                finalize_provider_attempt(attempt, normal_completion=True)
            except ProviderAccountingPersistenceError:
                pass
            else:
                raise AssertionError("persistence failure must be visible to the caller")
        finally:
            property_db.finalize_provider_attempt = original_finalize
            reset_active_provider_attempt(token)

    row = property_db.get_provider_attempts_for_trace(trace_id)[0]
    normalized = row["usage_normalized"]
    check("persistence failure receives exact usage state", row["usage_status"] == "unavailable_persistence_failure")
    check("persistence failure does not retain precise cost", row["estimated_cost_cny"] is None and row["cost_source"] is None)
    check("persistence failure leaves structural evidence", normalized["persistence_status"] == "failed" and normalized["error_evidence"]["phase"] == "persistence_finalize")
    check("failed finalization remains in request scope evidence", scope.attempts[-1]["usage_status"] == "unavailable_persistence_failure")


def test_reasoning_is_output_subset() -> None:
    _, normalized, _ = _complete_attempt(
        trace_id="s10d-reasoning-subset",
        stage="darwin",
        request_id="req-s10d-reasoning-subset",
        hit=10,
        miss=20,
        output=30,
        reasoning=25,
        stream=False,
    )
    check("reasoning is retained separately", normalized["reasoning_tokens"] == 25)
    check("reasoning is not added again to total", normalized["total_tokens"] == 60 and normalized["calculated_split_total_tokens"] == 60)
    check("consistent reasoning subset is not marked inconsistent", normalized["provider_usage_inconsistent"] is False)


def test_legacy_missing_usage_is_not_backfilled() -> None:
    trace_id = "s10d-legacy-missing"
    legacy = property_db.record_model_call(
        trace_id=trace_id,
        stage="legacy_stage",
        model_id=MODEL_ID,
        usage_source="unavailable",
        usage_status="cause_unconfirmed",
        status="failed",
        record_kind="legacy",
        input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        cached_tokens=None,
        total_tokens=None,
        estimated_cost_cny=None,
        usage_normalized={
            "record_kind": "legacy",
            "usage_status": "cause_unconfirmed",
            "token_source": "unavailable",
            "historical_evidence": "provider_response_not_preserved",
        },
    )
    current = property_db.get_model_call(legacy["id"])
    check("legacy missing usage stays unavailable", all(
        current.get(key) is None
        for key in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "total_tokens", "estimated_cost_cny")
    ))
    check("legacy row is never upgraded to Provider attempt", current["record_kind"] == "legacy" and property_db.get_provider_attempts_for_trace(trace_id) == [])


def test_gateway_retry_and_preflight_contracts() -> None:
    settings_source = _source("app/settings.py")
    build_model_node = _function_node(settings_source, "build_model")
    build_segment = ast.get_source_segment(settings_source, build_model_node) or ""
    check("hidden SDK retries are disabled", "max_retries=0" in build_segment.replace(" ", ""))

    invoke_node = _function_node(settings_source, "invoke")
    invoke_segment = ast.get_source_segment(settings_source, invoke_node) or ""
    check("durable preflight precedes paid synchronous invoke", invoke_segment.index("begin_provider_attempt") < invoke_segment.index("super().invoke"))

    paid_dispatches: list[str] = []
    original_create = property_db.create_provider_attempt

    def fail_preflight(*args: Any, **kwargs: Any) -> None:
        raise sqlite3.OperationalError("fixture preflight unavailable")

    property_db.create_provider_attempt = fail_preflight
    try:
        with provider_accounting_scope(
            trace_id="s10d-preflight-block",
            stage="router",
            price_snapshot=PRICE_SNAPSHOT,
        ):
            try:
                begin_provider_attempt(
                    requested_model=MODEL_ID,
                    thinking_enabled=True,
                    stream=False,
                )
                paid_dispatches.append("would-have-called-provider")
            except ProviderAccountingError:
                pass
            else:
                raise AssertionError("preflight failure must block the paid request")
    finally:
        property_db.create_provider_attempt = original_create
    check("preflight persistence failure sends no Provider request", paid_dispatches == [])


def test_error_evidence_is_non_sensitive() -> None:
    class FixtureHTTPError(RuntimeError):
        status_code = 503

    forbidden = (
        "fixture-secret-api-key",
        "Authorization",
        "Bearer",
        "private prompt body",
    )
    error = FixtureHTTPError(
        "Authorization: Bearer fixture-secret-api-key; private prompt body"
    )
    evidence = safe_error_evidence(error, phase="provider_dispatch")
    serialized = json.dumps(evidence, ensure_ascii=False)
    check("error evidence keeps only structural status", evidence == {
        "phase": "provider_dispatch",
        "exception_type": "FixtureHTTPError",
        "cause_type": None,
        "http_status": 503,
    })
    check("error evidence excludes credential and prompt text", all(value not in serialized for value in forbidden))


def test_provider_request_id_unique_schema() -> None:
    db_source = _source("db/property_db.py")
    check("Provider request id has a non-null unique index", all(
        phrase in db_source
        for phrase in (
            "idx_model_calls_provider_request_unique",
            "ON model_calls(provider_request_id)",
            "WHERE provider_request_id IS NOT NULL",
        )
    ))


def _deny_network(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("S10-D deterministic test attempted network access")


def main() -> None:
    property_db.init_db()
    original_create_connection = socket.create_connection
    socket.create_connection = _deny_network
    try:
        tests: Iterable[Any] = (
            test_content_and_usage_same_id_one_row,
            test_parsed_without_id_raw_with_id,
            test_none_never_overwrites_complete_usage,
            test_stream_exhausted_without_usage,
            test_stream_interruption_after_chunk,
            test_concurrent_contextvar_isolation,
            test_explicit_retry_two_attempts,
            test_tool_loop_two_attempts,
            test_badcase_operations_use_central_scope,
            test_retest_logical_aggregate_excluded,
            test_explicit_handoff_zero_provider_requests,
            test_finalize_persistence_failure_is_visible,
            test_reasoning_is_output_subset,
            test_legacy_missing_usage_is_not_backfilled,
            test_gateway_retry_and_preflight_contracts,
            test_error_evidence_is_non_sensitive,
            test_provider_request_id_unique_schema,
        )
        for test in tests:
            test()
    finally:
        socket.create_connection = original_create_connection
    print(
        "PASS: V1.8.2-S10-D Provider per-request reconciliation "
        f"({len(CHECKS)} deterministic assertions, Provider calls=0)"
    )


if __name__ == "__main__":
    main()
