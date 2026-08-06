"""Full-dependency, no-network S10-H quality-closure behavior tests.

The suite uses real FastAPI routes and a fresh SQLite database.  Chat runtime
boundaries are replaced with deterministic SSE generators, so no Provider is
ever contacted and no synthetic Provider attempt is written.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TEST_DATA = tempfile.TemporaryDirectory(prefix="yiai-s10h-data-")
_TEST_DATA_DIR = Path(_TEST_DATA.name).resolve()
_NORMALIZED_TEST_DATA_DIR = str(_TEST_DATA_DIR).replace("\\", "/").lower()
if (
    _NORMALIZED_TEST_DATA_DIR == "/app/data"
    or _NORMALIZED_TEST_DATA_DIR.startswith("/app/data/")
    or _NORMALIZED_TEST_DATA_DIR == "/volume3/docker/agno-demo-os"
    or _NORMALIZED_TEST_DATA_DIR.startswith("/volume3/docker/agno-demo-os/")
):
    raise RuntimeError("unsafe S10-H PROPERTY_DATA_DIR")

# Never inherit production data or a Provider credential into this process.
os.environ["PROPERTY_DATA_DIR"] = str(_TEST_DATA_DIR)
for _provider_key in (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "AGNO_API_KEY",
    "DASHSCOPE_API_KEY",
    "PARALLEL_API_KEY",
):
    os.environ[_provider_key] = ""

_ORIGINAL_CREATE_CONNECTION = socket.create_connection


def _blocked_connection(*_args: Any, **_kwargs: Any):
    raise AssertionError("network access is forbidden in S10-H deterministic tests")


socket.create_connection = _blocked_connection

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from db import property_db as db  # noqa: E402

# Initialize only the freshly forced temporary fixture database before app
# settings resolve model metadata from it.
db.init_db()

from app import badcases as badcase_api  # noqa: E402
from app import chat as chat_api  # noqa: E402
from app import evaluations as evaluation_api  # noqa: E402


CHECKS: list[str] = []
AVAILABLE_BUDGET = {
    "budget_status": "available",
    "statistics_status": "consistent",
    "data_quality_status": "normal",
    "alert_level": "none",
    "allowed": True,
    "http_status": None,
    "detail": "deterministic S10-H budget fixture",
    "today_cost": 0.001,
    "month_cost": 0.002,
}


def check(name: str, condition: object) -> None:
    if not condition:
        raise AssertionError(name)
    CHECKS.append(name)


def provider_attempt_count() -> int:
    connection = db._get_conn()
    try:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM model_calls "
                "WHERE record_kind = 'provider_attempt'"
            ).fetchone()[0]
        )
    finally:
        connection.close()


def _sse(event: str, payload: Dict[str, Any]) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


def _persist_trace(
    trace_id: str,
    session_id: str,
    message: str,
    *,
    status: str,
) -> None:
    db.ensure_chat_session(session_id)
    if not db.get_chat_trace(trace_id):
        db.create_chat_trace(trace_id, session_id, message, run_type="evaluation")
    db.update_chat_trace(
        trace_id,
        status=status,
        intent="customer_service",
        agent_name="客服 Agent",
        agent_id="customer_service",
    )


def _stream_stub(
    chunks: Iterable[str],
    *,
    trace_id: str | None = None,
    trace_status: str = "complete",
    trace_session_override: str | None = None,
) -> Callable[..., AsyncIterator[str]]:
    async def stream(
        message: str,
        session_id: str,
        _user_id: str = "retest",
    ) -> AsyncIterator[str]:
        if trace_id:
            _persist_trace(
                trace_id,
                trace_session_override or session_id,
                message,
                status=trace_status,
            )
        for chunk in chunks:
            yield chunk

    return stream


def _new_retest_case(
    label: str,
    *,
    last_applied_at: str,
    retest_response: str | None = None,
    retest_trace_id: str | None = None,
    retest_context: Dict[str, Any] | None = None,
    last_retest_at: str | None = None,
    linked_evaluation_case_id: int | None = None,
) -> Dict[str, Any]:
    case = db.create_badcase(
        title=f"S10-H {label}",
        description="deterministic retest fixture",
        category="other",
        status="verifying",
        source="manual",
        original_query=f"S10-H retest question {label}",
        linked_evaluation_case_id=linked_evaluation_case_id,
    )
    return db.update_badcase(
        int(case["id"]),
        last_applied_at=last_applied_at,
        retest_response=retest_response,
        retest_trace_id=retest_trace_id,
        retest_context_json=(
            json.dumps(retest_context, ensure_ascii=False)
            if retest_context is not None
            else None
        ),
        last_retest_at=last_retest_at,
    ) or case


def _successful_evaluation_stub(
    trace_prefix: str,
) -> Callable[[str, str], Any]:
    async def run(message: str, session_id: str):
        trace_id = f"{trace_prefix}-{session_id[-8:]}"
        _persist_trace(trace_id, session_id, message, status="complete")
        return (
            "客服已依据真实流程完成回答。",
            {
                "status": "complete",
                "trace_id": trace_id,
                "current_agent": "客服 Agent",
                "current_agent_id": "customer_service",
                "route_intent": "customer_service",
                "activated_skills": [],
                "tool_calls": [],
                "mcp_calls": [],
                "citations": [],
                "handoff": False,
                "decision_summary": {
                    "agent": {
                        "status": "selected",
                        "agent_id": "customer_service",
                    }
                },
            },
        )

    return run


def assert_evaluation_http_contract(client: TestClient) -> None:
    before_provider = provider_attempt_count()
    persistence_calls: list[Dict[str, Any]] = []
    original_create_run = evaluation_api.create_evaluation_run

    def counting_create_run(*args: Any, **kwargs: Any):
        persistence_calls.append(dict(kwargs))
        return original_create_run(*args, **kwargs)

    evaluation_api.create_evaluation_run = counting_create_run
    try:
        passing_case = db.create_evaluation_case(
            case_key="S10H-EVALUATION-PASS",
            title="S10-H Evaluation success",
            user_message="请给出客服流程回答",
            expected_agent_id="customer_service",
            required_terms=["真实流程"],
            status="active",
            version_label="V1.8.2-S10-H",
        )
        evaluation_api._run_real_chat = _successful_evaluation_stub(
            "s10h-evaluation-pass"
        )
        before_calls = len(persistence_calls)
        response = client.post(
            f"/api/evaluations/cases/{passing_case['id']}/run",
            json={},
        )
        check("Evaluation success HTTP is 200", response.status_code == 200)
        payload = response.json()
        check(
            "Evaluation success returns run, rules and the real budget gate",
            isinstance(payload.get("run"), dict)
            and isinstance(payload.get("rule_results"), list)
            and payload.get("budget") == AVAILABLE_BUDGET,
        )
        check(
            "Evaluation success is persisted exactly once",
            len(persistence_calls) == before_calls + 1
            and len(db.list_evaluation_runs(int(passing_case["id"]))) == 1,
        )
        check(
            "Evaluation success response assembly does not create a second run",
            payload["run"]["id"]
            == db.list_evaluation_runs(int(passing_case["id"]))[0]["id"],
        )
        check(
            "Evaluation deterministic success remains passed",
            payload["run"]["status"] == "passed"
            and not any(
                item.get("status") == "fail"
                for item in payload["rule_results"]
            ),
        )

        failing_case = db.create_evaluation_case(
            case_key="S10H-EVALUATION-FAIL",
            title="S10-H Evaluation runtime failure",
            user_message="模拟真实运行失败",
            status="active",
            version_label="V1.8.2-S10-H",
        )

        async def failed_runtime(message: str, session_id: str):
            trace_id = f"s10h-evaluation-failed-{session_id[-8:]}"
            _persist_trace(trace_id, session_id, message, status="failed")
            raise evaluation_api.RuntimeExecutionError(
                "deterministic runtime failure",
                "",
                {
                    "status": "failed",
                    "trace_id": trace_id,
                    "error_code": "deterministic_failure",
                },
            )

        evaluation_api._run_real_chat = failed_runtime
        before_calls = len(persistence_calls)
        failed_response = client.post(
            f"/api/evaluations/cases/{failing_case['id']}/run",
            json={},
        )
        check(
            "Evaluation runtime failure is a controlled saved result",
            failed_response.status_code == 200,
        )
        failed_payload = failed_response.json()
        check(
            "Evaluation failure is persisted exactly once as failed",
            len(persistence_calls) == before_calls + 1
            and len(db.list_evaluation_runs(int(failing_case["id"]))) == 1
            and failed_payload["run"]["status"] == "failed",
        )
        check(
            "Evaluation failure never fabricates PASS",
            failed_payload["rule_results"]
            and all(
                item.get("status") == "fail"
                for item in failed_payload["rule_results"]
            ),
        )
        check(
            "Evaluation success and failure create zero Provider attempts",
            provider_attempt_count() == before_provider,
        )
    finally:
        evaluation_api.create_evaluation_run = original_create_run


def assert_successful_badcase_retest(client: TestClient) -> None:
    case = _new_retest_case(
        "complete",
        last_applied_at="2000-01-01 00:00",
    )
    trace_id = "s10h-real-retest-complete"
    chat_api.stream_chat_response = _stream_stub(
        (
            _sse("delta", {"content": "partial answer that must be replaced"}),
            _sse(
                "final",
                {
                    "content": "final truthful repaired answer",
                    "trace_id": trace_id,
                },
            ),
            _sse(
                "done",
                {
                    "status": "complete",
                    "trace_id": trace_id,
                    "current_agent": "客服 Agent",
                    "current_agent_id": "customer_service",
                    "route_intent": "customer_service",
                    "token_detail": {},
                },
            ),
        ),
        trace_id=trace_id,
    )
    before_provider = provider_attempt_count()
    response = client.post(f"/api/badcases/{case['id']}/retest", json={})
    check("complete Badcase SSE returns 200", response.status_code == 200)
    payload = response.json()
    updated = payload["badcase"]
    context = payload["retest_context"]
    check(
        "complete Badcase SSE stores the real Trace only",
        updated["retest_trace_id"] == trace_id
        and db.get_chat_trace(trace_id) is not None,
    )
    check(
        "final SSE replaces delta without duplicating the successful answer",
        payload["retest_response"] == "final truthful repaired answer"
        and "partial answer" not in payload["retest_response"]
        and context.get("run_status") == "complete"
        and context.get("trace_id") == trace_id,
    )
    check(
        "complete Badcase SSE binds persisted Trace to the retest session",
        context.get("trace_persisted") is True
        and context.get("trace_status") == "complete"
        and context.get("session_id")
        and context.get("session_id") == context.get("trace_session_id"),
    )
    check(
        "complete Badcase SSE records a post-apply start anchor",
        bool(context.get("retest_started_at"))
        and updated["last_retest_at"] >= context["retest_started_at"]
        and context["retest_started_at"] >= updated["last_applied_at"],
    )
    check(
        "complete post-apply retest exposes manual verify-pass",
        updated["last_retest_at"] >= updated["last_applied_at"]
        and "verify-pass" in updated["allowed_actions"],
    )
    verify = client.post(
        f"/api/badcases/{case['id']}/verify",
        json={"passed": True, "note": "S10-H 人工验证通过"},
    )
    check(
        "operator can verify only after the complete retest",
        verify.status_code == 200
        and verify.json()["badcase"]["status"] == "released",
    )
    check(
        "successful Badcase retest creates zero Provider attempts",
        provider_attempt_count() == before_provider,
    )


def _assert_failed_retest(
    client: TestClient,
    *,
    label: str,
    chunks: Iterable[str],
    trace_id: str | None,
    trace_status: str = "failed",
    trace_session_override: str | None = None,
) -> None:
    sentinel_trace = f"s10h-old-real-{label}"
    sentinel_session = f"s10h-old-session-{label}"
    _persist_trace(
        sentinel_trace,
        sentinel_session,
        f"old evidence {label}",
        status="complete",
    )
    case = _new_retest_case(
        label,
        last_applied_at="2099-01-01 00:00",
        retest_response=f"old response {label}",
        retest_trace_id=sentinel_trace,
        retest_context={
            "run_status": "complete",
            "trace_id": sentinel_trace,
            "trace_persisted": True,
            "trace_status": "complete",
            "session_id": sentinel_session,
            "trace_session_id": sentinel_session,
            "retest_started_at": "2000-01-01 00:00",
        },
        last_retest_at="2000-01-01 00:00",
    )
    before = db.get_badcase(int(case["id"])) or {}
    protected_fields = {
        key: before.get(key)
        for key in (
            "status",
            "retest_response",
            "retest_context_json",
            "retest_trace_id",
            "last_retest_at",
        )
    }
    chat_api.stream_chat_response = _stream_stub(
        chunks,
        trace_id=trace_id,
        trace_status=trace_status,
        trace_session_override=trace_session_override,
    )
    before_provider = provider_attempt_count()
    response = client.post(f"/api/badcases/{case['id']}/retest", json={})
    check(f"{label} retest returns controlled 502", response.status_code == 502)
    after = db.get_badcase(int(case["id"])) or {}
    check(
        f"{label} retest preserves lifecycle and success fields",
        all(after.get(key) == value for key, value in protected_fields.items()),
    )
    check(
        f"{label} retest does not manufacture a retest Trace id",
        after.get("retest_trace_id") == sentinel_trace
        and not str(after.get("retest_trace_id") or "").startswith("retest-"),
    )
    verify = client.post(
        f"/api/badcases/{case['id']}/verify",
        json={"passed": True, "note": f"must remain blocked: {label}"},
    )
    check(
        f"{label} retest cannot unlock verify-pass",
        verify.status_code == 400
        and (db.get_badcase(int(case["id"])) or {}).get("status")
        == "verifying",
    )
    check(
        f"{label} retest creates zero Provider attempts",
        provider_attempt_count() == before_provider,
    )


def assert_failed_badcase_retests(client: TestClient) -> None:
    _assert_failed_retest(
        client,
        label="event-error",
        chunks=(
            _sse(
                "error",
                {
                    "status": "failed",
                    "error": "deterministic stream error",
                    "trace_id": "s10h-real-error-trace",
                },
            ),
            _sse(
                "done",
                {
                    "status": "failed",
                    "trace_id": "s10h-real-error-trace",
                },
            ),
        ),
        trace_id="s10h-real-error-trace",
    )
    _assert_failed_retest(
        client,
        label="failed-done",
        chunks=(
            _sse("delta", {"content": "partial failed answer"}),
            _sse(
                "done",
                {
                    "status": "failed",
                    "trace_id": "s10h-real-failed-done-trace",
                },
            ),
        ),
        trace_id="s10h-real-failed-done-trace",
    )
    _assert_failed_retest(
        client,
        label="missing-done",
        chunks=(_sse("delta", {"content": "orphaned answer"}),),
        trace_id="s10h-real-missing-done-trace",
    )
    _assert_failed_retest(
        client,
        label="missing-trace",
        chunks=(
            _sse("delta", {"content": "answer without trace"}),
            _sse("done", {"status": "complete"}),
        ),
        trace_id=None,
        trace_status="complete",
    )
    _assert_failed_retest(
        client,
        label="missing-answer",
        chunks=(
            _sse(
                "done",
                {
                    "status": "complete",
                    "trace_id": "s10h-real-missing-answer-trace",
                },
            ),
        ),
        trace_id="s10h-real-missing-answer-trace",
        trace_status="complete",
    )
    _assert_failed_retest(
        client,
        label="trace-db-failed",
        chunks=(
            _sse("delta", {"content": "answer backed by a failed Trace"}),
            _sse(
                "done",
                {
                    "status": "complete",
                    "trace_id": "s10h-done-complete-trace-db-failed",
                },
            ),
        ),
        trace_id="s10h-done-complete-trace-db-failed",
        trace_status="failed",
    )
    _assert_failed_retest(
        client,
        label="trace-other-runtime-session",
        chunks=(
            _sse("delta", {"content": "answer backed by another session"}),
            _sse(
                "done",
                {
                    "status": "complete",
                    "trace_id": "s10h-done-complete-trace-other-session",
                },
            ),
        ),
        trace_id="s10h-done-complete-trace-other-session",
        trace_status="complete",
        trace_session_override="s10h-unrelated-runtime-session",
    )
    _assert_failed_retest(
        client,
        label="double-done",
        chunks=(
            _sse("delta", {"content": "ambiguous duplicate terminal answer"}),
            _sse(
                "done",
                {
                    "status": "complete",
                    "trace_id": "s10h-real-double-done-trace",
                },
            ),
            _sse(
                "done",
                {
                    "status": "complete",
                    "trace_id": "s10h-real-double-done-trace",
                },
            ),
        ),
        trace_id="s10h-real-double-done-trace",
        trace_status="complete",
    )


def assert_retest_concurrent_apply_change(client: TestClient) -> None:
    case = _new_retest_case(
        "concurrent-apply-change",
        last_applied_at="2000-01-01 00:00",
    )
    trace_id = "s10h-concurrent-apply-change-trace"

    async def changed_during_stream(
        message: str,
        session_id: str,
        _user_id: str = "retest",
    ) -> AsyncIterator[str]:
        _persist_trace(trace_id, session_id, message, status="complete")
        db.update_badcase(case["id"], last_applied_at="2099-01-01 00:00")
        yield _sse("delta", {"content": "stale result after a new apply"})
        yield _sse(
            "done",
            {"status": "complete", "trace_id": trace_id},
        )

    chat_api.stream_chat_response = changed_during_stream
    before_provider = provider_attempt_count()
    response = client.post(f"/api/badcases/{case['id']}/retest", json={})
    check("concurrent apply anchor change rejects the stale retest", response.status_code == 502)
    current = db.get_badcase(case["id"])
    check(
        "concurrent apply anchor change never stores successful retest evidence",
        current is not None
        and current["status"] == "verifying"
        and current["last_applied_at"] == "2099-01-01 00:00"
        and not current.get("last_retest_at")
        and not current.get("retest_response")
        and not current.get("retest_trace_id"),
    )
    check(
        "concurrent apply anchor change creates zero Provider attempts",
        provider_attempt_count() == before_provider,
    )
    _assert_failed_retest(
        client,
        label="failed-then-complete",
        chunks=(
            _sse("delta", {"content": "ambiguous terminal answer"}),
            _sse(
                "done",
                {
                    "status": "failed",
                    "trace_id": "s10h-real-failed-then-complete-trace",
                },
            ),
            _sse(
                "done",
                {
                    "status": "complete",
                    "trace_id": "s10h-real-failed-then-complete-trace",
                },
            ),
        ),
        trace_id="s10h-real-failed-then-complete-trace",
        trace_status="complete",
    )


def assert_retest_evidence_gate(client: TestClient) -> None:
    old_trace = "s10h-real-preapply-trace"
    old_session = "s10h-preapply-session"
    _persist_trace(
        old_trace,
        old_session,
        "old retest",
        status="complete",
    )
    old = _new_retest_case(
        "preapply",
        last_applied_at="2026-08-06 10:00",
        retest_response="old but complete answer",
        retest_trace_id=old_trace,
        retest_context={
            "run_status": "complete",
            "trace_id": old_trace,
            "trace_persisted": True,
            "trace_status": "complete",
            "session_id": old_session,
            "trace_session_id": old_session,
            "retest_started_at": "2026-08-06 09:59",
        },
        # Completion after apply cannot rescue a run that started before apply.
        last_retest_at="2026-08-06 10:01",
    )
    old_detail = client.get(f"/api/badcases/{old['id']}")
    check("pre-apply retest detail is readable", old_detail.status_code == 200)
    check(
        "pre-apply retest never exposes verify-pass",
        "verify-pass"
        not in old_detail.json()["badcase"].get("allowed_actions", []),
    )
    old_verify = client.post(
        f"/api/badcases/{old['id']}/verify",
        json={"passed": True, "note": "old retest must not pass"},
    )
    check("pre-apply retest verify endpoint is blocked", old_verify.status_code == 400)

    evidence_cases = (
        {
            "label": "missing-response",
            "answer": "",
            "trace_id": "s10h-gate-response-trace",
            "session_id": "s10h-gate-session-missing-response",
            "trace_session_id": "s10h-gate-session-missing-response",
            "run_status": "complete",
            "trace_status": "complete",
            "trace_persisted": True,
            "persist": True,
        },
        {
            "label": "missing-trace-id",
            "answer": "answer",
            "trace_id": "",
            "session_id": "s10h-gate-session-missing-trace-id",
            "trace_session_id": "",
            "run_status": "complete",
            "trace_status": None,
            "trace_persisted": False,
            "persist": False,
        },
        {
            "label": "run-status-failed",
            "answer": "answer",
            "trace_id": "s10h-gate-run-failed-trace",
            "session_id": "s10h-gate-session-run-status-failed",
            "trace_session_id": "s10h-gate-session-run-status-failed",
            "run_status": "failed",
            "trace_status": "complete",
            "trace_persisted": True,
            "persist": True,
        },
        {
            "label": "trace-not-complete",
            "answer": "answer",
            "trace_id": "s10h-gate-incomplete-trace",
            "session_id": "s10h-gate-session-trace-not-complete",
            "trace_session_id": "s10h-gate-session-trace-not-complete",
            "run_status": "complete",
            "trace_status": "failed",
            "trace_persisted": True,
            "persist": True,
        },
        {
            "label": "trace-other-session",
            "answer": "answer",
            "trace_id": "s10h-gate-other-session-trace",
            "session_id": "s10h-gate-claimed-session",
            "trace_session_id": "s10h-gate-actual-session",
            "run_status": "complete",
            "trace_status": "complete",
            "trace_persisted": True,
            "persist": True,
        },
        {
            "label": "trace-not-persisted",
            "answer": "answer",
            "trace_id": "s10h-gate-nonexistent-trace",
            "session_id": "s10h-gate-session-nonexistent",
            "trace_session_id": "s10h-gate-session-nonexistent",
            "run_status": "complete",
            "trace_status": "complete",
            # A self-consistent JSON claim cannot replace a real DB Trace.
            "trace_persisted": True,
            "persist": False,
        },
    )
    for item in evidence_cases:
        label = str(item["label"])
        trace_id = str(item["trace_id"])
        if item["persist"] and trace_id:
            _persist_trace(
                trace_id,
                str(item["trace_session_id"]),
                label,
                status=str(item["trace_status"]),
            )
        case = _new_retest_case(
            label,
            last_applied_at="2026-08-06 10:00",
            retest_response=str(item["answer"]),
            retest_trace_id=trace_id,
            retest_context={
                "run_status": item["run_status"],
                "trace_id": trace_id,
                "trace_persisted": item["trace_persisted"],
                "trace_status": item["trace_status"],
                "session_id": item["session_id"],
                "trace_session_id": item["trace_session_id"],
                "retest_started_at": "2026-08-06 10:00",
            },
            last_retest_at="2026-08-06 10:01",
        )
        detail = client.get(f"/api/badcases/{case['id']}")
        check(f"{label} evidence detail is readable", detail.status_code == 200)
        check(
            f"{label} evidence does not expose verify-pass",
            "verify-pass"
            not in detail.json()["badcase"].get("allowed_actions", []),
        )
        verify = client.post(
            f"/api/badcases/{case['id']}/verify",
            json={"passed": True, "note": f"block {label}"},
        )
        check(f"{label} evidence verify endpoint is blocked", verify.status_code == 400)


def assert_evaluation_linked_retest_compatibility(client: TestClient) -> None:
    case = db.create_evaluation_case(
        case_key="S10H-LINKED-RETEST",
        title="S10-H linked Evaluation retest",
        user_message="linked retest question",
        expected_agent_id="customer_service",
        required_terms=["真实流程"],
        status="active",
        version_label="V1.8.2-S10-H",
    )
    badcase = _new_retest_case(
        "evaluation-linked",
        last_applied_at="2000-01-01 00:00",
        linked_evaluation_case_id=int(case["id"]),
    )
    evaluation_api._run_real_chat = _successful_evaluation_stub(
        "s10h-linked-evaluation"
    )
    before_provider = provider_attempt_count()
    response = client.post(
        f"/api/evaluations/cases/{case['id']}/run",
        json={"linked_badcase_id": int(badcase["id"])},
    )
    check("linked Evaluation retest HTTP is 200", response.status_code == 200)
    payload = response.json()
    check(
        "linked Evaluation retest remains a passed Evaluation run",
        payload["run"]["status"] == "passed"
        and payload["run"]["badcase_id"] == int(badcase["id"]),
    )
    detail = client.get(f"/api/badcases/{badcase['id']}")
    check("linked Evaluation Badcase detail is readable", detail.status_code == 200)
    linked = detail.json()["badcase"]
    check(
        "linked Evaluation retest carries complete evidence",
        linked["retest_response"]
        and linked["retest_trace_id"] == payload["run"]["trace_id"]
        and linked["retest_context"].get("run_status") == "complete"
        and linked["retest_context"].get("trace_persisted") is True
        and linked["retest_context"].get("trace_status") == "complete"
        and linked["retest_context"].get("session_id")
        == linked["retest_context"].get("trace_session_id")
        and bool(linked["retest_context"].get("retest_started_at")),
    )
    check(
        "linked Evaluation retest remains eligible for operator verify",
        "verify-pass" in linked.get("allowed_actions", []),
    )
    verify = client.post(
        f"/api/badcases/{badcase['id']}/verify",
        json={"passed": True, "note": "linked Evaluation retest verified"},
    )
    check(
        "raw stored linked retest context passes the verify endpoint",
        verify.status_code == 200
        and verify.json()["badcase"]["status"] == "released",
    )
    check(
        "linked Evaluation retest creates zero Provider attempts",
        provider_attempt_count() == before_provider,
    )


def main() -> None:
    original_evaluation_gate = evaluation_api._background_budget_gate
    original_badcase_gate = badcase_api._background_budget_gate
    original_evaluation_runtime = evaluation_api._run_real_chat
    original_chat_stream = chat_api.stream_chat_response
    initial_provider_attempts = provider_attempt_count()
    app = FastAPI()
    app.include_router(evaluation_api.router)
    app.include_router(badcase_api.router, prefix="/api/badcases")
    client = TestClient(app, raise_server_exceptions=False)
    try:
        evaluation_api._background_budget_gate = (
            lambda _operation: dict(AVAILABLE_BUDGET)
        )
        badcase_api._background_budget_gate = (
            lambda _operation: dict(AVAILABLE_BUDGET)
        )
        assert_evaluation_http_contract(client)
        assert_successful_badcase_retest(client)
        assert_failed_badcase_retests(client)
        assert_retest_concurrent_apply_change(client)
        assert_retest_evidence_gate(client)
        assert_evaluation_linked_retest_compatibility(client)
        check(
            "S10-H full suite has zero Provider attempt delta",
            provider_attempt_count() == initial_provider_attempts,
        )
    finally:
        evaluation_api._background_budget_gate = original_evaluation_gate
        badcase_api._background_budget_gate = original_badcase_gate
        evaluation_api._run_real_chat = original_evaluation_runtime
        chat_api.stream_chat_response = original_chat_stream
        client.close()
        socket.create_connection = _ORIGINAL_CREATE_CONNECTION
        _TEST_DATA.cleanup()
    print(
        "PASS: V1.8.2-S10-H quality closure truthfulness "
        f"({len(CHECKS)} behavior checks; Provider attempts delta=0)"
    )


if __name__ == "__main__":
    main()
