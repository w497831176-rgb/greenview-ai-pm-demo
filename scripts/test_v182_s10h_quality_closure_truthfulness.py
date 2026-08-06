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


def _raw_sse(event: str, raw_payload: str) -> str:
    """Build an SSE frame whose data is intentionally not normalized JSON."""
    return f"event: {event}\ndata: {raw_payload}\n\n"


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


def _evaluation_stream_stub(
    chunks: Iterable[str],
    *,
    entries: list[Dict[str, str]],
    trace_id: str | None = None,
    trace_status: str = "complete",
    trace_session_override: str | None = None,
    raise_after: bool = False,
    evidence_setup: Callable[[str, str], None] | None = None,
) -> Callable[..., AsyncIterator[str]]:
    """Replace only the SSE source below the real Evaluation runtime parser."""

    async def stream(
        message: str,
        session_id: str,
        user_id: str = "evaluation",
    ) -> AsyncIterator[str]:
        entries.append(
            {
                "message": message,
                "session_id": session_id,
                "user_id": user_id,
            }
        )
        if trace_id:
            _persist_trace(
                trace_id,
                trace_session_override or session_id,
                message,
                status=trace_status,
            )
        if evidence_setup:
            evidence_setup(session_id, trace_id or "")
        for chunk in chunks:
            yield chunk
        if raise_after:
            raise RuntimeError("deterministic stream interruption")

    return stream


def assert_evaluation_http_contract(client: TestClient) -> None:
    before_provider = provider_attempt_count()
    persistence_calls: list[Dict[str, Any]] = []
    original_create_run = evaluation_api.create_evaluation_run
    real_runtime = evaluation_api._run_real_chat

    def counting_create_run(*args: Any, **kwargs: Any):
        persistence_calls.append(dict(kwargs))
        return original_create_run(*args, **kwargs)

    evaluation_api.create_evaluation_run = counting_create_run
    try:
        passing_case = db.create_evaluation_case(
            case_key="S10H1-EVALUATION-PASS",
            title="S10-H.1 Evaluation terminal success",
            user_message="Give the truthful-flow customer-service answer.",
            expected_agent_id="customer_service",
            required_terms=["truthful-flow"],
            status="active",
            version_label="V1.8.2-S10-H.1",
        )
        success_trace_id = "s10h1-evaluation-pass-trace"
        success_entries: list[Dict[str, str]] = []
        chat_api.stream_chat_response = _evaluation_stream_stub(
            (
                _sse("delta", {"content": "partial answer"}),
                _sse(
                    "final",
                    {
                        "content": "truthful-flow final answer",
                        "current_agent": "Customer Service Agent",
                        "current_agent_id": "customer_service",
                    },
                ),
                _sse(
                    "done",
                    {
                        "status": "complete",
                        "trace_id": success_trace_id,
                        "current_agent": "Customer Service Agent",
                        "current_agent_id": "customer_service",
                        "route_intent": "customer_service",
                        "activated_skills": [],
                        "tool_calls": [],
                        "mcp_calls": [],
                        "citations": [],
                        "handoff": False,
                    },
                ),
            ),
            entries=success_entries,
            trace_id=success_trace_id,
        )
        before_calls = len(persistence_calls)
        success_provider_before = provider_attempt_count()
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
        check(
            "Evaluation success enters the SSE source exactly once",
            len(success_entries) == 1
            and success_entries[0]["user_id"] == "evaluation",
        )
        check(
            "Evaluation success uses the real runtime parser",
            evaluation_api._run_real_chat is real_runtime,
        )
        check(
            "Evaluation success creates zero Provider attempts",
            provider_attempt_count() == success_provider_before,
        )

        failure_specs: list[Dict[str, Any]] = []

        def add_failure(
            label: str,
            chunks: Iterable[str],
            *,
            expected_reason: str,
            trace_id: str | None = None,
            trace_status: str = "complete",
            trace_session_override: str | None = None,
            expected_trace_id: str | None = None,
            raise_after: bool = False,
        ) -> None:
            failure_specs.append(
                {
                    "label": label,
                    "chunks": tuple(chunks),
                    "expected_reason": expected_reason,
                    "trace_id": trace_id,
                    "trace_status": trace_status,
                    "trace_session_override": trace_session_override,
                    "expected_trace_id": expected_trace_id,
                    "raise_after": raise_after,
                }
            )

        trace = "s10h1-error-without-field"
        add_failure(
            "error-event-without-error-field",
            (
                _sse("final", {"content": "answer", "trace_id": trace}),
                _sse("error", {"trace_id": trace}),
                _sse("done", {"status": "complete", "trace_id": trace}),
            ),
            expected_reason="evaluation_error_event",
            trace_id=trace,
            expected_trace_id=trace,
        )
        trace = "s10h1-malformed-error"
        add_failure(
            "malformed-error-event",
            (
                _sse("final", {"content": "answer", "trace_id": trace}),
                _raw_sse("error", "{malformed-error-json"),
                _sse("done", {"status": "complete", "trace_id": trace}),
            ),
            expected_reason="evaluation_error_event_malformed",
            trace_id=trace,
            expected_trace_id=trace,
        )
        trace = "s10h1-missing-done"
        add_failure(
            "missing-done",
            (_sse("delta", {"content": "answer", "trace_id": trace}),),
            expected_reason="evaluation_done_missing",
            trace_id=trace,
            expected_trace_id=trace,
        )
        trace = "s10h1-duplicate-done"
        add_failure(
            "duplicate-done",
            (
                _sse("delta", {"content": "answer"}),
                _sse("done", {"status": "complete", "trace_id": trace}),
                _sse("done", {"status": "complete", "trace_id": trace}),
            ),
            expected_reason="evaluation_multiple_done_events",
            trace_id=trace,
            expected_trace_id=trace,
        )
        trace = "s10h1-malformed-done"
        add_failure(
            "malformed-done-json",
            (
                _sse("final", {"content": "answer", "trace_id": trace}),
                _raw_sse("done", "{malformed-done-json"),
            ),
            expected_reason="evaluation_done_malformed",
            trace_id=trace,
            expected_trace_id=trace,
        )
        trace = "s10h1-done-list"
        add_failure(
            "done-json-list",
            (
                _sse("final", {"content": "answer", "trace_id": trace}),
                _raw_sse("done", "[]"),
            ),
            expected_reason="evaluation_done_malformed",
            trace_id=trace,
            expected_trace_id=trace,
        )
        trace = "s10h1-done-scalar"
        add_failure(
            "done-json-scalar",
            (
                _sse("final", {"content": "answer", "trace_id": trace}),
                _raw_sse("done", '"complete"'),
            ),
            expected_reason="evaluation_done_malformed",
            trace_id=trace,
            expected_trace_id=trace,
        )
        for label, status_payload in (
            ("done-missing-status", {}),
            ("done-unknown-status", {"status": "unknown"}),
            ("done-failed-status", {"status": "failed"}),
            ("done-running-status", {"status": "running"}),
            ("done-created-status", {"status": "created"}),
        ):
            trace = f"s10h1-{label}"
            add_failure(
                label,
                (
                    _sse("delta", {"content": "answer"}),
                    _sse("done", {**status_payload, "trace_id": trace}),
                ),
                expected_reason=(
                    "evaluation_done_status_missing"
                    if label == "done-missing-status"
                    else "evaluation_done_not_complete"
                ),
                trace_id=trace,
                expected_trace_id=trace,
            )
        add_failure(
            "empty-object-done-missing-status",
            (
                _sse("delta", {"content": "answer"}),
                _sse("done", {}),
            ),
            expected_reason="evaluation_done_status_missing",
        )
        trace = "s10h1-final-only"
        add_failure(
            "final-cannot-replace-done",
            (_sse("final", {"content": "answer", "trace_id": trace}),),
            expected_reason="evaluation_done_missing",
            trace_id=trace,
            expected_trace_id=trace,
        )
        trace = "s10h1-message-id-only"
        add_failure(
            "message-id-cannot-replace-done",
            (
                _sse(
                    "final",
                    {
                        "content": "answer",
                        "message_id": 12345,
                        "trace_id": trace,
                    },
                ),
            ),
            expected_reason="evaluation_done_missing",
            trace_id=trace,
            expected_trace_id=trace,
        )
        trace = "s10h1-missing-answer"
        add_failure(
            "missing-answer",
            (_sse("done", {"status": "complete", "trace_id": trace}),),
            expected_reason="evaluation_answer_missing",
            trace_id=trace,
            expected_trace_id=trace,
        )
        add_failure(
            "missing-trace-id",
            (
                _sse("delta", {"content": "answer"}),
                _sse("done", {"status": "complete"}),
            ),
            expected_reason="evaluation_trace_missing",
        )
        trace = "s10h1-trace-not-found"
        add_failure(
            "trace-not-persisted",
            (
                _sse("delta", {"content": "answer"}),
                _sse("done", {"status": "complete", "trace_id": trace}),
            ),
            expected_reason="evaluation_trace_not_persisted",
            expected_trace_id=trace,
        )
        trace = "s10h1-trace-failed"
        add_failure(
            "trace-status-failed",
            (
                _sse("delta", {"content": "answer"}),
                _sse("done", {"status": "complete", "trace_id": trace}),
            ),
            expected_reason="evaluation_trace_not_complete",
            trace_id=trace,
            trace_status="failed",
            expected_trace_id=trace,
        )
        trace = "s10h1-trace-other-session"
        add_failure(
            "trace-session-mismatch",
            (
                _sse("delta", {"content": "answer"}),
                _sse("done", {"status": "complete", "trace_id": trace}),
            ),
            expected_reason="evaluation_trace_session_mismatch",
            trace_id=trace,
            trace_session_override="s10h1-unrelated-session",
            expected_trace_id=trace,
        )
        trace = "s10h1-error-then-complete"
        add_failure(
            "error-then-complete-done",
            (
                _sse("delta", {"content": "answer"}),
                _sse(
                    "error",
                    {"error": "deterministic failure", "trace_id": trace},
                ),
                _sse("done", {"status": "complete", "trace_id": trace}),
            ),
            expected_reason="evaluation_error_event",
            trace_id=trace,
            expected_trace_id=trace,
        )
        trace = "s10h1-stream-interrupted"
        add_failure(
            "stream-interrupted-after-real-evidence",
            (_sse("final", {"content": "answer", "trace_id": trace}),),
            expected_reason="evaluation_stream_interrupted",
            trace_id=trace,
            expected_trace_id=trace,
            raise_after=True,
        )
        trace = "s10h1-done-before-late-final"
        add_failure(
            "event-after-complete-done",
            (
                _sse("done", {"status": "complete", "trace_id": trace}),
                _sse(
                    "final",
                    {
                        "content": "late answer must not rescue the run",
                        "trace_id": "s10h1-late-final-other-trace",
                    },
                ),
            ),
            expected_reason="evaluation_event_after_done",
            trace_id=trace,
            expected_trace_id=trace,
        )
        for label, trailing_field in (
            (
                "data-only-after-complete-done",
                'data: {"status":"failed"}\n\n',
            ),
            ("id-after-complete-done", "id: late-event-id\n\n"),
            ("retry-after-complete-done", "retry: 1000\n\n"),
            ("unknown-field-after-complete-done", "x-runtime: late\n\n"),
        ):
            trace = f"s10h2-{label}"
            add_failure(
                label,
                (
                    _sse("delta", {"content": "answer"}),
                    _sse(
                        "done",
                        {"status": "complete", "trace_id": trace},
                    ),
                    trailing_field,
                ),
                expected_reason="evaluation_event_after_done",
                trace_id=trace,
                expected_trace_id=trace,
            )

        check(
            "Evaluation strict-terminal matrix covers at least 27 failures",
            len(failure_specs) >= 27,
        )
        for index, spec in enumerate(failure_specs, start=1):
            label = str(spec["label"])
            failing_case = db.create_evaluation_case(
                case_key=f"S10H1-FAIL-{index:02d}",
                title=f"S10-H.1 {label}",
                user_message=f"deterministic Evaluation failure {label}",
                status="active",
                version_label="V1.8.2-S10-H.1",
            )
            linked_badcase = _new_retest_case(
                f"evaluation-{label}",
                last_applied_at="2000-01-01 00:00",
                linked_evaluation_case_id=int(failing_case["id"]),
            )
            entries: list[Dict[str, str]] = []
            chat_api.stream_chat_response = _evaluation_stream_stub(
                spec["chunks"],
                entries=entries,
                trace_id=spec.get("trace_id"),
                trace_status=str(spec.get("trace_status") or "complete"),
                trace_session_override=spec.get("trace_session_override"),
                raise_after=bool(spec.get("raise_after")),
            )
            calls_before = len(persistence_calls)
            provider_before = provider_attempt_count()
            failed_response = client.post(
                f"/api/evaluations/cases/{failing_case['id']}/run",
                json={"linked_badcase_id": int(linked_badcase["id"])},
            )
            check(
                f"{label}: Evaluation failure is controlled HTTP 200",
                failed_response.status_code == 200,
            )
            failed_payload = failed_response.json()
            runs = db.list_evaluation_runs(int(failing_case["id"]))
            check(
                f"{label}: SSE source is entered exactly once",
                len(entries) == 1 and entries[0]["user_id"] == "evaluation",
            )
            check(
                f"{label}: exactly one failed Run and zero passed Runs",
                len(runs) == 1
                and runs[0]["status"] == "failed"
                and sum(item["status"] == "passed" for item in runs) == 0
                and len(persistence_calls) == calls_before + 1
                and failed_payload["run"]["id"] == runs[0]["id"],
            )
            check(
                f"{label}: failure has a non-empty explainable reason",
                str((runs[0].get("evidence") or {}).get("runtime_error") or "")
                .startswith(str(spec["expected_reason"])),
            )
            expected_trace_id = spec.get("expected_trace_id")
            if expected_trace_id:
                check(
                    f"{label}: obtainable real Trace evidence is retained",
                    runs[0].get("trace_id") == expected_trace_id,
                )
            check(
                f"{label}: failure creates zero Provider attempts",
                provider_attempt_count() == provider_before,
            )
            detail = client.get(f"/api/badcases/{linked_badcase['id']}")
            check(
                f"{label}: linked Badcase remains readable",
                detail.status_code == 200,
            )
            linked = detail.json()["badcase"]
            check(
                f"{label}: linked Badcase never exposes verify-pass",
                linked.get("status") != "released"
                and (linked.get("retest_context") or {}).get("run_status")
                == "failed"
                and "verify-pass" not in linked.get("allowed_actions", []),
            )
            verify = client.post(
                f"/api/badcases/{linked_badcase['id']}/verify",
                json={"passed": True, "note": f"must block {label}"},
            )
            check(
                f"{label}: linked Badcase cannot be verified as passed",
                verify.status_code != 200,
            )
            check(
                f"{label}: real Evaluation runtime parser remains installed",
                evaluation_api._run_real_chat is real_runtime,
            )
        check(
            "Evaluation success and strict failure matrix create zero Provider attempts",
            provider_attempt_count() == before_provider,
        )
    finally:
        evaluation_api.create_evaluation_run = original_create_run


def assert_evaluation_runtime_status_contract(client: TestClient) -> None:
    """Exercise complete/paused/completed through the real parser and route."""

    real_runtime = evaluation_api._run_real_chat
    initial_provider_attempts = provider_attempt_count()
    no_expectation = object()

    def make_case(label: str, expected: object = no_expectation) -> Dict[str, Any]:
        rubric: Dict[str, Any] = {}
        if expected is not no_expectation:
            rubric = {
                "deterministic_assertions": {
                    "expected_runtime_status": expected,
                }
            }
        return db.create_evaluation_case(
            case_key=f"S10H2-{label.upper()}",
            title=f"S10-H.2 runtime status {label}",
            user_message=f"deterministic runtime status question {label}",
            rubric=rubric,
            status="active",
            version_label="V1.8.2-S10-H.2",
        )

    def controlled_setup(
        proposal_id: str,
        *,
        committed: bool,
        receipt_id: str | None = None,
        resource_id: str | None = None,
        gateway_phase: str | None = None,
        proposal_trace_id: str | None = None,
    ) -> Callable[[str, str], None]:
        def setup(session_id: str, trace_id: str) -> None:
            idempotency_key = f"idem-{proposal_id}"
            source_trace_id = proposal_trace_id or trace_id
            if source_trace_id != trace_id:
                _persist_trace(
                    source_trace_id,
                    session_id,
                    f"prior proposal fixture {proposal_id}",
                    status="complete",
                )
            db.create_action_proposal(
                proposal_id,
                session_id,
                "work_order.create",
                "L2",
                {"description": f"fixture {proposal_id}"},
                idempotency_key,
                trace_id=source_trace_id,
                release_id="rr-test-s10h2",
            )
            if committed:
                db.record_action_approval(
                    proposal_id,
                    "approved",
                    "owner:s10h2",
                    "deterministic fixture",
                )
                db.save_action_receipt(
                    receipt_id or f"receipt-{proposal_id}",
                    proposal_id,
                    idempotency_key,
                    "committed",
                    result={"success": True},
                    resource_type="work_order",
                    resource_id=resource_id or f"WO-{proposal_id}",
                )
            if gateway_phase:
                if gateway_phase not in {"awaiting_confirmation", "committed"}:
                    raise AssertionError(f"unsupported gateway fixture phase: {gateway_phase}")
                if (gateway_phase == "committed") != committed:
                    raise AssertionError("gateway fixture phase must match Receipt state")
                db.record_trace_event(
                    trace_id,
                    "action_gateway",
                    "success",
                    metadata={
                        "proposal_id": proposal_id,
                        "receipt_id": (
                            receipt_id or f"receipt-{proposal_id}"
                            if committed else None
                        ),
                        "resource_id": (
                            resource_id or f"WO-{proposal_id}"
                            if committed else None
                        ),
                        "workflow_status": (
                            "completed"
                            if gateway_phase == "committed"
                            else "paused"
                        ),
                    },
                )

        return setup

    def execute(
        label: str,
        case: Dict[str, Any],
        chunks: Iterable[str],
        trace_id: str,
        *,
        expected_run_status: str,
        expected_runtime_status: str,
        evidence_setup: Callable[[str, str], None] | None = None,
        linked_failure: bool = False,
    ) -> Dict[str, Any]:
        linked_badcase = None
        body: Dict[str, Any] = {}
        if linked_failure:
            linked_badcase = _new_retest_case(
                f"runtime-status-{label}",
                last_applied_at="2000-01-01 00:00",
                linked_evaluation_case_id=int(case["id"]),
            )
            body["linked_badcase_id"] = int(linked_badcase["id"])
        entries: list[Dict[str, str]] = []
        before_provider = provider_attempt_count()
        chat_api.stream_chat_response = _evaluation_stream_stub(
            chunks,
            entries=entries,
            trace_id=trace_id,
            evidence_setup=evidence_setup,
        )
        response = client.post(
            f"/api/evaluations/cases/{case['id']}/run",
            json=body,
        )
        check(f"{label}: Evaluation HTTP is controlled 200", response.status_code == 200)
        payload = response.json()
        runs = db.list_evaluation_runs(int(case["id"]))
        check(
            f"{label}: exactly one Run is persisted",
            len(runs) == 1 and payload["run"]["id"] == runs[0]["id"],
        )
        check(
            f"{label}: expected Run status is truthful",
            runs[0]["status"] == expected_run_status
            and sum(item["status"] == "passed" for item in runs)
            == (1 if expected_run_status == "passed" else 0),
        )
        runtime_rules = [
            item for item in (runs[0].get("rule_results") or [])
            if item.get("key") == "runtime_status"
        ]
        check(
            f"{label}: runtime_status rule is visible with raw actual",
            len(runtime_rules) == 1
            and runtime_rules[0].get("actual") == expected_runtime_status,
        )
        check(
            f"{label}: real parser and lower SSE source are each used once",
            evaluation_api._run_real_chat is real_runtime
            and len(entries) == 1
            and entries[0]["user_id"] == "evaluation",
        )
        check(
            f"{label}: Provider attempt delta is zero",
            provider_attempt_count() == before_provider,
        )
        if linked_badcase:
            detail = client.get(f"/api/badcases/{linked_badcase['id']}")
            check(f"{label}: linked Badcase remains readable", detail.status_code == 200)
            badcase = detail.json()["badcase"]
            check(
                f"{label}: linked Badcase does not expose verify-pass",
                (badcase.get("retest_context") or {}).get("run_status") == "failed"
                and "verify-pass" not in badcase.get("allowed_actions", []),
            )
            verify = client.post(
                f"/api/badcases/{linked_badcase['id']}/verify",
                json={"passed": True, "note": f"must block {label}"},
            )
            check(f"{label}: verify-pass endpoint remains blocked", verify.status_code != 200)
        return payload

    completed_case = make_case("completed-pass", "completed")
    completed_trace = "s10h2-completed-pass-trace"
    completed_proposal = "proposal-s10h2-completed-pass"
    completed_receipt = "receipt-s10h2-completed-pass"
    completed_payload = execute(
        "completed-with-committed-receipt",
        completed_case,
        (
            _sse("delta", {"content": "controlled write completed"}),
            _sse(
                "done",
                {
                    "status": "completed",
                    "trace_id": completed_trace,
                    "runtime_path": "controlled_action",
                    "proposal_id": completed_proposal,
                    "action_receipts": [
                        {
                            "receipt_id": completed_receipt,
                            "proposal_id": completed_proposal,
                            "status": "committed",
                            "resource_id": "WO-S10H2-COMPLETED",
                        }
                    ],
                    "tool_calls": [
                        {
                            "tool_name": "action_gateway",
                            "arguments": {
                                "proposal_id": completed_proposal,
                                "phase": "committed",
                            },
                        }
                    ],
                },
            ),
        ),
        completed_trace,
        expected_run_status="passed",
        expected_runtime_status="completed",
        evidence_setup=controlled_setup(
            completed_proposal,
            committed=True,
            receipt_id=completed_receipt,
            resource_id="WO-S10H2-COMPLETED",
            gateway_phase="committed",
            proposal_trace_id="s10h2-completed-prior-proposal-trace",
        ),
    )
    completed_rules = completed_payload["rule_results"]
    check(
        "completed requires and exposes a committed persisted Receipt",
        any(
            item.get("key") == "completed_receipt_evidence"
            and item.get("status") == "pass"
            for item in completed_rules
        )
        and completed_payload["run"]["evidence"]["runtime_status"] == "completed"
        and completed_payload["run"]["evidence"]["controlled_action_evidence"]["receipt"]["resource_id"]
        == "WO-S10H2-COMPLETED"
        and completed_payload["run"]["evidence"]["controlled_action_evidence"]["proposal"]["trace_matches"]
        is False,
    )

    missing_receipt_case = make_case("completed-missing-receipt", "completed")
    missing_receipt_trace = "s10h2-completed-missing-receipt-trace"
    missing_receipt_proposal = "proposal-s10h2-completed-missing-receipt"
    missing_receipt_payload = execute(
        "completed-without-persisted-receipt",
        missing_receipt_case,
        (
            _sse("delta", {"content": "claimed completed without receipt"}),
            _sse(
                "done",
                {
                    "status": "completed",
                    "trace_id": missing_receipt_trace,
                    "runtime_path": "controlled_action",
                    "proposal_id": missing_receipt_proposal,
                    "action_receipts": [
                        {
                            "receipt_id": "forged-payload-only",
                            "status": "committed",
                            "resource_id": "FORGED-RESOURCE",
                        }
                    ],
                },
            ),
        ),
        missing_receipt_trace,
        expected_run_status="failed",
        expected_runtime_status="completed",
        evidence_setup=controlled_setup(
            missing_receipt_proposal,
            committed=False,
        ),
        linked_failure=True,
    )
    check(
        "completed payload cannot forge a successful Receipt",
        any(
            item.get("key") == "completed_receipt_evidence"
            and item.get("status") == "fail"
            for item in missing_receipt_payload["rule_results"]
        )
        and missing_receipt_payload["run"]["evidence"]["controlled_action_evidence"]["receipt"] is None,
    )

    missing_commit_event_case = make_case(
        "completed-missing-current-commit-event",
        "completed",
    )
    missing_commit_event_trace = "s10h2-completed-missing-commit-event-trace"
    missing_commit_event_proposal = "proposal-s10h2-completed-missing-commit-event"
    missing_commit_event_payload = execute(
        "completed-with-receipt-but-no-current-commit-event",
        missing_commit_event_case,
        (
            _sse("delta", {"content": "claimed completion from an old receipt"}),
            _sse(
                "done",
                {
                    "status": "completed",
                    "trace_id": missing_commit_event_trace,
                    "runtime_path": "controlled_action",
                    "proposal_id": missing_commit_event_proposal,
                },
            ),
        ),
        missing_commit_event_trace,
        expected_run_status="failed",
        expected_runtime_status="completed",
        evidence_setup=controlled_setup(
            missing_commit_event_proposal,
            committed=True,
            receipt_id="receipt-s10h2-missing-current-commit-event",
            resource_id="WO-S10H2-OLD-RECEIPT",
        ),
        linked_failure=True,
    )
    missing_commit_evidence = missing_commit_event_payload["run"]["evidence"][
        "controlled_action_evidence"
    ]
    check(
        "completed cannot reuse a persisted Receipt without current Trace commit evidence",
        missing_commit_evidence["receipt"]["status"] == "committed"
        and missing_commit_evidence["gateway_committed"] is False
        and any(
            item.get("key") == "completed_receipt_evidence"
            and item.get("status") == "fail"
            for item in missing_commit_event_payload["rule_results"]
        ),
    )

    paused_create = client.post(
        "/api/evaluations/cases",
        json={
            "case_key": "S10H2-PAUSED-PASS",
            "title": "S10-H.2 paused expected by operator",
            "user_message": "create a controlled draft and wait",
            "rubric": {
                "deterministic_assertions": {
                    "expected_runtime_status": "paused",
                }
            },
            "status": "active",
        },
    )
    check(
        "manual-case API stores expected_runtime_status in existing rubric",
        paused_create.status_code == 200
        and paused_create.json()["case"]["rubric"]["deterministic_assertions"]["expected_runtime_status"]
        == "paused",
    )
    paused_case = paused_create.json()["case"]
    paused_trace = "s10h2-paused-pass-trace"
    paused_proposal = "proposal-s10h2-paused-pass"
    paused_payload = execute(
        "paused-with-explicit-expectation-and-proposal",
        paused_case,
        (
            _sse("delta", {"content": "waiting for owner confirmation"}),
            _sse(
                "done",
                {
                    "status": "paused",
                    "trace_id": paused_trace,
                    "runtime_path": "controlled_action",
                    "proposal_id": paused_proposal,
                    "action_receipts": [],
                    "tool_calls": [
                        {
                            "tool_name": "action_gateway",
                            "arguments": {
                                "proposal_id": paused_proposal,
                                "phase": "awaiting_confirmation",
                            },
                        }
                    ],
                },
            ),
        ),
        paused_trace,
        expected_run_status="passed",
        expected_runtime_status="paused",
        evidence_setup=controlled_setup(
            paused_proposal,
            committed=False,
            gateway_phase="awaiting_confirmation",
        ),
    )
    check(
        "paused requires and exposes controlled waiting evidence",
        any(
            item.get("key") == "paused_controlled_action_evidence"
            and item.get("status") == "pass"
            for item in paused_payload["rule_results"]
        )
        and paused_payload["run"]["evidence"]["runtime_status"] == "paused",
    )

    for label, expected in (
        ("paused-with-default-complete", no_expectation),
        ("paused-while-completed-expected", "completed"),
    ):
        case = make_case(label, expected)
        trace = f"s10h2-{label}-trace"
        proposal = f"proposal-s10h2-{label}"
        payload = execute(
            label,
            case,
            (
                _sse("delta", {"content": "waiting for owner confirmation"}),
                _sse(
                    "done",
                    {
                        "status": "paused",
                        "trace_id": trace,
                        "runtime_path": "controlled_action",
                        "proposal_id": proposal,
                        "tool_calls": [
                            {
                                "tool_name": "action_gateway",
                                "arguments": {
                                    "proposal_id": proposal,
                                    "phase": "awaiting_confirmation",
                                },
                            }
                        ],
                    },
                ),
            ),
            trace,
            expected_run_status="failed",
            expected_runtime_status="paused",
            evidence_setup=controlled_setup(
                proposal,
                committed=False,
                gateway_phase="awaiting_confirmation",
            ),
            linked_failure=True,
        )
        check(
            f"{label}: runtime_status mismatch is the visible failure",
            any(
                item.get("key") == "runtime_status"
                and item.get("status") == "fail"
                for item in payload["rule_results"]
            ),
        )

    awaiting_parameters_case = make_case(
        "paused-awaiting-parameters-without-confirmation",
        "paused",
    )
    awaiting_parameters_trace = "s10h2-paused-awaiting-parameters-trace"

    def awaiting_parameters_setup(session_id: str, trace_id: str) -> None:
        del session_id
        db.record_trace_event(
            trace_id,
            "action_gateway",
            "success",
            metadata={
                "workflow_status": "paused",
                "proposal_id": None,
            },
        )

    awaiting_parameters_payload = execute(
        "paused-awaiting-parameters-is-not-confirmation",
        awaiting_parameters_case,
        (
            _sse("delta", {"content": "please provide missing parameters"}),
            _sse(
                "done",
                {
                    "status": "paused",
                    "trace_id": awaiting_parameters_trace,
                    "runtime_path": "controlled_action",
                    "tool_calls": [
                        {
                            "tool_name": "action_gateway",
                            "arguments": {"phase": "awaiting_parameters"},
                        }
                    ],
                },
            ),
        ),
        awaiting_parameters_trace,
        expected_run_status="failed",
        expected_runtime_status="paused",
        evidence_setup=awaiting_parameters_setup,
        linked_failure=True,
    )
    check(
        "paused awaiting_parameters without Proposal/Draft is not confirmation evidence",
        any(
            item.get("key") == "paused_controlled_action_evidence"
            and item.get("status") == "fail"
            for item in awaiting_parameters_payload["rule_results"]
        ),
    )

    comment_case = make_case("complete-with-transport-comments")
    comment_trace = "s10h2-complete-comments-trace"
    execute(
        "complete-with-only-post-done-comments",
        comment_case,
        (
            _sse("delta", {"content": "normal complete answer"}),
            _sse(
                "done",
                {"status": "complete", "trace_id": comment_trace},
            ),
            ": transport-flush\n\n",
            "\n: keep-alive\n\n",
        ),
        comment_trace,
        expected_run_status="passed",
        expected_runtime_status="complete",
    )

    invalid_expected = client.post(
        "/api/evaluations/cases",
        json={
            "case_key": "S10H2-INVALID-EXPECTED-STATUS",
            "title": "invalid expected runtime status",
            "user_message": "must be rejected without running",
            "rubric": {
                "deterministic_assertions": {
                    "expected_runtime_status": "unknown",
                }
            },
        },
    )
    check(
        "invalid expected_runtime_status is rejected at case configuration",
        invalid_expected.status_code == 400,
    )
    check(
        "runtime-status contract suite has zero Provider attempt delta",
        provider_attempt_count() == initial_provider_attempts,
    )


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
        required_terms=["truthful-flow"],
        status="active",
        version_label="V1.8.2-S10-H",
    )
    badcase = _new_retest_case(
        "evaluation-linked",
        last_applied_at="2000-01-01 00:00",
        linked_evaluation_case_id=int(case["id"]),
    )
    trace_id = "s10h-linked-evaluation-trace"
    entries: list[Dict[str, str]] = []
    chat_api.stream_chat_response = _evaluation_stream_stub(
        (
            _sse("final", {"content": "truthful-flow linked answer"}),
            _sse(
                "done",
                {
                    "status": "complete",
                    "trace_id": trace_id,
                    "current_agent": "Customer Service Agent",
                    "current_agent_id": "customer_service",
                    "route_intent": "customer_service",
                },
            ),
        ),
        entries=entries,
        trace_id=trace_id,
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
    check(
        "linked Evaluation retest enters only the lower SSE source once",
        len(entries) == 1 and entries[0]["user_id"] == "evaluation",
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


def assert_frontend_runtime_status_source_contract() -> None:
    """Supplement behavior tests with a narrow, explicitly static UI contract."""

    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    check(
        "frontend source contract has the expected-runtime-status select",
        "evaluation-case-runtime-status" in html,
    )
    check(
        "frontend source contract exposes the three business labels",
        all(
            label in html
            for label in (
                "正常完成回复（complete，默认）",
                "等待业主确认（paused）",
                "受控写入完成（completed）",
            )
        ),
    )
    check(
        "frontend source contract stores status in the existing rubric",
        "expected_runtime_status: $('#evaluation-case-runtime-status').value"
        in html,
    )
    check(
        "frontend source contract renders expected and actual runtime status",
        "runtimeStatusRule" in html
        and "ruleValueText(rule, 'expected')" in html
        and "ruleValueText(rule, 'actual')" in html,
    )


def main() -> None:
    original_evaluation_gate = evaluation_api._background_budget_gate
    original_badcase_gate = badcase_api._background_budget_gate
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
        assert_evaluation_runtime_status_contract(client)
        assert_successful_badcase_retest(client)
        assert_failed_badcase_retests(client)
        assert_retest_concurrent_apply_change(client)
        assert_retest_evidence_gate(client)
        assert_evaluation_linked_retest_compatibility(client)
        assert_frontend_runtime_status_source_contract()
        check(
            "S10-H full suite has zero Provider attempt delta",
            provider_attempt_count() == initial_provider_attempts,
        )
    finally:
        evaluation_api._background_budget_gate = original_evaluation_gate
        badcase_api._background_budget_gate = original_badcase_gate
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
