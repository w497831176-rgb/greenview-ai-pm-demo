"""Deterministic checks for the three narrow V1.8.7 demo fixes.

The checks use symbolic records and mocks only.  They do not call a Provider,
an HTTP service, the production database, or any business workflow.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST_DATA = tempfile.TemporaryDirectory(prefix="yiai-v187-three-narrow-fixes-")
os.environ["PROPERTY_DATA_DIR"] = TEST_DATA.name
os.environ["DEEPSEEK_API_KEY"] = ""

from db import property_db  # noqa: E402

property_db.init_db()

import app.badcases as badcases  # noqa: E402
import app.evaluations as evaluations  # noqa: E402


def _valid_darwin_analysis() -> dict[str, Any]:
    return {
        "recommended_category": "other",
        "root_cause_hypothesis": "symbolic-root-cause",
        "repair_path_suggestion": "symbolic-repair-path",
        "suggested_actions": ["symbolic-action"],
        "model": "deepseek-v4-pro",
        "trace_id": "symbolic-existing-darwin-trace",
    }


def test_saved_darwin_suggestion_is_reused_without_provider_or_action() -> None:
    analysis = _valid_darwin_analysis()
    case = {
        "id": 697,
        "status": "pending",
        "title": "symbolic-title",
        "darwin_analysis": json.dumps(analysis),
        "darwin_trace_id": analysis["trace_id"],
    }
    calls = {"provider": 0, "action": 0, "operation": 0, "budget": 0}

    async def forbidden_provider(*_args, **_kwargs):
        calls["provider"] += 1
        raise AssertionError("saved suggestion must not call Provider")

    originals = {
        "get": badcases.db_get_badcase,
        "enrich": badcases._enrich_badcase,
        "provider": badcases._llm_generate,
        "action": badcases._record_action,
        "operation": badcases.start_darwin_operation,
        "budget": badcases._darwin_budget_gate,
    }
    badcases.db_get_badcase = lambda _case_id: dict(case)
    badcases._enrich_badcase = lambda value: value
    badcases._llm_generate = forbidden_provider
    badcases._record_action = lambda *_args, **_kwargs: calls.__setitem__(
        "action", calls["action"] + 1
    )
    badcases.start_darwin_operation = lambda **_kwargs: calls.__setitem__(
        "operation", calls["operation"] + 1
    )
    badcases._darwin_budget_gate = lambda: calls.__setitem__(
        "budget", calls["budget"] + 1
    )
    try:
        result = asyncio.run(
            badcases.darwin_alias_frontend(697, badcases.DarwinFixRequest())
        )
    finally:
        badcases.db_get_badcase = originals["get"]
        badcases._enrich_badcase = originals["enrich"]
        badcases._llm_generate = originals["provider"]
        badcases._record_action = originals["action"]
        badcases.start_darwin_operation = originals["operation"]
        badcases._darwin_budget_gate = originals["budget"]

    assert calls == {"provider": 0, "action": 0, "operation": 0, "budget": 0}
    assert result["reused"] is True
    assert result["analysis"] == analysis
    assert result["badcase"]["status"] == "pending"
    assert result["status_changed"] is False


def test_missing_darwin_suggestion_keeps_original_mock_generation_path() -> None:
    case = {
        "id": 701,
        "status": "pending",
        "title": "symbolic-title",
        "category": "other",
        "description": "symbolic-description",
        "feedback_reason": "symbolic-feedback",
        "original_query": "symbolic-query",
        "ai_response": "symbolic-answer",
        "context_json": "{}",
    }
    generated = _valid_darwin_analysis()
    generated.pop("model")
    generated.pop("trace_id")
    calls = {"provider": 0, "action": 0}
    saved: dict[str, Any] = {}

    async def fake_provider(_prompt: str, **kwargs):
        calls["provider"] += 1
        assert kwargs["model_id"] == "deepseek-v4-pro"
        assert kwargs["stage"] == "darwin"
        return json.dumps(generated), {}

    def fake_update(case_id: int, **kwargs):
        saved.update(kwargs)
        return {**case, **kwargs, "id": case_id}

    originals = {
        "get": badcases.db_get_badcase,
        "find": badcases._find_darwin_skill,
        "start": badcases.start_darwin_operation,
        "budget": badcases._darwin_budget_gate,
        "provider": badcases._llm_generate,
        "attempts": badcases.get_provider_attempts_for_trace,
        "update": badcases.db_update_badcase,
        "action": badcases._record_action,
        "persist": badcases.persist_darwin_operation,
        "enrich": badcases._enrich_badcase,
    }
    badcases.db_get_badcase = lambda _case_id: dict(case)
    badcases._find_darwin_skill = lambda: None
    badcases.start_darwin_operation = lambda **_kwargs: None
    badcases._darwin_budget_gate = lambda: {"allowed": True}
    badcases._llm_generate = fake_provider
    badcases.get_provider_attempts_for_trace = lambda _trace_id: []
    badcases.db_update_badcase = fake_update
    badcases._record_action = lambda *_args, **_kwargs: calls.__setitem__(
        "action", calls["action"] + 1
    )
    badcases.persist_darwin_operation = lambda **_kwargs: None
    badcases._enrich_badcase = lambda value: value
    try:
        result = asyncio.run(badcases.darwin_fix(701, badcases.DarwinFixRequest()))
    finally:
        badcases.db_get_badcase = originals["get"]
        badcases._find_darwin_skill = originals["find"]
        badcases.start_darwin_operation = originals["start"]
        badcases._darwin_budget_gate = originals["budget"]
        badcases._llm_generate = originals["provider"]
        badcases.get_provider_attempts_for_trace = originals["attempts"]
        badcases.db_update_badcase = originals["update"]
        badcases._record_action = originals["action"]
        badcases.persist_darwin_operation = originals["persist"]
        badcases._enrich_badcase = originals["enrich"]

    assert calls == {"provider": 1, "action": 1}
    assert set(saved) == {"darwin_analysis", "darwin_trace_id"}
    assert result["status_changed"] is False
    assert result["badcase"]["status"] == "pending"


def _budget_result(
    *,
    reason_code: str | None,
    daily: float | None = None,
    monthly: float | None = None,
    allowed: bool = False,
) -> dict[str, Any]:
    return {
        "allowed": allowed,
        "reason_code": reason_code,
        "daily_threshold_cny": daily,
        "monthly_threshold_cny": monthly,
        "alert_level": "blocked" if not allowed else "normal",
        "http_status": 503 if not allowed else None,
        "detail": {"code": reason_code} if not allowed else None,
    }


def test_evaluation_budget_gate_only_relaxes_disabled_reconciliation_attention() -> None:
    original = evaluations._background_budget_gate
    try:
        evaluations._background_budget_gate = lambda _strategy: _budget_result(
            reason_code="budget_reconciliation_attention", daily=None, monthly=0
        )
        advisory = evaluations._evaluation_budget_gate()
        assert advisory["allowed"] is True
        assert advisory["alert_level"] == "warning"
        assert advisory["warning_code"] == "budget_reconciliation_attention"

        blocked_inputs = (
            _budget_result(
                reason_code="budget_reconciliation_attention", daily=1, monthly=None
            ),
            _budget_result(
                reason_code="budget_reconciliation_attention", daily=None, monthly=1
            ),
            _budget_result(reason_code="budget_threshold_exceeded", daily=1),
            _budget_result(reason_code="query_failed"),
            _budget_result(reason_code="data_quality_error"),
            _budget_result(reason_code="unknown_reason"),
        )
        for expected in blocked_inputs:
            evaluations._background_budget_gate = lambda _strategy, value=expected: value
            assert evaluations._evaluation_budget_gate() == expected
    finally:
        evaluations._background_budget_gate = original


def test_evaluation_run_entry_uses_local_gate_before_provider() -> None:
    provider_calls = 0

    async def forbidden_chat(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("blocked evaluation must not enter chat runtime")

    originals = {
        "get": evaluations.get_evaluation_case,
        "linked": evaluations._validate_linked_retest,
        "local_gate": evaluations._evaluation_budget_gate,
        "global_gate": evaluations._background_budget_gate,
        "chat": evaluations._run_real_chat,
    }
    evaluations.get_evaluation_case = lambda _case_id: {
        "id": 6,
        "case_key": "symbolic-case",
        "status": "active",
        "user_message": "symbolic-message",
    }
    evaluations._validate_linked_retest = lambda *_args, **_kwargs: None
    evaluations._evaluation_budget_gate = lambda: _budget_result(
        reason_code="query_failed"
    )
    evaluations._background_budget_gate = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("run_case bypassed its evaluation-local gate")
    )
    evaluations._run_real_chat = forbidden_chat
    try:
        try:
            asyncio.run(
                evaluations.run_case(6, evaluations.EvaluationRunRequest())
            )
            raise AssertionError("blocked evaluation unexpectedly ran")
        except evaluations.HTTPException as exc:
            assert exc.status_code == 503
    finally:
        evaluations.get_evaluation_case = originals["get"]
        evaluations._validate_linked_retest = originals["linked"]
        evaluations._evaluation_budget_gate = originals["local_gate"]
        evaluations._background_budget_gate = originals["global_gate"]
        evaluations._run_real_chat = originals["chat"]

    assert provider_calls == 0


def test_frontend_darwin_single_flight_contract() -> None:
    source = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    start = source.index("async function renderBadcaseDetailPage")
    end = source.index("async function renderLegacyEvaluationsPage", start)
    detail = source[start:end]

    guard = "if (darwinSuggestionInFlight || aiSuggest.disabled) return;"
    post = "await apiPost(`/api/badcases/${id}/darwin`, {});"
    assert guard in detail
    assert detail.index(guard) < detail.index("darwinSuggestionInFlight = true;")
    assert detail.index("darwinSuggestionInFlight = true;") < detail.index(post)
    assert detail.index("aiSuggest.disabled = true;") < detail.index(post)
    assert detail.count(post) == 1
    assert "生成中，请稍候（Darwin · Pro）…" in detail
    assert "${hasDarwinAdvice ? '' : '<button id=\"badcase-ai-suggest\"" in detail
    assert "const latest = await apiGet(`/api/badcases/${id}`);" in detail
    assert "hasDisplayableDarwinAdvice(latestBadcase.darwin_analysis_parsed || {})" in detail
    assert detail.index("const latest = await apiGet") < detail.index(
        "hasDisplayableDarwinAdvice(latestBadcase.darwin_analysis_parsed || {})"
    )
    assert "darwinSuggestionInFlight = false;" in detail


def test_frontend_citation_labels_execute_for_all_three_states() -> None:
    source = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    start = source.index("function renderCitations(citations, content = '')")
    end = source.index("const CITATION_SNAPSHOTS", start)
    function_source = source[start:end]
    harness = f"""
const getUsedCitationIndices = () => new Set();
const escapeHtml = value => String(value ?? '');
const registerCitationSnapshot = () => {{}};
{function_source}
const make = citation => renderCitations([{{
  doc_title: 'symbolic-document', used_in_answer: true, ...citation
}}], '');
const values = [
  make({{retrieval_score: 0.4567, retrieval_mode: 'symbolic-seed'}}),
  make({{retrieval_score: null, retrieval_mode: 'runtime_release_snapshot_adjacent'}}),
  make({{retrieval_score: null, retrieval_mode: 'symbolic-history'}})
];
process.stdout.write(JSON.stringify(values));
"""
    completed = subprocess.run(
        ["node", "-e", harness],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    numeric, adjacent, historical = json.loads(completed.stdout)
    assert "相关度 0.457" in numeric
    assert "相邻上下文（未单独评分）" in adjacent
    assert "相关度未记录" in historical
    assert "相关度 0.457" not in adjacent


def main() -> None:
    tests = (
        test_saved_darwin_suggestion_is_reused_without_provider_or_action,
        test_missing_darwin_suggestion_keeps_original_mock_generation_path,
        test_evaluation_budget_gate_only_relaxes_disabled_reconciliation_attention,
        test_evaluation_run_entry_uses_local_gate_before_provider,
        test_frontend_darwin_single_flight_contract,
        test_frontend_citation_labels_execute_for_all_three_states,
    )
    try:
        for test in tests:
            test()
            print(f"PASS {test.__name__}")
        print(f"Three narrow fixes: PASS ({len(tests)} checks; Provider calls: 0)")
    finally:
        TEST_DATA.cleanup()


if __name__ == "__main__":
    main()
