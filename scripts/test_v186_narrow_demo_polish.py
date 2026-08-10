"""Focused deterministic checks for the seven-item demo polish.

All data is symbolic and isolated.  The checks do not call a Provider, an HTTP
service, RAG, MCP, RuntimeRelease, or production business data.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST_DATA = tempfile.TemporaryDirectory(prefix="yiai-v186-narrow-polish-")
os.environ["PROPERTY_DATA_DIR"] = TEST_DATA.name
os.environ["DEEPSEEK_API_KEY"] = ""

from db import property_db  # noqa: E402

property_db.init_db()

import app.badcases as badcases  # noqa: E402
import app.observability as observability  # noqa: E402


def _symbolic_trace_events(loaded_chunks: int) -> list[dict]:
    return [
        {
            "span_name": "router",
            "metadata": {"visible_message_count": 3, "candidate_count": 2},
        },
        {
            "span_name": "agent_frozen",
            "metadata": {
                "bound_skill_ids": [13],
                "automatic_retry_count": 0,
            },
        },
        {
            "span_name": "skill.13.get_skill_instructions",
            "metadata": {"tool_name": "get_skill_instructions"},
        },
        {
            "span_name": "capability_decision",
            "metadata": {
                "decision_summary": {
                    "tool": {
                        "tool_calls": ["get_skill_instructions"]
                    }
                }
            },
        },
        {
            "span_name": "retrieval",
            "metadata": {
                "candidate_count": loaded_chunks + 2,
                "evidence": [
                    {"evidence_id": f"evidence-symbolic-{index}"}
                    for index in range(loaded_chunks)
                ],
                "loaded_character_count": loaded_chunks * 10,
            },
        },
        {
            "span_name": "final_response",
            "metadata": {
                "answer_status": "answered",
                "activated_skill_ids": [13],
                "automatic_retry_count": 0,
                "second_agent_request_count": 0,
                "citation_violations": [],
            },
        },
    ]


def test_trace_uses_only_persisted_rag_citations() -> None:
    calls = [
        {"stage": "router", "included_in_provider_summary": True},
        {"stage": "vertical_agent", "included_in_provider_summary": True},
        {"stage": "vertical_agent", "included_in_provider_summary": True},
        {"stage": "vertical_agent", "included_in_provider_summary": False},
    ]
    four_citations = {
        "ledger": {
            "citation_links": [
                {
                    "evidence_type": "rag_document_chunk",
                    "evidence_id": f"citation-symbolic-{index}",
                }
                for index in range(4)
            ]
        }
    }
    loaded_23 = observability._trace_cost_quality_control(
        _symbolic_trace_events(23), calls, [], four_citations
    )
    assert loaded_23["context_loading"]["rag_loaded_chunk_count"] == 23
    assert loaded_23["quality_evidence"]["rag_citation_used_count"] == 4
    assert loaded_23["call_reduction"] == {
        "router_requests": 1,
        "agent_requests": 2,
        "tool_follow_up_requests": 1,
        "selector_requests": 0,
        "resolver_requests": 0,
        "second_agent_requests": 0,
        "automatic_retries": 0,
        "other_provider_requests": 0,
    }
    assert loaded_23["context_loading"]["skill_used_count"] == 1
    assert loaded_23["context_loading"]["mcp_call_count"] == 0
    assert loaded_23["context_loading"]["business_tool_call_count"] == 0

    loaded_19 = observability._trace_cost_quality_control(
        _symbolic_trace_events(19), calls, [], {"ledger": {"citation_links": []}}
    )
    assert loaded_19["context_loading"]["rag_loaded_chunk_count"] == 19
    assert loaded_19["quality_evidence"]["rag_citation_used_count"] == 0

    historical = observability._trace_cost_quality_control(
        _symbolic_trace_events(7), calls, [], {"ledger": {}}
    )
    assert historical["context_loading"]["rag_loaded_chunk_count"] == 7
    assert historical["quality_evidence"]["rag_citation_used_count"] is None

    malformed = observability._trace_cost_quality_control(
        _symbolic_trace_events(7),
        calls,
        [],
        {
            "ledger": {
                "citation_links": [
                    {"evidence_type": "rag_document_chunk"}
                ]
            }
        },
    )
    assert malformed["quality_evidence"]["rag_citation_used_count"] is None

    capability_meta = {
        "decision_summary": {
            "tool": {
                "tool_calls": [
                    "get_skill_instructions",
                    "symbolic-business-tool",
                    "symbolic-mcp-tool",
                ]
            }
        }
    }
    skill_events = [
        {
            "span_name": "skill.13.get_skill_instructions",
            "metadata": {"tool_name": "get_skill_instructions"},
        }
    ]
    assert observability._business_tool_call_count(
        capability_meta,
        skill_events,
        [{"tool_name": "symbolic-mcp-tool"}],
    ) == 1


def test_darwin_budget_attention_is_only_advisory_when_disabled() -> None:
    original = badcases._background_budget_gate
    try:
        badcases._background_budget_gate = lambda _strategy: {
            "allowed": False,
            "budget_status": "unavailable",
            "alert_level": "warning",
            "reason_code": "budget_reconciliation_attention",
            "daily_threshold_cny": None,
            "monthly_threshold_cny": 0,
            "http_status": 503,
            "detail": {"code": "budget_status_unavailable"},
        }
        advisory = badcases._darwin_budget_gate()
        assert advisory["allowed"] is True
        assert advisory["warning_code"] == "budget_reconciliation_attention"

        badcases._background_budget_gate = lambda _strategy: {
            "allowed": False,
            "budget_status": "unavailable",
            "alert_level": "warning",
            "reason_code": "budget_reconciliation_attention",
            "daily_threshold_cny": 5,
            "monthly_threshold_cny": None,
            "http_status": 503,
            "detail": {"code": "budget_status_unavailable"},
        }
        assert badcases._darwin_budget_gate()["allowed"] is False

        badcases._background_budget_gate = lambda _strategy: {
            "allowed": False,
            "budget_status": "available",
            "alert_level": "blocked",
            "reason_code": None,
            "daily_threshold_cny": 5,
            "monthly_threshold_cny": 10,
            "http_status": 403,
            "detail": "symbolic budget limit",
        }
        assert badcases._darwin_budget_gate()["allowed"] is False

        badcases._background_budget_gate = lambda _strategy: {
            "allowed": False,
            "budget_status": "unavailable",
            "alert_level": "warning",
            "reason_code": "query_failed",
            "daily_threshold_cny": None,
            "monthly_threshold_cny": None,
            "http_status": 503,
            "detail": {"code": "budget_status_unavailable"},
        }
        assert badcases._darwin_budget_gate()["allowed"] is False
    finally:
        badcases._background_budget_gate = original


def test_pending_darwin_is_one_pro_suggestion_without_state_change() -> None:
    case = {
        "id": 42,
        "status": "pending",
        "title": "symbolic-title",
        "category": "other",
        "description": "symbolic-description",
        "feedback_reason": "symbolic-feedback",
        "original_query": "symbolic-query",
        "ai_response": "symbolic-answer",
        "context_json": "{}",
    }
    analysis = {
        "recommended_category": "other",
        "root_cause_hypothesis": "symbolic-root-cause",
        "repair_path_suggestion": "symbolic-repair-path",
        "suggested_actions": ["symbolic-action"],
        "root_cause_domain": "unknown",
    }
    llm_calls: list[dict] = []
    saved: dict = {}
    actions: list[tuple] = []

    async def fake_llm(_prompt: str, **kwargs):
        llm_calls.append(kwargs)
        return json.dumps(analysis), {"total_tokens": 11}

    def fake_update(case_id: int, **kwargs):
        saved.update(kwargs)
        return {**case, **kwargs, "id": case_id}

    patches = {
        "db_get_badcase": badcases.db_get_badcase,
        "find_darwin": badcases._find_darwin_skill,
        "start": badcases.start_darwin_operation,
        "gate": badcases._darwin_budget_gate,
        "llm": badcases._llm_generate,
        "attempts": badcases.get_provider_attempts_for_trace,
        "update": badcases.db_update_badcase,
        "action": badcases._record_action,
        "persist": badcases.persist_darwin_operation,
        "enrich": badcases._enrich_badcase,
    }
    badcases.db_get_badcase = lambda _case_id: dict(case)
    badcases._find_darwin_skill = lambda: {"name": "symbolic-darwin", "instructions": ""}
    badcases.start_darwin_operation = lambda **_kwargs: None
    badcases._darwin_budget_gate = lambda: {
        "allowed": True,
        "warning_code": "budget_reconciliation_attention",
    }
    badcases._llm_generate = fake_llm
    badcases.get_provider_attempts_for_trace = lambda _trace_id: [
        {
            "estimated_cost_cny": 0.0001,
            "cost_source": "platform_price_snapshot",
            "usage_source": "provider_actual",
            "total_tokens": 11,
        }
    ]
    badcases.db_update_badcase = fake_update
    badcases._record_action = lambda *args, **kwargs: actions.append((args, kwargs))
    badcases.persist_darwin_operation = lambda **_kwargs: None
    badcases._enrich_badcase = lambda value: value
    try:
        result = asyncio.run(
            badcases.darwin_fix(42, badcases.DarwinFixRequest())
        )
    finally:
        badcases.db_get_badcase = patches["db_get_badcase"]
        badcases._find_darwin_skill = patches["find_darwin"]
        badcases.start_darwin_operation = patches["start"]
        badcases._darwin_budget_gate = patches["gate"]
        badcases._llm_generate = patches["llm"]
        badcases.get_provider_attempts_for_trace = patches["attempts"]
        badcases.db_update_badcase = patches["update"]
        badcases._record_action = patches["action"]
        badcases.persist_darwin_operation = patches["persist"]
        badcases._enrich_badcase = patches["enrich"]

    assert len(llm_calls) == 1
    assert llm_calls[0]["model_id"] == "deepseek-v4-pro"
    assert llm_calls[0]["stage"] == "darwin"
    assert llm_calls[0]["model_selection_reason"] == (
        "仅低频Darwin深度分析，优先复杂分析质量，成本和耗时更高。"
    )
    persisted = json.loads(saved["darwin_analysis"])
    assert persisted["recommended_category"] == "other"
    assert persisted["root_cause_hypothesis"] == "symbolic-root-cause"
    assert persisted["repair_path_suggestion"] == "symbolic-repair-path"
    assert persisted["suggested_actions"] == ["symbolic-action"]
    assert set(saved) == {"darwin_analysis", "darwin_trace_id"}
    assert result["status_changed"] is False
    assert result["badcase"]["status"] == "pending"
    assert result["drafts"] == []
    assert result["budget_warning_code"] == "budget_reconciliation_attention"
    assert len(actions) == 1
    assert actions[0][0][1] == "ai-suggestion"
    assert actions[0][0][3:5] == ("pending", "pending")


def test_enabled_budget_blocks_before_darwin_provider() -> None:
    case = {
        "id": 43,
        "status": "pending",
        "title": "symbolic-title",
        "category": "other",
        "description": "symbolic-description",
        "context_json": "{}",
    }
    llm_call_count = 0

    async def forbidden_llm(*_args, **_kwargs):
        nonlocal llm_call_count
        llm_call_count += 1
        return "{}", {}

    patches = {
        "db_get_badcase": badcases.db_get_badcase,
        "find_darwin": badcases._find_darwin_skill,
        "start": badcases.start_darwin_operation,
        "gate": badcases._darwin_budget_gate,
        "llm": badcases._llm_generate,
        "persist": badcases.persist_darwin_operation,
    }
    badcases.db_get_badcase = lambda _case_id: dict(case)
    badcases._find_darwin_skill = lambda: None
    badcases.start_darwin_operation = lambda **_kwargs: None
    badcases._darwin_budget_gate = lambda: {
        "allowed": False,
        "budget_status": "available",
        "alert_level": "blocked",
        "reason": "symbolic budget limit",
        "http_status": 403,
        "detail": "symbolic budget limit",
    }
    badcases._llm_generate = forbidden_llm
    badcases.persist_darwin_operation = lambda **_kwargs: None
    try:
        try:
            asyncio.run(badcases.darwin_fix(43, badcases.DarwinFixRequest()))
            raise AssertionError("enabled budget limit did not block Darwin")
        except badcases.HTTPException as exc:
            assert exc.status_code == 403
    finally:
        badcases.db_get_badcase = patches["db_get_badcase"]
        badcases._find_darwin_skill = patches["find_darwin"]
        badcases.start_darwin_operation = patches["start"]
        badcases._darwin_budget_gate = patches["gate"]
        badcases._llm_generate = patches["llm"]
        badcases.persist_darwin_operation = patches["persist"]
    assert llm_call_count == 0


def test_frontend_narrow_contract() -> None:
    source = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    assert 'id="chat-handoff"' not in source
    assert "$('#chat-handoff').addEventListener" not in source
    assert 'id="chat-handoff-banner"' in source

    detail = source[
        source.index("async function renderBadcaseDetailPage") : source.index(
            "async function renderLegacyEvaluationsPage"
        )
    ]
    assert detail.count('id="badcase-ai-suggest"') == 1
    assert "生成AI处理建议（Darwin · Pro）" in detail
    assert "`/api/badcases/${id}/darwin`" in detail
    assert "badcase-darwin-suggest" not in detail
    assert 'data-badcase-trace-id="${escapeHtml(bc.trace_id)}"' in detail
    assert 'data-badcase-trace-id="${escapeHtml(bc.retest_trace_id)}"' in detail
    assert "openTraceDrawer(button.dataset.badcaseTraceId)" in detail

    trace_detail = source[
        source.index("async function showTraceDetailSlim") : source.index(
            "function renderGovernancePrinciples"
        )
    ]
    assert "同Agent Tool follow-up" not in trace_detail
    assert "同Agent能力续答" in trace_detail
    assert (
        "按需读取Skill或调用MCP/Tool后，同一个Agent需要再次基于结果回答，因此可能多一次Provider请求，不代表换Agent。"
        in trace_detail
    )
    assert "RAG装载" in trace_detail
    assert "回答实际采用" in trace_detail
    assert "mcp_call_count" in trace_detail
    assert "business_tool_call_count" in trace_detail
    assert "other_provider_requests" in trace_detail
    assert "业务Tool已选择" not in trace_detail
    assert (
        "普通高频业务，控制成本和响应时间；物业事实仍受实际Skill/RAG/MCP/Tool证据约束。"
        in source
    )
    assert (
        "仅低频Darwin深度分析，优先复杂分析质量，成本和耗时更高。"
        in source
    )


def main() -> None:
    tests = (
        test_trace_uses_only_persisted_rag_citations,
        test_darwin_budget_attention_is_only_advisory_when_disabled,
        test_pending_darwin_is_one_pro_suggestion_without_state_change,
        test_enabled_budget_blocks_before_darwin_provider,
        test_frontend_narrow_contract,
    )
    try:
        for test in tests:
            test()
            print(f"PASS {test.__name__}")
        print(f"Narrow demo polish: PASS ({len(tests)} checks; Provider calls: 0)")
    finally:
        TEST_DATA.cleanup()


if __name__ == "__main__":
    main()
