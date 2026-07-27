"""No-model deterministic contract checks for V1.8.2-S5-Fix1."""

from __future__ import annotations

import json
import os
import tempfile


def main() -> None:
    checks = []
    with tempfile.TemporaryDirectory(prefix="yiai-s5-fix1-") as data_dir:
        os.environ["PROPERTY_DATA_DIR"] = data_dir

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from db.property_db import (
            _get_conn,
            create_badcase,
            get_badcase,
            get_chat_trace,
            get_evidence_ledger,
            get_model_calls_for_trace,
            get_skill_prompt_draft,
            init_db,
            list_skill_prompt_drafts,
        )

        init_db()
        import app.badcases as badcases
        from app.runtime.darwin_evidence import persist_darwin_operation

        badcases._check_budget = lambda _strategy: {"alert_level": "none"}
        badcases._find_darwin_skill = lambda: None

        provider_model = "deepseek-v4-pro-provider"
        badcases.get_enabled_price_for_model = lambda model_id: {
            "model_id": model_id,
            "currency": "CNY",
            "effective_date": "2026-07-27",
            "input_price_per_1m": 3.0,
            "cached_input_price_per_1m": 0.025,
            "output_price_per_1m": 6.0,
            "reasoning_price_per_1m": None,
            "source_note": "deterministic fixture",
        }

        analysis = {
            "phenomenon_impact": "fixture impact",
            "root_cause_hypothesis": "fixture root cause",
            "root_cause_domain": "model_instruction",
            "evidence_uncertainties": "fixture only",
            "repair_path_suggestion": "skill_prompt",
            "recommended_category": "other",
            "expected_impact": "fixture improvement",
            "risks": "human review required",
            "suggested_actions": ["review fixture draft"],
            "drafts": [
                {
                    "type": "skill_prompt",
                    "title": "fixture draft",
                    "skill_name": "fixture skill",
                    "prompt_content": "fixture content",
                    "trigger_keywords": "fixture",
                }
            ],
        }
        usage = {
            "provider_response_model": provider_model,
            "provider_request_id": "fixture-request-id",
            "thinking_enabled": True,
            "input_cache_hit_tokens": 2688,
            "input_cache_miss_tokens": 24,
            "output_tokens": 1958,
            "total_tokens": 4670,
        }

        async def provider_success(_prompt, model_id):
            assert model_id == "deepseek-v4-pro"
            return json.dumps(analysis, ensure_ascii=False), dict(usage)

        badcases._llm_generate = provider_success
        case = create_badcase(
            title="fixture success",
            description="fixture",
            category="other",
            status="classified",
            source="manual",
            original_query="fixture query",
            ai_response="fixture response",
        )
        app = FastAPI()
        app.include_router(badcases.router, prefix="/api/badcases")
        client = TestClient(app)
        response = client.post(f"/api/badcases/{case['id']}/darwin-fix", json={})
        assert response.status_code == 200, response.text
        checks.append("1 Darwin HTTP 200")

        payload = response.json()
        trace_id = payload["darwin_trace_id"]
        trace = get_chat_trace(trace_id)
        calls = get_model_calls_for_trace(trace_id)
        ledger_row = get_evidence_ledger(trace_id)
        ledger = ledger_row["ledger"]
        assert trace["status"] == "complete" and trace["run_type"] == "badcase_darwin"
        checks.append("2 top-level Trace complete")
        assert len(calls) == 1 and calls[0]["status"] == "success"
        checks.append("3 child model_call success")
        assert ledger_row["status"] == "complete"
        checks.append("4 Evidence Ledger exists")
        assert trace["trace_id"] == calls[0]["trace_id"] == ledger["trace_id"]
        checks.append("5 trace_id consistent")

        model_evidence = ledger["model_calls"][0]
        assert model_evidence["requested_model"] == "deepseek-v4-pro"
        assert model_evidence["provider_response_model"] == provider_model
        assert model_evidence["requested_model"] != model_evidence["provider_response_model"]
        checks.append("6 requested and Provider models separate")
        assert model_evidence["provider_usage"] == {
            "input_cache_hit_tokens": 2688,
            "input_cache_miss_tokens": 24,
            "output_tokens": 1958,
        }
        assert ledger["cost_entries"][0]["amount"] == 0.0118872
        checks.append("7 exact Usage and cost consistent")

        draft_link = ledger["badcase_links"][0]["drafts"][0]
        draft = get_skill_prompt_draft(draft_link["draft_id"])
        assert ledger["badcase_links"][0]["badcase_id"] == case["id"]
        assert draft["badcase_id"] == case["id"] and draft["status"] == "draft"
        checks.append("8 Badcase and draft linked")
        counts = ledger["badcase_links"][0]["side_effect_counts"]
        assert all(value == 0 for value in counts.values()), counts
        assert ledger["action_proposals"] == []
        assert ledger["action_receipts"] == []
        assert ledger["tool_invocations"] == []
        checks.append("9 ActionGateway Receipt MCP Handoff zero")

        persist_darwin_operation(
            trace_id=trace_id,
            badcase_id=case["id"],
            model_call=calls[0],
            operation_status="complete",
            started_at=trace["created_at"],
            completed_at=trace["updated_at"],
            drafts=[{"type": "skill_prompt", "draft": draft}],
            status_before="classified",
            status_after="fixing",
        )
        conn = _get_conn()
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM run_evidence_ledgers WHERE trace_id = ?",
            (trace_id,),
        ).fetchone()
        conn.close()
        assert row["count"] == 1
        checks.append("10 Evidence persistence idempotent")

        async def provider_failure(_prompt, model_id):
            raise RuntimeError("simulated provider failure")

        badcases._llm_generate = provider_failure
        failed_case = create_badcase(
            title="fixture failure",
            description="fixture",
            category="other",
            status="classified",
            source="manual",
            original_query="fixture query",
            ai_response="fixture response",
        )
        failed_response = client.post(
            f"/api/badcases/{failed_case['id']}/darwin-fix", json={}
        )
        assert failed_response.status_code == 502
        conn = _get_conn()
        failed_trace = dict(
            conn.execute(
                """
                SELECT * FROM chat_traces
                WHERE session_id LIKE ? ORDER BY rowid DESC LIMIT 1
                """,
                (f"badcase-darwin:{failed_case['id']}:%",),
            ).fetchone()
        )
        conn.close()
        assert failed_trace["status"] == "failed"
        checks.append("11 Provider failure marks top Trace failed")
        assert get_badcase(failed_case["id"])["status"] == "classified"
        assert list_skill_prompt_drafts(badcase_id=failed_case["id"]) == []
        assert get_evidence_ledger(failed_trace["trace_id"])["status"] == "failed"
        checks.append("12 Provider failure creates no success draft or complete state")

        badcases._llm_generate = provider_success
        original_persist = badcases.persist_darwin_operation

        def visible_evidence_failure(**_kwargs):
            raise RuntimeError("simulated evidence persistence failure")

        badcases.persist_darwin_operation = visible_evidence_failure
        persistence_case = create_badcase(
            title="fixture persistence failure",
            description="fixture",
            category="other",
            status="classified",
            source="manual",
            original_query="fixture query",
            ai_response="fixture response",
        )
        error_client = TestClient(app, raise_server_exceptions=False)
        persistence_response = error_client.post(
            f"/api/badcases/{persistence_case['id']}/darwin-fix", json={}
        )
        badcases.persist_darwin_operation = original_persist
        assert persistence_response.status_code == 500
        checks.append("13 Evidence persistence failure is visible")

    for item in checks:
        print(f"PASS {item}")
    print(f"PASS all={len(checks)} model_calls=simulated-only")


if __name__ == "__main__":
    main()
