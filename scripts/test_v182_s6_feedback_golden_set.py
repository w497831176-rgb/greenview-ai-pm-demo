"""Deterministic V1.8.2-S6 feedback and Golden Set contract checks.

The script uses an isolated temporary SQLite database and mocked runtime
responses.  It never calls a model Provider or writes production data.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid


def main() -> None:
    temp_dir = tempfile.TemporaryDirectory(prefix="yiai-s6-")
    os.environ["PROPERTY_DATA_DIR"] = temp_dir.name

    from app import evaluations as evaluation_api
    from app.runtime.legacy_chat import FeedbackRequest, chat_feedback
    from db import property_db as db

    db.init_db()
    evaluation_api._check_budget = lambda _operation: {"alert_level": "ok"}

    def provider_attempt(stage: str, sequence: int) -> dict:
        request_id = f"fixture-s6-{stage}-{sequence}"
        core = {
            "stage": stage,
            "record_kind": "provider_attempt",
            "include_in_provider_aggregate": True,
            "provider_request_sent": True,
            "provider_request_id": request_id,
            "usage_status": "provider_actual",
        }
        return {**core, "usage_normalized": dict(core)}

    def assistant_message(session_id: str, trace_id: str, answer: str = "测试回答") -> dict:
        db.ensure_chat_session(session_id)
        db.save_chat_message(session_id, "user", "原始业主问题")
        return db.save_chat_message(
            session_id,
            "assistant",
            answer,
            trace_id=trace_id,
            current_agent="维修 Agent",
            current_agent_id="maintenance",
            status="success",
        )

    # 1-2. Like accepts no reason and never creates a Badcase.
    like_msg = assistant_message("s6-like", "trace-like")
    before_badcases = len(db.list_badcases())
    liked = asyncio.run(chat_feedback(FeedbackRequest(
        session_id="s6-like", message_id=like_msg["id"], type="thumb_up"
    )))
    assert liked["status"] == "ok" and liked["feedback"]["feedback_type"] == "thumb_up"
    assert liked["badcase"] is None and len(db.list_badcases()) == before_badcases

    # 3. Dislike persists the selected reason and creates exactly one Badcase.
    down_msg = assistant_message("s6-down", "trace-down", "存在事实错误的测试回答")
    disliked = asyncio.run(chat_feedback(FeedbackRequest(
        session_id="s6-down", message_id=down_msg["id"],
        type="thumb_down", reason="事实错误",
    )))
    assert disliked["badcase"]["feedback_reason"] == "事实错误"

    # 4. Missing negative reason receives an explicit, non-empty default.
    default_msg = assistant_message("s6-default", "trace-default")
    defaulted = asyncio.run(chat_feedback(FeedbackRequest(
        session_id="s6-default", message_id=default_msg["id"], type="thumb_down"
    )))
    assert defaulted["feedback"]["reason"] == "未提供具体原因"

    # 5. The Badcase retains original message/session/Trace and both texts.
    badcase = disliked["badcase"]
    assert badcase["session_id"] == "s6-down"
    assert badcase["message_id"] == down_msg["id"]
    assert badcase["trace_id"] == "trace-down"
    assert badcase["original_query"] == "原始业主问题"
    assert badcase["ai_response"] == "存在事实错误的测试回答"

    # 6. Repeating the same dislike is idempotent.
    repeated = asyncio.run(chat_feedback(FeedbackRequest(
        session_id="s6-down", message_id=down_msg["id"],
        type="thumb_down", reason="事实错误",
    )))
    assert repeated["already_recorded"] is True
    assert repeated["badcase"]["id"] == badcase["id"]
    assert len([
        item for item in db.list_badcases(source="user_feedback")
        if item.get("session_id") == "s6-down" and item.get("message_id") == down_msg["id"]
    ]) == 1

    async def mocked_success(message: str, session_id: str):
        trace_id = f"s6-test-{uuid.uuid4().hex[:12]}"
        db.ensure_chat_session(session_id)
        db.create_chat_trace(trace_id, session_id, message, run_type="evaluation")
        db.update_chat_trace(
            trace_id, status="complete", intent="maintenance",
            agent_name="维修 Agent", agent_id="maintenance",
        )
        db.record_model_call(
            trace_id=trace_id, stage="router", model_id="deepseek-v4-flash",
            status="success", total_tokens=10, usage_source="provider_actual",
            estimated_cost_cny=0.00001,
            local_attempt_id=f"fixture-local-{trace_id}-router",
            provider_request_id=f"fixture-request-{trace_id}-router",
            record_kind="provider_attempt", usage_status="provider_actual",
            cost_source="platform_price_snapshot",
            usage_normalized={
                **provider_attempt("router", 1),
                "local_attempt_id": f"fixture-local-{trace_id}-router",
                "provider_request_id": f"fixture-request-{trace_id}-router",
            },
        )
        db.record_model_call(
            trace_id=trace_id, stage="vertical_agent", model_id="deepseek-v4-flash",
            status="success", total_tokens=20, usage_source="provider_actual",
            estimated_cost_cny=0.00002,
            local_attempt_id=f"fixture-local-{trace_id}-agent",
            provider_request_id=f"fixture-request-{trace_id}-agent",
            record_kind="provider_attempt", usage_status="provider_actual",
            cost_source="platform_price_snapshot",
            usage_normalized={
                **provider_attempt("vertical_agent", 2),
                "local_attempt_id": f"fixture-local-{trace_id}-agent",
                "provider_request_id": f"fixture-request-{trace_id}-agent",
            },
        )
        db.save_evidence_ledger(
            trace_id, session_id,
            {"model_evidence": [{"stage": "router"}, {"stage": "vertical_agent"}]},
            status="complete", runtime_path="consultation",
        )
        answer = "5分钟内响应，30分钟内到场。工单 WO-20260714-001 当前已完成。"
        done = {
            "status": "complete", "trace_id": trace_id,
            "current_agent": "维修 Agent", "current_agent_id": "maintenance",
            "route_intent": "maintenance", "route_reason": "mocked",
            "activated_skills": [], "tool_calls": [], "mcp_calls": [],
            "citations": [{
                "doc_title": "物业维修服务承诺",
                "content_snapshot": "紧急维修5分钟内响应，30分钟内到场。",
            }],
            "handoff": False,
            "decision_summary": {
                "agent": {"status": "selected", "reason": "matched_intent", "agent_id": "maintenance"},
                "skill": {"status": "skipped", "reason": "no_match"},
                "rag": {"status": "selected", "reason": "knowledge_evidence_required", "evidence_decision": "accepted"},
                "tool": {"status": "skipped", "reason": "not_required"},
                "handoff": {"status": "skipped", "reason": "no_handoff_intent"},
            },
            "round_token_count": 30, "usage_source": "provider_actual",
        }
        return answer, done

    evaluation_api._run_real_chat = mocked_success

    # 7. A passing evaluation persists no Badcase.
    passing_case = db.create_evaluation_case(
        case_key="S6-TEST-PASS", title="S6 deterministic pass",
        user_message="维修承诺是什么", expected_agent_id="maintenance",
        expected_handoff=False, status="active", version_label="V1.8.2-S6",
    )
    passed = asyncio.run(evaluation_api.run_case(passing_case["id"]))
    assert passed["run"]["status"] == "passed"
    assert passed["run"]["badcase_id"] is None and passed["badcase"] is None

    # 8-10. Failure creates exactly one linked Badcase with case/run/trace,
    # failed assertions, answer and evidence; repeated persistence is idempotent.
    failing_case = db.create_evaluation_case(
        case_key="S6-TEST-FAIL", title="S6 deterministic fail",
        user_message="维修承诺是什么", expected_agent_id="customer_service",
        status="active", version_label="V1.8.2-S6",
    )
    failed = asyncio.run(evaluation_api.run_case(failing_case["id"]))
    failed_run = failed["run"]
    assert failed_run["status"] == "failed" and failed_run["badcase_id"]
    linked = db.get_badcase_by_evaluation_run(failed_run["id"])
    assert linked and linked["source"] == "evaluation"
    again = evaluation_api._ensure_badcase_for_run(failed_run["id"])
    assert again["id"] == linked["id"]
    assert len([
        item for item in db.list_badcases(source="evaluation")
        if item.get("linked_evaluation_run_id") == failed_run["id"]
    ]) == 1
    context = json.loads(linked["context_json"])
    assert context["evaluation_case"]["id"] == failing_case["id"]
    assert context["evaluation_run"]["id"] == failed_run["id"]
    assert context["evaluation_run"]["trace_id"] == failed_run["trace_id"]
    assert context["failed_assertions"] and context["answer"] and context["runtime_evidence"]

    # 11. The five Golden Set path contracts map to current runtime evidence.
    base_side_effects = {
        "business_writes": 0, "work_orders": 0, "work_order_drafts": 0,
        "action_proposals": 0, "action_receipts": 0,
    }
    five_contracts = [
        (
            {
                "expected_agent_id": "maintenance", "expected_citation_docs": ["物业维修服务承诺"],
                "required_terms": ["5分钟", "30分钟"], "expected_handoff": False,
                "rubric": {"deterministic_assertions": {
                    "expected_decision_summary": {
                        "rag": {"status": "selected", "evidence_decision": "accepted"},
                        "tool": {"status": "skipped"}, "handoff": {"status": "skipped"},
                    },
                    "citation_required_terms": ["5分钟", "30分钟"],
                    "forbid_business_side_effects": True, "expected_model_call_count": 2,
                    "expected_mcp_call_count": 0,
                }},
            },
            "5分钟内响应，30分钟内到场。",
            {
                "current_agent_id": "maintenance", "citations": [{"doc_title": "物业维修服务承诺", "content_snapshot": "5分钟响应，30分钟到场"}],
                "handoff": False, "side_effects": base_side_effects,
                "model_calls": [provider_attempt("router", 1), provider_attempt("vertical_agent", 2)],
                "decision_summary": {"rag": {"status": "selected", "evidence_decision": "accepted"}, "tool": {"status": "skipped"}, "handoff": {"status": "skipped"}},
            },
        ),
        (
            {
                "expected_agent_id": "customer_service", "expected_handoff": False,
                "rubric": {"deterministic_assertions": {
                    "expected_decision_summary": {"rag": {"status": "selected", "evidence_decision": "rejected_insufficient"}},
                    "require_knowledge_insufficient": True,
                    "forbid_business_side_effects": True, "expected_model_call_count": 1,
                    "expected_mcp_call_count": 0, "expected_citation_count": 0,
                }},
            },
            "当前知识依据不足，无法提供小区搬家服务细节。",
            {
                "current_agent_id": "customer_service", "handoff": False,
                "side_effects": base_side_effects, "model_calls": [provider_attempt("router", 1)],
                "decision_summary": {"rag": {"status": "selected", "evidence_decision": "rejected_insufficient"}},
            },
        ),
        (
            {
                "expected_agent_id": "maintenance", "expected_tools": ["get_my_work_order_by_id"],
                "required_terms": ["WO-20260714-001", "已完成"], "expected_handoff": False,
                "rubric": {"deterministic_assertions": {
                    "expected_decision_summary": {
                        "skill": {"status": "skipped", "reason": "structured_realtime_query"},
                        "rag": {"status": "skipped", "reason": "structured_realtime_query"},
                        "tool": {"status": "selected", "reason": "exact_workorder_lookup"},
                    },
                    "require_mcp_business_success": True,
                    "forbid_business_side_effects": True, "expected_model_call_count": 2,
                    "expected_mcp_call_count": 1, "expected_citation_count": 0,
                }},
            },
            "WO-20260714-001 当前已完成。",
            {
                "current_agent_id": "maintenance", "handoff": False,
                "mcp_calls": [{"server_name": "workorder-server", "tool_name": "get_my_work_order_by_id", "invocation_status": "success", "business_status": "success"}],
                "side_effects": base_side_effects,
                "model_calls": [provider_attempt("router", 1), provider_attempt("vertical_agent", 2)],
                "decision_summary": {
                    "skill": {"status": "skipped", "reason": "structured_realtime_query"},
                    "rag": {"status": "skipped", "reason": "structured_realtime_query"},
                    "tool": {"status": "selected", "reason": "exact_workorder_lookup"},
                },
            },
        ),
        (
            {
                "expected_agent_id": "human_copilot", "expected_handoff": True,
                "rubric": {"deterministic_assertions": {
                    "expected_decision_summary": {
                        "agent": {"status": "skipped", "reason": "handoff_preempted"},
                        "handoff": {"status": "selected", "reason": "owner_requested"},
                    },
                    "forbid_normal_business_answer": True,
                    "forbid_business_side_effects": True, "expected_model_call_count": 0,
                    "expected_mcp_call_count": 0, "expected_citation_count": 0,
                }},
            },
            "已转人工协同。",
            {
                "current_agent_id": "human_copilot", "handoff": True,
                "side_effects": base_side_effects, "model_calls": [],
                "decision_summary": {"agent": {"status": "skipped", "reason": "handoff_preempted"}, "handoff": {"status": "selected", "reason": "owner_requested"}},
            },
        ),
        (
            {
                "expected_agent_id": "human_copilot", "expected_handoff": True,
                "rubric": {"deterministic_assertions": {
                    "expected_decision_summary": {
                        "agent": {"status": "skipped", "reason": "handoff_preempted"},
                        "handoff": {"status": "selected", "reason": "safety_risk", "matched_signals": ["safety_override"]},
                    },
                    "forbid_normal_business_answer": True,
                    "forbid_business_side_effects": True, "expected_model_call_count": 0,
                    "expected_mcp_call_count": 0, "expected_citation_count": 0,
                }},
            },
            "安全风险优先，已转人工协同。",
            {
                "current_agent_id": "human_copilot", "handoff": True,
                "side_effects": base_side_effects, "model_calls": [],
                "decision_summary": {"agent": {"status": "skipped", "reason": "handoff_preempted"}, "handoff": {"status": "selected", "reason": "safety_risk", "matched_signals": ["燃气泄漏", "safety_override"]}},
            },
        ),
    ]
    for case, answer, done in five_contracts:
        rules, status = evaluation_api.evaluate_runtime_evidence(case, answer, done)
        assert status == "passed", [item for item in rules if item["status"] == "fail"]

    # 12. Controlled-failure canary is visible but excluded from Golden stats.
    before_summary = db.evaluation_summary()
    canary_case = db.create_evaluation_case(
        case_key="S6-TEST-CANARY", title="S6验收用故障注入-不计入黄金集",
        user_message="只读查询维修承诺", status="active", version_label="V1.8.2-S6",
        rubric={
            "controlled_failure_canary": True,
            "deterministic_assertions": {
                "controlled_failure": {"expected": "故意错误", "actual": "真实只读结果"}
            },
        },
    )
    canary = asyncio.run(evaluation_api.run_case(canary_case["id"]))
    after_summary = db.evaluation_summary()
    assert canary["run"]["status"] == "failed" and canary["run"]["badcase_id"]
    assert after_summary["golden_runs_total"] == before_summary["golden_runs_total"]
    assert after_summary["controlled_failure_canary_runs"] == before_summary["controlled_failure_canary_runs"] + 1

    # 13. API rule results expose expected, actual and failure reason per assertion.
    assert failed["rule_results"]
    assert all("expected" in item and "actual" in item and "note" in item for item in failed["rule_results"])
    assert all(item["note"] for item in failed["rule_results"] if item["status"] == "fail")

    # 14. Provider/runtime failure cannot become PASS and links a Badcase.
    async def mocked_failure(message: str, session_id: str):
        trace_id = f"s6-failed-{uuid.uuid4().hex[:12]}"
        db.ensure_chat_session(session_id)
        db.create_chat_trace(trace_id, session_id, message, run_type="evaluation")
        db.update_chat_trace(trace_id, status="failed")
        raise evaluation_api.RuntimeExecutionError(
            "mock provider failure", "", {"status": "failed", "trace_id": trace_id}
        )

    evaluation_api._run_real_chat = mocked_failure
    provider_case = db.create_evaluation_case(
        case_key="S6-TEST-PROVIDER-FAIL", title="Provider failure",
        user_message="模拟失败", status="active", version_label="V1.8.2-S6",
    )
    provider_failed = asyncio.run(evaluation_api.run_case(provider_case["id"]))
    assert provider_failed["run"]["status"] == "failed"
    assert provider_failed["run"]["badcase_id"] and provider_failed["badcase"]["source"] == "evaluation"

    temp_dir.cleanup()
    print("PASS: V1.8.2-S6 deterministic feedback and Golden Set contract (14 checks)")


if __name__ == "__main__":
    main()
