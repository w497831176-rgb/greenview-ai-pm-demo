"""Low-cost contract checks for the V1.6 quality-governance release.

This is deliberately not an end-to-end model test: it validates the state
machine and the deterministic Golden Set rule engine with synthetic runtime
evidence, so running it never consumes provider Tokens or alters business data.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.badcase_schema import (  # noqa: E402
    effective_allowed_actions,
    is_terminal_status,
    validate_status_transition,
)
from app.evaluations import evaluate_runtime_evidence  # noqa: E402


def expect_raises(fn, text: str) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(text)


def test_lifecycle() -> None:
    validate_status_transition("pending", "classified")
    validate_status_transition("classified", "investigating")
    validate_status_transition("investigating", "fixing")
    validate_status_transition("fixing", "verifying")
    validate_status_transition("verifying", "released")
    validate_status_transition("released", "closed")
    validate_status_transition("released", "fixing")
    expect_raises(lambda: validate_status_transition("verifying", "closed"), "must observe release before closure")
    expect_raises(lambda: validate_status_transition("closed", "fixing"), "terminal status must not reopen")
    assert is_terminal_status("accepted_limitation")
    assert is_terminal_status("duplicate")
    assert "verify-pass" not in effective_allowed_actions({"status": "verifying", "last_applied_at": "2026-07-18 10:00:00", "last_retest_at": None, "retest_response": ""})
    assert "verify-pass" in effective_allowed_actions({"status": "verifying", "last_applied_at": "2026-07-18 10:00:00", "last_retest_at": "2026-07-18 10:01:00", "retest_response": "修复后回答"})
    assert "close" in effective_allowed_actions({"status": "released"})
    assert "verify-fail" in effective_allowed_actions({"status": "released"})


def test_rule_engine() -> None:
    case = {
        "expected_agent_id": "maintenance",
        "expected_skills": ["维修工单处理"],
        "expected_tools": ["get_current_weather"],
        "expected_citation_docs": ["物业维修服务承诺"],
        "required_terms": ["30分钟"],
        "forbidden_terms": ["已立即创建工单"],
        "expected_handoff": False,
        "rubric": {},
    }
    done = {
        "current_agent_id": "maintenance",
        "activated_skills": ["维修工单处理"],
        "mcp_calls": [{"server_name": "weather-server", "tool_name": "get_current_weather"}],
        "citations": [{"doc_title": "物业维修服务承诺"}],
        "handoff": False,
    }
    rules, status = evaluate_runtime_evidence(case, "紧急维修承诺30分钟内到场。", done)
    assert status == "passed", (status, rules)
    assert all(item["status"] in {"pass", "not_configured"} for item in rules), rules

    failed_case = dict(case, expected_tools=["count_work_orders"])
    _, failed_status = evaluate_runtime_evidence(failed_case, "紧急维修承诺30分钟内到场。", done)
    assert failed_status == "failed"

    manual_case = dict(case, rubric={"operator_rubric": "话术是否足够安抚"})
    _, manual_status = evaluate_runtime_evidence(manual_case, "紧急维修承诺30分钟内到场。", done)
    assert manual_status == "needs_manual_review"


if __name__ == "__main__":
    test_lifecycle()
    test_rule_engine()
    print("V1.6 quality-governance contract checks passed.")
