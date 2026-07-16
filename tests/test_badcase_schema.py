"""
Unit tests for the Badcase schema and state machine helpers.

These tests require no external services; they run with only the Python stdlib
and the app.badcase_schema module.
"""

import json
import sys
from pathlib import Path

# Make app.badcase_schema importable when running from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.badcase_schema import (
    CATEGORY_LABELS,
    STATUS_LABELS,
    _enrich_badcase,
    _format_action,
    allowed_actions,
    allowed_target_statuses,
    effective_allowed_actions,
    is_draft_editable,
    is_draft_terminal,
    is_terminal_status,
    repair_path_for_category,
    require_status,
    validate_draft_status_transition,
    validate_status_transition,
)


def test_enrich_badcase_exposes_stable_schema():
    raw = {
        "id": 1,
        "title": "缴费失败",
        "original_query": "物业费怎么交",
        "ai_response": "不知道",
        "category": "knowledge_gap",
        "status": "verifying",
        "source": "manual",
        "context_json": json.dumps({"route_intent": "billing"}),
        "retest_context_json": json.dumps({"current_agent": "billing"}),
        "retest_response": "已修复",
        "retest_trace_id": "trace-123",
        "root_cause": "缺缴费指引",
        "fix_plan": "补充知识",
        "evidence": "用户反馈",
        "actions": [
            {
                "action_type": "classify",
                "action_detail": json.dumps({"category": "knowledge_gap"}),
                "status_before": "pending",
                "status_after": "classified",
                "created_at": "2026-07-16T10:00:00",
            }
        ],
    }
    bc = _enrich_badcase(raw)

    assert bc["query"] == "物业费怎么交"
    assert bc["category_label"] == CATEGORY_LABELS["knowledge_gap"]
    assert bc["status_label"] == STATUS_LABELS["verifying"]
    assert bc["source_label"] == "人工反馈"
    assert bc["context"]["route_intent"] == "billing"
    assert bc["retest_context"]["current_agent"] == "billing"
    assert bc["retest_response"] == "已修复"
    assert bc["retest_trace_id"] == "trace-123"
    assert bc["analysis_evidence"] == "用户反馈"
    assert bc["actions"][0]["status_before_label"] == "待分类"
    assert bc["actions"][0]["action_detail_parsed"]["category"] == "knowledge_gap"
    assert "retest_result" not in bc
    assert bc["allowed_actions"] == effective_allowed_actions(raw)
    assert not bc["is_terminal"]


def test_enrich_badcase_action_parsing_fallback():
    raw = {
        "id": 2,
        "status": "pending",
        "actions": [
            {"action_type": "transition", "action_detail": "not-json", "status_before": "pending", "status_after": "classified"}
        ],
    }
    bc = _enrich_badcase(raw)
    assert bc["actions"][0]["action_detail_parsed"] == {"raw": "not-json"}


def test_state_machine_transitions():
    assert allowed_target_statuses("pending") == ["classified", "rejected"]
    assert allowed_target_statuses("classified") == ["fixing", "rejected"]
    assert allowed_target_statuses("fixing") == ["rejected", "verifying"]
    assert allowed_target_statuses("verifying") == ["closed", "fixing", "rejected"]
    assert allowed_target_statuses("closed") == []
    assert allowed_target_statuses("rejected") == []


def test_invalid_transitions_raise():
    invalid_pairs = [
        ("pending", "fixing"),
        ("pending", "verifying"),
        ("pending", "closed"),
        ("classified", "verifying"),
        ("fixing", "closed"),
        ("closed", "pending"),
        ("rejected", "pending"),
    ]
    for from_s, to_s in invalid_pairs:
        try:
            validate_status_transition(from_s, to_s)
            raise AssertionError(f"expected ValueError for {from_s} -> {to_s}")
        except ValueError:
            pass


def test_allowed_actions_per_status():
    assert set(allowed_actions("pending")) == {"classify", "reject", "transition"}
    assert set(allowed_actions("classified")) == {"darwin-fix", "extract-knowledge", "reject", "transition"}
    assert set(allowed_actions("fixing")) == {
        "accept-capability-gap",
        "apply-draft",
        "edit-draft",
        "reject",
        "retest",
        "review-draft",
        "transition",
    }
    assert set(allowed_actions("verifying")) == {"reject", "retest", "transition", "verify-fail", "verify-pass"}
    assert allowed_actions("closed") == []
    assert allowed_actions("rejected") == []


def test_require_status_rejects_wrong_status():
    try:
        require_status("pending", "retest", {"verifying"})
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "pending" in str(exc)
        assert "retest" in str(exc)


def test_is_terminal_status():
    assert is_terminal_status("closed")
    assert is_terminal_status("rejected")
    assert not is_terminal_status("pending")
    assert not is_terminal_status("fixing")


def test_repair_path_for_category():
    assert repair_path_for_category("knowledge_gap") == "knowledge"
    assert repair_path_for_category("skill_prompt") == "skill_prompt"
    assert repair_path_for_category("routing") == "skill_prompt"
    assert repair_path_for_category("response_quality") == "skill_prompt"
    assert repair_path_for_category("mcp_capability") == "capability_gap"
    assert repair_path_for_category("other") == "darwin"


def test_terminal_allowed_actions_are_empty():
    assert effective_allowed_actions({"status": "closed"}) == []
    assert effective_allowed_actions({"status": "rejected"}) == []


def test_effective_allowed_actions_hides_verify_pass_without_retest():
    bc = {"status": "verifying", "retest_response": "", "last_applied_at": "2026-07-16 10:00:00"}
    actions = set(effective_allowed_actions(bc))
    assert "verify-pass" not in actions
    assert "retest" in actions
    assert "verify-fail" in actions
    assert "reject" in actions


def test_effective_allowed_actions_shows_verify_pass_with_post_apply_retest():
    bc = {
        "status": "verifying",
        "retest_response": "已修复",
        "last_applied_at": "2026-07-16 10:00:00",
        "last_retest_at": "2026-07-16 11:00:00",
    }
    actions = set(effective_allowed_actions(bc))
    assert "verify-pass" in actions
    assert "retest" in actions
    assert "verify-fail" in actions
    assert "reject" in actions


def test_effective_allowed_actions_hides_verify_pass_when_retest_is_before_apply():
    bc = {
        "status": "verifying",
        "retest_response": "旧复测",
        "last_applied_at": "2026-07-16 12:00:00",
        "last_retest_at": "2026-07-16 10:00:00",
    }
    actions = set(effective_allowed_actions(bc))
    assert "verify-pass" not in actions
    assert "retest" in actions


def test_effective_allowed_actions_for_non_verifying_unchanged():
    bc = {"status": "fixing", "retest_response": ""}
    assert set(effective_allowed_actions(bc)) == set(allowed_actions("fixing"))


def test_draft_status_transitions_knowledge_and_skill():
    # happy path
    validate_draft_status_transition("knowledge", "draft", "under_review")
    validate_draft_status_transition("knowledge", "under_review", "approved")
    validate_draft_status_transition("knowledge", "approved", "published")
    validate_draft_status_transition("skill_prompt", "draft", "rejected")
    validate_draft_status_transition("skill_prompt", "under_review", "rejected")

    # illegal direct to published
    for draft_type in ("knowledge", "skill_prompt"):
        for illegal in ("draft", "under_review", "rejected"):
            try:
                validate_draft_status_transition(draft_type, illegal, "published")
                raise AssertionError(f"expected ValueError for {draft_type} {illegal} -> published")
            except ValueError:
                pass


def test_draft_status_transitions_capability_gap():
    validate_draft_status_transition("capability_gap", "draft", "under_review")
    validate_draft_status_transition("capability_gap", "under_review", "approved")
    validate_draft_status_transition("capability_gap", "approved", "accepted")
    validate_draft_status_transition("capability_gap", "under_review", "rejected")

    for illegal in ("draft", "under_review", "rejected"):
        try:
            validate_draft_status_transition("capability_gap", illegal, "accepted")
            raise AssertionError(f"expected ValueError for capability_gap {illegal} -> accepted")
        except ValueError:
            pass


def test_draft_terminal_and_editable():
    assert is_draft_terminal("knowledge", "published")
    assert is_draft_terminal("knowledge", "rejected")
    assert not is_draft_terminal("knowledge", "approved")
    assert is_draft_terminal("capability_gap", "accepted")

    assert is_draft_editable("knowledge", "draft")
    assert is_draft_editable("knowledge", "under_review")
    assert is_draft_editable("knowledge", "approved")
    assert not is_draft_editable("knowledge", "published")
    assert not is_draft_editable("knowledge", "rejected")


def test_draft_under_review_can_return_to_draft():
    """All three draft types allow under_review -> draft rollback."""
    for draft_type in ("knowledge", "skill_prompt", "capability_gap"):
        validate_draft_status_transition(draft_type, "under_review", "draft")


def test_draft_terminal_cannot_escape_to_non_terminal():
    """Terminal draft statuses must not transition back to editable states."""
    terminal_cases = [
        ("knowledge", "published", ["draft", "under_review", "approved"]),
        ("knowledge", "rejected", ["draft", "under_review", "approved", "published"]),
        ("skill_prompt", "published", ["draft", "under_review", "approved"]),
        ("skill_prompt", "rejected", ["draft", "under_review", "approved", "published"]),
        ("capability_gap", "accepted", ["draft", "under_review", "approved"]),
        ("capability_gap", "rejected", ["draft", "under_review", "approved", "accepted"]),
    ]
    for draft_type, terminal_status, illegal_targets in terminal_cases:
        for target in illegal_targets:
            try:
                validate_draft_status_transition(draft_type, terminal_status, target)
                raise AssertionError(
                    f"expected ValueError for {draft_type} {terminal_status} -> {target}"
                )
            except ValueError:
                pass


if __name__ == "__main__":
    test_enrich_badcase_exposes_stable_schema()
    test_enrich_badcase_action_parsing_fallback()
    test_state_machine_transitions()
    test_invalid_transitions_raise()
    test_allowed_actions_per_status()
    test_require_status_rejects_wrong_status()
    test_is_terminal_status()
    test_repair_path_for_category()
    test_terminal_allowed_actions_are_empty()
    test_effective_allowed_actions_hides_verify_pass_without_retest()
    test_effective_allowed_actions_shows_verify_pass_with_post_apply_retest()
    test_effective_allowed_actions_hides_verify_pass_when_retest_is_before_apply()
    test_effective_allowed_actions_for_non_verifying_unchanged()
    test_draft_status_transitions_knowledge_and_skill()
    test_draft_status_transitions_capability_gap()
    test_draft_terminal_and_editable()
    test_draft_under_review_can_return_to_draft()
    test_draft_terminal_cannot_escape_to_non_terminal()
    print("All badcase_schema tests passed.")
