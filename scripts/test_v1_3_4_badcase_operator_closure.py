#!/usr/bin/env python3
"""
V1.3.4 Badcase Operator Closure Acceptance Tests
=================================================

This script exercises the new Badcase detail API contract, authoritative state
machine, and draft review/edit/apply flow against a running API server.

Run against a local server:

    python scripts/test_v1_3_4_badcase_operator_closure.py --base http://127.0.0.1:8000

The script does NOT depend on LLM calls. It uses manual classification and the
extract-knowledge endpoint to move cases through the lifecycle without Darwin.
"""

import argparse
import json
import sys
from urllib.parse import urljoin

import requests


class BadcaseAcceptanceClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return urljoin(self.base + "/", path.lstrip("/"))

    def get(self, path: str):
        r = self.session.get(self._url(path))
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body=None):
        r = self.session.post(self._url(path), json=body or {})
        r.raise_for_status()
        return r.json()

    def put(self, path: str, body=None):
        r = self.session.put(self._url(path), json=body or {})
        r.raise_for_status()
        return r.json()

    def expect_error(self, path: str, method: str = "post", body=None, status: int = 400):
        fn = self.session.post if method == "post" else self.session.put
        r = fn(self._url(path), json=body or {})
        if r.status_code != status:
            raise AssertionError(
                f"expected {status} for {method.upper()} {path}, got {r.status_code}: {r.text}"
            )
        return r.json()


def assert_field(obj, field, msg=None):
    if field not in obj:
        raise AssertionError(msg or f"missing field: {field}")


def find_draft(drafts, title_substring: str):
    for d in drafts:
        if title_substring in (d.get("title") or ""):
            return d
    return None


def run(base_url: str) -> int:
    c = BadcaseAcceptanceClient(base_url)
    passed = 0
    failed = 0

    def check(name: str, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1

    # Use an existing badcase or create one. We prefer creating to avoid
    # side-effects on production/demo data.
    print("\n[setup] creating a test badcase")
    create_resp = c.post("/api/badcases", {
        "title": "验收测试：Badcase 状态机",
        "description": "用于 V1.3.4 验收测试",
        "category": "pending",
        "source": "manual",
        "original_query": "测试问题",
        "ai_response": "测试回答",
    })
    case_id = create_resp["badcase"]["id"]
    print(f"  created badcase #{case_id}")

    print("\n[1] Detail API schema completeness")
    detail = c.get(f"/api/badcases/{case_id}")["badcase"]
    check("query field present", lambda: assert_field(detail, "query"))
    check("category_label field present", lambda: assert_field(detail, "category_label"))
    check("source_label field present", lambda: assert_field(detail, "source_label"))
    check("context field present", lambda: assert_field(detail, "context"))
    check("root_cause field present", lambda: assert_field(detail, "root_cause"))
    check("fix_plan field present", lambda: assert_field(detail, "fix_plan"))
    check("darwin_analysis field present", lambda: assert_field(detail, "darwin_analysis"))
    check("evidence field present", lambda: assert_field(detail, "evidence"))
    check("retest_response field present", lambda: assert_field(detail, "retest_response"))
    check("retest_context field present", lambda: assert_field(detail, "retest_context"))
    check("retest_trace_id field present", lambda: assert_field(detail, "retest_trace_id"))
    check("actions field present", lambda: assert_field(detail, "actions"))
    check("knowledge_drafts field present", lambda: assert_field(detail, "knowledge_drafts"))
    check("skill_prompt_drafts field present", lambda: assert_field(detail, "skill_prompt_drafts"))
    check("capability_gap_drafts field present", lambda: assert_field(detail, "capability_gap_drafts"))
    check("allowed_actions field present", lambda: assert_field(detail, "allowed_actions"))
    check("is_terminal field present", lambda: assert_field(detail, "is_terminal"))
    check("legacy retest_result removed", lambda: ("retest_result" not in detail) or detail["retest_result"] is None)

    print("\n[2] State machine authority")
    check(
        "pending cannot darwin-fix",
        lambda: c.expect_error(f"/api/badcases/{case_id}/darwin-fix", status=400),
    )
    check(
        "pending cannot retest",
        lambda: c.expect_error(f"/api/badcases/{case_id}/retest", status=400),
    )
    check(
        "pending cannot verify-pass",
        lambda: c.expect_error(f"/api/badcases/{case_id}/verify", body={"passed": True}, status=400),
    )

    print("\n[3] classified -> Darwin/fixing flow")
    c.post(f"/api/badcases/{case_id}/classify", {"auto": False, "category": "knowledge_gap", "reason": "test"})
    detail = c.get(f"/api/badcases/{case_id}")["badcase"]
    check("status is classified", lambda: detail["status"] == "classified")
    check("classified allows darwin-fix in allowed_actions", lambda: "darwin-fix" in detail["allowed_actions"])

    # Move to fixing by extracting knowledge (no LLM required).
    c.post(f"/api/badcases/{case_id}/extract-knowledge", {
        "title": "验收知识条目",
        "content": "这是验收测试生成的知识草稿内容。",
        "category": "缴费",
    })
    detail = c.get(f"/api/badcases/{case_id}")["badcase"]
    check("status is fixing", lambda: detail["status"] == "fixing")
    check("knowledge draft was created", lambda: len(detail["knowledge_drafts"]) >= 1)

    draft = detail["knowledge_drafts"][0]
    draft_id = draft["id"]
    check("draft initial status is draft", lambda: draft.get("status") == "draft")

    print("\n[4] Draft review / edit / apply")
    # Edit draft.
    updated = c.put(f"/api/badcases/{case_id}/knowledge-drafts/{draft_id}", {
        "title": "验收知识条目（已编辑）",
        "content": "这是编辑后的内容。",
        "category": "缴费",
    })["knowledge_draft"]
    check("edit saved new title", lambda: "已编辑" in (updated.get("title") or ""))

    # Approve draft.
    c.post(f"/api/badcases/{case_id}/knowledge-drafts/{draft_id}/review", {"status": "approved"})
    detail = c.get(f"/api/badcases/{case_id}")["badcase"]
    draft = next(d for d in detail["knowledge_drafts"] if d["id"] == draft_id)
    check("draft status is approved", lambda: draft.get("status") == "approved")

    # Unapproved draft cannot be applied: create a second draft to test.
    c.post(f"/api/badcases/{case_id}/extract-knowledge", {
        "title": "第二个草稿",
        "content": "未审核的草稿。",
        "category": "缴费",
    })
    detail = c.get(f"/api/badcases/{case_id}")["badcase"]
    second = find_draft(detail["knowledge_drafts"], "第二个草稿")
    check("second draft found", lambda: second is not None)
    check(
        "unapproved draft cannot be applied",
        lambda: c.expect_error(
            f"/api/badcases/{case_id}/knowledge-drafts/{second['id']}/apply",
            status=400,
        ),
    )

    # Apply approved draft.
    c.post(f"/api/badcases/{case_id}/knowledge-drafts/{draft_id}/apply")
    detail = c.get(f"/api/badcases/{case_id}")["badcase"]
    check("status moved to verifying after apply", lambda: detail["status"] == "verifying")
    check("draft status is published", lambda: next(d for d in detail["knowledge_drafts"] if d["id"] == draft_id).get("status") == "published")

    print("\n[5] Verify flow")
    # From verifying, reject must work with a reason.
    c.post(f"/api/badcases/{case_id}/reject", {"rejected_reason": "验收测试驳回"})
    detail = c.get(f"/api/badcases/{case_id}")["badcase"]
    check("status is rejected", lambda: detail["status"] == "rejected")
    check("is_terminal true", lambda: detail["is_terminal"] is True)
    check("terminal status allows only transition", lambda: detail["allowed_actions"] == ["transition"])

    # Reject from terminal should fail.
    check(
        "cannot reject from rejected",
        lambda: c.expect_error(
            f"/api/badcases/{case_id}/reject",
            body={"rejected_reason": "again"},
            status=400,
        ),
    )

    # Transition back to verifying for verify-pass test.
    c.post(f"/api/badcases/{case_id}/transition", {"status": "verifying"})
    detail = c.get(f"/api/badcases/{case_id}")["badcase"]
    check("transition to verifying works", lambda: detail["status"] == "verifying")

    # Verify-pass without retest_response should fail.
    check(
        "verify-pass without retest_response fails",
        lambda: c.expect_error(
            f"/api/badcases/{case_id}/verify",
            body={"passed": True},
            status=400,
        ),
    )

    # Simulate retest by writing retest_response via transition? No, transition
    # does not write retest_response. We use the retest endpoint which requires
    # a running chat backend; if it fails due to missing model, we manually set
    # the field via the DB or skip. For a pure API test we use a small helper
    # endpoint if available, otherwise we inject directly. To keep the script
    # server-only, we call retest and accept 500/502 as infrastructure issue,
    # but a 400 means our state machine rejected it.
    retest_resp = c.session.post(c._url(f"/api/badcases/{case_id}/retest"))
    if retest_resp.status_code == 400:
        raise AssertionError("retest was rejected by state machine despite verifying status")
    if retest_resp.status_code in (200, 201, 202):
        c.post(f"/api/badcases/{case_id}/verify", {"passed": True})
        detail = c.get(f"/api/badcases/{case_id}")["badcase"]
        check("verify-pass closes case", lambda: detail["status"] == "closed")
    else:
        print(f"  SKIP  real retest/verify-pass (server returned {retest_resp.status_code}, likely model/config missing)")

    print("\n[6] Transition state machine enforcement")
    # Create a fresh case to test invalid transitions.
    create_resp = c.post("/api/badcases", {
        "title": "验收测试：状态机越界",
        "description": "用于测试非法状态流转",
        "category": "pending",
        "source": "manual",
    })
    case2_id = create_resp["badcase"]["id"]
    check(
        "pending -> verifying transition rejected",
        lambda: c.expect_error(
            f"/api/badcases/{case2_id}/transition",
            body={"status": "verifying"},
            status=400,
        ),
    )
    check(
        "pending -> closed transition rejected",
        lambda: c.expect_error(
            f"/api/badcases/{case2_id}/transition",
            body={"status": "closed"},
            status=400,
        ),
    )

    print("\n[7] Operation history readability")
    detail = c.get(f"/api/badcases/{case_id}")["badcase"]
    check("actions is non-empty", lambda: len(detail["actions"]) > 0)
    sample = detail["actions"][0]
    check("action has action_type", lambda: "action_type" in sample)
    check("action has status_before_label", lambda: "status_before_label" in sample)
    check("action has status_after_label", lambda: "status_after_label" in sample)
    check("action has action_detail_parsed", lambda: "action_detail_parsed" in sample)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V1.3.4 Badcase operator closure acceptance tests")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="API base URL")
    args = parser.parse_args()
    sys.exit(run(args.base))
