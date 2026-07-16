#!/usr/bin/env python3
"""
V1.3.4 Badcase Operator Closure Acceptance Tests
=================================================

This script exercises the new Badcase detail API contract, authoritative state
machine, and draft review/edit/apply flow against a running API server.

Run against a local server:

    python scripts/test_v1_3_4_badcase_operator_closure.py --base http://127.0.0.1:8000

The script avoids LLM calls where possible. It uses manual classification and
the extract-knowledge endpoint to move cases through the lifecycle.

All created test data is prefixed with DEMO_TEST_V134_ and the script attempts
to clean it up in the finally block. Any residual test data causes the script
to exit with a non-zero code so that "test items passed" cannot mask leftover
DEMO_TEST artifacts.
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

TEST_PREFIX = "DEMO_TEST_V134_"


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

    def delete(self, path: str):
        r = self.session.delete(self._url(path))
        return r.status_code, r.text

    def expect_error(self, path: str, method: str = "post", body=None, status: int = 400):
        fn = self.session.post if method == "post" else self.session.put
        if method == "delete":
            fn = self.session.delete
        r = fn(self._url(path), json=body or {})
        if r.status_code != status:
            raise AssertionError(
                f"expected {status} for {method.upper()} {path}, got {r.status_code}: {r.text}"
            )
        try:
            return r.json()
        except Exception:
            return {"raw": r.text}


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
    created_case_ids: List[int] = []
    created_doc_ids: List[int] = []
    residuals: List[Dict[str, Any]] = []

    def record_residual(resource_type: str, resource_id: int, status_code: Optional[int] = None, error: Optional[str] = None):
        residual: Dict[str, Any] = {"type": resource_type, "id": resource_id}
        if status_code is not None:
            residual["status_code"] = status_code
        if error is not None:
            residual["error"] = error
        residuals.append(residual)

    def check(name: str, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1

    def new_case(title: str, category: str = "pending", **extra) -> Dict[str, Any]:
        payload = {
            "title": f"{TEST_PREFIX}{title}",
            "description": f"{TEST_PREFIX}acceptance test",
            "category": category,
            "source": "manual",
            "original_query": "测试问题",
            "ai_response": "测试回答",
        }
        payload.update(extra)
        resp = c.post("/api/badcases", payload)
        case = resp["badcase"]
        created_case_ids.append(case["id"])
        print(f"  created badcase #{case['id']} {case['title']}")
        return case

    try:
        print("\n[setup] creating primary test badcase")
        case = new_case("Badcase 状态机", category="pending")
        case_id = case["id"]

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

        print("\n[3] classified -> extract-knowledge -> fixing")
        c.post(f"/api/badcases/{case_id}/classify", {"auto": False, "category": "knowledge_gap", "reason": "test"})
        detail = c.get(f"/api/badcases/{case_id}")["badcase"]
        check("status is classified", lambda: detail["status"] == "classified")
        check("classified allows extract-knowledge for knowledge_gap", lambda: "extract-knowledge" in detail["allowed_actions"])

        # Move to fixing by extracting knowledge (no LLM required).
        c.post(f"/api/badcases/{case_id}/extract-knowledge", {
            "title": f"{TEST_PREFIX}知识草稿",
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
        # Edit draft while draft.
        updated = c.put(f"/api/badcases/{case_id}/knowledge-drafts/{draft_id}", {
            "title": f"{TEST_PREFIX}知识草稿（已编辑）",
            "content": "这是编辑后的内容。",
            "category": "缴费",
        })["knowledge_draft"]
        check("edit saved new title", lambda: TEST_PREFIX in (updated.get("title") or ""))

        # Strict review flow: draft -> under_review -> approved -> apply.
        c.post(f"/api/badcases/{case_id}/knowledge-drafts/{draft_id}/review", {"status": "under_review"})
        detail = c.get(f"/api/badcases/{case_id}")["badcase"]
        draft = next(d for d in detail["knowledge_drafts"] if d["id"] == draft_id)
        check("draft status is under_review", lambda: draft.get("status") == "under_review")

        # under_review can return to draft.
        c.post(f"/api/badcases/{case_id}/knowledge-drafts/{draft_id}/review", {"status": "draft"})
        detail = c.get(f"/api/badcases/{case_id}")["badcase"]
        draft = next(d for d in detail["knowledge_drafts"] if d["id"] == draft_id)
        check("under_review can return to draft", lambda: draft.get("status") == "draft")

        # Move back to under_review then approved.
        c.post(f"/api/badcases/{case_id}/knowledge-drafts/{draft_id}/review", {"status": "under_review"})
        c.post(f"/api/badcases/{case_id}/knowledge-drafts/{draft_id}/review", {"status": "approved"})
        detail = c.get(f"/api/badcases/{case_id}")["badcase"]
        draft = next(d for d in detail["knowledge_drafts"] if d["id"] == draft_id)
        check("draft status is approved", lambda: draft.get("status") == "approved")

        # Editing approved resets to draft.
        c.put(f"/api/badcases/{case_id}/knowledge-drafts/{draft_id}", {
            "title": f"{TEST_PREFIX}知识草稿（approved 编辑）",
            "content": "approved 编辑后内容。",
            "category": "缴费",
        })
        detail = c.get(f"/api/badcases/{case_id}")["badcase"]
        draft = next(d for d in detail["knowledge_drafts"] if d["id"] == draft_id)
        check("editing approved draft resets to draft", lambda: draft.get("status") == "draft")

        # Re-approve for apply test.
        c.post(f"/api/badcases/{case_id}/knowledge-drafts/{draft_id}/review", {"status": "under_review"})
        c.post(f"/api/badcases/{case_id}/knowledge-drafts/{draft_id}/review", {"status": "approved"})

        # Unapproved draft cannot be applied: use a fresh independent case in classified.
        unapproved_case = new_case("未审核草稿不能应用", category="pending")
        c.post(f"/api/badcases/{unapproved_case['id']}/classify", {"auto": False, "category": "knowledge_gap", "reason": "test"})
        c.post(f"/api/badcases/{unapproved_case['id']}/extract-knowledge", {
            "title": f"{TEST_PREFIX}未审核草稿",
            "content": "未审核的草稿。",
            "category": "缴费",
        })
        detail = c.get(f"/api/badcases/{unapproved_case['id']}")["badcase"]
        second = find_draft(detail["knowledge_drafts"], TEST_PREFIX)
        check("second draft found", lambda: second is not None)
        check(
            "unapproved draft cannot be applied",
            lambda: c.expect_error(
                f"/api/badcases/{unapproved_case['id']}/knowledge-drafts/{second['id']}/apply",
                status=400,
            ),
        )

        # Apply approved draft on primary case.
        apply_resp = c.post(f"/api/badcases/{case_id}/knowledge-drafts/{draft_id}/apply")
        knowledge_doc = apply_resp.get("knowledge_doc") or {}
        created_doc_id = knowledge_doc.get("id")
        created_doc_title = knowledge_doc.get("title") or ""
        if created_doc_id and TEST_PREFIX in created_doc_title:
            created_doc_ids.append(created_doc_id)
        elif created_doc_id:
            print(f"  WARNING applied knowledge doc {created_doc_id} title does not contain {TEST_PREFIX}; not scheduling cleanup")
        detail = c.get(f"/api/badcases/{case_id}")["badcase"]
        check("status moved to verifying after apply", lambda: detail["status"] == "verifying")
        check("draft status is published", lambda: next(d for d in detail["knowledge_drafts"] if d["id"] == draft_id).get("status") == "published")

        print("\n[5] Verify flow")
        # Verify-pass should be hidden from allowed_actions without retest_response.
        check(
            "verify-pass not in allowed_actions without retest_response",
            lambda: "verify-pass" not in detail["allowed_actions"],
        )
        # Verify-pass endpoint should reject without retest_response.
        check(
            "verify-pass without retest_response fails",
            lambda: c.expect_error(
                f"/api/badcases/{case_id}/verify",
                body={"passed": True},
                status=400,
            ),
        )
        # Verify-fail should work from verifying and return to fixing.
        c.post(f"/api/badcases/{case_id}/verify", {"passed": False, "note": "复测回答仍不满足"})
        detail = c.get(f"/api/badcases/{case_id}")["badcase"]
        check("verify-fail returns to fixing", lambda: detail["status"] == "fixing")

        # Reject from fixing must work with a reason.
        c.post(f"/api/badcases/{case_id}/reject", {"rejected_reason": "验收测试驳回"})
        detail = c.get(f"/api/badcases/{case_id}")["badcase"]
        check("status is rejected", lambda: detail["status"] == "rejected")
        check("is_terminal true", lambda: detail["is_terminal"] is True)
        check("terminal status allows no actions", lambda: detail["allowed_actions"] == [])

        # Reject from terminal should fail.
        check(
            "cannot reject from rejected",
            lambda: c.expect_error(
                f"/api/badcases/{case_id}/reject",
                body={"rejected_reason": "again"},
                status=400,
            ),
        )
        # Transition out of terminal should fail.
        check(
            "cannot transition out of terminal rejected",
            lambda: c.expect_error(
                f"/api/badcases/{case_id}/transition",
                body={"status": "fixing"},
                status=400,
            ),
        )

        print("\n[6] Skill / capability gap reality checks")
        # Create a skill_prompt case and seed a skill draft via Darwin if possible.
        skill_case = new_case("Skill 草稿拦截", category="skill_prompt")
        skill_case_id = skill_case["id"]
        c.post(f"/api/badcases/{skill_case_id}/classify", {"auto": False, "category": "skill_prompt", "reason": "test"})
        darwin_resp = c.session.post(c._url(f"/api/badcases/{skill_case_id}/darwin-fix"))
        if darwin_resp.status_code in (200, 201, 202):
            detail = c.get(f"/api/badcases/{skill_case_id}")["badcase"]
            skill_draft = next((d for d in detail.get("skill_prompt_drafts", [])), None)
            if skill_draft:
                c.post(f"/api/badcases/{skill_case_id}/skill-prompt-drafts/{skill_draft['id']}/review", {"status": "under_review"})
                c.post(f"/api/badcases/{skill_case_id}/skill-prompt-drafts/{skill_draft['id']}/review", {"status": "approved"})
                check(
                    "skill prompt apply blocked with 409",
                    lambda: c.expect_error(
                        f"/api/badcases/{skill_case_id}/skill-prompt-drafts/{skill_draft['id']}/apply",
                        status=409,
                    ),
                )
                detail = c.get(f"/api/badcases/{skill_case_id}")["badcase"]
                check("skill apply does not move case to verifying", lambda: detail["status"] == "fixing")
            else:
                print("  SKIP  no skill_prompt draft generated by Darwin")
        else:
            print(f"  SKIP  Darwin unavailable for skill draft seeding (status {darwin_resp.status_code})")

        # Create an mcp_capability case and seed a capability gap draft.
        gap_case = new_case("能力缺口保持 fixing", category="mcp_capability")
        gap_case_id = gap_case["id"]
        c.post(f"/api/badcases/{gap_case_id}/classify", {"auto": False, "category": "mcp_capability", "reason": "test"})
        darwin_resp = c.session.post(c._url(f"/api/badcases/{gap_case_id}/darwin-fix"))
        if darwin_resp.status_code in (200, 201, 202):
            detail = c.get(f"/api/badcases/{gap_case_id}")["badcase"]
            gap_draft = next((d for d in detail.get("capability_gap_drafts", [])), None)
            if gap_draft:
                c.post(f"/api/badcases/{gap_case_id}/capability-gap-drafts/{gap_draft['id']}/review", {"status": "under_review"})
                c.post(f"/api/badcases/{gap_case_id}/capability-gap-drafts/{gap_draft['id']}/review", {"status": "approved"})
                c.post(f"/api/badcases/{gap_case_id}/capability-gap-drafts/{gap_draft['id']}/apply")
                detail = c.get(f"/api/badcases/{gap_case_id}")["badcase"]
                check("capability gap accepted keeps case fixing", lambda: detail["status"] == "fixing")
                gap_draft_refreshed = next(d for d in detail["capability_gap_drafts"] if d["id"] == gap_draft["id"])
                check("capability gap draft status accepted", lambda: gap_draft_refreshed.get("status") == "accepted")
            else:
                print("  SKIP  no capability_gap draft generated by Darwin")
        else:
            print(f"  SKIP  Darwin unavailable for capability gap seeding (status {darwin_resp.status_code})")

        print("\n[7] Draft review cannot bypass apply")
        bypass_case = new_case("review 不能绕过 apply", category="pending")
        c.post(f"/api/badcases/{bypass_case['id']}/classify", {"auto": False, "category": "knowledge_gap", "reason": "test"})
        c.post(f"/api/badcases/{bypass_case['id']}/extract-knowledge", {
            "title": f"{TEST_PREFIX}绕过测试草稿",
            "content": "用于测试 review 不能绕过 apply。",
            "category": "缴费",
        })
        detail = c.get(f"/api/badcases/{bypass_case['id']}")["badcase"]
        bypass = find_draft(detail["knowledge_drafts"], TEST_PREFIX)
        if bypass:
            check(
                "review cannot set knowledge draft to published",
                lambda: c.expect_error(
                    f"/api/badcases/{bypass_case['id']}/knowledge-drafts/{bypass['id']}/review",
                    body={"status": "published"},
                    status=400,
                ),
            )
            check(
                "review cannot set knowledge draft to approved directly from draft",
                lambda: c.expect_error(
                    f"/api/badcases/{bypass_case['id']}/knowledge-drafts/{bypass['id']}/review",
                    body={"status": "approved"},
                    status=400,
                ),
            )

        print("\n[8] Transition state machine enforcement")
        trans_case = new_case("状态机越界", category="pending")
        trans_case_id = trans_case["id"]
        check(
            "pending -> verifying transition rejected",
            lambda: c.expect_error(
                f"/api/badcases/{trans_case_id}/transition",
                body={"status": "verifying"},
                status=400,
            ),
        )
        check(
            "pending -> closed transition rejected",
            lambda: c.expect_error(
                f"/api/badcases/{trans_case_id}/transition",
                body={"status": "closed"},
                status=400,
            ),
        )

        print("\n[9] Operation history readability")
        detail = c.get(f"/api/badcases/{case_id}")["badcase"]
        check("actions is non-empty", lambda: len(detail["actions"]) > 0)
        sample = detail["actions"][0]
        check("action has action_type", lambda: "action_type" in sample)
        check("action has status_before_label", lambda: "status_before_label" in sample)
        check("action has status_after_label", lambda: "status_after_label" in sample)
        check("action has action_detail_parsed", lambda: "action_detail_parsed" in sample)

    finally:
        print("\n[cleanup] attempting to remove test data")
        for doc_id in created_doc_ids:
            try:
                code, _ = c.delete(f"/api/knowledge/docs/{doc_id}")
                if 200 <= code < 300:
                    print(f"  deleted knowledge doc {doc_id}: HTTP {code}")
                else:
                    print(f"  failed to delete knowledge doc {doc_id}: HTTP {code}")
                    record_residual("knowledge_doc", doc_id, status_code=code)
            except Exception as exc:
                print(f"  failed to delete knowledge doc {doc_id}: {exc}")
                record_residual("knowledge_doc", doc_id, error=str(exc))
        for cid in created_case_ids:
            try:
                code, _ = c.delete(f"/api/badcases/{cid}")
                if 200 <= code < 300:
                    print(f"  deleted badcase {cid}: HTTP {code}")
                else:
                    print(f"  failed to delete badcase {cid}: HTTP {code}")
                    record_residual("badcase", cid, status_code=code)
            except Exception as exc:
                print(f"  failed to delete badcase {cid}: {exc}")
                record_residual("badcase", cid, error=str(exc))

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Created Badcase IDs: {created_case_ids}")
    print(f"Created Knowledge Doc IDs: {created_doc_ids}")
    print(f"Residuals: {residuals}")

    if failed:
        return 1
    if residuals:
        print("\nFAILURE: test data residuals remain; see Residuals list above.")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V1.3.4 Badcase operator closure acceptance tests")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="API base URL")
    args = parser.parse_args()
    sys.exit(run(args.base))
