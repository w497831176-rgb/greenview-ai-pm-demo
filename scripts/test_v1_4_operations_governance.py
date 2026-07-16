#!/usr/bin/env python3
"""
V1.4.1 Operations Integrity Unified Acceptance Tests
====================================================

This script verifies the v1.4.1 integrity patch:
- Badcase repair-after-apply lifecycle (apply -> verifying -> retest -> verify-pass -> closed)
- Monthly budget governance alongside daily budget
- Acceptance script itself does not mutate real config and records residuals

Run against a local server (default is inside the demo-os-api container):

    python scripts/test_v1_4_operations_governance.py --base http://127.0.0.1:8000

All created test data is prefixed with DEMO_TEST_V141_. The script saves the
original budget configuration and restores it in the finally block. Any
residual test data or failed restore causes a non-zero exit.
"""

import argparse
import json
import math
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests

TEST_PREFIX = "DEMO_TEST_V141_"


class AcceptanceClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return urljoin(self.base + "/", path.lstrip("/"))

    def get(self, path: str, params: Optional[Dict[str, Any]] = None):
        r = self.session.get(self._url(path), params=params)
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

    def raw_post(self, path: str, body=None):
        return self.session.post(self._url(path), json=body or {})

    def expect_error(self, path: str, method: str = "post", body=None, status: int = 400):
        fn = {"post": self.session.post, "put": self.session.put, "delete": self.session.delete}.get(method, self.session.post)
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


def chat_sse(c: AcceptanceClient, message: str, session_id: Optional[str] = None, timeout: int = 240) -> Dict[str, Any]:
    payload = {"message": message, "stream": True, "enable_rag": True}
    if session_id:
        payload["session_id"] = session_id
    resp = c.session.post(c._url("/api/chat/stream"), json=payload, stream=True)
    resp.raise_for_status()

    text_parts: List[str] = []
    events: List[Dict[str, Any]] = []
    done: Dict[str, Any] = {}
    current_event: Optional[str] = None
    start = time.time()
    for line in resp.iter_lines(decode_unicode=True):
        if time.time() - start > timeout:
            raise TimeoutError("SSE chat timed out")
        if not line:
            current_event = None
            continue
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
            continue
        if line.startswith("data:"):
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                evt = json.loads(data)
            except json.JSONDecodeError:
                continue
            evt["_event"] = current_event
            events.append(evt)
            if current_event == "delta":
                text_parts.append(evt.get("content", ""))
            elif current_event == "done":
                done = evt
                break
    return {"text": "".join(text_parts), "events": events, "done": done}


def check_no_secret(text: str) -> Optional[str]:
    lowered = text.lower()
    keywords = ["sk-", "api_key", "apikey", "api-secret", "api_secret", "secret_key", "bearer "]
    for kw in keywords:
        idx = lowered.find(kw)
        if idx != -1:
            return text[max(0, idx - 10):idx + 50]
    return None


def thresholds_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    for key in ("per_call_threshold_cny", "daily_threshold_cny", "monthly_threshold_cny"):
        av = a.get(key)
        bv = b.get(key)
        if av is None and bv is None:
            continue
        if av is None or bv is None:
            return False
        if not math.isclose(float(av), float(bv), rel_tol=1e-9, abs_tol=1e-12):
            return False
    return True


def run(base_url: str) -> int:
    c = AcceptanceClient(base_url)
    passed = 0
    failed = 0
    created_case_ids: List[int] = []
    created_doc_ids: List[int] = []
    created_skill_ids: List[int] = []
    created_agent_skill_bindings: List[Dict[str, Any]] = []
    residuals: List[Dict[str, Any]] = []
    original_budget: Optional[Dict[str, Any]] = None

    def record_residual(resource_type: str, resource_id: Any, status_code: Optional[int] = None, error: Optional[str] = None):
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

    def save_original_budget():
        nonlocal original_budget
        original_budget = c.get("/api/observability/budget").get("budget", {})
        print(f"  saved original budget: per_call={original_budget.get('per_call_threshold_cny')} "
              f"daily={original_budget.get('daily_threshold_cny')} monthly={original_budget.get('monthly_threshold_cny')}")

    def restore_original_budget():
        if original_budget is None:
            return
        try:
            c.put("/api/observability/budget", {
                "per_call_threshold_cny": original_budget.get("per_call_threshold_cny"),
                "daily_threshold_cny": original_budget.get("daily_threshold_cny"),
                "monthly_threshold_cny": original_budget.get("monthly_threshold_cny"),
            })
            current = c.get("/api/observability/budget").get("budget", {})
            if not thresholds_equal(current, original_budget):
                record_residual("budget_config", "original", error="restored budget does not match original")
                print(f"  WARNING: restored budget {current} does not match original {original_budget}")
            else:
                print("  restored original budget")
        except Exception as exc:
            record_residual("budget_config", "original", error=str(exc))
            print(f"  failed to restore original budget: {exc}")

    def set_high_budget():
        c.put("/api/observability/budget", {
            "per_call_threshold_cny": 1000000,
            "daily_threshold_cny": 1000000,
            "monthly_threshold_cny": 1000000,
        })

    def pick_target_agent() -> Tuple[str, str]:
        agents = c.get("/api/agents").get("agents", [])
        for agent in agents:
            if agent.get("category") == "vertical" and agent.get("enabled"):
                return agent["agent_id"], agent["name"]
        if agents:
            return agents[0]["agent_id"], agents[0]["name"]
        raise AssertionError("no agent available for skill binding test")

    def assert_retest_has_runtime_fields(detail: Dict[str, Any]):
        if not detail.get("retest_response"):
            raise AssertionError("retest_response missing")
        ctx = detail.get("retest_context") or {}
        required = ("model_id", "trace_id", "token_count", "token_detail", "total_tokens", "usage_source", "estimated_cost_cny", "price_snapshot")
        missing = [k for k in required if k not in ctx]
        if missing:
            raise AssertionError(f"retest_context missing fields: {missing}; got {ctx}")
        if not ctx.get("model_id"):
            raise AssertionError("retest_context.model_id is empty")
        if not ctx.get("trace_id"):
            raise AssertionError("retest_context.trace_id is empty")
        if not (ctx.get("token_count") or ctx.get("total_tokens")):
            raise AssertionError("retest_context has no token_count or total_tokens")

    def assert_retest_trace_has_retest_model_call(trace_id: str):
        trace_detail = c.get(f"/api/observability/traces/{trace_id}")
        model_calls = trace_detail.get("model_calls", [])
        retest_calls = [mc for mc in model_calls if mc.get("stage") == "retest"]
        if not retest_calls:
            raise AssertionError(f"no stage=retest model_call found in trace {trace_id}; got {model_calls}")

    def ensure_knowledge_draft(case_id: int) -> Tuple[Dict[str, Any], int]:
        detail = c.get(f"/api/badcases/{case_id}")["badcase"]
        if detail.get("knowledge_drafts"):
            return detail, detail["knowledge_drafts"][0]["id"]
        c.post(f"/api/badcases/{case_id}/extract-knowledge", {
            "title": f"{TEST_PREFIX}知识草稿",
            "content": f"{TEST_PREFIX} 这是验收测试生成的知识草稿内容，用于验证检索。",
            "category": "缴费",
        })
        detail = c.get(f"/api/badcases/{case_id}")["badcase"]
        return detail, detail["knowledge_drafts"][0]["id"]

    def approve_knowledge_draft(case_id: int, draft_id: int):
        c.post(f"/api/badcases/{case_id}/knowledge-drafts/{draft_id}/review", {"status": "under_review"})
        c.post(f"/api/badcases/{case_id}/knowledge-drafts/{draft_id}/review", {"status": "approved"})

    def assert_allowed_after_apply_no_retest(detail: Dict[str, Any]):
        if "verify-pass" in detail.get("allowed_actions", []):
            raise AssertionError("verify-pass should be hidden before post-apply retest")
        if "retest" not in detail.get("allowed_actions", []):
            raise AssertionError("retest should be available in verifying")

    try:
        # ------------------------------------------------------------------
        # Setup: save original budget and set high thresholds for tests
        # ------------------------------------------------------------------
        print("\n[setup] save original budget and set high thresholds")
        save_original_budget()
        set_high_budget()

        # ------------------------------------------------------------------
        # A. Knowledge gap: apply -> verifying -> retest -> verify-pass -> closed
        # ------------------------------------------------------------------
        print("\n[A] Knowledge gap post-apply retest lifecycle")
        case_a = new_case("知识库缺口闭环", category="knowledge_gap")
        case_a_id = case_a["id"]

        c.post(f"/api/badcases/{case_a_id}/classify", {"auto": False, "category": "knowledge_gap", "reason": "test"})
        detail = c.get(f"/api/badcases/{case_a_id}")["badcase"]
        check("A classified", lambda: detail["status"] == "classified")

        # Generate fixing state with a draft
        darwin_resp = c.raw_post(f"/api/badcases/{case_a_id}/darwin-fix")
        if darwin_resp.status_code not in (200, 201, 202):
            print(f"  SKIP  Darwin unavailable (status {darwin_resp.status_code}); seeding knowledge draft manually")
        detail, draft_a_id = ensure_knowledge_draft(case_a_id)
        check("A status is fixing with draft", lambda: detail["status"] == "fixing")
        approve_knowledge_draft(case_a_id, draft_a_id)

        # Pre-apply retest: allowed as diagnosis, but must not unlock verify-pass later
        c.post(f"/api/badcases/{case_a_id}/retest")
        detail = c.get(f"/api/badcases/{case_a_id}")["badcase"]
        check("A stays fixing after pre-apply retest", lambda: detail["status"] == "fixing")
        pre_apply_retest_at = detail.get("last_retest_at")

        # Apply the draft -> moves to verifying and clears old retest evidence
        apply_resp = c.post(f"/api/badcases/{case_a_id}/knowledge-drafts/{draft_a_id}/apply")
        doc = apply_resp.get("knowledge_doc") or {}
        doc_id = doc.get("id")
        if doc_id and TEST_PREFIX in (doc.get("title") or ""):
            created_doc_ids.append(doc_id)
        detail = c.get(f"/api/badcases/{case_a_id}")["badcase"]
        check("A status moved to verifying after apply", lambda: detail["status"] == "verifying")
        check("A old retest evidence cleared", lambda: not detail.get("retest_response"))
        check("A last_applied_at recorded", lambda: bool(detail.get("last_applied_at")))
        check("A verify-pass blocked before post-apply retest", lambda: assert_allowed_after_apply_no_retest(detail))
        check("A verify-pass endpoint returns 400 before post-apply retest", lambda: c.expect_error(
            f"/api/badcases/{case_a_id}/verify", body={"passed": True, "note": "should fail"}, status=400
        ))

        # Post-apply retest
        c.post(f"/api/badcases/{case_a_id}/retest")
        detail = c.get(f"/api/badcases/{case_a_id}")["badcase"]
        check("A stays verifying after post-apply retest", lambda: detail["status"] == "verifying")
        check("A post-apply retest has runtime fields", lambda: assert_retest_has_runtime_fields(detail))
        check("A post-apply retest is newer than apply", lambda: detail.get("last_retest_at", "") >= detail.get("last_applied_at", ""))
        check("A retest trace has stage=retest model_call", lambda: assert_retest_trace_has_retest_model_call(detail.get("retest_trace_id")))

        # Verify-pass -> closed
        c.post(f"/api/badcases/{case_a_id}/verify", {"passed": True, "note": "验收通过"})
        detail = c.get(f"/api/badcases/{case_a_id}")["badcase"]
        check("A status closed", lambda: detail["status"] == "closed")
        check("A terminal has no actions", lambda: detail["allowed_actions"] == [])

        # ------------------------------------------------------------------
        # B. Skill prompt: apply -> verifying -> retest -> verify-pass -> closed
        # ------------------------------------------------------------------
        print("\n[B] Skill prompt post-apply retest lifecycle")
        case_b = new_case("Skill Prompt 绑定", category="skill_prompt")
        case_b_id = case_b["id"]
        target_agent_id, target_agent_name = pick_target_agent()
        print(f"  target agent: {target_agent_id} ({target_agent_name})")

        c.post(f"/api/badcases/{case_b_id}/classify", {"auto": False, "category": "skill_prompt", "reason": "test"})

        darwin_resp = c.raw_post(f"/api/badcases/{case_b_id}/darwin-fix")
        if darwin_resp.status_code not in (200, 201, 202):
            print(f"  SKIP  Darwin unavailable (status {darwin_resp.status_code}); seeding skill draft manually")
        detail = c.get(f"/api/badcases/{case_b_id}")["badcase"]

        if not detail.get("skill_prompt_drafts"):
            check("B skill_prompt draft exists", lambda: False)
        else:
            skill_draft = detail["skill_prompt_drafts"][0]
            skill_draft_id = skill_draft["id"]

            trigger_kw = "V141_SKILL_TRIGGER"
            original_query = f"请使用测试技能 {trigger_kw}"
            c.put(f"/api/badcases/{case_b_id}/skill-prompt-drafts/{skill_draft_id}", {
                "title": f"{TEST_PREFIX}Skill草稿",
                "skill_name": f"{TEST_PREFIX}测试Skill",
                "prompt_content": f"当用户消息包含 '{trigger_kw}' 时，请在回答中明确提到 '{TEST_PREFIX}测试Skill已激活' 并给出简短确认。",
                "trigger_keywords": trigger_kw,
            })

            c.post(f"/api/badcases/{case_b_id}/skill-prompt-drafts/{skill_draft_id}/review", {"status": "under_review"})
            c.post(f"/api/badcases/{case_b_id}/skill-prompt-drafts/{skill_draft_id}/review", {"status": "approved"})
            detail = c.get(f"/api/badcases/{case_b_id}")["badcase"]
            skill_draft = next(d for d in detail["skill_prompt_drafts"] if d["id"] == skill_draft_id)
            check("B skill draft approved", lambda: skill_draft.get("status") == "approved")

            # Pre-apply retest
            c.post(f"/api/badcases/{case_b_id}/retest")
            detail = c.get(f"/api/badcases/{case_b_id}")["badcase"]
            check("B stays fixing after pre-apply retest", lambda: detail["status"] == "fixing")

            # Record original agent skills before binding
            original_agent_skills = c.get(f"/api/agents/{target_agent_id}")["agent"].get("skill_ids", [])

            # Apply skill draft to target agent -> verifying
            c.post(
                f"/api/badcases/{case_b_id}/skill-prompt-drafts/{skill_draft_id}/apply",
                {"target_agent_id": target_agent_id},
            )
            detail = c.get(f"/api/badcases/{case_b_id}")["badcase"]
            check("B case moved to verifying after apply", lambda: detail["status"] == "verifying")
            check("B old retest evidence cleared", lambda: not detail.get("retest_response"))
            check("B verify-pass blocked before post-apply retest", lambda: assert_allowed_after_apply_no_retest(detail))

            # Post-apply retest
            c.post(f"/api/badcases/{case_b_id}/retest")
            detail = c.get(f"/api/badcases/{case_b_id}")["badcase"]
            check("B stays verifying after post-apply retest", lambda: detail["status"] == "verifying")
            check("B post-apply retest has runtime fields", lambda: assert_retest_has_runtime_fields(detail))
            check("B retest trace has stage=retest model_call", lambda: assert_retest_trace_has_retest_model_call(detail.get("retest_trace_id")))

            # Verify skill created and bound
            agent_detail = c.get(f"/api/agents/{target_agent_id}")["agent"]
            created_skill_id = None
            for sid in agent_detail.get("skill_ids", []):
                skill = c.get(f"/api/skills/{sid}").get("skill", {})
                if TEST_PREFIX in (skill.get("name") or ""):
                    created_skill_id = sid
                    break
            check("B skill created and bound to agent", lambda: created_skill_id is not None)
            if created_skill_id:
                created_skill_ids.append(created_skill_id)
                created_agent_skill_bindings.append({
                    "agent_id": target_agent_id,
                    "skill_id": created_skill_id,
                    "original_skill_ids": original_agent_skills,
                })

            # Verify-pass -> closed
            c.post(f"/api/badcases/{case_b_id}/verify", {"passed": True, "note": "验收通过"})
            detail = c.get(f"/api/badcases/{case_b_id}")["badcase"]
            check("B status closed", lambda: detail["status"] == "closed")

        # ------------------------------------------------------------------
        # C. MCP capability gap stays fixing after apply
        # ------------------------------------------------------------------
        print("\n[C] MCP capability gap")
        case_c = new_case("MCP能力缺口", category="mcp_capability")
        case_c_id = case_c["id"]

        c.post(f"/api/badcases/{case_c_id}/classify", {"auto": False, "category": "mcp_capability", "reason": "test"})
        darwin_resp = c.raw_post(f"/api/badcases/{case_c_id}/darwin-fix")
        if darwin_resp.status_code not in (200, 201, 202):
            print(f"  SKIP  Darwin unavailable (status {darwin_resp.status_code}); seeding capability gap manually")
        detail = c.get(f"/api/badcases/{case_c_id}")["badcase"]
        check("C status is fixing", lambda: detail["status"] == "fixing")

        if not detail.get("capability_gap_drafts"):
            check("C capability_gap draft exists", lambda: False)

        if detail.get("capability_gap_drafts"):
            gap_draft = detail["capability_gap_drafts"][0]
            gap_draft_id = gap_draft["id"]

            c.post(f"/api/badcases/{case_c_id}/capability-gap-drafts/{gap_draft_id}/review", {"status": "under_review"})
            c.post(f"/api/badcases/{case_c_id}/capability-gap-drafts/{gap_draft_id}/review", {"status": "approved"})
            c.post(f"/api/badcases/{case_c_id}/capability-gap-drafts/{gap_draft_id}/apply")
            detail = c.get(f"/api/badcases/{case_c_id}")["badcase"]
            gap_draft = next(d for d in detail["capability_gap_drafts"] if d["id"] == gap_draft_id)
            check("C case stays fixing", lambda: detail["status"] == "fixing")
            check("C gap draft accepted", lambda: gap_draft.get("status") == "accepted")

        # ------------------------------------------------------------------
        # D. Verifying without post-apply retest cannot verify-pass
        # ------------------------------------------------------------------
        print("\n[D] Verifying without post-apply retest")
        case_d = new_case("无复测不能通过", category="knowledge_gap")
        case_d_id = case_d["id"]
        c.post(f"/api/badcases/{case_d_id}/classify", {"auto": False, "category": "knowledge_gap", "reason": "test"})
        detail, draft_d_id = ensure_knowledge_draft(case_d_id)
        approve_knowledge_draft(case_d_id, draft_d_id)
        c.post(f"/api/badcases/{case_d_id}/knowledge-drafts/{draft_d_id}/apply")
        detail = c.get(f"/api/badcases/{case_d_id}")["badcase"]
        check("D status is verifying", lambda: detail["status"] == "verifying")
        check("D verify-pass not in allowed_actions", lambda: "verify-pass" not in detail.get("allowed_actions", []))
        err = c.expect_error(f"/api/badcases/{case_d_id}/verify", body={"passed": True, "note": "no retest"}, status=400)
        check("D verify-pass endpoint returns 400", lambda: "复测" in (err.get("detail") or err.get("raw", "")))

        # ------------------------------------------------------------------
        # E. Terminal state enforcement
        # ------------------------------------------------------------------
        print("\n[E] Terminal state enforcement")
        case_e = new_case("终态不可转移", category="other")
        case_e_id = case_e["id"]
        c.post(f"/api/badcases/{case_e_id}/reject", {"rejected_reason": "验收测试驳回"})
        detail = c.get(f"/api/badcases/{case_e_id}")["badcase"]
        check("E rejected is terminal", lambda: detail["status"] == "rejected" and detail.get("is_terminal") is True)
        check("E rejected allowed_actions empty", lambda: detail.get("allowed_actions") == [])
        check("E cannot transition out of rejected", lambda: c.expect_error(
            f"/api/badcases/{case_e_id}/transition", body={"status": "fixing"}, status=400
        ))

        detail = c.get(f"/api/badcases/{case_a_id}")["badcase"]
        check("E closed is terminal", lambda: detail["status"] == "closed" and detail.get("is_terminal") is True)
        check("E closed allowed_actions empty", lambda: detail.get("allowed_actions") == [])
        check("E cannot transition out of closed", lambda: c.expect_error(
            f"/api/badcases/{case_a_id}/transition", body={"status": "fixing"}, status=400
        ))

        # ------------------------------------------------------------------
        # F. Auto-capture evidence
        # ------------------------------------------------------------------
        print("\n[F] Auto-capture evidence")
        auto_created = False
        auto_case_id = None
        try:
            nonsense_term = "V141_NONSENSE_XYZ_99999"
            chat_resp = chat_sse(c, nonsense_term)
            auto_case_id = chat_resp.get("done", {}).get("auto_badcase_id")
            if auto_case_id:
                created_case_ids.append(auto_case_id)
                auto_created = True
                print(f"  auto-captured badcase #{auto_case_id}")
        except Exception as exc:
            print(f"  auto-capture trigger failed: {exc}")

        if not auto_created:
            print("  fallback: creating source=auto badcase manually")
            simulated = c.post("/api/badcases", {
                "title": f"{TEST_PREFIX}自动发现缺口感知",
                "description": "模拟自动捕获",
                "category": "knowledge_gap",
                "source": "auto",
                "original_query": "V141_AUTO_QUERY",
                "ai_response": "",
            })["badcase"]
            auto_case_id = simulated["id"]
            created_case_ids.append(auto_case_id)

        auto_cases = c.get("/api/badcases", params={"source": "auto", "category": "knowledge_gap"}).get("badcases", [])
        check("F list filter finds auto knowledge_gap case", lambda: any(bc["id"] == auto_case_id for bc in auto_cases))

        # ------------------------------------------------------------------
        # G. Cost governance: overview, trace formula, daily/monthly budget block
        # ------------------------------------------------------------------
        print("\n[G] Cost governance")
        set_high_budget()

        # Ensure there is some cost in today/this_month by running an A/B test
        ab = c.post("/api/model-configs/ab-test", {"prompt": "我要投诉楼下噪音太大，物业不作为"})
        ab_trace_id = ab.get("trace_id")
        check("G A/B test returns trace_id", lambda: bool(ab_trace_id))

        overview = c.get("/api/observability/overview")
        for field in ("today", "last_7_days", "this_month", "by_model", "by_stage"):
            check(f"G overview has {field}", lambda f=field: f in overview)

        if ab_trace_id:
            trace_detail = c.get(f"/api/observability/traces/{ab_trace_id}")
            model_calls = trace_detail.get("model_calls", [])
            check("G trace detail has model_calls", lambda: len(model_calls) > 0)
            check("G trace model_call has cost_formula", lambda: all("cost_formula" in mc for mc in model_calls))

        # Test daily budget block
        c.put("/api/observability/budget", {"per_call_threshold_cny": 0.000001, "daily_threshold_cny": 0.000001, "monthly_threshold_cny": 1000000})
        budget = c.get("/api/observability/budget").get("budget", {})
        check("G daily budget returns monthly_threshold_cny", lambda: "monthly_threshold_cny" in budget)

        case_g = new_case("日预算拦截测试", category="knowledge_gap")
        case_g_id = case_g["id"]
        c.post(f"/api/badcases/{case_g_id}/classify", {"auto": False, "category": "knowledge_gap", "reason": "test"})
        darwin_resp = c.raw_post(f"/api/badcases/{case_g_id}/darwin-fix")
        check("G Darwin returns 403 when daily budget blocked", lambda: darwin_resp.status_code == 403)

        # AI-assisted classification should also be blocked
        classify_resp = c.raw_post(f"/api/badcases/{case_g_id}/classify", {"auto": True})
        check("G AI classify returns 403 when daily budget blocked", lambda: classify_resp.status_code == 403)

        # Owner Flash chat should still work
        chat_resp = chat_sse(c, "你好，请自我介绍")
        check("G owner Flash chat still works when daily budget blocked", lambda: len(chat_resp.get("text", "")) > 0)

        # Restore high budget
        set_high_budget()

        # Test monthly budget block
        c.put("/api/observability/budget", {"per_call_threshold_cny": 1000000, "daily_threshold_cny": 1000000, "monthly_threshold_cny": 0.000001})
        case_g2 = new_case("月预算拦截测试", category="knowledge_gap")
        case_g2_id = case_g2["id"]
        c.post(f"/api/badcases/{case_g2_id}/classify", {"auto": False, "category": "knowledge_gap", "reason": "test"})
        darwin_resp2 = c.raw_post(f"/api/badcases/{case_g2_id}/darwin-fix")
        check("G Darwin returns 403 when monthly budget blocked", lambda: darwin_resp2.status_code == 403)

        # Owner Flash chat should still work
        chat_resp2 = chat_sse(c, "月预算测试")
        check("G owner Flash chat still works when monthly budget blocked", lambda: len(chat_resp2.get("text", "")) > 0)

        # Restore high budget for remaining tests
        set_high_budget()

        # ------------------------------------------------------------------
        # H. API Key leak prevention
        # ------------------------------------------------------------------
        print("\n[H] API Key leak prevention")
        chat_resp = chat_sse(c, "你好，请自我介绍")
        full_sse_text = json.dumps(chat_resp.get("events", []) + [chat_resp.get("done", {})], ensure_ascii=False)
        check("H SSE does not expose API key", lambda: check_no_secret(full_sse_text) is None)

        prices_text = json.dumps(c.get("/api/observability/prices"), ensure_ascii=False)
        check("H prices endpoint does not expose API key", lambda: check_no_secret(prices_text) is None)

        configs_text = json.dumps(c.get("/api/model-configs"), ensure_ascii=False)
        check("H model-configs does not expose API key", lambda: check_no_secret(configs_text) is None)

    finally:
        print("\n[cleanup] attempting to remove test data and restore configuration")
        # Restore agent skill bindings before deleting skills
        for binding in created_agent_skill_bindings:
            try:
                c.put(f"/api/agents/{binding['agent_id']}", {"skill_ids": binding["original_skill_ids"]})
                # Verify restoration
                agent_detail = c.get(f"/api/agents/{binding['agent_id']}")["agent"]
                restored_ids = sorted(agent_detail.get("skill_ids", []))
                expected_ids = sorted(binding["original_skill_ids"])
                if restored_ids != expected_ids:
                    record_residual(
                        "agent_skill_binding",
                        binding["agent_id"],
                        error=f"restored skill_ids {restored_ids} != expected {expected_ids}",
                    )
                    print(f"  failed to restore agent {binding['agent_id']} skills to {expected_ids}; got {restored_ids}")
                else:
                    print(f"  restored agent {binding['agent_id']} skills to {expected_ids}")
            except Exception as exc:
                record_residual("agent_skill_binding", binding["agent_id"], error=str(exc))
                print(f"  failed to restore agent skills for {binding['agent_id']}: {exc}")

        for skill_id in created_skill_ids:
            try:
                code, _ = c.delete(f"/api/skills/{skill_id}")
                if 200 <= code < 300:
                    print(f"  deleted skill {skill_id}")
                else:
                    print(f"  failed to delete skill {skill_id}: HTTP {code}")
                    record_residual("skill", skill_id, status_code=code)
            except Exception as exc:
                print(f"  failed to delete skill {skill_id}: {exc}")
                record_residual("skill", skill_id, error=str(exc))

        for doc_id in created_doc_ids:
            try:
                code, _ = c.delete(f"/api/knowledge/docs/{doc_id}")
                if 200 <= code < 300:
                    print(f"  deleted knowledge doc {doc_id}")
                else:
                    print(f"  failed to delete knowledge doc {doc_id}: HTTP {code}")
                    record_residual("knowledge_doc", doc_id, status_code=code)
            except Exception as exc:
                print(f"  failed to delete knowledge doc {doc_id}: {exc}")
                record_residual("knowledge_doc", doc_id, error=str(exc))

        for cid in created_case_ids:
            try:
                detail = c.get(f"/api/badcases/{cid}")["badcase"]
                title = detail.get("title", "")
                if not (title.startswith("DEMO_TEST_V140_") or title.startswith("DEMO_TEST_V141_")):
                    print(f"  skipping deletion of non-test badcase {cid}: {title}")
                    continue
                code, _ = c.delete(f"/api/badcases/{cid}")
                if 200 <= code < 300:
                    print(f"  deleted badcase {cid}")
                else:
                    print(f"  failed to delete badcase {cid}: HTTP {code}")
                    record_residual("badcase", cid, status_code=code)
            except Exception as exc:
                print(f"  failed to delete badcase {cid}: {exc}")
                record_residual("badcase", cid, error=str(exc))

        restore_original_budget()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Created Badcase IDs: {created_case_ids}")
    print(f"Created Knowledge Doc IDs: {created_doc_ids}")
    print(f"Created Skill IDs: {created_skill_ids}")
    print(f"Created Agent Skill Bindings: {created_agent_skill_bindings}")
    print(f"Residuals: {residuals}")

    if failed:
        return 1
    if residuals:
        print("\nFAILURE: test data residuals or config restore failures remain; see Residuals list above.")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V1.4.1 Operations Integrity acceptance tests")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="API base URL")
    args = parser.parse_args()
    sys.exit(run(args.base))
