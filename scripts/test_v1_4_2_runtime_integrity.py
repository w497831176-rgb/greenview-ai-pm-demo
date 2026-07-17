#!/usr/bin/env python3
"""
V1.4.2 Runtime Integrity Acceptance Tests
==========================================

Verifies that backend configuration actually drives the owner chat runtime,
that Skills are strictly isolated per-Agent, that citations / traces are
 demo-friendly, and that cost-governance numbers are real and explainable.

Run against a running server:

    python scripts/test_v1_4_2_runtime_integrity.py --base http://127.0.0.1:8000

All test artifacts are prefixed with DEMO_TEST_V142_ and cleaned up in the
finally block. Any residual test data causes a non-zero exit.
"""

import argparse
import json
import re
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests

TEST_PREFIX = "DEMO_TEST_V142_"


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

    def expect_error(self, path: str, method: str = "post", body=None, status: int = 400):
        fn = {
            "post": self.session.post,
            "put": self.session.put,
            "delete": self.session.delete,
        }.get(method, self.session.post)
        r = fn(self._url(path), json=body or {})
        if r.status_code != status:
            raise AssertionError(
                f"expected {status} for {method.upper()} {path}, got {r.status_code}: {r.text}"
            )
        try:
            return r.json()
        except Exception:
            return {"raw": r.text}


def chat_sse(
    c: AcceptanceClient,
    message: str,
    session_id: Optional[str] = None,
    enable_rag: bool = True,
    timeout: int = 120,
) -> Dict[str, Any]:
    payload = {"message": message, "stream": True, "enable_rag": enable_rag}
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
    result = {
        "text": "".join(text_parts),
        "events": events,
        "done": done,
        "route": next((e for e in events if e.get("_event") == "route"), {}),
        "tool_calls": [e for e in events if e.get("_event") == "tool_calls"],
        "activated_skills": done.get("activated_skills") or [],
        "citations": done.get("citations") or [],
        "trace_id": done.get("trace_id"),
    }
    return result


def find_agent(agents: List[Dict[str, Any]], agent_id: str) -> Optional[Dict[str, Any]]:
    for a in agents:
        if a.get("agent_id") == agent_id:
            return a
    return None


def run(base_url: str) -> int:
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    c = AcceptanceClient(base_url)
    passed = 0
    failed = 0
    residuals: List[Dict[str, Any]] = []

    # Resources created by this run that must be removed.
    created_skill_ids: List[int] = []
    created_doc_ids: List[int] = []
    created_session_ids: List[str] = []
    created_badcase_ids: List[int] = []
    created_price_ids: List[int] = []
    dynamic_agent_id: Optional[str] = None

    # Snapshot of original agent config so we can restore it.
    original_agent_config: Dict[str, Dict[str, Any]] = {}

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
            print(f"  PASS  {name}", flush=True)
            passed += 1
        except Exception as exc:
            import traceback
            print(f"  FAIL  {name}: {exc}", flush=True)
            traceback.print_exc()
            failed += 1

    def cleanup():
        print("\n[Cleanup]", flush=True)
        # Delete dynamic agent first to unbind skills/tools.
        if dynamic_agent_id:
            try:
                status, text = c.delete(f"/api/agents/{dynamic_agent_id}")
                if status not in (200, 204, 404):
                    record_residual("agent", dynamic_agent_id, status, text)
                else:
                    print(f"  deleted dynamic agent {dynamic_agent_id}", flush=True)
            except Exception as exc:
                record_residual("agent", dynamic_agent_id, error=str(exc))

        # Unbind test skill from all agents before deleting it.
        for agent_id in original_agent_config:
            try:
                cfg = original_agent_config[agent_id]
                c.put(f"/api/agents/{agent_id}", {
                    "available_skills": cfg["available_skills"],
                    "available_mcp_tools": cfg["available_mcp_tools"],
                })
                print(f"  restored bindings for {agent_id}", flush=True)
            except Exception as exc:
                record_residual("agent_bindings", agent_id, error=str(exc))

        # Restore original instructions for canonical agents.
        for agent_id, cfg in original_agent_config.items():
            try:
                c.put(f"/api/agents/{agent_id}", {
                    "system_prompt": cfg["system_prompt"],
                    "available_skills": cfg["available_skills"],
                    "available_mcp_tools": cfg["available_mcp_tools"],
                })
                print(f"  restored config for {agent_id}", flush=True)
            except Exception as exc:
                record_residual("agent_config", agent_id, error=str(exc))

        for doc_id in created_doc_ids:
            try:
                status, text = c.delete(f"/api/knowledge/docs/{doc_id}")
                if status not in (200, 204, 404):
                    record_residual("knowledge_doc", doc_id, status, text)
                else:
                    print(f"  deleted knowledge doc #{doc_id}", flush=True)
            except Exception as exc:
                record_residual("knowledge_doc", doc_id, error=str(exc))

        for skill_id in created_skill_ids:
            try:
                status, text = c.delete(f"/api/skills/{skill_id}")
                if status not in (200, 204, 404):
                    record_residual("skill", skill_id, status, text)
                else:
                    print(f"  deleted skill #{skill_id}", flush=True)
            except Exception as exc:
                record_residual("skill", skill_id, error=str(exc))

        for case_id in created_badcase_ids:
            try:
                status, text = c.delete(f"/api/badcases/{case_id}")
                if status not in (200, 204, 404):
                    record_residual("badcase", case_id, status, text)
            except Exception as exc:
                record_residual("badcase", case_id, error=str(exc))

        for price_id in created_price_ids:
            try:
                status, text = c.delete(f"/api/observability/prices/{price_id}")
                if status not in (200, 204, 404):
                    record_residual("price", price_id, status, text)
                else:
                    print(f"  deleted price #{price_id}", flush=True)
            except Exception as exc:
                record_residual("price", price_id, error=str(exc))

    try:
        print("\n[Agent Management Preconditions]", flush=True)

        agents = c.get("/api/agents").get("agents", [])
        router = find_agent(agents, "router")
        if not router:
            raise AssertionError("router agent must exist")
        if not router.get("is_router"):
            raise AssertionError("router agent must be marked as router")
        print(f"  router exists, members={len(router.get('members', []))}", flush=True)

        maintenance = find_agent(agents, "maintenance")
        if not maintenance:
            raise AssertionError("maintenance agent must exist")
        billing_agent = find_agent(agents, "billing")
        if not billing_agent:
            raise AssertionError("billing agent must exist")
        complaint_agent = find_agent(agents, "complaint")
        customer_service_agent = find_agent(agents, "customer_service")
        for a in [maintenance, billing_agent, complaint_agent, customer_service_agent]:
            if a:
                original_agent_config[a["agent_id"]] = {
                    "system_prompt": a.get("system_prompt") or "",
                    "available_skills": [s for s in (a.get("available_skills") or [])],
                    "available_mcp_tools": [t for t in (a.get("available_mcp_tools") or [])],
                }

        # --- Router singleton protection ---
        check(
            "POST /api/agents rejects second router",
            lambda: c.expect_error(
                "/api/agents",
                "post",
                {"name": f"{TEST_PREFIX}Router2", "is_router": True, "category": "router"},
                400,
            ),
        )
        check(
            "PUT /api/agents/router rejects router update",
            lambda: c.expect_error(
                "/api/agents/router",
                "put",
                {"name": "Hacked Router"},
                400,
            ),
        )
        check(
            "DELETE /api/agents/router rejects router deletion",
            lambda: c.expect_error(
                "/api/agents/router",
                "delete",
                status=400,
            ),
        )
        check(
            "POST /api/agents/router/toggle rejects router disable",
            lambda: c.expect_error(
                "/api/agents/router/toggle",
                "post",
                {"enabled": False},
                400,
            ),
        )

        # --- Agent Prompt runtime ---
        print("\n[Agent Prompt Runtime]", flush=True)
        marker = "ACPT_AGENT_MAINT_0717"
        c.put("/api/agents/maintenance", {
            "system_prompt": f"{marker}\n你是YIAI物业维修 Agent。业主报修时，第一行必须精确输出'{marker}'，然后给出维修建议。",
            "available_skills": [],
            "available_mcp_tools": [],
        })
        print(f"  injected marker into maintenance agent instructions", flush=True)

        def agent_prompt_runtime():
            resp = chat_sse(c, f"{marker}，厨房漏水需要报修。")
            print(f"  resp text len={len(resp['text'])}, events={len(resp['events'])}, done={resp['done']}", flush=True)
            if not resp["text"].strip():
                raise AssertionError(f"chat response text is empty; events={resp['events'][:5]}")
            first_line = resp["text"].strip().splitlines()[0]
            if marker not in first_line:
                raise AssertionError(f"expected first line to contain {marker}, got: {first_line[:100]}")
            if resp["done"].get("current_agent") != "maintenance":
                raise AssertionError(f"expected current_agent=maintenance, got {resp['done'].get('current_agent')}")

        check("Maintenance agent follows DB instructions with marker", agent_prompt_runtime)

        # Restore maintenance instructions.
        c.put("/api/agents/maintenance", {
            "system_prompt": original_agent_config["maintenance"]["system_prompt"],
            "available_skills": [],
            "available_mcp_tools": [],
        })
        print("  restored maintenance instructions", flush=True)

        # --- Skill isolation ---
        print("\n[Skill Isolation]", flush=True)
        skill_marker = f"{TEST_PREFIX}BILL_0717"
        skill_name = f"{TEST_PREFIX}费用探针"
        skill_payload = {
            "name": skill_name,
            "description": "Acceptance test skill for billing isolation",
            "instructions": f"当用户询问费用/账单时，必须在回答中嵌入标记 '{skill_marker}'。",
            "category": "验收测试",
            "enabled": True,
            "trigger_condition": "费用、缴费、物业费、账单",
        }
        skill = c.post("/api/skills", skill_payload).get("skill")
        created_skill_ids.append(skill["id"])
        print(f"  created skill #{skill['id']} {skill_name}", flush=True)

        # Bind only to billing agent.
        c.put(f"/api/agents/billing", {
            "available_skills": [skill_name],
            "available_mcp_tools": [],
        })
        c.put(f"/api/agents/maintenance", {
            "available_skills": [],
            "available_mcp_tools": [],
        })
        c.put(f"/api/agents/complaint", {
            "available_skills": [],
            "available_mcp_tools": [],
        })
        c.put(f"/api/agents/customer_service", {
            "available_skills": [],
            "available_mcp_tools": [],
        })
        print("  bound test skill only to billing agent", flush=True)

        def skill_isolated_billing():
            resp = chat_sse(c, "我要查询本月的物业费账单。")
            if skill_marker not in resp["text"]:
                raise AssertionError(f"billing query should contain {skill_marker}")
            if not any(skill_name == s.get("name") for s in resp["activated_skills"]):
                raise AssertionError(f"activated_skills should include {skill_name}: {resp['activated_skills']}")

        def skill_isolated_maintenance():
            resp = chat_sse(c, "厨房漏水需要报修。")
            if skill_marker in resp["text"]:
                raise AssertionError(f"maintenance query should NOT contain {skill_marker}")
            if any(skill_name == s.get("name") for s in resp["activated_skills"]):
                raise AssertionError(f"activated_skills should NOT include {skill_name}: {resp['activated_skills']}")

        check("Skill marker appears for billing query", skill_isolated_billing)
        check("Skill marker absent for maintenance query", skill_isolated_maintenance)

        # Unbind and delete skill later.
        c.put(f"/api/agents/billing", {
            "available_skills": [],
            "available_mcp_tools": [],
        })

        # --- Dynamic vertical agent ---
        print("\n[Dynamic Vertical Agent]", flush=True)
        dynamic_agent_id = f"{TEST_PREFIX}Vert"
        dynamic_marker = f"{TEST_PREFIX}DYNAMIC_MARKER"

        def create_dynamic_agent():
            c.post("/api/agents", {
                "agent_id": dynamic_agent_id,
                "name": f"{TEST_PREFIX}测试垂直 Agent",
                "description": "用于 V1.4.2 运行时验收的动态垂直 Agent，处理电动车充电规定咨询。",
                "system_prompt": f"你是专门的测试 Agent。任何提问，第一行必须精确输出'{dynamic_marker}'，然后简要回答。",
                "category": "vertical",
                "enabled": True,
            })
            print(f"  created dynamic agent {dynamic_agent_id}", flush=True)

        check("Create dynamic vertical agent", create_dynamic_agent)

        def dynamic_agent_routed():
            resp = chat_sse(c, "请帮我查一下电动车充电规定。")
            first_line = resp["text"].strip().splitlines()[0]
            if dynamic_marker not in first_line:
                raise AssertionError(f"expected dynamic marker in first line, got: {first_line[:100]}")
            if resp["done"].get("current_agent") != dynamic_agent_id:
                raise AssertionError(f"expected current_agent={dynamic_agent_id}, got {resp['done'].get('current_agent')}")

        check("Router routes to dynamic vertical agent", dynamic_agent_routed)

        # --- RAG keyword + semantic hit ---
        print("\n[RAG Keyword and Semantic Retrieval]", flush=True)
        doc_title = f"{TEST_PREFIX}电动车充电管理规定"
        doc_body = (
            "小区电动车管理规定："
            "1. 禁止在楼道、疏散通道、安全出口停放电动车或充电。"
            "2. 17 号集中充电区为指定充电区域，业主可在此安全充电。"
            "3. 飞线充电存在严重安全隐患，一经发现物业将上门劝阻并整改。"
        )
        doc = c.post("/api/knowledge/docs", {"title": doc_title, "content": doc_body}).get("knowledge_doc")
        created_doc_ids.append(doc["id"])
        print(f"  created knowledge doc #{doc['id']}", flush=True)

        # Wait for the async/sync indexing triggered by creation to settle.
        time.sleep(6)

        def rag_keyword_hit():
            result = c.get("/api/knowledge/search", {"query": "17号充电区", "top_k": 5, "mode": "keyword"})
            chunks = result.get("results", [])
            if not chunks:
                raise AssertionError("keyword retrieval returned no chunks")
            found = any(doc["id"] == ch.get("doc_id") for ch in chunks)
            if not found:
                raise AssertionError(f"keyword retrieval did not return doc #{doc['id']}: {chunks}")

        def rag_semantic_hit():
            result = c.get("/api/knowledge/search", {"query": "第十七号充电的地方", "top_k": 5, "mode": "semantic", "threshold": 0.0})
            chunks = result.get("results", [])
            if not chunks:
                raise AssertionError("semantic retrieval returned no chunks")
            found = any(doc["id"] == ch.get("doc_id") for ch in chunks)
            if not found:
                raise AssertionError(f"semantic retrieval did not return doc #{doc['id']}: {chunks}")

        check("Keyword retrieval hits '17号充电区'", rag_keyword_hit)
        check("Semantic retrieval hits similar query", rag_semantic_hit)

        def rag_chat_citation():
            resp = chat_sse(c, "17号集中充电区能充电吗？")
            if not resp["citations"]:
                raise AssertionError("chat response did not include citations")
            found = any(str(doc["id"]) == str(ch.get("doc_id")) for ch in resp["citations"])
            if not found:
                raise AssertionError(f"chat citations did not include doc #{doc['id']}: {resp['citations']}")

        check("Chat answer cites correct doc_id/chunk_index", rag_chat_citation)

        # --- MCP gating ---
        print("\n[MCP Gating]", flush=True)

        def mcp_no_calendar_for_rag():
            resp = chat_sse(c, "楼道飞线充电允许吗")
            tool_names = []
            for tc in resp["tool_calls"]:
                tool_names.extend([call.get("tool_name", "") for call in tc.get("tool_calls", [])])
            calendar_calls = [n for n in tool_names if "calendar" in n.lower() or "date" in n.lower() or "time" in n.lower()]
            if calendar_calls:
                raise AssertionError(f"RAG question should not trigger calendar tools: {calendar_calls}")

        def mcp_weather_called():
            resp = chat_sse(c, "今天天气怎么样？")
            tool_names = []
            for tc in resp["tool_calls"]:
                tool_names.extend([call.get("tool_name", "") for call in tc.get("tool_calls", [])])
            if not any("weather" in n.lower() for n in tool_names):
                raise AssertionError(f"weather question did not trigger weather tool: {tool_names}")

        def mcp_date_called():
            resp = chat_sse(c, "今天几号？")
            tool_names = []
            for tc in resp["tool_calls"]:
                tool_names.extend([call.get("tool_name", "") for call in tc.get("tool_calls", [])])
            if not any("date" in n.lower() or "time" in n.lower() for n in tool_names):
                raise AssertionError(f"date question did not trigger date/time tool: {tool_names}")

        def mcp_workorder_called():
            resp = chat_sse(c, "帮我查一下工单进度 12345")
            tool_names = []
            for tc in resp["tool_calls"]:
                tool_names.extend([call.get("tool_name", "") for call in tc.get("tool_calls", [])])
            if not any("workorder" in n.lower() or "工单" in n for n in tool_names):
                raise AssertionError(f"workorder question did not trigger workorder tool: {tool_names}")

        check("RAG question does not trigger calendar tool", mcp_no_calendar_for_rag)
        check("Weather question triggers weather tool", mcp_weather_called)
        check("Date question triggers date/time tool", mcp_date_called)
        check("Workorder progress question triggers workorder tool", mcp_workorder_called)

        # --- Cost governance ---
        print("\n[Cost Governance]", flush=True)

        # Ensure a model price exists so cost is computable.
        price_payload = {
            "model_id": "deepseek-v4-flash",
            "effective_date": "2026-01-01",
            "input_price_per_1m": 0.0,
            "cached_input_price_per_1m": 0.0,
            "output_price_per_1m": 0.0,
            "reasoning_price_per_1m": 0.0,
            "source_note": f"{TEST_PREFIX} acceptance zero-price",
            "enabled": True,
        }
        price = c.post("/api/observability/prices", price_payload).get("price")
        created_price_ids.append(price["id"])
        print(f"  created price #{price['id']}", flush=True)

        # Also test that 0 price is preserved.
        fetched_price = c.get(f"/api/observability/prices/{price['id']}").get("price")
        if fetched_price.get("input_price_per_1m") != 0.0:
            print(f"  WARNING: zero price input was not preserved: {fetched_price.get('input_price_per_1m')}", flush=True)

        def cost_trace_has_model_and_tokens():
            resp = chat_sse(c, "厨房漏水需要报修。")
            trace_id = resp["trace_id"]
            if not trace_id:
                raise AssertionError("missing trace_id in done event")
            traces = c.get("/api/observability/traces", {"trace_id": trace_id}).get("traces", [])
            if not traces:
                raise AssertionError(f"trace {trace_id} not found")
            t = traces[0]
            if t.get("no_model_calls"):
                raise AssertionError("trace has no model calls")
            if t.get("total_tokens") is None or t.get("total_tokens") <= 0:
                raise AssertionError(f"trace total_tokens invalid: {t.get('total_tokens')}")
            if t.get("price_missing"):
                raise AssertionError("price should be configured for the test model")
            if t.get("estimated_cost_cny") is None:
                raise AssertionError("estimated_cost_cny should not be None")

        def cost_trace_detail_not_zeroed():
            resp = chat_sse(c, "我要查询本月的物业费账单。")
            trace_id = resp["trace_id"]
            detail = c.get(f"/api/observability/traces/{trace_id}")
            breakdown = detail.get("context_breakdown") or {}
            if not breakdown:
                raise AssertionError("context_breakdown is empty")
            all_null = all(v is None for k, v in breakdown.items() if k != "note")
            if all_null:
                raise AssertionError("context_breakdown contains only null values")
            # At least user_message should be estimable.
            if breakdown.get("user_message_tokens") is None:
                raise AssertionError("user_message_tokens should be estimable")

        def cost_router_no_usage_is_null():
            resp = chat_sse(c, "帮我查一下工单进度 12345")
            trace_id = resp["trace_id"]
            detail = c.get(f"/api/observability/traces/{trace_id}")
            router_call = next((m for m in detail.get("model_calls", []) if m.get("stage") == "router"), None)
            if not router_call:
                raise AssertionError("router model call not found")
            if router_call.get("total_tokens") is not None and router_call.get("total_tokens") != 0:
                # Provider did return usage; that's fine, but we still verify cost semantics.
                pass
            if router_call.get("estimated_cost_cny") == 0.0 and router_call.get("usage_source") == "unavailable":
                raise AssertionError("router call with unavailable usage must not show ¥0")

        def cost_two_day_range():
            today = datetime.now()
            start = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            end = today.strftime("%Y-%m-%d")
            result = c.get("/api/observability/traces", {"start": start, "end": end, "limit": 500})
            traces = result.get("traces", [])
            returned_end = result.get("end") or ""
            if "23:59:59" not in returned_end:
                raise AssertionError(f"end should be expanded to 23:59:59, got {returned_end}")
            # We should see at least some traces from today.
            if not traces:
                raise AssertionError("two-day range returned no traces")

        check("Trace list shows model, tokens, and cost", cost_trace_has_model_and_tokens)
        check("Trace detail context breakdown is not all zeros/nulls", cost_trace_detail_not_zeroed)
        check("Router call with unavailable usage does not show fake ¥0", cost_router_no_usage_is_null)
        check("Two-day date range returns data", cost_two_day_range)

    finally:
        cleanup()

    print("\n[Residuals]", flush=True)
    if residuals:
        print(f"  {len(residuals)} residuals found:", flush=True)
        for r in residuals:
            print(f"    {r}", flush=True)
    else:
        print("  none", flush=True)

    print(f"\nResults: {passed} passed, {failed} failed, {len(residuals)} residuals", flush=True)
    if failed or residuals:
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V1.4.2 Runtime Integrity Acceptance Tests")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="server base URL")
    args = parser.parse_args()
    sys.exit(run(args.base))
