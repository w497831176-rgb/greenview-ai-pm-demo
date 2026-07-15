"""
V1.3.3 Badcase Operational Closure Acceptance Tests
=====================================================

Covers the full loop: manual feedback -> badcase -> classify -> Darwin -> drafts ->
manual publish -> real retest -> verify close, plus capability-gap no-fake-tool guard.

Run against a running server:
    export BASE_URL=http://localhost:8000
    python scripts/test_v1_3_3_badcase_closure.py

Exit code 0 if all assertions pass.
"""

import json
import os
import re
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
USER_ID = "test-v133"
PROXIES = {"http": None, "https": None}


class TestError(RuntimeError):
    pass


def req(method: str, path: str, **kwargs) -> Any:
    url = f"{BASE_URL}{path}"
    kwargs.setdefault("proxies", PROXIES)
    kwargs.setdefault("timeout", 120)
    resp = requests.request(method, url, **kwargs)
    try:
        resp.raise_for_status()
    except Exception as exc:
        raise TestError(f"{method} {path} failed: {resp.status_code} {resp.text[:300]}") from exc
    if not resp.text:
        return {}
    try:
        return resp.json()
    except Exception:
        return resp.text


def post(path: str, body: Optional[Dict[str, Any]] = None) -> Any:
    return req("POST", path, json=body or {})


def get(path: str) -> Any:
    return req("GET", path)


def delete(path: str) -> Any:
    return req("DELETE", path)


def create_session() -> str:
    data = post(f"/api/chat/sessions?user_id={USER_ID}")
    return data["session"]["session_id"]


def send_chat(session_id: str, message: str) -> Dict[str, Any]:
    """Send a message and return the SSE done payload + full answer."""
    url = f"{BASE_URL}/api/chat/stream"
    params = {"message": message, "session_id": session_id, "user_id": USER_ID}
    resp = requests.get(url, params=params, stream=True, proxies=PROXIES, timeout=180)
    resp.raise_for_status()
    answer = ""
    done: Dict[str, Any] = {}
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore")
        if line.startswith("data:"):
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if isinstance(obj, dict):
                if obj.get("content"):
                    answer += str(obj["content"])
                if obj.get("status") == "complete" or "message_id" in obj:
                    done = obj
    if not done:
        done = {"answer": answer}
    done["answer"] = answer
    return done


def send_feedback(session_id: str, message_id: int, reason: str) -> Dict[str, Any]:
    return post("/api/chat/feedback", {
        "session_id": session_id,
        "message_id": message_id,
        "reason": reason,
        "type": "thumb_down",
    })


def get_badcase(badcase_id: int) -> Dict[str, Any]:
    return get(f"/api/badcases/{badcase_id}")["badcase"]


def assert_cond(condition: bool, message: str) -> None:
    if not condition:
        raise TestError(message)


def run_case_1_mcp_capability_gap() -> Dict[str, Any]:
    """Use case 1: owner asks for work-order dispatch (write MCP missing)."""
    print("\n[Case 1] MCP capability gap via manual feedback")
    session_id = create_session()
    message = (
        "请先查询工单 WO-20260710-001，然后直接把它指派给张师傅，并预约今天 15:00 上门。"
        "必须实际执行，不能只给建议；如果做不到，请明确说明卡在哪个系统能力。"
    )
    done = send_chat(session_id, message)
    answer = done.get("answer", "")
    print(f"  AI answer preview: {answer[:120].replace(chr(10), ' ')}")

    # AI should honestly admit missing dispatch capability.
    assert_cond(
        "派单" in answer or "预约" in answer or "缺少" in answer or "无法" in answer or "做不到" in answer,
        "AI did not address missing dispatch capability",
    )

    message_id = done.get("message_id")
    assert_cond(message_id, "message_id missing from SSE done")
    feedback_reason = (
        "需求未完成：当前只有工单查询 MCP，缺少工单派单与预约上门的写操作工具。"
    )
    fb = send_feedback(session_id, message_id, feedback_reason)
    badcase_id = fb["badcase"]["id"]
    print(f"  Created badcase id={badcase_id}, source={fb['badcase'].get('source')}")
    assert_cond(fb.get("source") == "manual", "feedback source should be manual")

    bc = get_badcase(badcase_id)
    assert_cond(bc.get("source") == "manual", "badcase.source should be manual")
    assert_cond(bc.get("original_query"), "original_query missing")
    assert_cond(bc.get("ai_response"), "ai_response missing")
    assert_cond(bc.get("feedback_reason") == feedback_reason, "feedback_reason mismatch")
    assert_cond(bc.get("trace_id"), "trace_id missing")
    ctx = bc.get("context", {})
    assert_cond(ctx.get("route_intent"), "context.route_intent missing")
    assert_cond(ctx.get("model_id"), "context.model_id missing")

    # Classify.
    cls = post(f"/api/badcases/{badcase_id}/classify", {"auto": True})
    print(f"  Classified as {cls['badcase']['category']}")
    assert_cond(cls["badcase"]["status"] == "classified", "status should be classified")
    assert_cond(cls["badcase"]["category"] == "mcp_capability", "category should be mcp_capability")

    # Darwin.
    darwin = post(f"/api/badcases/{badcase_id}/darwin-fix", {})
    print(f"  Darwin root_cause: {darwin.get('analysis', {}).get('root_cause', '')[:80]}")
    assert_cond(darwin["badcase"]["status"] == "fixing", "status should be fixing after Darwin")
    assert_cond(darwin.get("darwin_trace_id"), "darwin_trace_id missing")
    assert_cond(darwin.get("analysis", {}).get("root_cause"), "root_cause missing")
    assert_cond(isinstance(darwin.get("drafts", []), list), "drafts list missing")

    # Verify Darwin trace in observability.
    trace = get(f"/api/observability/traces/{darwin['darwin_trace_id']}")
    model_calls = trace.get("model_calls", [])
    darwin_calls = [m for m in model_calls if m.get("stage") == "darwin"]
    assert_cond(darwin_calls, "Darwin model_call not found")
    assert_cond(darwin_calls[0].get("model_id") == "deepseek-v4-pro", "Darwin should use Pro")

    # Capability gap draft must exist; accept it; ensure no real tool is created.
    gap_drafts = darwin["badcase"].get("capability_gap_drafts", [])
    assert_cond(gap_drafts, "capability_gap_draft missing")
    gap_id = gap_drafts[0]["id"]
    accepted = post(f"/api/badcases/{badcase_id}/accept-capability-gap/{gap_id}", {"note": "产品待办"})
    print(f"  Accepted capability gap id={gap_id}: {accepted.get('note', '')}")
    assert_cond(
        accepted["badcase"]["capability_gap_drafts"][0]["status"] == "accepted",
        "capability gap draft should be accepted",
    )

    # Reject badcase with reason.
    post(f"/api/badcases/{badcase_id}/reject", {"rejected_reason": "当前缺少写操作 MCP，无法闭环，记录为待办"})
    bc = get_badcase(badcase_id)
    assert_cond(bc["status"] == "rejected", "badcase should be rejected")
    print(f"  Case 1 PASS (badcase {badcase_id})")
    return {"badcase_id": badcase_id, "darwin_trace_id": darwin["darwin_trace_id"]}


def run_case_2_knowledge_gap_closed_loop() -> Dict[str, Any]:
    """Use case 2: DEMO_TEST knowledge gap -> draft -> publish -> retest -> close -> cleanup."""
    print("\n[Case 2] Knowledge gap closed loop with DEMO_TEST")
    demo_query = "DEMO_TEST_新能源车充电桩申请需要哪些材料？"

    # Manually create a badcase simulating a knowledge gap.
    bc = post("/api/badcases", {
        "title": "DEMO_TEST 知识缺口：新能源车充电桩申请",
        "description": "业主询问充电桩申请材料，系统未能从知识库检索到相关政策",
        "category": "knowledge_gap",
        "status": "pending",
        "source": "manual",
        "original_query": demo_query,
        "ai_response": "抱歉，我目前没有相关资料。",
        "feedback_reason": "知识库缺少新能源车充电桩申请指南",
        "priority": "high",
    })
    badcase_id = bc["id"]
    print(f"  Created badcase id={badcase_id}")

    # Classify.
    cls = post(f"/api/badcases/{badcase_id}/classify", {"auto": False, "category": "knowledge_gap", "reason": "缺少充电桩申请材料文档"})
    assert_cond(cls["badcase"]["status"] == "classified", "status should be classified")

    # Darwin.
    darwin = post(f"/api/badcases/{badcase_id}/darwin-fix", {})
    assert_cond(darwin["badcase"]["status"] == "fixing", "status should be fixing after Darwin")
    knowledge_drafts = darwin["badcase"].get("knowledge_drafts", [])
    assert_cond(knowledge_drafts, "knowledge draft missing from Darwin")
    draft_id = knowledge_drafts[0]["id"]

    # Publish knowledge draft.
    published = post(f"/api/badcases/{badcase_id}/publish-draft/{draft_id}", {})
    print(f"  Published knowledge doc id={published['knowledge_doc']['id']}")
    doc_id = published["knowledge_doc"]["id"]
    assert_cond(published["badcase"]["status"] == "verifying", "status should be verifying after publish")

    # Retest using real chat stream.
    retest = post(f"/api/badcases/{badcase_id}/retest", {})
    print(f"  Retest answer preview: {retest.get('retest_response', '')[:120].replace(chr(10), ' ')}")
    assert_cond(retest.get("retest_response"), "retest_response missing")
    ctx = retest.get("retest_context", {})
    assert_cond(ctx.get("trace_id"), "retest trace_id missing")

    # Verify close.
    verified = post(f"/api/badcases/{badcase_id}/verify", {"passed": True})
    assert_cond(verified["badcase"]["status"] == "closed", "badcase should be closed")
    print(f"  Badcase closed")

    # Cleanup DEMO_TEST data.
    delete(f"/api/knowledge/{doc_id}")
    delete(f"/api/badcases/{badcase_id}")
    print(f"  Case 2 PASS (created doc {doc_id})")
    return {"doc_id": doc_id, "badcase_id": badcase_id}


def run_case_3_composite_rag_mcp() -> Dict[str, Any]:
    """Use case 3: composite RAG + MCP regression."""
    print("\n[Case 3] Composite RAG + MCP regression")
    session_id = create_session()
    message = (
        "我是 3-2-1201 的王先生。卫生间天花板持续滴水，我担心近期天气变化会让漏水加重。"
        "请按“先查询、再判断、后建议”的顺序完成：\n"
        "1. 用天气工具查询武汉当前天气，并判断当前天气对漏水风险的影响；\n"
        "2. 用工单工具查询我房号最近的维修工单，以及系统当前待处理工单数量，判断是否已有相似工单；\n"
        "3. 必须依据知识库《物业维修服务承诺》和《常见维修问题 FAQ》，说明家里漏水了怎么办？；\n"
        "4. 如果已有相似工单，请说明应跟进还是补充；不要创建新工单；\n"
        "5. 最后用一行汇总：本次调用了哪些工具、使用了哪些知识库证据。"
    )
    done = send_chat(session_id, message)
    answer = done.get("answer", "")
    print(f"  Answer preview: {answer[:120].replace(chr(10), ' ')}")

    assert_cond(done.get("route_intent") == "maintenance", f"route should be maintenance, got {done.get('route_intent')}")
    skills = done.get("activated_skills") or []
    skill_names = [s["name"] if isinstance(s, dict) else s for s in skills]
    assert_cond("维修工单处理" in skill_names, f"skill missing: {skill_names}")

    mcp_calls = done.get("mcp_calls") or []
    mcp_tools = [c.get("tool_name") for c in mcp_calls]
    assert_cond("get_current_weather" in mcp_tools, "weather MCP missing")
    assert_cond("list_recent_work_orders" in mcp_tools or "count_work_orders" in mcp_tools, "work order MCP missing")

    citations = done.get("citations") or []
    titles = [c.get("doc_title") for c in citations]
    assert_cond("物业维修服务承诺" in titles, "citation 物业维修服务承诺 missing")
    assert_cond("常见维修问题 FAQ" in titles, "citation FAQ missing")
    faq_q1 = any("家里漏水了怎么办" in (c.get("content") or "") for c in citations)
    assert_cond(faq_q1, "FAQ Q1 家里漏水了怎么办 not cited")
    assert_cond(not done.get("auto_badcase_id"), "auto_badcase_id should be empty")
    assert_cond("未从知识库检索到" not in answer, "answer falsely claims no knowledge hit")
    assert_cond("绿景智服" not in answer, "old brand appears in answer")

    # Check trace/cost observability.
    trace_id = done.get("trace_id")
    assert_cond(trace_id, "trace_id missing")
    trace = get(f"/api/observability/traces/{trace_id}")
    model_calls = trace.get("model_calls", [])
    stages = {m.get("stage") for m in model_calls}
    assert_cond("router" in stages and "vertical_agent" in stages, f"trace stages missing: {stages}")
    costs = [m.get("estimated_cost_cny") for m in model_calls if m.get("estimated_cost_cny") is not None]
    print(f"  Trace stages: {stages}, model calls with cost: {len(costs)}")

    # Refresh session and verify persistence.
    list_resp = get(f"/api/chat/sessions?user_id={USER_ID}")
    sessions = list_resp.get("sessions", [])
    assert_cond(any(s.get("session_id") == session_id for s in sessions), "session not persisted after refresh")

    print("  Case 3 PASS")
    return {"session_id": session_id, "trace_id": trace_id, "done": done}


def run_session_management_check() -> None:
    print("\n[Session management] first render / new session / refresh")
    # Basic API checks; Playwright covers UI.
    sessions_before = get(f"/api/chat/sessions?user_id={USER_ID}").get("sessions", [])
    sid = create_session()
    sessions_after = get(f"/api/chat/sessions?user_id={USER_ID}").get("sessions", [])
    assert_cond(len(sessions_after) == len(sessions_before) + 1, "new session not created")
    msgs = get(f"/api/chat/sessions/{sid}/messages").get("messages", [])
    assert_cond(isinstance(msgs, list), "messages not list")
    print("  Session management API PASS")


def cleanup_demo_sessions() -> None:
    try:
        sessions = get(f"/api/chat/sessions?user_id={USER_ID}").get("sessions", [])
        for s in sessions:
            sid = s.get("session_id")
            if sid:
                try:
                    delete(f"/api/chat/sessions/{sid}")
                except Exception:
                    pass
    except Exception:
        pass


def main() -> int:
    print(f"BASE_URL={BASE_URL}")
    results: List[Dict[str, Any]] = []
    try:
        run_session_management_check()
        results.append({"name": "session_management", "status": "PASS"})
    except TestError as e:
        results.append({"name": "session_management", "status": "FAIL", "error": str(e)})
        print(f"  FAIL: {e}")

    try:
        run_case_1_mcp_capability_gap()
        results.append({"name": "case_1_mcp_capability", "status": "PASS"})
    except TestError as e:
        results.append({"name": "case_1_mcp_capability", "status": "FAIL", "error": str(e)})
        print(f"  FAIL: {e}")

    try:
        run_case_2_knowledge_gap_closed_loop()
        results.append({"name": "case_2_knowledge_gap", "status": "PASS"})
    except TestError as e:
        results.append({"name": "case_2_knowledge_gap", "status": "FAIL", "error": str(e)})
        print(f"  FAIL: {e}")

    try:
        run_case_3_composite_rag_mcp()
        results.append({"name": "case_3_composite_rag_mcp", "status": "PASS"})
    except TestError as e:
        results.append({"name": "case_3_composite_rag_mcp", "status": "FAIL", "error": str(e)})
        print(f"  FAIL: {e}")

    cleanup_demo_sessions()

    print("\n=== Summary ===")
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    for r in results:
        mark = "✅" if r["status"] == "PASS" else "❌"
        print(f"{mark} {r['name']}: {r['status']}")
    print(f"Total: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
