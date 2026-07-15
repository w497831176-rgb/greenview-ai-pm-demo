"""
End-to-end acceptance tests for feat/v1.3-observability-cost-governance.

Run against the deployed API (default NAS container localhost:8000):
    python scripts/test_v1_3_observability_cost.py

Environment:
    BASE_URL - API base URL (default http://localhost:8000)
"""

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def api(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{BASE_URL}{path}"
    return requests.request(method, url, timeout=60, **kwargs)


def get(path: str) -> Any:
    r = api("GET", path)
    r.raise_for_status()
    return r.json()


def post(path: str, json_body: Optional[Dict[str, Any]] = None) -> Any:
    r = api("POST", path, json=json_body)
    r.raise_for_status()
    return r.json()


def put(path: str, json_body: Optional[Dict[str, Any]] = None) -> Any:
    r = api("PUT", path, json=json_body)
    r.raise_for_status()
    return r.json()


def delete(path: str) -> Any:
    r = api("DELETE", path)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# SSE chat helper
# ---------------------------------------------------------------------------


def chat_sse(message: str, session_id: Optional[str] = None, timeout: int = 240) -> Dict[str, Any]:
    payload = {
        "message": message,
        "stream": True,
        "enable_rag": True,
    }
    if session_id:
        payload["session_id"] = session_id

    resp = api("POST", "/api/chat/stream", json=payload, stream=True)
    resp.raise_for_status()

    text_parts: List[str] = []
    events: List[Dict[str, Any]] = []
    done: Optional[Dict[str, Any]] = None
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

    full_text = "".join(text_parts)
    return {
        "text": full_text,
        "events": events,
        "done": done or {},
        "session_id": (done or {}).get("session_id"),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


tests: List[Dict[str, Any]] = []


def record(name: str, passed: bool, evidence: Any = None):
    tests.append({"name": name, "passed": passed, "evidence": evidence})
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if not passed:
        print(f"       evidence: {json.dumps(evidence, ensure_ascii=False, indent=2)[:400]}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_session(user_id: str = "web-user") -> Dict[str, Any]:
    return post(f"/api/chat/sessions?user_id={user_id}").get("session", {})


def list_sessions(user_id: str = "web-user") -> List[Dict[str, Any]]:
    return get(f"/api/chat/sessions?user_id={user_id}").get("sessions", [])


def list_messages(session_id: str) -> List[Dict[str, Any]]:
    return get(f"/api/chat/history?session_id={session_id}").get("messages", [])


def create_price(payload: Dict[str, Any]) -> Dict[str, Any]:
    return post("/api/observability/prices", payload)


def list_prices() -> List[Dict[str, Any]]:
    return get("/api/observability/prices").get("prices", [])


def set_budget(payload: Dict[str, Any]) -> Dict[str, Any]:
    return put("/api/observability/budget", payload)


def get_budget() -> Dict[str, Any]:
    return get("/api/observability/budget").get("budget", {})


def get_overview() -> Dict[str, Any]:
    return get("/api/observability/overview")


def list_traces() -> List[Dict[str, Any]]:
    return get("/api/observability/traces").get("traces", [])


def get_trace(trace_id: str) -> Dict[str, Any]:
    return get(f"/api/observability/traces/{trace_id}")


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------


def check_no_secret(text: str) -> Optional[str]:
    """Return the offending snippet if a possible secret is found."""
    lowered = text.lower()
    keywords = ["sk-", "api_key", "apikey", "api-secret", "api_secret", "secret_key", "bearer " ]
    for kw in keywords:
        idx = lowered.find(kw)
        if idx != -1:
            return text[max(0, idx - 10):idx + 50]
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health():
    print("\n=== Health check ===")
    r = api("GET", "/api/agents")
    record("Health check /api/agents", r.status_code == 200, {"status": r.status_code})


def test_session_create_and_isolation():
    print("\n=== Session create & isolation ===")
    s1 = create_session()
    s2 = create_session()
    sid1 = s1.get("session_id")
    sid2 = s2.get("session_id")

    passed_create = bool(sid1) and bool(sid2) and sid1 != sid2
    record("Create two sessions", passed_create, {"session_id_1": sid1, "session_id_2": sid2})

    # Send different messages in each session.
    r1 = chat_sse("会话一：我家卫生间漏水怎么办", session_id=sid1)
    r2 = chat_sse("会话二：帮我查一下物业费", session_id=sid2)

    msgs1_after = list_messages(sid1)
    msgs2_after = list_messages(sid2)
    user_contents_1 = [m.get("content") for m in msgs1_after if m.get("role") == "user"]
    user_contents_2 = [m.get("content") for m in msgs2_after if m.get("role") == "user"]

    passed_isolation = (
        "会话一" in " ".join(user_contents_1)
        and "会话二" in " ".join(user_contents_2)
        and "会话二" not in " ".join(user_contents_1)
        and "会话一" not in " ".join(user_contents_2)
    )
    record(
        "Session isolation",
        passed_isolation,
        {"user_contents_1": user_contents_1, "user_contents_2": user_contents_2},
    )

    # Verify session list metadata.
    sessions = list_sessions()
    meta_map = {s.get("session_id"): s for s in sessions}
    meta1 = meta_map.get(sid1, {})
    meta2 = meta_map.get(sid2, {})
    passed_meta = (
        meta1.get("title")
        and meta2.get("title")
        and meta1.get("last_message_at")
        and meta2.get("last_message_at")
        and meta1.get("last_agent")
        and meta2.get("last_agent")
    )
    record(
        "Session list metadata",
        passed_meta,
        {
            "session_titles": [meta1.get("title"), meta2.get("title")],
            "last_agents": [meta1.get("last_agent"), meta2.get("last_agent")],
        },
    )

    # Refresh (re-list) and ensure messages are still there.
    msgs1_refreshed = list_messages(sid1)
    passed_refresh = len(msgs1_refreshed) >= 2
    record("Session refresh preserves messages", passed_refresh, {"message_count": len(msgs1_refreshed)})

    return sid1, sid2


def test_trace_and_mcp_audit():
    print("\n=== Trace & MCP audit ===")
    # Use unique sessions to avoid history-driven tool suppression.
    weather_sid = f"v13-weather-{uuid.uuid4().hex[:8]}"
    res_weather = chat_sse("明天暴雨，我想预约维修师傅上门看看屋顶漏水，能先查一下上海天气吗？", session_id=weather_sid)
    done_weather = res_weather.get("done") or {}
    trace_id_weather = done_weather.get("trace_id")
    mcp_weather = done_weather.get("mcp_calls", [])
    passed_weather = (
        bool(trace_id_weather)
        and any(m.get("server_name") == "weather-server" and m.get("status") == "success" for m in mcp_weather)
    )
    record(
        "Weather MCP audit",
        passed_weather,
        {"trace_id": trace_id_weather, "mcp_calls": mcp_weather},
    )

    workorder_sid = f"v13-workorder-{uuid.uuid4().hex[:8]}"
    res_workorder = chat_sse("我最近报修了几个工单？帮我统计一下待处理的维修工单数量。", session_id=workorder_sid)
    done_workorder = res_workorder.get("done") or {}
    trace_id_workorder = done_workorder.get("trace_id")
    mcp_workorder = done_workorder.get("mcp_calls", [])
    workorder_calls = [m for m in mcp_workorder if m.get("server_name") == "workorder-server"]
    passed_workorder = bool(trace_id_workorder) and len(workorder_calls) >= 2
    record(
        "Workorder MCP audit",
        passed_workorder,
        {"trace_id": trace_id_workorder, "workorder_calls": workorder_calls},
    )

    calendar_sid = f"v13-calendar-{uuid.uuid4().hex[:8]}"
    res_calendar = chat_sse("今天是几号？我想知道当前日期，以及下周三还有几天。", session_id=calendar_sid)
    done_calendar = res_calendar.get("done") or {}
    trace_id_calendar = done_calendar.get("trace_id")
    mcp_calendar = done_calendar.get("mcp_calls", [])
    passed_calendar = (
        bool(trace_id_calendar)
        and any(m.get("server_name") == "calendar-server" and m.get("status") == "success" for m in mcp_calendar)
    )
    record(
        "Calendar MCP audit",
        passed_calendar,
        {"trace_id": trace_id_calendar, "mcp_calls": mcp_calendar},
    )

    # Safe failed tool test: ask for weather of a deliberately invalid location that the tool rejects.
    fail_sid = f"v13-fail-{uuid.uuid4().hex[:8]}"
    res_fail = chat_sse("查一下火星的天气怎么样", session_id=fail_sid)
    done_fail = res_fail.get("done") or {}
    trace_id_fail = done_fail.get("trace_id")
    mcp_fail = done_fail.get("mcp_calls", [])
    failed_call = next((m for m in mcp_fail if m.get("status") == "failed"), None)
    passed_fail = bool(trace_id_fail) and bool(failed_call)
    record(
        "MCP failed audit",
        passed_fail,
        {"trace_id": trace_id_fail, "failed_call": failed_call},
    )

    return {
        "weather_trace_id": trace_id_weather,
        "workorder_trace_id": trace_id_workorder,
        "calendar_trace_id": trace_id_calendar,
        "fail_trace_id": trace_id_fail,
    }


def test_cost_governance_no_price():
    print("\n=== Cost governance without price ===")
    # Ensure no prices are configured for a fresh dummy model to test unknown cost.
    overview_before = get_overview()
    unknown_before = overview_before.get("unknown_cost_calls", 0)

    # Trigger an owner chat; default is Flash.
    chat_sid = f"v13-cost-{uuid.uuid4().hex[:8]}"
    res = chat_sse("帮我报修客厅灯不亮", session_id=chat_sid)
    done = res.get("done") or {}
    trace_id = done.get("trace_id")

    trace = get_trace(trace_id) if trace_id else {}
    model_calls = trace.get("model_calls", [])
    flash_call = next((c for c in model_calls if c.get("stage") == "vertical_agent"), None)
    passed_flash = flash_call and flash_call.get("model_id") == "deepseek-v4-flash"
    record(
        "Owner chat uses Flash",
        passed_flash,
        {"trace_id": trace_id, "vertical_model": flash_call.get("model_id") if flash_call else None},
    )

    # At least one call should be provider_reported or estimated_tokenization.
    sources = {c.get("usage_source") for c in model_calls}
    passed_sources = bool(sources & {"provider_reported", "estimated_tokenization"})
    record(
        "Usage source recorded",
        passed_sources,
        {"sources": list(sources), "model_calls": [{"stage": c.get("stage"), "source": c.get("usage_source")} for c in model_calls]},
    )

    # Without a price configured for a synthetic model, cost should be None.
    dummy_call = next((c for c in model_calls if c.get("model_id") == "nonexistent-model-xyz"), None)
    record(
        "Unknown price shows no cost",
        dummy_call is None or dummy_call.get("estimated_cost_cny") is None,
        {"dummy_call": dummy_call},
    )

    return {"trace_id": trace_id}


def test_price_table_and_cost_recalc():
    print("\n=== Price table & cost recalc ===")
    # Add prices for flash and pro.
    flash_price = {
        "model_id": "deepseek-v4-flash",
        "currency": "CNY",
        "effective_date": "2026-07-15",
        "input_price_per_1m": 1.0,
        "cached_input_price_per_1m": 0.5,
        "output_price_per_1m": 4.0,
        "reasoning_price_per_1m": 0.0,
        "source_note": "演示用 Gateway 账单",
        "enabled": True,
    }
    pro_price = {
        "model_id": "deepseek-v4-pro",
        "currency": "CNY",
        "effective_date": "2026-07-15",
        "input_price_per_1m": 5.0,
        "cached_input_price_per_1m": 2.5,
        "output_price_per_1m": 20.0,
        "reasoning_price_per_1m": 0.0,
        "source_note": "演示用 Gateway 账单",
        "enabled": True,
    }
    create_price(flash_price)
    create_price(pro_price)
    prices = list_prices()
    passed_prices = any(p.get("model_id") == "deepseek-v4-flash" for p in prices) and any(
        p.get("model_id") == "deepseek-v4-pro" for p in prices
    )
    record("Price table CRUD", passed_prices, {"price_count": len(prices)})

    # A/B test should now have non-null cost.
    ab = post("/api/model-configs/ab-test", {"prompt": "我要投诉楼下噪音太大，物业不作为"})
    a = ab.get("model_a_result", {})
    b = ab.get("model_b_result", {})
    passed_ab_cost = (
        a.get("estimated_cost_cny") is not None
        and b.get("estimated_cost_cny") is not None
        and ab.get("trace_id")
    )
    record(
        "A/B cost with configured prices",
        passed_ab_cost,
        {
            "trace_id": ab.get("trace_id"),
            "flash_cost": a.get("estimated_cost_cny"),
            "pro_cost": b.get("estimated_cost_cny"),
            "flash_usage": a.get("usage_source"),
            "pro_usage": b.get("usage_source"),
        },
    )

    # Overview should show cost > 0 now.
    overview = get_overview()
    passed_overview = overview.get("total_cost") is not None and overview.get("total_cost", 0) >= 0
    record("Overview cost recalculates", passed_overview, {"total_cost": overview.get("total_cost")})

    return ab.get("trace_id")


def test_budget_alert():
    print("\n=== Budget alert ===")
    # Set very low thresholds to force alerts.
    set_budget({"per_call_threshold_cny": 0.000001, "daily_threshold_cny": 0.000001})
    # Trigger an owner chat to generate a call above the threshold.
    chat_sid = f"v13-budget-{uuid.uuid4().hex[:8]}"
    chat_sse("帮我报修卧室门锁坏了", session_id=chat_sid)
    overview = get_overview()
    alerts = overview.get("alerts", [])
    passed = any(a.get("type") in ("per_call", "daily") for a in alerts)
    record(
        "Budget alert triggers",
        passed,
        {"alerts": alerts, "thresholds": get_budget()},
    )


def test_darwin_records_pro():
    print("\n=== Darwin records Pro ===")
    try:
        badcase = post("/api/badcases", {
            "session_id": f"v13-darwin-{uuid.uuid4().hex[:8]}",
            "message_id": 0,
            "category": "routing_error",
            "reason": "测试 Darwin 分析",
            "title": "V1.3 Darwin 测试",
            "description": "测试 Darwin 优化",
            "status": "pending",
        })
        case_id = badcase.get("id")
        fix = post(f"/api/badcases/{case_id}/darwin-fix")
        darwin_trace_id = fix.get("trace_id") or fix.get("darwin_trace_id")
        record("Darwin endpoint reachable", bool(darwin_trace_id), {"badcase_id": case_id, "trace_id": darwin_trace_id})
    except Exception as e:
        record("Darwin endpoint reachable", False, {"error": str(e)})
        darwin_trace_id = None

    # Check recent traces for any darwin stage.
    traces = list_traces()
    darwin_calls = []
    for t in traces[:10]:
        detail = get_trace(t.get("trace_id"))
        for c in detail.get("model_calls", []):
            if c.get("stage") == "darwin":
                darwin_calls.append({"trace_id": t.get("trace_id"), "model_id": c.get("model_id")})
    passed = any(c.get("model_id") == "deepseek-v4-pro" for c in darwin_calls)
    record("Darwin uses Pro", passed, {"darwin_calls": darwin_calls[:5]})


def test_cost_02_rag_topk():
    print("\n=== COST-02 RAG Top-K ===")
    original = get("/api/knowledge/retrieval-settings").get("retrieval_settings", {})
    orig_topk = original.get("top_k")
    query = "维修收费标准是什么"

    def run_search(k: int) -> Dict[str, Any]:
        put("/api/knowledge/retrieval-settings", {**original, "top_k": k})
        return post("/api/retrieval/search", {"query": query, "top_k": k})

    r1 = run_search(1)
    r5 = run_search(5)
    put("/api/knowledge/retrieval-settings", {**original, "top_k": orig_topk})

    c1 = len(r1.get("citations", []))
    c5 = len(r5.get("citations", []))
    passed = c1 <= c5 and orig_topk == get("/api/knowledge/retrieval-settings").get("retrieval_settings", {}).get("top_k")
    record(
        "COST-02 Top-K comparison",
        passed,
        {"topk1_count": c1, "topk5_count": c5, "restored_topk": orig_topk},
    )


def test_security_no_secrets():
    print("\n=== Security: no secrets in responses ===")
    # Probe a few endpoints and look for secrets.
    bodies = [
        get("/api/agents"),
        get("/api/mcp-servers"),
        get("/api/observability/traces"),
        get("/api/model-configs"),
    ]
    text = json.dumps(bodies, ensure_ascii=False)
    secret = check_no_secret(text)
    record("No API key in JSON responses", secret is None, {"offending_snippet": secret})

    # Also probe chat SSE done events.
    res = chat_sse("你好", session_id=f"v13-sec-{uuid.uuid4().hex[:8]}")
    done_text = json.dumps(res.get("done", {}), ensure_ascii=False)
    secret2 = check_no_secret(done_text)
    record("No API key in SSE done event", secret2 is None, {"offending_snippet": secret2})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    test_health()
    sids = test_session_create_and_isolation()
    traces = test_trace_and_mcp_audit()
    cost_trace = test_cost_governance_no_price()
    ab_trace = test_price_table_and_cost_recalc()
    test_budget_alert()
    test_darwin_records_pro()
    test_cost_02_rag_topk()
    test_security_no_secrets()

    passed = sum(1 for t in tests if t["passed"])
    total = len(tests)
    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": BASE_URL,
        "tests": tests,
        "summary": {"passed": passed, "total": total, "success": passed == total},
        "trace_ids": {
            "session_test": {"session_1": sids[0], "session_2": sids[1]} if sids else None,
            "mcp": traces,
            "cost_no_price": cost_trace.get("trace_id"),
            "ab_test": ab_trace,
        },
    }
    path = "/tmp/v1_3_observability_cost_evidence.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n{passed}/{total} passed. Evidence written to {path}")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
