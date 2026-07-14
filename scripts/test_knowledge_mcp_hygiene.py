"""
End-to-end acceptance tests for fix/knowledge-mcp-console-hygiene.

Run against the deployed API (default NAS container localhost:8000):
    python scripts/test_knowledge_mcp_hygiene.py

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


def patch(path: str, json_body: Optional[Dict[str, Any]] = None) -> Any:
    r = api("PATCH", path, json=json_body)
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


def list_mcp_servers() -> List[Dict[str, Any]]:
    return get("/api/mcp-servers").get("mcp_servers", [])


def list_agents() -> List[Dict[str, Any]]:
    return get("/api/agents").get("agents", [])


def find_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    for a in list_agents():
        if a.get("agent_id") == agent_id:
            return a
    return None


def get_retrieval_settings() -> Dict[str, Any]:
    return get("/api/knowledge/retrieval-settings").get("retrieval_settings", {})


def set_retrieval_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    return post("/api/knowledge/retrieval-settings", settings).get("retrieval_settings", {})


def debug_retrieval(query: str, **overrides) -> Dict[str, Any]:
    body = {
        "query": query,
        "top_k": 5,
        "keyword_weight": 0.3,
        "semantic_weight": 0.7,
        "rrf_k": 60,
        "enable_rerank": False,
        "score_threshold": None,
        "context_threshold": 0.0,
    }
    body.update(overrides)
    return post("/api/retrieval/debug", body)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health():
    print("\n=== Health check ===")
    r = api("GET", "/api/agents")
    record("Health check /api/agents", r.status_code == 200, {"status": r.status_code})


def test_mcp_hygiene():
    print("\n=== MCP server hygiene ===")
    servers = list_mcp_servers()
    names = [s.get("name") for s in servers]
    test_names = [n for n in names if "test" in n.lower()]
    canonical = {"weather-server", "workorder-server", "calendar-server"}
    kept = [n for n in names if n in canonical]
    passed = (
        len(servers) == 3
        and set(kept) == canonical
        and not test_names
        and all(s.get("enabled") for s in servers)
    )
    record(
        "MCP server hygiene",
        passed,
        {
            "server_names": names,
            "test_servers_found": test_names,
            "count": len(servers),
        },
    )


def test_mcp_discovery():
    print("\n=== MCP discovery ===")
    servers = list_mcp_servers()
    evidence: Dict[str, Any] = {}
    passed = True
    for s in servers:
        name = s.get("name")
        # Trigger real tool discovery and use the cached tool list.
        try:
            discovered = post(f"/api/mcp-servers/{s.get('id')}/discover")
            tools = discovered.get("discovered", [])
        except Exception as e:
            tools = []
            evidence[name + "_discover_error"] = str(e)
        evidence[name] = {
            "tool_count": len(tools),
            "tools": [
                {
                    "name": t.get("name"),
                    "description": t.get("description"),
                    "input_schema": t.get("input_schema"),
                }
                for t in tools
            ],
        }
        if len(tools) == 0:
            passed = False
    record("MCP discovery", passed, evidence)


def test_agent_bindings():
    print("\n=== Agent MCP bindings ===")
    maintenance = find_agent("maintenance")
    customer_service = find_agent("customer_service")
    evidence: Dict[str, Any] = {}
    if maintenance:
        evidence["maintenance"] = maintenance.get("available_mcp_tools", [])
    if customer_service:
        evidence["customer_service"] = customer_service.get("available_mcp_tools", [])

    maint_ok = maintenance and set(maintenance.get("available_mcp_tools", [])) >= {
        "weather-server",
        "workorder-server",
    }
    cs_ok = customer_service and "calendar-server" in customer_service.get("available_mcp_tools", [])
    record("Agent MCP bindings", maint_ok and cs_ok, evidence)


def test_rag_debug_stages():
    print("\n=== RAG debug four-stage visibility ===")
    data = debug_retrieval("YIAI-RAG-927，楼道飞线给电动车充电允许吗？")
    # The debug endpoint returns the four RAG stages at the top level.
    stages = {
        "keyword_results": bool(data.get("keyword_results")),
        "semantic_results": bool(data.get("semantic_results")),
        "fused_results": bool(data.get("fused_results")),
        "results": bool(data.get("results")),
    }
    missing = [k for k, v in stages.items() if not v]
    final_titles = [r.get("doc_title") for r in (data.get("results") or [])]
    record(
        "RAG debug four-stage visibility",
        len(missing) == 0,
        {"missing_stages": missing, "final_titles": final_titles},
    )


def test_chat_rag_citations():
    print("\n=== Chat RAG citations ===")
    set_retrieval_settings({"top_k": 5})
    res = chat_sse("YIAI-RAG-927，楼道飞线给电动车充电允许吗？")
    done = res.get("done") or {}
    citations = done.get("citations", [])
    rule_hit = any(
        "飞线充电" in (c.get("content", "") + c.get("doc_title", ""))
        for c in citations
    )
    price_hit = any(
        "20元" in (c.get("content", "") + c.get("doc_title", ""))
        for c in citations
    )
    passed = len(citations) > 0 and rule_hit and price_hit
    record(
        "Chat RAG citations (Top-K=5)",
        passed,
        {
            "citation_count": len(citations),
            "rule_hit": rule_hit,
            "price_hit": price_hit,
            "citations": [
                {
                    "doc_id": c.get("doc_id"),
                    "doc_title": c.get("doc_title"),
                    "chunk_index": c.get("chunk_index"),
                    "content_snippet": c.get("content", "")[:200],
                }
                for c in citations
            ],
        },
    )


def test_top_k_consistency():
    print("\n=== Top-K consistency ===")
    set_retrieval_settings({"top_k": 1})
    res1 = chat_sse("YIAI-RAG-927，电动车充电")
    count1 = len((res1.get("done") or {}).get("citations", []))

    set_retrieval_settings({"top_k": 5})
    res5 = chat_sse("YIAI-RAG-927，电动车充电", session_id=None)
    count5 = len((res5.get("done") or {}).get("citations", []))

    passed = count1 <= 1 and count5 >= 2
    record(
        "Top-K consistency (1 vs 5)",
        passed,
        {"topk1_count": count1, "topk5_count": count5},
    )


def _find_tool_call(done: Dict[str, Any], predicate) -> bool:
    for tc in done.get("tool_calls", []):
        name = tc.get("tool_name", "")
        if predicate(name):
            return True
    return False


def test_weather_tool_call():
    print("\n=== Weather MCP tool call ===")
    query = "明天暴雨，我想预约维修师傅上门看看屋顶漏水，能先查一下上海天气吗？"
    res = chat_sse(query, session_id=f"hygiene-weather-{uuid.uuid4().hex[:8]}")
    done = res.get("done") or {}
    has_weather = _find_tool_call(done, lambda n: "weather" in n.lower() or n == "get_current_weather")
    passed = (
        done.get("route_intent") == "maintenance"
        and done.get("current_agent") == "维修 Agent"
        and has_weather
    )
    record(
        "Weather MCP tool call",
        passed,
        {
            "route_intent": done.get("route_intent"),
            "current_agent": done.get("current_agent"),
            "tool_calls": done.get("tool_calls", []),
        },
    )


def test_workorder_tool_call():
    print("\n=== Workorder MCP tool call ===")
    query = "我最近报修了几个工单？帮我统计一下待处理的维修工单数量。"
    res = chat_sse(query, session_id=f"hygiene-workorder-{uuid.uuid4().hex[:8]}")
    done = res.get("done") or {}
    has_workorder = _find_tool_call(
        done,
        lambda n: "work_order" in n.lower() or "workorder" in n.lower() or "count_work_orders" in n,
    )
    passed = (
        done.get("route_intent") == "maintenance"
        and done.get("current_agent") == "维修 Agent"
        and has_workorder
    )
    record(
        "Workorder MCP tool call",
        passed,
        {
            "route_intent": done.get("route_intent"),
            "current_agent": done.get("current_agent"),
            "tool_calls": done.get("tool_calls", []),
        },
    )


def test_calendar_tool_call():
    print("\n=== Calendar MCP tool call ===")
    query = "今天是几号？我想知道当前日期，以及下周三还有几天。"
    res = chat_sse(query, session_id=f"hygiene-calendar-{uuid.uuid4().hex[:8]}")
    done = res.get("done") or {}
    has_calendar = _find_tool_call(
        done,
        lambda n: "date" in n.lower() or "calendar" in n.lower() or "add_days" in n,
    )
    passed = has_calendar and len(done.get("tool_calls", [])) > 0
    record(
        "Calendar MCP tool call",
        passed,
        {
            "route_intent": done.get("route_intent"),
            "current_agent": done.get("current_agent"),
            "tool_calls": done.get("tool_calls", []),
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    test_health()
    test_mcp_hygiene()
    test_mcp_discovery()
    test_agent_bindings()
    test_rag_debug_stages()
    test_chat_rag_citations()
    test_top_k_consistency()
    test_weather_tool_call()
    test_workorder_tool_call()
    test_calendar_tool_call()

    passed = sum(1 for t in tests if t["passed"])
    total = len(tests)
    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": BASE_URL,
        "tests": tests,
        "summary": {"passed": passed, "total": total, "success": passed == total},
    }
    path = "/tmp/knowledge_mcp_hygiene_evidence.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n{passed}/{total} passed. Evidence written to {path}")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
