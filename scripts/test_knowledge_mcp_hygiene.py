#!/usr/bin/env python3
"""
End-to-end acceptance for fix/knowledge-mcp-console-hygiene.

Covers:
- RAG citation contract (doc_title, doc_id, chunk_index)
- Top-K consistency (1 vs 5)
- RAG debug four-stage visibility
- MCP server hygiene (no Test Server, canonical 3 servers)
- MCP real tool calls (weather, workorder, calendar)
- Agent MCP binding correctness

Run inside the demo-os-api container against localhost:8000.
"""

import json
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

BASE = "http://localhost:8000"
EVIDENCE: Dict[str, Any] = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "tests": [],
}


def log_test(name: str, passed: bool, evidence: Dict[str, Any]) -> None:
    EVIDENCE["tests"].append({"name": name, "passed": passed, "evidence": evidence})
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")


def get_sse_done(payload: Dict[str, Any], timeout: int = 180) -> Optional[Dict[str, Any]]:
    resp = requests.post(f"{BASE}/api/chat/stream", json=payload, stream=True, timeout=timeout)
    done: Optional[Dict[str, Any]] = None
    for line in resp.iter_lines():
        if not line:
            continue
        s = line.decode("utf-8")
        if s.startswith("data:"):
            data = s[5:].strip()
            try:
                obj = json.loads(data)
                if obj.get("status") == "complete":
                    done = obj
            except Exception:
                pass
    return done


def set_top_k(top_k: int) -> Dict[str, Any]:
    settings = requests.get(f"{BASE}/api/knowledge/retrieval/settings", timeout=10).json()["retrieval_settings"]
    payload = {
        "top_k": top_k,
        "keyword_weight": settings["keyword_weight"],
        "semantic_weight": settings["semantic_weight"],
        "rrf_k": settings["rrf_k"],
        "enable_rerank": settings["enable_rerank"],
        "rerank_model": settings.get("rerank_model"),
        "score_threshold": settings["score_threshold"],
        "context_threshold": settings["context_threshold"],
    }
    r = requests.post(f"{BASE}/api/knowledge/retrieval/settings", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()["retrieval_settings"]


def test_health() -> None:
    r = requests.get(f"{BASE}/api/agents", timeout=10)
    r.raise_for_status()
    log_test("Health check /api/agents", r.status_code == 200, {"status": r.status_code})


def test_mcp_hygiene() -> None:
    r = requests.get(f"{BASE}/api/mcp-servers", timeout=10)
    r.raise_for_status()
    servers = r.json()["mcp_servers"]
    names = {s["name"] for s in servers}
    bad_names = {s["name"] for s in servers if "test" in s["name"].lower()}
    canonical = {"weather-server", "workorder-server", "calendar-server"}
    passed = (
        not bad_names
        and names == canonical
        and len(servers) == 3
        and all(s["enabled"] for s in servers)
    )
    log_test(
        "MCP server hygiene",
        passed,
        {"server_names": list(names), "test_servers_found": list(bad_names), "count": len(servers)},
    )


def test_mcp_discovery() -> None:
    r = requests.get(f"{BASE}/api/mcp-servers", timeout=10)
    servers = r.json()["mcp_servers"]
    discovery: Dict[str, Any] = {}
    all_have_tools = True
    for s in servers:
        if not s.get("enabled"):
            continue
        tools_resp = requests.get(f"{BASE}/api/mcp-servers/{s['id']}/tools", timeout=30)
        tools = tools_resp.json().get("tools", [])
        discovery[s["name"]] = {"tool_count": len(tools), "tools": tools}
        if not tools:
            all_have_tools = False
    log_test(
        "MCP discovery",
        all_have_tools and len(discovery) == 3,
        discovery,
    )


def test_agent_bindings() -> None:
    r = requests.get(f"{BASE}/api/agents", timeout=10)
    agents = {a["agent_id"]: a for a in r.json()["agents"]}
    maintenance = agents.get("maintenance", {})
    customer_service = agents.get("customer_service", {})
    maintenance_tools = set(maintenance.get("available_mcp_tools", []))
    customer_tools = set(customer_service.get("available_mcp_tools", []))
    passed = (
        maintenance_tools == {"weather-server", "workorder-server"}
        and customer_tools == {"calendar-server"}
    )
    log_test(
        "Agent MCP bindings",
        passed,
        {
            "maintenance": list(maintenance_tools),
            "customer_service": list(customer_tools),
        },
    )


def test_rag_debug() -> None:
    payload = {
        "query": "YIAI-RAG-927，楼道飞线给电动车充电允许吗？",
        "top_k": 5,
        "keyword_weight": 0.3,
        "semantic_weight": 0.7,
        "rrf_k": 60,
        "enable_rerank": False,
        "score_threshold": 0.0,
        "context_threshold": 0.2,
    }
    r = requests.post(f"{BASE}/api/retrieval/debug", json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    stages = ["keyword_results", "semantic_results", "fused_results", "final_results"]
    missing = [k for k in stages if k not in data or not isinstance(data[k], list)]
    # Expect rule doc and price doc in final results
    titles = {x.get("doc_title", "") for x in data.get("final_results", [])}
    has_rule = any("YIAI-RAG-927" in t for t in titles)
    has_price = any("物业费收费标准" in t for t in titles)
    passed = not missing and has_rule and has_price
    log_test(
        "RAG debug four-stage visibility",
        passed,
        {"missing_stages": missing, "final_titles": list(titles)[:10]},
    )


def test_chat_citations() -> None:
    payload = {
        "message": "YIAI-RAG-927，楼道飞线给电动车充电允许吗？",
        "session_id": f"rag-citation-{int(time.time())}",
        "user_id": "owner-001",
        "stream": True,
    }
    done = get_sse_done(payload)
    citations: List[Dict[str, Any]] = done.get("citations", []) if done else []
    rule_hit = any(
        "严禁在楼道" in c.get("content", "") and "YIAI-RAG-927" in c.get("doc_title", "")
        for c in citations
    )
    price_hit = any(
        "电动车充电管理费：20元" in c.get("content", "") for c in citations
    )
    passed = done is not None and rule_hit and price_hit and len(citations) >= 2
    log_test(
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
                    "content_snippet": c.get("content", "")[:120],
                }
                for c in citations
            ],
        },
    )


def test_top_k_consistency() -> bool:
    original = requests.get(f"{BASE}/api/knowledge/retrieval/settings", timeout=10).json()["retrieval_settings"]

    set_top_k(1)
    payload = {
        "message": "YIAI-RAG-927，楼道飞线给电动车充电允许吗？",
        "session_id": f"rag-topk1-{int(time.time())}",
        "user_id": "owner-001",
        "stream": True,
    }
    done1 = get_sse_done(payload)
    citations1 = done1.get("citations", []) if done1 else []

    set_top_k(5)
    payload["session_id"] = f"rag-topk5-{int(time.time())}"
    done5 = get_sse_done(payload)
    citations5 = done5.get("citations", []) if done5 else []

    # restore original if different
    if original.get("top_k") != 5:
        set_top_k(original.get("top_k", 5))

    passed = len(citations1) <= 1 and len(citations5) >= 2
    log_test(
        "Top-K consistency (1 vs 5)",
        passed,
        {"topk1_count": len(citations1), "topk5_count": len(citations5)},
    )
    return passed


def extract_tool_calls(done: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not done:
        return []
    return done.get("tool_calls", []) or []


def test_weather_tool_call() -> None:
    payload = {
        "message": "今天天气怎么样？我想安排维修",
        "session_id": f"weather-{int(time.time())}",
        "user_id": "owner-001",
        "stream": True,
    }
    done = get_sse_done(payload)
    calls = extract_tool_calls(done)
    weather_calls = [c for c in calls if c.get("tool_name") == "weather-server" or "get_current_weather" in str(c.get("function", ""))]
    passed = bool(weather_calls)
    log_test(
        "Weather MCP tool call",
        passed,
        {
            "route_intent": done.get("route_intent") if done else None,
            "current_agent": done.get("current_agent") if done else None,
            "tool_calls": calls,
        },
    )


def test_workorder_tool_call() -> None:
    payload = {
        "message": "帮我查一下当前有多少待办维修工单？",
        "session_id": f"workorder-{int(time.time())}",
        "user_id": "owner-001",
        "stream": True,
    }
    done = get_sse_done(payload)
    calls = extract_tool_calls(done)
    workorder_calls = [c for c in calls if c.get("tool_name") == "workorder-server" or "work_order" in str(c.get("function", ""))]
    passed = bool(workorder_calls)
    log_test(
        "Workorder MCP tool call",
        passed,
        {
            "route_intent": done.get("route_intent") if done else None,
            "current_agent": done.get("current_agent") if done else None,
            "tool_calls": calls,
        },
    )


def test_calendar_tool_call() -> None:
    payload = {
        "message": "今天是几号？",
        "session_id": f"calendar-{int(time.time())}",
        "user_id": "owner-001",
        "stream": True,
    }
    done = get_sse_done(payload)
    calls = extract_tool_calls(done)
    calendar_calls = [c for c in calls if c.get("tool_name") == "calendar-server" or "get_current_date" in str(c.get("function", ""))]
    passed = bool(calendar_calls)
    log_test(
        "Calendar MCP tool call",
        passed,
        {
            "route_intent": done.get("route_intent") if done else None,
            "current_agent": done.get("current_agent") if done else None,
            "tool_calls": calls,
        },
    )


def main() -> int:
    test_health()
    test_mcp_hygiene()
    test_mcp_discovery()
    test_agent_bindings()
    test_rag_debug()
    test_chat_citations()
    test_top_k_consistency()
    test_weather_tool_call()
    test_workorder_tool_call()
    test_calendar_tool_call()

    passed = sum(1 for t in EVIDENCE["tests"] if t["passed"])
    total = len(EVIDENCE["tests"])
    EVIDENCE["summary"] = {"passed": passed, "total": total, "success": passed == total}

    out_path = "/tmp/knowledge_mcp_hygiene_evidence.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(EVIDENCE, f, ensure_ascii=False, indent=2)
    print(f"\nEvidence written to {out_path}")
    print(f"Summary: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
