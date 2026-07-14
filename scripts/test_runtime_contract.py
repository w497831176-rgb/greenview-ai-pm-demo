"""
Runtime contract acceptance tests for fix/agent-skill-mcp-rag-runtime-contract.

Run against the NAS demo-os-api container:
    python scripts/test_runtime_contract.py

Environment:
    BASE_URL - API base URL (default http://localhost:8000)
"""

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")

# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------

def api(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{BASE_URL}{path}"
    resp = requests.request(method, url, timeout=60, **kwargs)
    return resp


def get(path: str):
    r = api("GET", path)
    r.raise_for_status()
    return r.json()


def post(path: str, json_body: Optional[Dict[str, Any]] = None):
    r = api("POST", path, json=json_body)
    r.raise_for_status()
    return r.json()


def put(path: str, json_body: Optional[Dict[str, Any]] = None):
    r = api("PUT", path, json=json_body)
    r.raise_for_status()
    return r.json()


def patch(path: str, json_body: Optional[Dict[str, Any]] = None):
    r = api("PATCH", path, json=json_body)
    r.raise_for_status()
    return r.json()


# -----------------------------------------------------------------------------
# SSE chat helper
# -----------------------------------------------------------------------------

def chat_sse(message: str, session_id: Optional[str] = None, timeout: int = 180) -> Dict[str, Any]:
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
        "session_id": done.get("session_id") if done else None,
    }


# -----------------------------------------------------------------------------
# Assertions / reporting
# -----------------------------------------------------------------------------

results: List[Dict[str, Any]] = []


def record(name: str, passed: bool, detail: str = ""):
    results.append({"name": name, "passed": passed, "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")


def ensure(passed: bool, name: str, detail: str = ""):
    record(name, passed, detail)
    return passed


# -----------------------------------------------------------------------------
# Agent helpers
# -----------------------------------------------------------------------------

def list_agents() -> List[Dict[str, Any]]:
    return get("/api/agents").get("agents", [])


def find_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    for a in list_agents():
        if a.get("agent_id") == agent_id:
            return a
    return None


def update_agent_bindings(agent_id: str, skills: List[str] = None, tools: List[str] = None):
    body: Dict[str, Any] = {}
    if skills is not None:
        body["available_skills"] = skills
    if tools is not None:
        body["available_mcp_tools"] = tools
    return put(f"/api/agents/{agent_id}", body)


# -----------------------------------------------------------------------------
# Skill helpers
# -----------------------------------------------------------------------------

def ensure_skill(name: str, trigger: str, instructions: str) -> Dict[str, Any]:
    skills = get("/api/skills").get("skills", [])
    for s in skills:
        if s.get("name") == name:
            put(f"/api/skills/{s['id']}", {
                "name": name,
                "description": s.get("description", ""),
                "instructions": instructions,
                "category": s.get("category", "业务 Skill"),
                "enabled": True,
                "trigger_condition": trigger,
                "model_id": s.get("model_id"),
            })
            return get(f"/api/skills/{s['id']}")["skill"]
    created = post("/api/skills", {
        "name": name,
        "description": "验收专用 Skill",
        "instructions": instructions,
        "category": "业务 Skill",
        "enabled": True,
        "trigger_condition": trigger,
        "model_id": None,
    })
    return created["skill"]


# -----------------------------------------------------------------------------
# MCP helpers
# -----------------------------------------------------------------------------

def list_mcp_servers() -> List[Dict[str, Any]]:
    return get("/api/mcp-servers").get("mcp_servers", [])


def find_mcp_server(name: str) -> Optional[Dict[str, Any]]:
    for s in list_mcp_servers():
        if s.get("name") == name:
            return s
    return None


# -----------------------------------------------------------------------------
# Knowledge helpers
# -----------------------------------------------------------------------------

def ensure_knowledge_doc(title: str, content: str, category: str = "管理规定") -> Dict[str, Any]:
    docs = get("/api/knowledge/docs").get("knowledge_docs", [])
    for d in docs:
        if d.get("title") == title:
            put(f"/api/knowledge/docs/{d['id']}", {
                "title": title,
                "content": content,
                "category": category,
                "chunk_size": 300,
                "chunk_overlap": 32,
                "split_strategy": "auto",
            })
            post(f"/api/knowledge/docs/{d['id']}/reindex")
            return get(f"/api/knowledge/docs/{d['id']}")["knowledge_doc"]
    created = post("/api/knowledge/docs", {
        "title": title,
        "content": content,
        "category": category,
        "chunk_size": 300,
        "chunk_overlap": 32,
        "split_strategy": "auto",
    })
    doc_id = created["knowledge_doc"]["id"]
    post(f"/api/knowledge/docs/{doc_id}/reindex")
    return get(f"/api/knowledge/docs/{doc_id}")["knowledge_doc"]


def debug_retrieval(query: str) -> Dict[str, Any]:
    return post("/api/retrieval/debug", {
        "query": query,
        "top_k": 5,
        "keyword_weight": 0.3,
        "semantic_weight": 0.7,
        "rrf_k": 60,
        "enable_rerank": False,
        "score_threshold": None,
        "context_threshold": 0.0,
    })


# -----------------------------------------------------------------------------
# Main test flow
# -----------------------------------------------------------------------------

def test_agent_cleanup():
    print("\n=== Agent cleanup ===")
    agents = list_agents()
    print(f"Total agents: {len(agents)}")
    print("| id | agent_id | name | category | is_router | skills | mcp_tools |")
    for a in agents:
        print(f"| {a.get('id')} | {a.get('agent_id')} | {a.get('name')} | {a.get('category')} | {a.get('is_router')} | {a.get('available_skills')} | {a.get('available_mcp_tools')} |")

    canonical_ids = {"router", "maintenance", "billing", "complaint", "customer_service"}
    actual_ids = {a.get("agent_id") for a in agents}
    ensure(actual_ids == canonical_ids, "Only canonical 5 agents exist", f"actual={actual_ids}")

    for bad in ["Temp Test Agent", "测试用路由 Agent", "测试用维修 Agent"]:
        if any(bad in (a.get("name") or "") for a in agents):
            ensure(False, f"No temp/test agent '{bad}' remains")
        else:
            ensure(True, f"No temp/test agent '{bad}' remains")

    router = find_agent("router")
    ensure(router is not None and router.get("is_router") is True, "Router agent is router", router)


def test_agent_binding_echo():
    print("\n=== Agent binding echo ===")
    maintenance = find_agent("maintenance")
    billing = find_agent("billing")
    ensure(maintenance is not None, "Maintenance agent exists")
    ensure(billing is not None, "Billing agent exists")

    # Maintenance should have 维修工单处理 skill after setup.
    maint_skills = maintenance.get("available_skills", [])
    echo_ok = "维修工单处理" in maint_skills
    ensure(echo_ok, "Maintenance agent echoes 维修工单处理 binding", maint_skills)

    # Billing should have 费用绑定探针.
    billing_skills = billing.get("available_skills", [])
    echo_ok = "费用绑定探针" in billing_skills
    ensure(echo_ok, "Billing agent echoes 费用绑定探针 binding", billing_skills)


def test_skill_per_agent():
    print("\n=== Skill per agent ===")
    fee_skill = ensure_skill(
        "费用绑定探针",
        "YIAI-ONLY-927",
        "当且仅当本 Skill 被注入时，回答首行输出：\n【SKILL-HIT:YIAI-ONLY-927】\n未注入时不得输出该标记。",
    )
    maint_skill = ensure_skill(
        "维修工单处理",
        "报修、维修、漏水、渗水、故障、损坏、上门、修",
        "你是维修处理助手。收到业主报修时，先安抚并确认地址、问题描述、紧急程度，再生成维修工单建议。",
    )

    # Bindings: fee only to billing; maintenance skill only to maintenance.
    update_agent_bindings("billing", skills=["费用绑定探针"], tools=[])
    update_agent_bindings("maintenance", skills=["维修工单处理"], tools=[])

    # Test A: billing query -> fee skill hit.
    res_a = chat_sse("YIAI-ONLY-927，请问本月物业费如何缴纳？")
    text_a = res_a["text"]
    done_a = res_a["done"]
    hit_a = "【SKILL-HIT:YIAI-ONLY-927】" in text_a
    activated_a = done_a.get("activated_skills") or []
    skill_hit_a = "费用绑定探针" in activated_a
    ensure(hit_a and skill_hit_a, "Skill test A: fee skill injected in billing chat", {
        "activated_skills": activated_a,
        "first_line": text_a[:80],
        "current_agent": done_a.get("current_agent"),
        "route_intent": done_a.get("route_intent"),
    })

    # Test B: maintenance query -> no fee skill hit, maintenance skill allowed.
    res_b = chat_sse("YIAI-ONLY-927，厨房漏水需要报修。")
    text_b = res_b["text"]
    done_b = res_b["done"]
    no_hit = "【SKILL-HIT:YIAI-ONLY-927】" not in text_b
    no_skill = "费用绑定探针" not in (done_b.get("activated_skills") or [])
    ensure(no_hit and no_skill, "Skill test B: fee skill not injected in maintenance chat", {
        "activated_skills": done_b.get("activated_skills"),
        "first_line": text_b[:80],
        "current_agent": done_b.get("current_agent"),
        "route_intent": done_b.get("route_intent"),
    })


def test_mcp_per_agent():
    print("\n=== MCP per agent ===")
    weather = find_mcp_server("weather-server")
    if weather is None:
        ensure(False, "weather-server MCP server exists")
        return
    if not weather.get("enabled"):
        patch(f"/api/mcp-servers/{weather['id']}", {"enabled": True})

    # Bind weather-server to maintenance.
    update_agent_bindings("maintenance", skills=["维修工单处理"], tools=["weather-server"])

    query = "明天有暴雨，外墙渗水报修前我该做什么？我是深圳福田区。请查询天气后给建议。"
    res = chat_sse(query)
    done = res["done"]
    tool_calls = done.get("tool_calls", [])
    has_weather = any("weather" in (tc.get("tool_name") or "").lower() for tc in tool_calls)
    ensure(len(tool_calls) > 0 and has_weather, "MCP test A: weather tool called", {
        "tool_calls": tool_calls,
        "current_agent": done.get("current_agent"),
    })

    # Reverse test: unbind weather.
    update_agent_bindings("maintenance", skills=["维修工单处理"], tools=[])
    res2 = chat_sse(query)
    done2 = res2["done"]
    no_calls = not done2.get("tool_calls")
    ensure(no_calls, "MCP test B: no tool calls after unbinding", {
        "tool_calls": done2.get("tool_calls"),
    })

    # Restore binding.
    update_agent_bindings("maintenance", skills=["维修工单处理"], tools=["weather-server"])


def test_rag_citations():
    print("\n=== RAG chunk-accurate citations ===")
    ensure_knowledge_doc(
        "YIAI-RAG-927 电动车充电临时规定",
        "一、业主只能在1号集中充电区为电动车充电，严禁在楼道、消防通道或住宅内飞线充电。\n"
        "二、发现飞线充电，物业应先劝阻并登记；屡劝不改时，按消防安全流程上报。\n"
        "三、充电桩发生故障时，业主请提交“公共设施报修”工单，物业应在24小时内受理。",
    )
    fee_doc = ensure_knowledge_doc(
        "物业费收费标准与缴纳流程",
        "第一节 物业费构成\n"
        "物业费包括公共区域保洁、秩序维护、绿化养护、设施设备运行维护等费用。\n"
        "第二节 停车费与充电管理费\n"
        "电动车充电管理费：20元/个·月。\n"
        "停车费按车位类型收取，详见缴费通知。",
    )

    # Debug retrieval to verify chunk-level stages.
    debug_a = debug_retrieval("YIAI-RAG-927，楼道飞线给电动车充电允许吗？")
    stages = ["keyword_results", "semantic_results", "fused_results", "results"]
    for stage in stages:
        arr = debug_a.get(stage)
        ensure(isinstance(arr, list), f"RAG debug stage '{stage}' exists", stage)
    final_a = debug_a.get("results", [])
    doc_titles = [r.get("doc_title") for r in final_a]
    ensure("YIAI-RAG-927 电动车充电临时规定" in doc_titles, "RAG test A final includes charging doc", doc_titles)

    res_a = chat_sse("YIAI-RAG-927，楼道飞线给电动车充电允许吗？")
    done_a = res_a["done"]
    citations_a = done_a.get("citations", [])
    # Find citation for charging doc and fee doc.
    charge_cit = next((c for c in citations_a if "YIAI-RAG-927" in (c.get("doc_title") or "")), None)
    fee_cit = next((c for c in citations_a if "物业费" in (c.get("doc_title") or "")), None)
    ensure(charge_cit is not None and charge_cit.get("chunk_index") is not None, "RAG test A citation has chunk_index", charge_cit)
    if charge_cit:
        chunk_text = (charge_cit.get("content") or "").lower()
        ensure("严禁" in chunk_text, "RAG test A charging citation points to correct chunk", charge_cit)

    # Test B: fee query targeting the charging-management fee chunk.
    res_b = chat_sse("电动车充电管理费多少钱一个月？")
    done_b = res_b["done"]
    citations_b = done_b.get("citations", [])
    fee_cit_b = next((c for c in citations_b if "物业费" in (c.get("doc_title") or "")), None)
    ensure(fee_cit_b is not None and fee_cit_b.get("chunk_index") is not None, "RAG test B citation has chunk_index", fee_cit_b)
    if fee_cit_b:
        chunk_text = (fee_cit_b.get("content") or "").lower()
        ensure("20元" in chunk_text, "RAG test B fee citation points to price chunk", fee_cit_b)


def test_brand_cleanup():
    print("\n=== Brand cleanup ===")
    docs = get("/api/knowledge/docs").get("knowledge_docs", [])
    bad = []
    for d in docs:
        for field in ("title", "content"):
            val = d.get(field) or ""
            if "绿景" in val:
                bad.append(f"doc {d['id']} field {field}")
    ensure(len(bad) == 0, "No '绿景' in knowledge docs", bad)

    agents = list_agents()
    for a in agents:
        for field in ("name", "description", "instructions"):
            val = a.get(field) or ""
            if "绿景" in val:
                bad.append(f"agent {a['agent_id']} field {field}")
    ensure(len(bad) == 0, "No '绿景' in agent fields", bad)

    skills = get("/api/skills").get("skills", [])
    for s in skills:
        for field in ("name", "description", "instructions", "trigger_condition"):
            val = s.get(field) or ""
            if "绿景" in val:
                bad.append(f"skill {s['id']} field {field}")
    ensure(len(bad) == 0, "No '绿景' in skill fields", bad)


def main():
    print(f"BASE_URL={BASE_URL}")
    # Smoke test.
    health = get("/api/agents")
    ensure(health is not None, "API reachable")

    test_agent_cleanup()
    test_agent_binding_echo()
    test_skill_per_agent()
    test_mcp_per_agent()
    test_rag_citations()
    test_brand_cleanup()

    print("\n=== Summary ===")
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"Passed {passed}/{total}")
    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
